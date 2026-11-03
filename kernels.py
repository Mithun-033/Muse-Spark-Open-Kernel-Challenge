import torch
import triton
import triton.language as tl


@triton.jit
def qkv_gate_splitk(
    x_ptr, w_ptr, c_ptr,
    M,
    K: tl.constexpr = 8192,
    N: tl.constexpr = 2304,
    stride_xm: tl.constexpr = 8192,
    stride_xk: tl.constexpr = 1,
    stride_wk: tl.constexpr = 2304,
    stride_wn: tl.constexpr = 1,
    stride_cm: tl.constexpr = 2304,
    stride_cn: tl.constexpr = 1,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
    SPLIT_K: tl.constexpr = 4,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Slice the K dimension across SPLIT_K blocks
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = k_start + k_per_split

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M

    for k in range(k_start, k_end, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Store into output using atomic add across the K splits
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def qkv_gate_dense(
    x_ptr, w_ptr, c_ptr,
    M,
    K: tl.constexpr = 8192,
    N: tl.constexpr = 2304,
    stride_xm: tl.constexpr = 8192,
    stride_xk: tl.constexpr = 1,
    stride_wk: tl.constexpr = 2304,
    stride_wn: tl.constexpr = 1,
    stride_cm: tl.constexpr = 2304,
    stride_cn: tl.constexpr = 1,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 128,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


def solve_qkv_gate(X: torch.Tensor, W: torch.Tensor, out: torch.Tensor):
    M = X.shape[0]

    if M <= 32:
        # Launch 144 blocks to occupy all 132 SMs of H200!
        SPLIT_K = 4
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(2304, BLOCK_N), SPLIT_K)
        out.zero_()  # Initialize buffer for atomic add
        qkv_gate_splitk[grid](
            X, W, out,
            M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
            num_warps=4, num_stages=3
        )
    else:
        # Large M compute-bound
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 128
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(2304, BLOCK_N),)
        qkv_gate_dense[grid](
            X, W, out,
            M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
            num_warps=8, num_stages=3
        )


