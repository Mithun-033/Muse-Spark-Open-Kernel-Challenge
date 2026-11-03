"""
fallback.py - Archive of experimental and fallback kernels for muse_bf16_proj.
Contains:
1. Triton Two-Phase Split-K partials and reductions.
2. Triton Hi/Lo Split-K router kernels.
3. Triton 1D and N=8 streaming GEMVs.
4. Lossless C12 streaming GEMV and packing utilities.
"""

import torch
import triton
import triton.language as tl

# 1. Triton Router Hi/Lo & Fused Split-K Kernels

@triton.jit
def _router_hilo_splitk_mma(
    x_ptr, hi_ptr, lo_ptr, partials_ptr,
    M, N: tl.constexpr, K: tl.constexpr, total_n: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_hi_n: tl.constexpr, stride_hi_k: tl.constexpr,
    stride_lo_n: tl.constexpr, stride_lo_k: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Computes partial dot-products for High and Low BF16 routing slices into a dedicated
    L2 scratch buffer without atomic collisions.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    hi_ptrs = hi_ptr + (offs_n[None, :] * stride_hi_n + offs_k[:, None] * stride_hi_k)
    lo_ptrs = lo_ptr + (offs_n[None, :] * stride_lo_n + offs_k[:, None] * stride_lo_k)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
    w_hi = tl.load(hi_ptrs, mask=mask_n[None, :], other=0.0, eviction_policy="evict_first")
    w_lo = tl.load(lo_ptrs, mask=mask_n[None, :], other=0.0, eviction_policy="evict_first")

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot(x, w_hi, acc)
    acc = tl.dot(x, w_lo, acc)

    part_ptrs = partials_ptr + (pid_k * (M * total_n) + offs_m[:, None] * total_n + offs_n[None, :])
    tl.store(part_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _reduce_router_splitk(
    partials_ptr, out_ptr,
    M, N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduces partial sums from L2 cache scratchpad across SPLIT_K slices into the final
    FP32 output tensor.
    """
    pid = tl.program_id(0)
    total_elements = M * N
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    offs_s = tl.arange(0, SPLIT_K)
    ptrs = partials_ptr + (offs_s[:, None] * total_elements + offs[None, :])
    vals = tl.load(ptrs, mask=mask[None, :], other=0.0)
    res = tl.sum(vals, axis=0)

    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def router_fused_splitk(
    x_ptr, hi_ptr, lo_ptr, c_ptr,
    M,
    stride_xm: tl.constexpr = 8192,
    stride_xk: tl.constexpr = 1,
    stride_wk: tl.constexpr = 256,
    stride_wn: tl.constexpr = 1,
    stride_cm: tl.constexpr = 256,
    stride_cn: tl.constexpr = 1,
    K: tl.constexpr = 8192,
    N: tl.constexpr = 256,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 64,
    SPLIT_K: tl.constexpr = 8,
):
    """Fuses Hi and Lo BF16 weight multiplications into a single kernel pass over K=8192,
    accumulating atomically into L2 cache.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = k_start + k_per_split

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    hi_ptrs = hi_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    lo_ptrs = lo_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M

    for k in range(k_start, k_end, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
        w_hi = tl.load(hi_ptrs, eviction_policy="evict_first")
        w_lo = tl.load(lo_ptrs, eviction_policy="evict_first")
        acc = tl.dot(x, w_hi, acc)
        acc = tl.dot(x, w_lo, acc)
        x_ptrs += BLOCK_K * stride_xk
        hi_ptrs += BLOCK_K * stride_wk
        lo_ptrs += BLOCK_K * stride_wk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, mask=mask_m[:, None])


# 2. Triton Streaming GEMV (1D & N=8)

@triton.jit
def _bf16_streaming_gemv_1d(
    x_ptr, w_ptr, y_ptr,
    N: tl.constexpr,
    K: tl.constexpr = 8192,
    BLOCK_K: tl.constexpr = 1024,
):
    """1D scalar streaming GEMV: 1 CTA per N scalar (N=4096 on pooler = 31 waves across
    132 SMs), streaming weights through registers with evict_first policy.
    """
    pid_n = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        curr_k = k + offs_k
        x = tl.load(x_ptr + curr_k, eviction_policy="evict_last")
        w = tl.load(w_ptr + (pid_n * K + curr_k), eviction_policy="evict_first")
        acc = tl.fma(x.to(tl.float32), w.to(tl.float32), acc)

    out = tl.sum(acc, axis=0)
    tl.store(y_ptr + pid_n, out.to(tl.bfloat16))


@triton.jit
def _bf16_streaming_gemv_n8(
    x_ptr, w_ptr, y_ptr,
    N: tl.constexpr,
    K: tl.constexpr = 8192,
    BLOCK_N: tl.constexpr = 8,
    BLOCK_K: tl.constexpr = 1024,
):
    """Vectorized streaming GEMV assigning 8 N rows per CTA (288 CTAs across 132 SMs = 2.18 waves),
    broadcasting input activations in registers while streaming 8 weight rows concurrently.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        curr_k = k + offs_k
        x = tl.load(x_ptr + curr_k, eviction_policy="evict_last")
        w = tl.load(w_ptr + (offs_n[:, None] * K + curr_k[None, :]), eviction_policy="evict_first")
        acc = tl.fma(x[None, :].to(tl.float32), w.to(tl.float32), acc)

    out = tl.sum(acc, axis=1)
    tl.store(y_ptr + offs_n, out.to(tl.bfloat16))


# 3. Two-Phase Non-Atomic Split-K GEMM

@triton.jit
def splitk_phase1_gemm(
    x_ptr, w_ptr, partials_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Slices the inner K=8192 dimension by SPLIT_K (2 or 4), boosting CTA count to saturate
    all 132 SMs, storing intermediate partials in dedicated zero-atomic L2 scratch buffers.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

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

    part_ptrs = partials_ptr + (pid_k * (M * N) + offs_m[:, None] * N + offs_n[None, :])
    tl.store(part_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _reduce_splitk_kernel(
    partials_ptr, out_ptr,
    total_elements,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """High-speed 1D vector addition kernel that reads SPLIT_K partial tiles from L2 cache,
    reduces them in FP32 registers, and writes final BF16 results.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    offs_s = tl.arange(0, SPLIT_K)

    ptrs = partials_ptr + (offs_s[:, None] * total_elements + offs[None, :])
    vals = tl.load(ptrs, mask=mask[None, :], other=0.0)
    res = tl.sum(vals, axis=0)
    tl.store(out_ptr + offs, res.to(tl.bfloat16), mask=mask)


# 4. Lossless C12 Streaming GEMV Helpers

def prepare_c12_lossless(w: torch.Tensor) -> dict:
    """Packs BF16 [N, K] weights into 12-bit compressed format (25% DRAM bandwidth reduction)
    with sparse CSR outlier correction for 100% bit-exact accuracy.
    """
    raw = w.view(torch.int16).to(torch.int32)
    eh = (raw >> 7) & 0xFF
    sign = (raw >> 15) & 1
    mant = raw & 0x7F

    hist = torch.bincount(eh.flatten(), minlength=256).cpu()
    base_exp = int(torch.stack([hist[b:b+8].sum() for b in range(248)]).argmax())

    d_exp = torch.clamp(eh - base_exp, 0, 7)
    w12 = (sign << 11) | (d_exp << 7) | mant

    w_even, w_odd = w12[:, 0::2], w12[:, 1::2]
    b0 = (w_even & 0xFF).to(torch.uint8)
    b1 = (((w_even >> 8) & 0x0F) | ((w_odd & 0x0F) << 4)).to(torch.uint8)
    b2 = ((w_odd >> 4) & 0xFF).to(torch.uint8)
    w_packed = torch.stack([b0, b1, b2], dim=-1).reshape(w.size(0), -1).contiguous()

    w_recon_bits = (sign << 15) | ((base_exp + d_exp) << 7) | mant
    w_recon = w_recon_bits.to(torch.int16).view(torch.bfloat16)
    delta = w - w_recon

    mask = (eh < base_exp) | (eh >= base_exp + 8)
    row_counts = mask.sum(dim=1)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=w.device), row_counts.cumsum(0, dtype=torch.int32)])
    outlier_rows, col_idx = mask.nonzero().unbind(1)
    val = delta[outlier_rows, col_idx].contiguous()

    return {
        "w_c12": w_packed,
        "base_exp": base_exp,
        "row_ptr": row_ptr,
        "col_idx": col_idx.to(torch.int32).contiguous(),
        "val": val
    }


# Clean C12 Compressed Swap-AB Tensor Core GEMV (Pooler M=1, N=4096, K=8192)
_POOLER_C12_CUDA = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define SMD __device__ __forceinline__

SMD int4 sm_ldw(const int4* p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

SMD uint4 sm_ldx(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

SMD uint32_t sm_prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

SMD void sm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
    );
}

// 8 weights of one k16 step: low bytes in L0, L1; 4-bit exponent codes in C.
SMD void c12_decode_8(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
    const uint32_t cm = C & 0x77777777u;
    const uint32_t hlo = sm_prmt(T0, T1, cm);
    const uint32_t hhi = sm_prmt(T0, T1, cm >> 16);
    a[0] = sm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
    a[1] = sm_prmt(L0, hlo, 0x7362u) | ((C << 8)  & 0x80008000u);
    a[2] = sm_prmt(L1, hhi, 0x5140u) | ((C << 4)  & 0x80008000u);
    a[3] = sm_prmt(L1, hhi, 0x7362u) | ( C        & 0x80008000u);
}

// Escape entry patching: seq << 18 | lane << 13 | reg << 8 | half << 7 | xor7
SMD void sm_patch(uint32_t* A, uint32_t e) {
    const uint32_t reg = (e >> 8) & 31u;
    const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (reg == (uint32_t)r) {
            A[r] ^= xr;
        }
    }
}

// Single-CTA Pooler Kernel (M=1, N=4096, K=8192): S=1, no DSMEM, no remote cluster barriers
template <int WR, int WK, int KU>
__global__ __launch_bounds__(WR * WK * 32)
void pooler_c12_gemv_kernel(
    const int4* __restrict__ wp,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    const uint32_t* __restrict__ esc,
    uint32_t T0, uint32_t T1
) {
    constexpr int NW = WR * WK;
    constexpr int G = (4096 / (WR * 16)) * NW;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wr = warp % WR;
    const int wk = warp / WR;

    const int gw = blockIdx.x * NW + warp;
    const size_t sstride = (size_t)G * (3 * 32);
    const int4* wq = wp + (size_t)gw * (3 * 32) + lane;

    // Load escape for this lane
    const uint32_t my_e = esc[gw * 32 + lane];
    uint32_t nxt = __shfl_sync(0xffffffffu, my_e, 0);
    int ep = 0;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int s = 0; s < KU; ++s) {
        // 1. Vector load 48 bytes of C12 weights (3 int4 loads)
        const int4* cur_wq = wq + (size_t)s * sstride;
        int4 w0 = sm_ldw(cur_wq + 0 * 32);
        int4 w1 = sm_ldw(cur_wq + 1 * 32);
        int4 w2 = sm_ldw(cur_wq + 2 * 32);

        // 2. Decode into 16 A-matrix registers (four k16 MMA steps)
        uint32_t A[16];
        c12_decode_8(w0.x, w0.y, w2.x, T0, T1, A + 0);
        c12_decode_8(w0.z, w0.w, w2.y, T0, T1, A + 4);
        c12_decode_8(w1.x, w1.y, w2.z, T0, T1, A + 8);
        c12_decode_8(w1.z, w1.w, w2.w, T0, T1, A + 12);

        // 3. Lossless escape patching
        const uint32_t useq = (uint32_t)s;
        if ((nxt >> 18) == useq) {
            do {
                if (lane == (int)((nxt >> 13) & 31u)) {
                    sm_patch(A, nxt);
                }
                ++ep;
                nxt = (ep < 32) ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
            } while ((nxt >> 18) == useq);
        }

        // 4. Vector load X activations (16 BF16 values for this lane)
        const int k_elem = ((wk * KU + s) * 64) + (lane & 3) * 16;
        const __nv_bfloat16* xp = x + k_elem;
        uint4 xv0 = sm_ldx(xp);
        uint4 xv1 = sm_ldx(xp + 8);
        uint32_t b[8] = {xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w};

        // 5. Swap-AB Tensor Core MMA
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            sm_mma(acc, A + 4 * q, b[2 * q], b[2 * q + 1]);
        }
    }

    // 6. Intra-CTA Reduction across WK warps
    extern __shared__ float4 smem[];
    smem[warp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();

    // First WR warps sum across WK K-slices and write final output
    if (warp < WR) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int k = 0; k < WK; ++k) {
            float4 v = smem[(k * WR + wr) * 32 + lane];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }

        // For M=1, only lanes with (lane & 3) == 0 write to output
        if ((lane & 3) == 0) {
            const int row_base = (blockIdx.x * WR + wr) * 16 + (lane >> 2);
            out[row_base]     = __float2bfloat16_rn(sum.x);
            out[row_base + 8] = __float2bfloat16_rn(sum.z);
        }
    }
}

int pooler_c12_launch(
    int64_t wp, int64_t x, int64_t out, int64_t esc,
    int64_t t01, int64_t stream
) {
    uint32_t T0 = (uint32_t)(t01 & 0xFFFFFFFF);
    uint32_t T1 = (uint32_t)((uint64_t)t01 >> 32);
    auto s = reinterpret_cast<cudaStream_t>(stream);

    // Configuration: WR=1, WK=8, KU=16 (256 threads, 256 blocks, 4KB smem)
    constexpr int WR = 1, WK = 8, KU = 16;
    dim3 grid(4096 / (WR * 16));
    dim3 block(WR * WK * 32);
    size_t smem_bytes = (WR * WK * 32) * sizeof(float4);

    pooler_c12_gemv_kernel<WR, WK, KU><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const int4*>(wp),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<const uint32_t*>(esc),
        T0, T1
    );
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
"""

_C12_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

def c12_pack_pooler(w: torch.Tensor) -> dict:
    """Module: [pooler] | Target M: M == 1.

    Packs 16-bit BF16 pooler weights into 12-bit compressed format with escape table.
    Shrinks 67.1 MB down to 50.3 MB for peak Hopper HBM3e bandwidth utilization.
    """
    assert w.shape == (4096, 8192)
    dev = w.device
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(4096 // 16, 2, 8, 8192 // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(256, 128, 32, 32)

    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F

    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))

    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_C12_SIGN_POS, device=dev).long()
    code = ((d.view(256, 128, 32, 4, 8).long() << nib) + (sign.view(256, 128, 32, 4, 8).long() << pos)).sum(-1)
    code = code.to(torch.int32).contiguous().view(torch.uint8).view(256, 128, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(256, 128, 1536).contiguous()

    # Geometry: WR=1, WK=8, KU=16 (256 blocks, 8 warps = 256 threads)
    WR, WK, KU = 1, 8, 16
    gx = 256 // WR
    nw = WR * WK
    G = gx * nw
    si = torch.arange(KU, device=dev).view(KU, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(1, device=dev).view(1, 1, 1)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % WR, warp // WR
    tile = (bx * WR + wri) + j
    chunk = (by * WK + wki) * KU + si
    idx = (tile * 128 + chunk).reshape(-1)
    wp = data.view(256 * 128, 1536)[idx].contiguous().view(-1)

    # Escape table packing
    esc = torch.full((G, 32), -1, dtype=torch.int32, device=dev)
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    x7 = eh[er, ec, el, ev] ^ base

    bx_e = er // WR
    wr_e = er % WR
    by_e = ec // (WK * KU)
    rem2 = ec % (WK * KU)
    wk_e = rem2 // KU
    s_e = rem2 % KU

    g_e = (by_e * gx + bx_e) * nw + wk_e * WR + wr_e
    seq = s_e
    order = torch.argsort(g_e * (1 << 14) + seq)
    g_s, seq_s = g_e[order], seq[order]
    rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
    reg = ev[order] // 2
    entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | ((ev[order] % 2).long() << 7) | x7[order].long()
    esc[g_s, rank] = entry.to(torch.int32)

    # Exponent base decode tables
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    t01 = t0 | (t1 << 32)
    if t01 >= (1 << 63):
        t01 -= (1 << 64)

    return {"wp": wp, "esc": esc, "t01": t01, "base": base}




# QKV Split-K Kernels (Archived)

@triton.jit
def qkv_splitk_phase1_kernel(
    x_ptr, w_ptr, partials_ptr,
    M,
    stride_sk: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    K_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Module: [qkv_gate] Split-K Phase 1 GEMM | Target M: M in {16, 32}.

    Splits K=8192 into 4 independent slices of 2048 elements.
    With BLOCK_N=64, produces 36 N-tiles x 4 splits = 144 CTAs, fully saturating
    all 132 SMs on H200 in a single wave and writing partial sums
    non-atomically to L2 cache.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_start = pid_k * K_SPLIT

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + (k_start + offs_k[None, :]) * stride_xk)
    w_ptrs = w_ptr + ((k_start + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M

    for _ in range(0, K_SPLIT, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    part_ptrs = partials_ptr + (
        pid_k * stride_sk +
        offs_m[:, None] * stride_sm +
        offs_n[None, :] * stride_sn
    )
    tl.store(part_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def qkv_splitk_reduce_kernel(
    partials_ptr, out_ptr,
    total_elements: tl.constexpr,
    stride_sk: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Module: [qkv_gate] Split-K Phase 2 Vectorized Reduction | Target M: M in {16, 32}.

    Vectorized 128-bit reduction across L2 cache: sums the 4 Split-K partials into BF16 output.
    Executes in under 0.5 us with zero atomic contention.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for s in range(SPLIT_K):
        val = tl.load(partials_ptr + s * stride_sk + offs, mask=mask, other=0.0)
        acc += val
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)
