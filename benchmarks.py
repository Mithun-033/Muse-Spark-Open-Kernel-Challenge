import torch

from kernels import solve_qkv_gate


def time_us(fn, iters=300, warmup=50):
    """Accurate GPU timing using CUDA events. Returns latency in µs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000.0


def benchmark_qkv(m: int):
    K, N = 8192, 2304
    device = "cuda"

    X = torch.randn((m, K), dtype=torch.bfloat16, device=device).contiguous()
    W = torch.randn((K, N), dtype=torch.bfloat16, device=device).contiguous()
    out_yours = torch.empty((m, N), dtype=torch.bfloat16, device=device).contiguous()
    out_ref = torch.empty((m, N), dtype=torch.bfloat16, device=device).contiguous()

    def ref_call():
        torch.mm(X, W, out=out_ref)
    def your_call():
        solve_qkv_gate(X, W, out_yours)

    # 3. Check correctness: Relative L2 error must be <= 1%
    ref_call()
    your_call()
    rel_l2 = float((out_yours.float() - out_ref.float()).norm() / out_ref.float().norm())
    ok = rel_l2 <= 0.01

    # 4. Latency timing
    t_ref = time_us(ref_call)
    t_yours = time_us(your_call)
    speedup = t_ref / t_yours

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] M={m:<4} | Rel L2: {rel_l2:.2e} | Ref: {t_ref:6.2f} µs | Yours: {t_yours:6.2f} µs | Speedup: {speedup:5.2f}x")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Testing qkv_gate (K=8192, N=2304)...")
    print("-" * 75)

    # Scored shapes for qkv_gate
    for m in [1, 8, 16, 32, 128, 256]:
        benchmark_qkv(m)