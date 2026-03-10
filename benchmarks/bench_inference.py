"""Benchmark GLQ inference kernel: memory and throughput.

Usage:
    python benchmarks/bench_inference.py          # requires CUDA
    python benchmarks/bench_inference.py --cpu     # CPU-only quality benchmark
"""

import argparse
import sys
import time

import torch

from glq.codebook import E8ShellCodebook
from glq.inference_kernel import glq_dequant_matmul, _fallback_dequant_matmul
from glq.quantize_model import quantize_layer_e8_shell_rht
from glq.quantized_linear import E8RHTLinear


# ──────────────────────────────────────────────────────────────────────
# 1. Kernel-level benchmark (CUDA only)
# ──────────────────────────────────────────────────────────────────────

def bench_single_layer(M, N, B, K=65536, n_warmup=10, n_iter=100):
    """Benchmark a single dequant+matmul layer."""
    device = "cuda"
    cb = torch.randn(K, 8, device=device, dtype=torch.float16)
    Qidxs = torch.randint(0, K, (M, N // 8), device=device, dtype=torch.int32).to(torch.int16)
    x = torch.randn(B, N, device=device, dtype=torch.float16)
    Wscale = 1.0

    # Memory baseline
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    # --- Fused kernel ---
    for _ in range(n_warmup):
        y_fused = glq_dequant_matmul(x, Qidxs, cb, Wscale)
    torch.cuda.synchronize()

    mem_fused = torch.cuda.max_memory_allocated() - mem_before

    start = time.perf_counter()
    for _ in range(n_iter):
        y_fused = glq_dequant_matmul(x, Qidxs, cb, Wscale)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - start) / n_iter

    # --- Naive fallback ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    for _ in range(n_warmup):
        y_naive = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
    torch.cuda.synchronize()

    mem_naive = torch.cuda.max_memory_allocated() - mem_before

    start = time.perf_counter()
    for _ in range(n_iter):
        y_naive = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
    torch.cuda.synchronize()
    t_naive = (time.perf_counter() - start) / n_iter

    # --- Dense bf16 matmul (upper bound) ---
    W_dense = torch.randn(M, N, device=device, dtype=torch.float16)
    for _ in range(n_warmup):
        y_dense = x.float() @ W_dense.float().T
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        y_dense = x.float() @ W_dense.float().T
    torch.cuda.synchronize()
    t_dense = (time.perf_counter() - start) / n_iter

    # Correctness check
    diff = (y_fused - y_naive).abs().max().item()

    return {
        "t_fused_ms": t_fused * 1000,
        "t_naive_ms": t_naive * 1000,
        "t_dense_ms": t_dense * 1000,
        "mem_fused_MB": mem_fused / 1e6,
        "mem_naive_MB": mem_naive / 1e6,
        "max_diff": diff,
        "speedup": t_naive / t_fused if t_fused > 0 else float("inf"),
    }


def run_kernel_benchmark():
    print("=" * 90)
    print("GLQ Kernel Benchmark (fused vs naive vs dense)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 90)

    configs = [
        # (M, N, B, label)
        (1024, 1024, 1, "SmolLM 1k×1k B=1"),
        (1024, 1024, 32, "SmolLM 1k×1k B=32"),
        (4096, 4096, 1, "LLaMA 4k×4k B=1"),
        (4096, 4096, 32, "LLaMA 4k×4k B=32"),
        (4096, 4096, 128, "LLaMA 4k×4k B=128"),
        (4096, 16384, 1, "LLaMA MLP 4k×16k B=1"),
        (4096, 16384, 32, "LLaMA MLP 4k×16k B=32"),
    ]

    header = (f"{'Config':<28s} {'Fused':>8s} {'Naive':>8s} {'Dense':>8s} "
              f"{'Speedup':>8s} {'MemF':>8s} {'MemN':>8s} {'MaxDiff':>10s}")
    print(header)
    print("-" * len(header))

    for M, N, B, label in configs:
        r = bench_single_layer(M, N, B)
        print(
            f"{label:<28s} "
            f"{r['t_fused_ms']:7.2f}ms "
            f"{r['t_naive_ms']:7.2f}ms "
            f"{r['t_dense_ms']:7.2f}ms "
            f"{r['speedup']:7.2f}x "
            f"{r['mem_fused_MB']:7.1f}MB "
            f"{r['mem_naive_MB']:7.1f}MB "
            f"{r['max_diff']:10.4f}"
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Memory footprint comparison (analytical)
# ──────────────────────────────────────────────────────────────────────

def run_memory_comparison():
    print("\n" + "=" * 70)
    print("Weight Storage: E8RHTLinear vs nn.Linear")
    print("=" * 70)

    models = [
        ("SmolLM2-360M", 960, 24, 7),
        ("Llama-3.2-1B", 2048, 16, 7),
        ("Llama-3.2-3B", 3072, 28, 7),
        ("Llama-3-8B", 4096, 32, 7),
        ("Llama-3-70B", 8192, 80, 7),
    ]

    header = (f"{'Model':<18s} {'fp16':>8s} {'2bpw':>8s} {'3bpw':>8s} "
              f"{'4bpw':>8s} {'2bpw ratio':>10s}")
    print(header)
    print("-" * len(header))

    for name, hidden, n_layers, linears_per_layer in models:
        n_linear = n_layers * linears_per_layer
        # Assume square weight matrices for simplicity
        N = hidden

        fp16_bytes = n_linear * N * N * 2
        bpw2_bytes = n_linear * (N * (N // 8) * 2 + N * 2 + N * 2 + 4)  # Qidxs + SU + SV + Wscale
        bpw3_bytes = n_linear * (N * (N // 8) * 2 * 2 + N * 2 + N * 2 + 8)  # 2× Qidxs
        bpw4_bytes = bpw3_bytes  # same storage, different codebook

        # Add codebook cost (shared, amortized)
        cb_bytes = 65536 * 8 * 2  # 1 MB

        print(
            f"{name:<18s} "
            f"{fp16_bytes / 1e9:7.2f}G "
            f"{(bpw2_bytes + cb_bytes) / 1e9:7.2f}G "
            f"{(bpw3_bytes + cb_bytes) / 1e9:7.2f}G "
            f"{(bpw4_bytes + cb_bytes) / 1e9:7.2f}G "
            f"{fp16_bytes / bpw2_bytes:9.1f}x"
        )


# ──────────────────────────────────────────────────────────────────────
# 3. Quantization quality benchmark (CPU-safe)
# ──────────────────────────────────────────────────────────────────────

def run_quality_benchmark():
    print("\n" + "=" * 70)
    print("Quantization Quality: 2bpw vs 3bpw vs 4bpw")
    print("=" * 70)

    codebook = E8ShellCodebook.build(device="cpu", verbose=False)

    configs = [
        (32, 64, "32×64"),
        (64, 128, "64×128"),
        (128, 256, "128×256"),
        (256, 512, "256×512"),
    ]

    header = (f"{'Shape':<12s} {'2bpw SQNR':>10s} {'3bpw SQNR':>10s} "
              f"{'4bpw SQNR':>10s} {'2bpw MSE':>10s} {'4bpw MSE':>10s}")
    print(header)
    print("-" * len(header))

    for m, n, label in configs:
        torch.manual_seed(42)
        W = torch.randn(m, n) * 0.1
        A = torch.randn(n, n)
        H = A @ A.T + 0.1 * torch.eye(n)

        _, _, m2 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        _, _, m3 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        _, _, m4 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=4)

        # Also measure output MSE through E8RHTLinear
        x = torch.randn(16, n)
        y_orig = x.float() @ W.float().T

        results = {}
        for bpw, metrics in [(2, m2), (3, m3), (4, m4)]:
            W_hat, arts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=bpw)
            layer = E8RHTLinear(n, m)
            layer.Qidxs.copy_(arts['Qidxs'])
            layer.SU.copy_(arts['SU'])
            layer.SV.copy_(arts['SV'])
            layer.Wscale.copy_(arts['Wscale'])
            cb2 = None
            if bpw == 3:
                cb2 = codebook.make_small(256)
            elif bpw == 4:
                cb2 = codebook
            if 'Qidxs2' in arts:
                layer.Qidxs2 = arts['Qidxs2']
                layer.inv_resid_scale = arts['inv_resid_scale']
            layer.set_codebook(codebook, codebook2=cb2)

            y_q = layer(x).float()
            mse = (y_q - y_orig).pow(2).mean().item()
            results[bpw] = mse

        print(
            f"{label:<12s} "
            f"{m2['sqnr']:9.1f}dB "
            f"{m3['sqnr']:9.1f}dB "
            f"{m4['sqnr']:9.1f}dB "
            f"{results[2]:10.6f} "
            f"{results[4]:10.6f}"
        )


# ──────────────────────────────────────────────────────────────────────
# 4. E8RHTLinear full forward benchmark (CPU timing)
# ──────────────────────────────────────────────────────────────────────

def run_e2e_layer_benchmark():
    print("\n" + "=" * 70)
    print("E8RHTLinear Full Forward (CPU, includes RHT transforms)")
    print("=" * 70)

    codebook = E8ShellCodebook.build(device="cpu", verbose=False)

    configs = [
        (64, 32, 1, "64→32 B=1"),
        (64, 32, 16, "64→32 B=16"),
        (256, 128, 1, "256→128 B=1"),
        (256, 128, 16, "256→128 B=16"),
        (512, 256, 1, "512→256 B=1"),
        (512, 256, 16, "512→256 B=16"),
    ]

    header = f"{'Config':<20s} {'GLQ':>10s} {'nn.Linear':>10s} {'Ratio':>8s}"
    print(header)
    print("-" * len(header))

    n_warmup = 3
    n_iter = 20

    for n_in, n_out, batch, label in configs:
        # Set up E8RHTLinear
        layer = E8RHTLinear(n_in, n_out)
        layer.set_codebook(codebook)

        # Set up nn.Linear reference
        ref = torch.nn.Linear(n_in, n_out, bias=False)

        x = torch.randn(batch, n_in)

        # Warmup
        for _ in range(n_warmup):
            layer(x)
            ref(x)

        # GLQ timing
        start = time.perf_counter()
        for _ in range(n_iter):
            layer(x)
        t_glq = (time.perf_counter() - start) / n_iter

        # Dense timing
        start = time.perf_counter()
        for _ in range(n_iter):
            ref(x)
        t_dense = (time.perf_counter() - start) / n_iter

        ratio = t_glq / t_dense if t_dense > 0 else float("inf")
        print(
            f"{label:<20s} "
            f"{t_glq*1000:9.2f}ms "
            f"{t_dense*1000:9.2f}ms "
            f"{ratio:7.1f}x"
        )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLQ Inference Benchmarks")
    parser.add_argument("--cpu", action="store_true",
                        help="Run CPU-only benchmarks (quality + E8RHTLinear timing)")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        if not args.cpu:
            print("No CUDA available, running CPU-only benchmarks.\n")
        run_memory_comparison()
        run_quality_benchmark()
        run_e2e_layer_benchmark()
    else:
        run_kernel_benchmark()
        run_memory_comparison()
        run_quality_benchmark()

    print("\nDone.")


if __name__ == "__main__":
    main()
