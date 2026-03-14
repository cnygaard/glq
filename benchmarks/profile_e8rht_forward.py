"""Profile E8RHTLinear.forward() end-to-end latency.

Measures the full forward pass (fused RHT + dequant+matmul) at B=1 and B=32.
"""

import torch
import math

from glq.codebook import E8ShellCodebook
from glq.quantized_linear import E8RHTLinear


def profile_forward(M, N, B, codebook, n_warmup=20, n_iter=200):
    """Profile E8RHTLinear.forward() end-to-end."""
    device = "cuda"
    K = codebook.codebook.shape[0]

    layer = E8RHTLinear(N, M, bias=False)

    # Fill with random quantized data
    m_pad = layer.m_pad
    n_pad = layer.n_pad
    layer.Qidxs = torch.randint(0, K, (m_pad, n_pad // 8), device=device, dtype=torch.int32).to(torch.int16)
    layer.SU = torch.where(torch.rand(m_pad, device=device) > 0.5,
                           torch.ones(1, device=device, dtype=torch.float16),
                           -torch.ones(1, device=device, dtype=torch.float16))
    layer.SV = torch.where(torch.rand(n_pad, device=device) > 0.5,
                           torch.ones(1, device=device, dtype=torch.float16),
                           -torch.ones(1, device=device, dtype=torch.float16))
    layer.Wscale = torch.tensor(0.01, dtype=torch.float32, device=device)
    layer.Qidxs2 = torch.zeros(m_pad, n_pad // 8, dtype=torch.int16, device=device)
    layer.inv_resid_scale = torch.zeros((), dtype=torch.float32, device=device)

    layer = layer.to(device)
    layer.set_codebook(codebook)

    x = torch.randn(B, N, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(n_warmup):
        _ = layer(x)
    torch.cuda.synchronize()

    # Timed runs
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start_evt.record()
        _ = layer(x)
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[9 * len(times) // 10]
    return median, p10, p90


def main():
    print("=" * 80)
    print("E8RHTLinear.forward() End-to-End Latency")
    print("GPU:", torch.cuda.get_device_name())
    print("=" * 80)

    codebook = E8ShellCodebook.build(device="cuda", verbose=False)

    configs = [
        (960, 960, 1, "SmolLM2 960x960 B=1"),
        (960, 960, 32, "SmolLM2 960x960 B=32"),
        (2560, 960, 1, "SmolLM2 2560x960 B=1"),
        (2560, 960, 32, "SmolLM2 2560x960 B=32"),
        (3072, 3072, 1, "3B q_proj 3072x3072 B=1"),
        (3072, 3072, 32, "3B q_proj 3072x3072 B=32"),
        (9216, 3072, 1, "3B gate 9216x3072 B=1"),
        (9216, 3072, 32, "3B gate 9216x3072 B=32"),
        (3072, 9216, 1, "3B down 3072x9216 B=1"),
        (3072, 9216, 32, "3B down 3072x9216 B=32"),
    ]

    header = f"{'Config':<30s} {'Median':>10s} {'P10':>10s} {'P90':>10s}"
    print(header)
    print("-" * len(header))

    for M, N, B, label in configs:
        med, p10, p90 = profile_forward(M, N, B, codebook)
        print(f"{label:<30s} {med*1000:9.0f}us {p10*1000:9.0f}us {p90*1000:9.0f}us")

    # Dense bf16 reference
    print("\n--- Dense bf16 matmul reference ---")
    ref_header = f"{'Config':<30s} {'Median':>10s}"
    print(ref_header)
    print("-" * len(ref_header))

    for M, N, B, label in configs:
        W = torch.randn(M, N, device="cuda", dtype=torch.float16)
        x = torch.randn(B, N, device="cuda", dtype=torch.float16)
        for _ in range(20):
            _ = x @ W.T
        torch.cuda.synchronize()

        times = []
        for _ in range(200):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            _ = x @ W.T
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        med = times[len(times) // 2]
        print(f"{label:<30s} {med*1000:9.0f}us")


if __name__ == "__main__":
    main()
