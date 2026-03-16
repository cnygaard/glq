"""Profile per-component latency of GLQ inference kernel.

Measures input FHT, dequant+matmul, output FHT, and overhead
using CUDA events at B=1 (decode) and B=32 (prefill).

Usage:
    python benchmarks/profile_inference.py
"""

import torch
import torch.nn.functional as F
import time

from glq.codebook import E8ShellCodebook
from glq.hadamard import fast_hadamard_transform
from glq.inference_kernel import glq_dequant_matmul, _triton_available


def profile_layer(M, N, B, codebook, n_warmup=20, n_iter=200):
    """Profile a single E8RHTLinear-style forward pass, component by component."""
    device = "cuda"

    # Simulate E8RHTLinear buffers
    m_pad = 1 << (M - 1).bit_length()
    n_pad = 1 << (N - 1).bit_length()
    K = codebook.codebook.shape[0]

    Qidxs = torch.randint(0, K, (m_pad, n_pad // 8), device=device, dtype=torch.int32).to(torch.int16)
    SU = torch.where(torch.rand(m_pad, device=device) > 0.5,
                     torch.ones(1, device=device, dtype=torch.float16),
                     -torch.ones(1, device=device, dtype=torch.float16))
    SV = torch.where(torch.rand(n_pad, device=device) > 0.5,
                     torch.ones(1, device=device, dtype=torch.float16),
                     -torch.ones(1, device=device, dtype=torch.float16))
    Wscale = 0.01
    cb_half = codebook.codebook_half.to(device)
    x = torch.randn(B, N, device=device, dtype=torch.float16)

    # CUDA events
    def make_events():
        return [torch.cuda.Event(enable_timing=True) for _ in range(6)]

    # Warmup
    for _ in range(n_warmup):
        x_f = x.float()
        x_pad = F.pad(x_f, (0, n_pad - N))
        sv = SV.float()
        x_rht = fast_hadamard_transform(x_pad * sv.unsqueeze(0))
        y_rht = glq_dequant_matmul(x_rht, Qidxs, cb_half, Wscale)
        su = SU.float()
        y = fast_hadamard_transform(y_rht) * su.unsqueeze(0)
        y = y[:, :M]

    torch.cuda.synchronize()

    # Timed runs
    times = {"cast_pad_sv": [], "input_fht": [], "dequant_matmul": [],
             "output_fht_su": [], "unpad_cast": [], "total": []}

    for _ in range(n_iter):
        evts = make_events()

        torch.cuda.synchronize()
        evts[0].record()

        # Phase 1: cast + pad + SV multiply
        x_f = x.float()
        x_pad = F.pad(x_f, (0, n_pad - N))
        sv = SV.float()
        x_sv = x_pad * sv.unsqueeze(0)
        evts[1].record()

        # Phase 2: input FHT
        x_rht = fast_hadamard_transform(x_sv)
        evts[2].record()

        # Phase 3: dequant + matmul
        y_rht = glq_dequant_matmul(x_rht, Qidxs, cb_half, Wscale)
        evts[3].record()

        # Phase 4: output FHT + SU multiply
        su = SU.float()
        y = fast_hadamard_transform(y_rht) * su.unsqueeze(0)
        evts[4].record()

        # Phase 5: unpad + cast
        y = y[:, :M]
        y = y.half()
        evts[5].record()

        torch.cuda.synchronize()

        times["cast_pad_sv"].append(evts[0].elapsed_time(evts[1]))
        times["input_fht"].append(evts[1].elapsed_time(evts[2]))
        times["dequant_matmul"].append(evts[2].elapsed_time(evts[3]))
        times["output_fht_su"].append(evts[3].elapsed_time(evts[4]))
        times["unpad_cast"].append(evts[4].elapsed_time(evts[5]))
        times["total"].append(evts[0].elapsed_time(evts[5]))

    # Compute medians (more stable than means for GPU timings)
    result = {}
    for key, vals in times.items():
        vals_sorted = sorted(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        result[key] = median

    return result


def main():
    print("=" * 90)
    print("GLQ Inference Kernel Profiling (per-component latency)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton available: {_triton_available}")
    print("=" * 90)

    codebook = E8ShellCodebook.build(device="cuda", verbose=False)

    configs = [
        # (M, N, B, label)
        # SmolLM2-360M dimensions
        (960, 960, 1, "SmolLM2 960x960 B=1"),
        (960, 960, 32, "SmolLM2 960x960 B=32"),
        (2560, 960, 1, "SmolLM2 2560x960 B=1"),
        (2560, 960, 32, "SmolLM2 2560x960 B=32"),
        # Ministral-3B / Llama-3B dimensions
        (3072, 3072, 1, "3B q_proj 3072x3072 B=1"),
        (3072, 3072, 4, "3B q_proj 3072x3072 B=4"),
        (3072, 3072, 8, "3B q_proj 3072x3072 B=8"),
        (3072, 3072, 16, "3B q_proj 3072x3072 B=16"),
        (3072, 3072, 32, "3B q_proj 3072x3072 B=32"),
        (3072, 3072, 64, "3B q_proj 3072x3072 B=64"),
        (9216, 3072, 1, "3B gate 9216x3072 B=1"),
        (9216, 3072, 8, "3B gate 9216x3072 B=8"),
        (9216, 3072, 32, "3B gate 9216x3072 B=32"),
        (9216, 3072, 64, "3B gate 9216x3072 B=64"),
        (3072, 9216, 1, "3B down 3072x9216 B=1"),
        (3072, 9216, 8, "3B down 3072x9216 B=8"),
        (3072, 9216, 32, "3B down 3072x9216 B=32"),
        (3072, 9216, 64, "3B down 3072x9216 B=64"),
    ]

    header = (f"{'Config':<30s} {'Cast+Pad':>9s} {'InFHT':>9s} {'Dequant':>9s} "
              f"{'OutFHT':>9s} {'Unpad':>9s} {'Total':>9s}")
    print(header)
    print("-" * len(header))

    for M, N, B, label in configs:
        r = profile_layer(M, N, B, codebook)
        print(
            f"{label:<30s} "
            f"{r['cast_pad_sv']*1000:8.0f}us "
            f"{r['input_fht']*1000:8.0f}us "
            f"{r['dequant_matmul']*1000:8.0f}us "
            f"{r['output_fht_su']*1000:8.0f}us "
            f"{r['unpad_cast']*1000:8.0f}us "
            f"{r['total']*1000:8.0f}us"
        )

    # Also measure dense bf16 matmul for reference
    print("\n--- Dense bf16 matmul reference ---")
    ref_header = f"{'Config':<30s} {'bf16 matmul':>12s}"
    print(ref_header)
    print("-" * len(ref_header))

    for M, N, B, label in configs:
        W = torch.randn(M, N, device="cuda", dtype=torch.float16)
        x = torch.randn(B, N, device="cuda", dtype=torch.float16)
        for _ in range(20):
            _ = x @ W.T
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(200):
            _ = x @ W.T
        end_evt.record()
        torch.cuda.synchronize()
        t_ms = start_evt.elapsed_time(end_evt) / 200

        print(f"{label:<30s} {t_ms*1000:11.0f}us")


if __name__ == "__main__":
    main()
