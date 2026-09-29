"""How much of a CPU decode step is the serial block-diagonal FHT?

Counter profiling showed 82.89% of CPU cycles at 48 threads sitting in OpenMP barrier wait.
That is a *symptom*: ATen ops at a 1-token decode are all below GRAIN_SIZE (32768) and open no
parallel region at all, so the only regions are the trellis matvecs — and between them
`blockdiag_fht_rows` runs fully serial while 47 threads spin. Working the cycle accounting
backwards (47 x serial_wall = 0.829 x 48 x 293.1 ms) puts serial wall time at ~248 ms of a
293.1 ms step.

`glq/csrc/cpu/glq_fht_cpu.cpp` is a candidate for the largest single piece of that:

  * fully serial -- no at::parallel_for, no omp
  * compiled with `-O3 -fopenmp` and **no -march** and no `#pragma GCC target`, unlike the
    trellis TUs, so it gets baseline x86-64 (SSE2, 4-wide) rather than AVX-512 (16-wide)
  * `:27` divides per element (`x[i] /= r`) instead of multiplying by a reciprocal

This measures it at the shapes an actual decode uses, so the "FHT dominates the serial time"
claim is tested rather than estimated from instruction counts.

    python benchmarks/_fht_cpu_probe.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

# Qwen3.8-Flash-Next: hidden 2560, w13_out 2*640, inter 640.
# expert_bracket does 4 FHTs per expert: in(hidden), out(w13_out), in(inter), out(hidden).
HIDDEN, W13_OUT, INTER = 2560, 1280, 640
PER_EXPERT = [("w13 in  (hidden)", HIDDEN), ("w13 out (w13_out)", W13_OUT),
              ("w2  in  (inter)", INTER), ("w2  out (hidden)", HIDDEN)]

# Per token, from the checkpoint's own layer_bpw: 48 layers x 10 routed experts = 480 expert
# invocations, plus 302 dense GLQ linears (2 FHTs each).
EXPERT_CALLS, DENSE_CALLS = 48 * 10, 302
STEP_MS_T48 = 293.1          # measured: 3.412 tok/s at T=48


def _meta(dim: int) -> torch.Tensor:
    """The (nblocks, 4) int32 block-diag metadata, via GLQ's own decomposition."""
    from glq.hadamard import _block_decompose
    blocks = _block_decompose(dim)
    rows, off = [], 0
    for bs in blocks:
        rows.append([off, bs, bs.bit_length() - 1, 0])
        off += bs
    return torch.tensor(rows, dtype=torch.int32), blocks


def _time(fn, reps=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)      # microseconds
    return statistics.median(ts), min(ts), max(ts)


def main():
    from glq import inference_kernel_cpu as ikc
    assert ikc._try_load_cpu_ext(), f"CPU extension unavailable: {ikc.cpu_ext_status()}"
    ext = ikc._glq_cpu
    torch.set_num_threads(1)     # it is serial anyway; pin it so the number is unambiguous
    print(f"torch {torch.__version__}  threads {torch.get_num_threads()}  "
          f"glq cpu ext: {ikc.cpu_ext_status()}")
    print("NOTE: the FHT TU has no `#pragma GCC target`, so it is baseline x86-64 "
          "regardless of the tier reported above.\n")

    hdr = f"{'FHT':22s} {'blocks':>18s} {'us (median)':>12s} {'min-max':>15s} {'Mflop/s':>9s}"
    print(hdr); print("-" * len(hdr))

    per_expert_us = 0.0
    seen = {}
    for label, dim in PER_EXPERT:
        meta, blocks = _meta(dim)
        x = torch.randn(1, dim).float().contiguous()
        med, lo, hi = _time(lambda: ext.glq_blockdiag_fht_cpu(x, meta))
        # butterflies: sum over blocks of (bs/2 * log2(bs)) pairs, 2 flops each, + bs divides
        flops = sum(bs // 2 * (bs.bit_length() - 1) * 2 + bs for bs in blocks)
        per_expert_us += med
        seen[dim] = med
        print(f"{label:22s} {str(blocks):>18s} {med:12.2f} {lo:6.1f}-{hi:6.1f} "
              f"{flops/med:9.1f}")

    dense_us = seen[HIDDEN] * 2      # a dense linear does in(hidden) + out(m); use hidden twice
    total_ms = (per_expert_us * EXPERT_CALLS + dense_us * DENSE_CALLS) / 1000.0
    print(f"\nper expert  ({len(PER_EXPERT)} FHTs): {per_expert_us:8.2f} us")
    print(f"per dense linear (2 FHTs, approx): {dense_us:8.2f} us")
    print(f"\nper decoded token:")
    print(f"  {EXPERT_CALLS} expert invocations x {per_expert_us:.2f} us = "
          f"{per_expert_us * EXPERT_CALLS / 1000:7.1f} ms")
    print(f"  {DENSE_CALLS} dense linears     x {dense_us:.2f} us = "
          f"{dense_us * DENSE_CALLS / 1000:7.1f} ms")
    print(f"  TOTAL serial FHT                = {total_ms:7.1f} ms")
    print(f"\nagainst a measured {STEP_MS_T48:.1f} ms step at T=48: "
          f"**{100 * total_ms / STEP_MS_T48:.1f}% of the step**")
    print(f"against the ~248 ms of serial wall time implied by the barrier share: "
          f"{100 * total_ms / 248:.1f}% of it")
    print("\nScope: this machine's clock and ISA, single-threaded, batch 1. The SHARE "
          "transfers better than the absolute us.")


if __name__ == "__main__":
    main()
