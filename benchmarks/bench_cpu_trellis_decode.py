"""CPU fused trellis decode benchmark: per-layer timings vs the paths it replaces.

Arms per (shape, K, B):
  fused[arith|lut]  the glq._C_cpu fused entry (in-op FHT + decode-GEMV/GEMM)
  dense-cached      today's fallback economics: torch matmul on a PRE-decoded fp32 weight
                    (the decode cost is excluded — this is the steady-state dense path)
  roofline          packed_bytes / measured DRAM bandwidth (STREAM-triad-ish)

Reported: us/layer, effective packed GB/s, % of roofline, and a tok/s-equivalent for a
model built of `--layers` copies of the shape. Numbers are machine-scoped by definition —
the header prints CPU, threads, and the active ISA tier.

Usage: python benchmarks/bench_cpu_trellis_decode.py [--shapes 3b|27b|both] [--iters 30]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from glq import inference_kernel_cpu as ikc  # noqa: E402

SHAPES = {
    "3b": [(2048, 2048), (2048, 11008)],
    "27b": [(5376, 5376), (5376, 21504)],
}


def _time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def dram_bandwidth_gbs(mib=512, iters=5):
    """Triad-ish: c = a + s*b over fp32 arrays far larger than LLC."""
    n = mib * 1024 * 1024 // 4
    a, b = torch.rand(n), torch.rand(n)
    c = torch.empty(n)
    t = _time(lambda: torch.add(a, b, alpha=1.7, out=c), iters, warmup=2)
    return 3 * n * 4 / t / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="3b", choices=["3b", "27b", "both"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--bs", default="1,4,8")
    ap.add_argument("--ks", default="2,3,4")
    args = ap.parse_args()

    ext = ikc.require_cpu_ext()
    bw = dram_bandwidth_gbs()
    print(f"threads={torch.get_num_threads()} isa={ext.glq_cpu_active_isa()} "
          f"dram~{bw:.1f} GB/s (triad)")
    shapes = SHAPES["3b"] + SHAPES["27b"] if args.shapes == "both" else SHAPES[args.shapes]

    for m, n in shapes:
        Wd = torch.randn(m, n)
        print(f"\n== {m}x{n} ==  (dense fp32 {m * n * 4 / 2 ** 20:.0f} MiB)")
        print(f"{'K':>2} {'B':>2} {'fused-arith':>12} {'fused-lut':>10} "
              f"{'dense-mm':>9} {'packed':>8} {'eff GB/s':>8} {'roofl%':>6}")
        for K in [int(k) for k in args.ks.split(",")]:
            packed = torch.randint(-32768, 32767, (m // 16 * (n // 16), 16 * K),
                                   dtype=torch.int16)
            pbytes = packed.numel() * 2
            for B in [int(b) for b in args.bs.split(",")]:
                x = torch.randn(B, n)
                if B == 1:
                    xv = x.view(n).contiguous()
                    fa = lambda: ext.glq_decode_matvec_trellis_3inst_cpu(xv, packed, m, n, 1.0)
                else:
                    fa = lambda: ext.glq_decode_matmul_trellis_3inst_cpu(x, packed, m, n, 1.0)
                ts = {}
                for variant in ("arith", "lut"):
                    ext.glq_cpu_set_decode_variant(variant)
                    ts[variant] = _time(fa, args.iters)
                ext.glq_cpu_set_decode_variant("auto")
                td = _time(lambda: x @ Wd.t(), max(5, args.iters // 3))
                best = min(ts.values())
                eff = pbytes / best / 1e9
                print(f"{K:>2} {B:>2} {ts['arith'] * 1e6:>10.0f}us {ts['lut'] * 1e6:>8.0f}us "
                      f"{td * 1e6:>7.0f}us {pbytes / 2 ** 20:>6.1f}MB {eff:>8.2f} "
                      f"{100 * eff / bw:>5.1f}%")


if __name__ == "__main__":
    main()
