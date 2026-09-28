"""Is GLQ's trellis matvec competitive at lm_head's shape?

`lm_head` is excluded from quantization by a bare constant (`hf_integration.py:96`
`MODULES_TO_NOT_CONVERT = ["lm_head"]`), and on Qwen3.8-Flash-Next it is 248320x2560 bf16 =
1.271 GB read every decode token — 32.3% of per-token weight traffic, and larger than all 10
routed experts across all 48 layers combined. Quantizing it is therefore the largest single
traffic lever available.

But traffic is not latency. GLQ trades bytes for instructions: the 3INST decode issues ~1.24
instructions per weight and only ~1 in 16 of those is an FMA, so replacing a tuned bf16 GEMV
with a trellis matvec is only a win where the machine is short of *bandwidth* rather than
*issue slots*. This probe measures which side of that trade a given machine is on, at the
actual shape, before any quantize-path work is done.

Deliberately uses the bare `glq_decode_matvec_trellis_3inst_cpu` rather than the full fused
linear: that omits the RHT bracket and the bf16->fp32 activation conversion, so the number is a
**lower bound** on what GLQ would really cost. If the lower bound already loses, the conclusion
is firm.

3INST decode cost is data-independent (a computed hash, no data-dependent branching and no
lookup table), so random packed data is a valid speed fixture — no Viterbi pass needed, which
is what makes probing m=248320 practical at all.

    python benchmarks/_lm_head_shape_probe.py [--reps 7]
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

# (label, m, n) — m is the output-row count, n the reduction dim. W is (m, n), y = x @ W.T
SHAPES = [
    ("lm_head        Qwen-Next", 248320, 2560),
    ("expert gate_up          ", 1280, 2560),
    ("dense 2560x2560         ", 2560, 2560),
]
K = 3  # bits per weight; the 3 bpw checkpoint


def _ext():
    from glq import inference_kernel_cpu as ikc
    assert ikc._try_load_cpu_ext(), f"CPU extension unavailable: {ikc.cpu_ext_status()}"
    return ikc._glq_cpu, ikc.cpu_ext_status()


def _time(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--isa", default="",
                    help="comma-separated tiers to sweep (e.g. avx2,avx512,avx512fp16). "
                         "Default: whatever 'auto' picks. Sweeping prices the tier gap, which "
                         "matters because an avx2 reading does not predict an avx512fp16 box.")
    args = ap.parse_args()

    ext, status = _ext()
    print(f"torch {torch.__version__}  threads {torch.get_num_threads()}  glq cpu ext: {status}")
    print("GLQ arm is a LOWER BOUND: bare matvec, no RHT bracket, no bf16->fp32 convert.\n")

    tiers = [t for t in args.isa.split(",") if t] or [None]
    hdr = (f"{'shape':26s} {'arm':10s} {'ms (median)':>12s} {'min-max':>15s} "
           f"{'weight MiB':>11s} {'GB/s':>7s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    for label, m, n in SHAPES:
        row = {}
        # ---- GLQ trellis, 3 bpw, once per requested tier ----
        packed = torch.randint(-32768, 32767, (m // 16 * (n // 16), 16 * K), dtype=torch.int16)
        x32 = (torch.randn(n) * 0.5).contiguous()
        glq_bytes = packed.numel() * 2
        for t in tiers:
            if t is not None:
                if not ext.glq_cpu_isa_available(t):
                    print(f"{label:26s} {'glq3/'+t:10s} {'unavailable on this CPU/build':>40s}")
                    continue
                ext.glq_cpu_set_isa(t)
            row["glq3" + (f"/{t}" if t else "")] = (*_time(
                lambda: ext.glq_decode_matvec_trellis_3inst_cpu(x32, packed, m, n, 1.0),
                args.reps), glq_bytes)
        ext.glq_cpu_set_isa("auto")
        del packed
        gc.collect()

        # ---- bf16 dense, what lm_head actually does today ----
        Wb = (torch.randn(m, n) * 0.05).to(torch.bfloat16)
        xb = x32.to(torch.bfloat16).unsqueeze(0)
        bf_bytes = Wb.numel() * 2
        row["bf16"] = (*_time(lambda: torch.nn.functional.linear(xb, Wb), args.reps), bf_bytes)
        del Wb, xb
        gc.collect()

        # ---- fp32 dense, for reference ----
        Wf = (torch.randn(m, n) * 0.05).float()
        row["fp32"] = (*_time(lambda: torch.nn.functional.linear(x32.unsqueeze(0), Wf),
                              args.reps), Wf.numel() * 4)
        del Wf
        gc.collect()

        base = row["bf16"][0]
        for arm in [k for k in row if k.startswith("glq3")] + ["bf16", "fp32"]:
            med, lo, hi, nbytes = row[arm]
            gbs = nbytes / (med / 1e3) / 1e9
            if arm == "bf16":
                verdict = "baseline (today)"
            else:
                r = base / med
                verdict = (f"{r:.2f}x vs bf16 — "
                           + ("FASTER" if r > 1.05 else "SLOWER" if r < 0.95 else "parity"))
            print(f"{label:26s} {arm:10s} {med:12.2f} {lo:6.1f}-{hi:6.1f} "
                  f"{nbytes/2**20:11.1f} {gbs:7.1f}  {verdict}")
        print()

    print("Scope: CPU only, batch 1, this ISA tier and thread count. CPU decode is core-bound")
    print("(measured 45.3% core-bound vs 19.3% memory-bound at ~1% of DRAM peak), so a CPU")
    print("result here does NOT predict GPU, where B=1 decode is bandwidth-bound and is where")
    print("GLQ's recorded 1.90x-on-an-L4 comes from. Read this as the CPU half of the trade.")


if __name__ == "__main__":
    main()
