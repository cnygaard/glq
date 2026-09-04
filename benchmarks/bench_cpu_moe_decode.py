#!/usr/bin/env python3
"""Fused CPU MoE decode: tok/s per ISA tier, against the roofline and the loop it replaces.

Shapes default to gemma-4-26B-A4B's real MoE block (hidden 2816, per-expert intermediate
704, 128 experts, top-k 8, gated => w13_out 1408), so the numbers map onto the model that
motivated the kernel. Packed bytes are random: trellis decode cost is content-independent,
which is what makes a fixture instant instead of an hour of LDLQ.

Two comparisons, because "the fused kernel is faster" is a claim, not an assumption:

  fused   the one-call MoE op (grouping + brackets + activation + reduce in C++)
  loop    the same work driven from Python, one dense fused-linear call per expert —
          i.e. what glq_vllm's _apply_trellis does today on CPU

and the per-token weight traffic against measured triad bandwidth, so a slow result can be
read as "bandwidth-bound, as expected" or "not bandwidth-bound, look at the decode".

    python benchmarks/bench_cpu_moe_decode.py                 # every available tier
    python benchmarks/bench_cpu_moe_decode.py --tiers avx512fp16 --tokens 1,8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from glq.hadamard import _block_decompose                      # noqa: E402
from glq.inference_kernel_cpu import _try_load_cpu_ext         # noqa: E402

TIERS = ("scalar", "avx2", "avx512", "avx512fp16")


def _ext():
    if not _try_load_cpu_ext():
        sys.exit("CPU extension not available (build it: GLQ_BUILD_CPU_EXT=1 pip install -e .)")
    from glq import inference_kernel_cpu as ikc
    return ikc._glq_cpu


def _meta(dim):
    rows, off = [], 0
    for bs in _block_decompose(dim):
        rows.append([off, bs, int(bs).bit_length() - 1, 0])
        off += bs
    return torch.tensor(rows, dtype=torch.int32)


def _time(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def triad_gbs(mib=256, iters=3):
    """Measured streaming bandwidth — the ceiling a decode-bound kernel is judged against."""
    n = mib * 1024 * 1024 // 4
    a, b = torch.randn(n), torch.randn(n)
    dt = _time(lambda: torch.add(a, b, alpha=2.0), iters, warmup=1)
    return (3 * n * 4) / dt / 1e9


def build(E, hidden, inter, K, seed=0):
    torch.manual_seed(seed)
    w13_out = 2 * inter
    rand = lambda *s: torch.randint(-32768, 32767, s, dtype=torch.int16)
    sign = lambda n: torch.where(torch.rand(n) < 0.5, -1.0, 1.0).half()
    return dict(
        w13_packed=rand(E, w13_out // 16 * (hidden // 16), 16 * K),
        w2_packed=rand(E, hidden // 16 * (inter // 16), 16 * K),
        w13_SU=torch.stack([sign(w13_out) for _ in range(E)]),
        w2_SU=torch.stack([sign(hidden) for _ in range(E)]),
        w13_SV=sign(hidden), w2_SV=sign(inter),
        w13_Wscale=torch.full((E,), 0.01), w2_Wscale=torch.full((E,), 0.01),
        meta_n_w13=_meta(hidden), meta_m_w13=_meta(w13_out),
        meta_n_w2=_meta(inter), meta_m_w2=_meta(hidden),
        hidden=hidden, inter=inter, w13_out=w13_out, E=E)


def route(T, E, topk, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(T)]).long()
    w = torch.rand(T, topk, generator=g)
    return ids, (w / w.sum(1, keepdim=True)).float()


def run_fused(ext, w, x, ids, wts, act=1):
    return ext.glq_fused_moe_trellis_3inst_cpu(
        x, ids, wts, w["w13_packed"], w["w13_SU"], w["w13_SV"], w["w13_Wscale"],
        w["w2_packed"], w["w2_SU"], w["w2_SV"], w["w2_Wscale"],
        w["hidden"], w["inter"], w["w13_out"],
        w["meta_n_w13"], w["meta_m_w13"], w["meta_n_w2"], w["meta_m_w2"], act)


def run_loop(ext, w, x, ids, wts):
    """What glq_vllm's _apply_trellis does on CPU today: one dense call per expert, driven
    from Python. The baseline the fused kernel has to beat to justify existing."""
    T, topk = ids.shape
    out = torch.zeros(T, w["hidden"])
    for e in ids.unique().tolist():
        mask = (ids == e)
        rows = mask.any(dim=1)
        ew = (wts * mask.float()).sum(dim=1)[rows]
        y = ext.glq_fused_linear_trellis_3inst_cpu(
            x[rows], w["w13_SV"], w["w13_SU"][e], w["w13_packed"][e], w["meta_n_w13"],
            w["meta_m_w13"], float(w["w13_Wscale"][e]), w["hidden"], w["w13_out"],
            w["hidden"], w["w13_out"])
        h = (torch.nn.functional.gelu(y[:, :w["inter"]], approximate="tanh")
             * y[:, w["inter"]:])
        z = ext.glq_fused_linear_trellis_3inst_cpu(
            h.contiguous(), w["w2_SV"], w["w2_SU"][e], w["w2_packed"][e], w["meta_n_w2"],
            w["meta_m_w2"], float(w["w2_Wscale"][e]), w["inter"], w["hidden"],
            w["inter"], w["hidden"])
        out[rows] += z * ew.unsqueeze(-1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--inter", type=int, default=704)
    p.add_argument("--bpw", type=int, default=4, help="K, bits/weight (packed width 16*K)")
    p.add_argument("--tokens", default="1,4,16")
    p.add_argument("--layers", type=int, default=30,
                   help="MoE layers in the model, for the per-token model-level estimate")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--tiers", default=",".join(TIERS))
    args = p.parse_args()

    ext = _ext()
    w = build(args.experts, args.hidden, args.inter, args.bpw)
    tokens = [int(t) for t in args.tokens.split(",")]
    bw = triad_gbs()

    # Weight bytes an expert's routings must stream, per token: topk * (w13 + w2).
    per_tok_bytes = args.topk * (2 * args.inter * args.hidden + args.hidden * args.inter) \
        * args.bpw / 8
    print(f"gemma-4-26B-A4B MoE block: hidden {args.hidden}, inter {args.inter}, "
          f"E {args.experts}, top-k {args.topk}, {args.bpw} bpw")
    print(f"triad bandwidth {bw:.1f} GB/s   weights/token/layer "
          f"{per_tok_bytes / 2**20:.1f} MiB   ({args.layers} layers => "
          f"{per_tok_bytes * args.layers / 2**30:.2f} GiB/token)\n")
    print(f"{'tier':<12} {'tok':>4} {'fused ms':>9} {'loop ms':>9} {'speedup':>8} "
          f"{'GB/s':>7} {'roofl%':>7} {'layer tok/s':>11} {'model tok/s':>11}")
    print("-" * 92)

    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        if not ext.glq_cpu_isa_available(tier):
            print(f"{tier:<12} (not available on this CPU/build)")
            continue
        ext.glq_cpu_set_isa(tier)
        for T in tokens:
            x = torch.randn(T, args.hidden)
            ids, wts = route(T, args.experts, args.topk)
            tf = _time(lambda: run_fused(ext, w, x, ids, wts), args.iters)
            tl = _time(lambda: run_loop(ext, w, x, ids, wts), max(2, args.iters // 3))
            gbs = per_tok_bytes * T / tf / 1e9
            print(f"{tier:<12} {T:>4} {tf * 1e3:>9.2f} {tl * 1e3:>9.2f} "
                  f"{tl / tf:>7.2f}x {gbs:>7.2f} {100 * gbs / bw:>6.1f}% "
                  f"{T / tf:>11.1f} {T / (tf * args.layers):>11.2f}")
    ext.glq_cpu_set_isa("auto")


if __name__ == "__main__":
    main()
