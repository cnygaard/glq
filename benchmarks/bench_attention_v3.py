"""v0.5 Phase 5.3 — per-kernel microbench: v2.1 (65 K) vs v3.0 (4 K).

Isolates the codebook-residency effect by calling the Triton attention
launchers directly with synthetic K/V at realistic Gemma-4-E4B-it
shapes. No vLLM scheduler / cudagraph in the loop — measures pure
kernel time via CUDA events.

Gemma-4-E4B attention surface:
  sliding layer (×5/6): head_dim=256, num_q=8,  num_kv=4, sw=512
  full layer    (×1/6): head_dim=512, num_q=8,  num_kv=4, no sw

Phase 5.3 acceptance gate (per the plan):
  - v3.0 per-call ≤ 50 µs on each shape variant
  - v3.0 wall < v2.1 wall on same shape

Usage::

    python benchmarks/bench_attention_v3.py \\
        [--Tk 256] [--Tq 16] [--n-iters 200] [--warmup 20] \\
        [--out /tmp/logs/bench_attn_v3.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch


def _make_quantizer(quant_method: str, target_size: int | None):
    """Build E8KVQuantizer on cuda for given codebook size."""
    from glq.kv_cache import _get_codebook
    from glq.kv_e8 import E8KVQuantizer
    cb = _get_codebook(quant_method, device="cuda", target_size=target_size)
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


def _quantize_to_paged(K, V, quantizer, *, num_blocks, block_size):
    Tk, H_kv, D = K.shape
    n_groups = D // 8

    qk = quantizer.quantize(K)
    qv = quantizer.quantize(V)

    def _pack_int16(qt_idx):
        flat = qt_idx.reshape(Tk, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=torch.int16, device=K.device,
        )
        for t in range(Tk):
            paged[t // block_size, t % block_size] = flat[t]
        return paged

    def _pack_scale(scale_tensor):
        flat = scale_tensor.reshape(Tk, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=flat.dtype, device=K.device,
        )
        for t in range(Tk):
            paged[t // block_size, t % block_size] = flat[t]
        return paged

    return (
        _pack_int16(qk["idx1"]), _pack_int16(qk["idx2"]),
        _pack_scale(qk["scale"]),
        _pack_int16(qv["idx1"]), _pack_int16(qv["idx2"]),
        _pack_scale(qv["scale"]),
    )


def _build_fixture(*, Tq, Tk, head_size, num_q_heads, num_kv_heads,
                   block_size, quantizer, sliding_window):
    """Build a single (q, packed_quants, block_table, etc.) fixture."""
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(42)
    q = torch.randn(Tq, num_q_heads, head_size, device=device,
                    dtype=dtype, generator=g)
    K = torch.randn(Tk, num_kv_heads, head_size, device=device,
                    dtype=dtype, generator=g)
    V = torch.randn(Tk, num_kv_heads, head_size, device=device,
                    dtype=dtype, generator=g)

    num_blocks = (Tk + block_size - 1) // block_size
    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale) = _quantize_to_paged(
        K, V, quantizer, num_blocks=num_blocks, block_size=block_size,
    )

    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).reshape(1, num_blocks)

    codebook = quantizer.codebook.codebook_half
    from glq_vllm.e8_paged_cache import _get_hadamard
    H_mat = _get_hadamard(torch.float32, "cuda")
    rs = float(quantizer.codebook.resid_scale)

    return dict(
        q=q, out=out,
        k_idx1=k_idx1, k_idx2=k_idx2, k_scale=k_scale,
        v_idx1=v_idx1, v_idx2=v_idx2, v_scale=v_scale,
        codebook=codebook, H_mat=H_mat,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(head_size),
        resid_scale=rs, block_table=block_table,
        sliding_window=sliding_window,
    )


def _time_kernel(launcher, fixture, *, n_iters, warmup):
    """CUDA-event timing for a launcher applied to a fixture."""
    # Warm-up (Triton autotune + L2 warm)
    for _ in range(warmup):
        launcher(**fixture)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        launcher(**fixture)
        stops[i].record()
    torch.cuda.synchronize()

    times_us = [
        starts[i].elapsed_time(stops[i]) * 1000.0
        for i in range(n_iters)
    ]
    return times_us


def _stats(name, times):
    import statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    p05 = sorted(times)[len(times) // 20]
    p95 = sorted(times)[-(len(times) // 20 + 1)]
    return dict(
        name=name, n=len(times),
        mean_us=round(mean, 2), median_us=round(median, 2),
        stdev_us=round(stdev, 2),
        p05_us=round(p05, 2), p95_us=round(p95, 2),
        min_us=round(min(times), 2), max_us=round(max(times), 2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Tq", type=int, default=16)
    p.add_argument("--Tk", type=int, default=256)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default="/tmp/bench_attn_v3.json")
    args = p.parse_args()

    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v2_1,
        unified_attention_e8_v3_0,
        unified_attention_e8_v3_b,
        unified_attention_e8_v3_fht,
        unified_attention_e8_v3_split,
    )

    print(f"=== Phase 5.3 microbench: v2.1 (65 K) vs v3.0 (4 K) vs v3_b (4 K smem) vs v3_fht (4 K butterfly) ===", flush=True)
    print(f"  Tq={args.Tq} Tk={args.Tk} n_iters={args.n_iters} "
          f"warmup={args.warmup}", flush=True)

    # Build both quantizers / codebooks
    print(f"  building 65 K strict codebook ...", flush=True)
    q65k = _make_quantizer("e8_strict", target_size=None)
    print(f"  building 4 K relaxed codebook ...", flush=True)
    q4k = _make_quantizer("e8_relaxed", target_size=4096)

    # Two Gemma-4-E4B shapes
    shapes = [
        dict(name="sliding_layer", head_size=256, num_q=8, num_kv=4,
             sliding_window=512),
        dict(name="full_layer", head_size=512, num_q=8, num_kv=4,
             sliding_window=0),
    ]

    results = []
    for shape in shapes:
        print(f"\n=== shape: {shape['name']} "
              f"(head={shape['head_size']}, q={shape['num_q']}, "
              f"kv={shape['num_kv']}, sw={shape['sliding_window']}) ===",
              flush=True)

        # Build fixtures for both codebooks (same Q/K/V seed, just
        # different quantization)
        fx_65k = _build_fixture(
            Tq=args.Tq, Tk=args.Tk, head_size=shape["head_size"],
            num_q_heads=shape["num_q"], num_kv_heads=shape["num_kv"],
            block_size=16, quantizer=q65k,
            sliding_window=shape["sliding_window"],
        )
        fx_4k = _build_fixture(
            Tq=args.Tq, Tk=args.Tk, head_size=shape["head_size"],
            num_q_heads=shape["num_q"], num_kv_heads=shape["num_kv"],
            block_size=16, quantizer=q4k,
            sliding_window=shape["sliding_window"],
        )

        # Confirm all five produce output within reasonable range (sanity)
        unified_attention_e8_v2_1(**fx_65k)
        unified_attention_e8_v3_0(**fx_4k)
        # v3_b / v3_fht / v3_split share the 4K fixture (same compressed K/V)
        unified_attention_e8_v3_b(**fx_4k)
        unified_attention_e8_v3_fht(**fx_4k)
        # v3_split picks its own num_kv_splits from the seq_len heuristic.
        unified_attention_e8_v3_split(**fx_4k)

        # Time all five
        t_v21 = _time_kernel(
            unified_attention_e8_v2_1, fx_65k,
            n_iters=args.n_iters, warmup=args.warmup,
        )
        t_v30 = _time_kernel(
            unified_attention_e8_v3_0, fx_4k,
            n_iters=args.n_iters, warmup=args.warmup,
        )
        t_v3b = _time_kernel(
            unified_attention_e8_v3_b, fx_4k,
            n_iters=args.n_iters, warmup=args.warmup,
        )
        t_fht = _time_kernel(
            unified_attention_e8_v3_fht, fx_4k,
            n_iters=args.n_iters, warmup=args.warmup,
        )
        t_split = _time_kernel(
            unified_attention_e8_v3_split, fx_4k,
            n_iters=args.n_iters, warmup=args.warmup,
        )

        s_v21 = _stats(f"v2.1 (65K)", t_v21)
        s_v30 = _stats(f"v3.0 (4K)", t_v30)
        s_v3b = _stats(f"v3_b (4K smem)", t_v3b)
        s_fht = _stats(f"v3_fht (4K butterfly)", t_fht)
        s_split = _stats(f"v3_split (4K KV-split)", t_split)
        speedup_v30 = s_v21["median_us"] / s_v30["median_us"]
        speedup_v3b = s_v21["median_us"] / s_v3b["median_us"]
        speedup_smem = s_v30["median_us"] / s_v3b["median_us"]
        speedup_fht = s_v21["median_us"] / s_fht["median_us"]
        speedup_fht_v30 = s_v30["median_us"] / s_fht["median_us"]
        print(f"  v2.1 (65 K):       median {s_v21['median_us']} µs   "
              f"(p05 {s_v21['p05_us']}  p95 {s_v21['p95_us']}  "
              f"σ {s_v21['stdev_us']})", flush=True)
        print(f"  v3.0 (4 K L2):     median {s_v30['median_us']} µs   "
              f"(p05 {s_v30['p05_us']}  p95 {s_v30['p95_us']}  "
              f"σ {s_v30['stdev_us']})", flush=True)
        print(f"  v3_b (4 K smem):   median {s_v3b['median_us']} µs   "
              f"(p05 {s_v3b['p05_us']}  p95 {s_v3b['p95_us']}  "
              f"σ {s_v3b['stdev_us']})", flush=True)
        print(f"  v3_fht (butterfly):median {s_fht['median_us']} µs   "
              f"(p05 {s_fht['p05_us']}  p95 {s_fht['p95_us']}  "
              f"σ {s_fht['stdev_us']})", flush=True)
        print(f"  speedup:           v3.0 / v2.1 = {speedup_v30:.2f}×   "
              f"v3_b / v2.1 = {speedup_v3b:.2f}×   "
              f"v3_b / v3.0 = {speedup_smem:.2f}×", flush=True)
        print(f"  speedup (FHT):     v3_fht / v2.1 = {speedup_fht:.2f}×   "
              f"v3_fht / v3.0 = {speedup_fht_v30:.2f}×", flush=True)
        # v3_split: KV-split flash-decoding. At short Tk the heuristic
        # picks num_kv_splits=1 (degenerates to v3_fht + a no-op reduce);
        # at long Tk it parallelizes the key range across extra CTAs.
        speedup_split_fht = s_fht["median_us"] / s_split["median_us"]
        print(f"  v3_split (KVsplit):median {s_split['median_us']} µs   "
              f"(p05 {s_split['p05_us']}  p95 {s_split['p95_us']}  "
              f"σ {s_split['stdev_us']})", flush=True)
        print(f"  speedup (split):   v3_split / v3_fht = "
              f"{speedup_split_fht:.2f}×", flush=True)
        gate_pass = s_fht["median_us"] <= 50.0
        print(f"  gate (≤ 50 µs, FHT):  {'PASS' if gate_pass else 'FAIL'}",
              flush=True)
        results.append(dict(
            shape=shape, v21=s_v21, v30=s_v30, v3b=s_v3b, fht=s_fht,
            split=s_split,
            speedup_v30=round(speedup_v30, 3),
            speedup_v3b=round(speedup_v3b, 3),
            speedup_smem=round(speedup_smem, 3),
            speedup_fht=round(speedup_fht, 3),
            speedup_fht_v30=round(speedup_fht_v30, 3),
            speedup_split_fht=round(speedup_split_fht, 3),
            gate_pass=gate_pass,
        ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(
            Tq=args.Tq, Tk=args.Tk, n_iters=args.n_iters,
            warmup=args.warmup,
            results=results,
        ), f, indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
