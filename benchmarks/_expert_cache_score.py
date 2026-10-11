"""Score a recorded GLQ routing trace: what hit rate would an expert cache of size S reach?

This is S0 of the expert-cache work, and it exists to answer that question **before** a cache
is built. Strata sized its own design this way rather than by reasoning, and the trace is what
told it the cache was *"never short of capacity — it was handing all of it to two layers"*.

Run the model with the trace on, then score it::

    GLQ_MOE_TRACE_ROUTING=/opt/dlami/nvme/routing.npz \\
        python benchmarks/run_model.py --model <ckpt> --batches 1 --decode 256 ...
    python benchmarks/_expert_cache_score.py /opt/dlami/nvme/routing.npz --gib 10

It replays the trace through the SAME `ResidencyPolicy` the live cache will use
(`glq_vllm/expert_cache.py`), so a hit rate measured here means something about the cache that
gets built. Reported for both admission policies and every eviction policy, because:

* a globally-shared slot counter measured **2.97%** for Strata where per-layer scored 21.4%
  (8 slots/layer) and 70.4% (64) — so the comparison is the headline result, not a footnote;
* and eviction is **not** obviously a win *a priori*. On synthetic routing in
  `tests/test_expert_cache_policy.py` LRU scored 12x *below* no-eviction when the cache was
  far smaller than the working set (thrashing). On GLQ's real decode routing it is the single
  biggest lever. Both are measurements; neither was guessable.

**★ `--phase decode` is the default, and `--phase all` is a trap.** vLLM runs a synthetic
max-model-len forward at startup to size its KV pool. On Flash-Next at `--max-model-len 4096`
that one pass is **89.3% of the trace** and routes to only **495** of 24,576 (layer, expert)
pairs, so every cache scores near-perfectly on it. Including it reported 0.9400 where real
decode was 0.5263. Decode is also the phase the 6.517 tok/s baseline measures.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq_vllm.expert_cache import score_trace            # noqa: E402
from glq_vllm.expert_trace import PHASES, RoutingTrace   # noqa: E402

#: Bytes one expert's pageable blob occupies, for translating GiB -> slots. Read off the
#: safetensors header of Flash-Next 3bpw (`tests/models/qwen3_8_flash_next_glq_3bpw.json`),
#: not estimated: per (layer, expert) it is 3 x trellis_packed I16[6400, 48] = 1.758 MiB plus
#: 0.018 MiB of SU/SV/Wscale = **1.776 MiB**, and 48 x 512 of them is 42.6 GiB. Override with
#: --blob-mib for another model.
#:
#: SV is listed in that 0.018 and should NOT be: `down_proj.SV` is [640] and `gate/up.SV` is
#: [2560] per expert here, so unlike the dense case it IS per-expert and does belong in the
#: arena. Keeping it in the blob is therefore correct, and the 1.0% it contributes is why this
#: is read from a header rather than reasoned about.
_DEFAULT_BLOB_MIB = 1.776


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="an .npz written by GLQ_MOE_TRACE_ROUTING")
    ap.add_argument("--slots", default=None,
                    help="comma list of TOTAL slot counts to score (default: a sweep)")
    ap.add_argument("--per-layer-slots", default=None,
                    help="comma list of slots PER LAYER instead (x n_layers)")
    ap.add_argument("--gib", type=float, default=None,
                    help="score the slot count a VRAM arena of this size affords")
    ap.add_argument("--blob-mib", type=float, default=_DEFAULT_BLOB_MIB,
                    help=f"MiB per expert blob (default {_DEFAULT_BLOB_MIB}, Flash-Next 3bpw)")
    ap.add_argument("--per-routing", action="store_true",
                    help="score the OLD per-routing policy instead of the step policy the "
                         "live arena runs (which protects the slots a step reads). The "
                         "per-routing form under-reports; kept only to reproduce S0.")
    ap.add_argument("--phase", default="decode", choices=PHASES,
                    help="which forwards to score (default decode — 'all' is dominated by "
                         "vLLM's dummy profiling pass and reports ~2x the truth)")
    args = ap.parse_args(argv)

    t = RoutingTrace.load(args.trace)
    if not len(t):
        print(f"{args.trace}: no records — was GLQ_MOE_TRACE_ROUTING set for a run that "
              f"actually routed experts?")
        return 1

    n_layers, n_experts = t.n_layers, t.n_experts
    recs = t.scored_records(args.phase)
    if not recs:
        print(f"{args.trace}: no records in phase {args.phase!r}. Widths seen: "
              f"{t.phase_breakdown()}")
        return 1
    top_k = max(len(ids) for _, ids in recs)
    pairs_touched = len({(li, e) for li, ids in recs for e in ids})
    print(f"trace      : {args.trace}")
    # The layer_keys list is 48 long-prefixed entries; it is provenance, not something to
    # read, and dumping it buries the table below.
    meta_short = t.meta_json if len(t.meta_json) < 300 else t.meta_json[:297] + "..."
    print(f"meta       : {meta_short}")
    print(f"phase      : {args.phase}  ({len(recs)} of {len(t)} records)")
    print(f"records    : {len(recs)}  layers={n_layers}  experts(seen)={n_experts}  "
          f"top_k={top_k}")
    print(f"lookups    : {sum(len(i) for _, i in recs)}")
    print(f"pairs seen : {pairs_touched} of {n_layers * n_experts} "
          f"({100.0 * pairs_touched / max(n_layers * n_experts, 1):.1f}% of {n_layers} x "
          f"{n_experts}) <- the working set; a cache cannot beat this many slots")
    print()

    # The composition of the trace, always, because this is what exposed the profiling run.
    print("forward widths (tokens per forward -> records). width 1 = a decode step;")
    print("a single very wide entry is vLLM's dummy KV-sizing pass, NOT real routing:")
    for w, c in sorted(t.phase_breakdown().items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {w:>6} tokens  {c:>9} records  ({100.0 * c / len(t):5.1f}%)")
    if t.widths_were_derived:
        print("  NOTE: widths were DERIVED from contiguous same-layer runs (this trace "
              "predates\n        the recorded `width` field), not read from the file.")
    print()

    if args.gib is not None:
        afforded = int(args.gib * 1024 / args.blob_mib)
        print(f"a {args.gib:g} GiB arena at {args.blob_mib:g} MiB/expert affords "
              f"{afforded} slots = {afforded // max(n_layers, 1)} per layer")
        print()

    if args.slots:
        slot_list = [int(s) for s in args.slots.split(",") if s.strip()]
    elif args.per_layer_slots:
        slot_list = [int(s) * n_layers for s in args.per_layer_slots.split(",") if s.strip()]
    elif args.gib is not None:
        slot_list = [int(args.gib * 1024 / args.blob_mib)]
    else:
        # A sweep spanning "below one token's lookups" to "everything resident", because the
        # interesting regime is exactly where the quota crosses top_k.
        per_layer = [1, 2, 4, 8, 16, 32, 64, 128]
        slot_list = sorted({q * n_layers for q in per_layer}
                           | {n_layers * n_experts})

    print(f"  {'slots':>7} {'per-lyr':>7}  {'admission':<10} {'eviction':<9} "
          f"{'hit_rate':>9}  {'evictions':>10}  min/max admitted")
    print("  " + "-" * 86)
    for slots in slot_list:
        for per_layer_mode in (False, True):
            for ev in (None, "lru", "lfu"):
                if not per_layer_mode and ev is not None:
                    continue          # the global counter is the baseline; don't cross it
                r = score_trace(recs, n_layers=n_layers, n_experts=n_experts, slots=slots,
                                per_layer=per_layer_mode, eviction=ev,
                                step_mode=not args.per_routing)
                print(f"  {slots:>7} {slots // max(n_layers, 1):>7}  "
                      f"{'per-layer' if per_layer_mode else 'GLOBAL':<10} "
                      f"{ev or 'none':<9} {r.hit_rate:>9.4f}  {r.evictions:>10}  "
                      f"{min(r.admitted_per_layer, default=0)}/"
                      f"{max(r.admitted_per_layer, default=0)}")
        print()

    print("How to read this, with what the Flash-Next L4 trace already said:")
    print("  * GLOBAL vs per-layer: MEASURED A NON-FINDING here (0.5200 vs 0.5263 at 115")
    print("    slots/layer). Strata's 2.97%-vs-21.4% needs slots < n_layers*top_k, i.e. a")
    print("    cache smaller than ONE position's lookups (its 256 vs 480). Ours affords")
    print("    5565 vs 480 = 11.6 positions, so the pathology cannot arise. Keep per-layer")
    print("    anyway: it costs nothing, and it is what makes the quota predictable.")
    print("  * Eviction is THE lever instead: 0.5263 -> 0.8233 with LRU at the same slots,")
    print("    and LRU > LFU at every size >= 16/layer. Opposite of the synthetic fixture,")
    print("    where LRU was 12x worse. This is why the policy was not chosen up front.")
    print("  * 'min admitted' of 0 under GLOBAL is the pathology: whole layers never got a")
    print("    slot, so their lookups missed for the entire run.")
    print("  * 'pairs seen' above is the ceiling. Decode touched 14,109 of 24,576 pairs in")
    print("    367 steps - routing is diffuse, so a small arena cannot do well no matter")
    print("    what the policy is.")
    print("  * A hit rate is NOT a speedup. Converting it needs the measured ms/token/GiB")
    print("    of PCIe traffic; the L4 baseline to beat is 6.517 tok/s at 40 GiB offload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
