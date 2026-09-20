"""Where does a CPU decode step actually go?

Qwen3.8-Flash-Next decodes at ~1.7 tok/s on a 32-vCPU Genoa. Both rooflines say that is
not a hardware limit:

    6B active params x 3 bpw = 2.25 GB/token  ->  ~3.8 GB/s at 1.7 tok/s
    Genoa DDR5 (12ch)        ~100-460 GB/s    ->  we use ~1-4% of bandwidth
    ~6B decode+FMA ops/token ->  single-digit % of what 16 AVX-512 cores can do

So the time is going somewhere other than weight fetch or arithmetic, and the prime
suspect is in every CPU run's log: 36 of 48 layers are GatedDeltaNet, and transformers
says their recurrent math is running as reference PyTorch ("correct but much slower").
GLQ's fused CPU kernel only covers the linear layers.

This attributes decode time to MODULE CLASSES rather than bare aten ops, because
"aten::bmm 31%" does not say whether that bmm is the delta rule or an expert matvec.
Module-level `record_function` wrappers give buckets you can act on.

Deliberately small: profiler overhead is real, so a handful of steps after a proper warmup.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=4, help="decode steps to profile")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device-map", dest="device_map", default="cpu")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--wrap-linears", dest="wrap_linears", action="store_true",
                    help="also wrap the ~49k leaf E8RHTLinears (adds real overhead)")
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    import glq.hf_integration  # noqa: F401  MUST precede from_pretrained
    from transformers import AutoConfig, AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_model import resolve_hf_class

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cls = resolve_hf_class(cfg)
    print(f"loading {cls.__name__} ...", flush=True)
    t0 = time.perf_counter()
    model = cls.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                device_map=args.device_map, trust_remote_code=True)
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- buckets by module NAME ------------------------------------------------------
    # Class names collapse every quantized matvec into "E8RHTLinear". The decision needs
    # hyper_connection (dense bf16, 1.23 GiB/token) separated from experts (3 bpw, sparse,
    # 0.85 GiB/token) and from the GDN path -- so match on the dotted path instead.
    # First match wins, so order matters: the specific before the general.
    # No catch-all ".mlp": `layers.N.mlp` is visited before `layers.N.mlp.experts`, so a
    # generic bucket claims the whole MoE subtree and the experts vanish inside a row
    # labelled "mlp (dense)". List the children instead, and let the parent go unclaimed.
    NAME_BUCKETS = [
        ("hyper_connection (bf16 dense)", "hyper_connection"),
        ("indexer", ".indexer"),
        ("experts (GLQ)", ".experts"),
        ("shared_expert (GLQ)", "shared_expert"),
        ("router", "gate"),
        ("ple", ".ple"),
        ("linear_attn (GDN)", "linear_attn"),
        ("self_attn (sparse)", "self_attn"),
    ]

    def _bucket_for(name: str):
        for label, needle in NAME_BUCKETS:
            if needle in name:
                return label
        return None
    live: dict[int, object] = {}
    handles = []

    def _pre_named(label):
        def _pre(mod, _inp):
            ctx = record_function(f"MOD::{label}")
            ctx.__enter__()
            live[id(mod)] = ctx
        return _pre

    def _post(mod, _inp, _out):
        ctx = live.pop(id(mod), None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    # Wrap only the TOP-most module in each bucket, so time is not double-counted down a
    # nested chain (an expert linear inside an experts container inside an MoE block).
    wrapped = collections.Counter()
    claimed: list[str] = []
    for name, m in model.named_modules():
        b = _bucket_for(name)
        if b is None:
            continue
        if any(name.startswith(c + ".") for c in claimed):
            continue                      # an ancestor already owns this subtree
        claimed.append(name)
        handles.append(m.register_forward_pre_hook(_pre_named(b)))
        handles.append(m.register_forward_hook(_post))
        wrapped[b] += 1
    print("wrapped buckets:", dict(wrapped.most_common(12)), flush=True)

    # ---- make the fused CPU kernel visible -------------------------------------------
    # `_trellis_linear_apply` is a staticmethod and the single choke point both HF and
    # vLLM's MoE fallback go through. The kernel behind it is a pybind11 call, which
    # torch.profiler cannot see at all -- so its absence from the first trace was expected,
    # not evidence that it never ran. Wrapping here also COUNTS invocations, which is what
    # proves the fused path is live.
    from glq.quantized_linear import E8RHTLinear
    _orig_apply = E8RHTLinear._trellis_linear_apply
    calls = {"n": 0}

    def _counted_apply(*a, **kw):
        calls["n"] += 1
        with record_function("GLQ::trellis_linear_apply"):
            return _orig_apply(*a, **kw)

    E8RHTLinear._trellis_linear_apply = staticmethod(_counted_apply)

    enc = tok(["The capital city of New Zealand is"], return_tensors="pt")
    enc = {k: v.to("cpu") for k, v in enc.items()}

    with torch.no_grad():
        print(f"warmup {args.warmup} tokens ...", flush=True)
        model.generate(**enc, max_new_tokens=args.warmup, min_new_tokens=args.warmup,
                       do_sample=False)

        print(f"profiling {args.steps} decode steps ...", flush=True)
        calls["n"] = 0          # warmup must not be counted into the per-step figure
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            model.generate(**enc, max_new_tokens=args.steps, min_new_tokens=args.steps,
                           do_sample=False)
        wall = time.perf_counter() - t0

    for h in handles:
        h.remove()
    E8RHTLinear._trellis_linear_apply = staticmethod(_orig_apply)
    print(f"\n_trellis_linear_apply invocations: {calls['n']} "
          f"({calls['n'] / max(1, args.steps):.0f}/step)"
          + ("   <-- ZERO: the fused path never ran" if calls["n"] == 0 else ""))

    ev = prof.key_averages()
    total_self = sum(e.self_cpu_time_total for e in ev) or 1.0

    print(f"\nwall for {args.steps} steps: {wall:.2f}s "
          f"({args.steps / wall:.2f} tok/s incl. profiler overhead)")

    print("\n=== module-class buckets (total CPU time, inclusive) ===")
    mods = [e for e in ev if e.key.startswith("MOD::")]
    for e in sorted(mods, key=lambda e: -e.cpu_time_total)[: args.top]:
        print(f"  {e.cpu_time_total / 1000:10.1f} ms  n={e.count:<6d} {e.key[5:]}")

    print("\n=== top aten ops by SELF cpu time ===")
    ops = [e for e in ev if not e.key.startswith("MOD::")]
    for e in sorted(ops, key=lambda e: -e.self_cpu_time_total)[: args.top]:
        pct = 100.0 * e.self_cpu_time_total / total_self
        print(f"  {e.self_cpu_time_total / 1000:10.1f} ms  {pct:5.1f}%  n={e.count:<7d} {e.key}")


if __name__ == "__main__":
    main()
