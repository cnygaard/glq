"""Paired A/B for the fused CPU MoE path on a real checkpoint, from ONE model load.

`GLQ_HF_MOE_CPU_FUSED` is read inside `_try_fused_cpu`, per forward — so both arms run in
the same process against the same weights. That matters for more than the 140 s load: a
two-process comparison cannot distinguish "the fused op is wrong" from "the two loads
differ", and on a 512-expert checkpoint that ambiguity is expensive to resolve.

Three measurements, in this order, because the middle one mutates state:

1. ``loop_before``  — flag off. The shipped per-expert Python path.
2. ``fused``        — flag on. First call also re-homes every expert's packed codes into
                      one contiguous (E, tiles, 16R) buffer and re-points the per-expert
                      linears at slices of it.
3. ``loop_after``   — flag off again, now running on those slices.

``loop_after`` must equal ``loop_before`` **exactly**. It is the same code on the same
values; anything else means the re-home corrupted or reordered the weights, and it is the
only check here that can be bit-exact. ``fused`` is compared to a tolerance instead: the
loop rounds the gate_up output to the activation dtype before the gated multiply while the
op stays fp32, and the two reduce in different orders.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT = "The capital city of New Zealand is"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tokens", type=int, default=8,
                    help="greedy tokens to also compare as text")
    args = ap.parse_args()

    import torch
    import glq.hf_integration  # noqa: F401  MUST precede from_pretrained
    from transformers import AutoConfig, AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_model import resolve_hf_class

    from glq.fused_experts import GLQStackedGatedExperts

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cls = resolve_hf_class(cfg)
    print(f"loading {cls.__name__} dtype={args.dtype} ...", flush=True)
    t0 = time.perf_counter()
    model = cls.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                device_map="cpu", trust_remote_code=True)
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    containers = [m for m in model.modules() if isinstance(m, GLQStackedGatedExperts)]
    print(f"GLQStackedGatedExperts containers: {len(containers)}", flush=True)
    if not containers:
        print("AB_FAIL no stacked gated expert containers — nothing to compare")
        return 2
    why = containers[0]._glq_moe_cpu_refusal()
    print(f"gate says: {why or 'eligible'}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tok([PROMPT], return_tensors="pt")

    def logits(flag: str):
        os.environ["GLQ_HF_MOE_CPU_FUSED"] = flag
        t = time.perf_counter()
        with torch.no_grad():
            out = model(**enc).logits[0, -1].float()
        return out, time.perf_counter() - t

    loop_before, t_loop = logits("0")
    fused, t_fused = logits("1")
    loop_after, _ = logits("0")

    # Did the fused path actually engage? A stale gate would make every number below agree
    # perfectly and mean nothing.
    engaged = containers[0]._stacked_is_live()
    print(f"stacked buffers live: {engaged}", flush=True)

    ok = True

    same = torch.equal(loop_before, loop_after)
    print(f"re-home is value-preserving: loop_after == loop_before -> {same}")
    if not same:
        d = (loop_after - loop_before).abs().max().item()
        print(f"  max abs diff {d:.3e}  <-- the re-home changed the weights")
        ok = False

    d = (fused - loop_before).abs()
    denom = loop_before.abs().max().item() or 1.0
    cos = torch.nn.functional.cosine_similarity(fused, loop_before, dim=0).item()
    print(f"fused vs loop: max_abs={d.max().item():.4e} "
          f"rel={d.max().item() / denom:.3e} cosine={cos:.9f}")
    print(f"top1 agree: {int(fused.argmax()) == int(loop_before.argmax())} "
          f"(fused={int(fused.argmax())} loop={int(loop_before.argmax())})")
    top5f = set(fused.topk(5).indices.tolist())
    top5l = set(loop_before.topk(5).indices.tolist())
    print(f"top5 overlap: {len(top5f & top5l)}/5")
    if cos < 0.9999:
        print("  cosine below 0.9999 -- treat as a failure, not rounding")
        ok = False

    print(f"single-forward wall: loop={t_loop:.2f}s fused={t_fused:.2f}s "
          f"({t_loop / t_fused:.2f}x)   # one forward, not a throughput number")

    if args.tokens:
        for flag in ("0", "1"):
            os.environ["GLQ_HF_MOE_CPU_FUSED"] = flag
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=args.tokens,
                                   min_new_tokens=args.tokens, do_sample=False)
            arm = "fused" if flag == "1" else "loop "
            print(f"{arm}: {tok.decode(o[0], skip_special_tokens=True)!r}", flush=True)

    print("AB_OK" if ok and engaged else "AB_FAIL")
    return 0 if (ok and engaged) else 1


if __name__ == "__main__":
    sys.exit(main())
