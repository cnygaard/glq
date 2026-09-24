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
    ap.add_argument("--flag", default="GLQ_HF_MOE_CPU_FUSED",
                    help="the env switch to A/B. Both GLQ CPU fast paths are read per "
                         "forward, so either one can be toggled against one set of weights")
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

    is_moe = args.flag == "GLQ_HF_MOE_CPU_FUSED"
    containers = [m for m in model.modules() if isinstance(m, GLQStackedGatedExperts)]
    if is_moe:
        print(f"GLQStackedGatedExperts containers: {len(containers)}", flush=True)
        if not containers:
            print("AB_FAIL no stacked gated expert containers — nothing to compare")
            return 2
        print(f"gate says: {containers[0]._glq_moe_cpu_refusal() or 'eligible'}", flush=True)
    else:
        gdn = [m for m in model.modules() if "GatedDeltaNet" in type(m).__name__]
        print(f"GatedDeltaNet modules: {len(gdn)}", flush=True)
        if not gdn:
            print(f"AB_FAIL no GatedDeltaNet modules — nothing for {args.flag} to do")
            return 2

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tok([PROMPT], return_tensors="pt")

    def logits(flag: str):
        """Logits of ONE token, from a fresh cache.

        For the GDN flag this MUST be a decode step, not a prompt forward: the shim only
        handles `seq_len == 1`, so comparing a prefill forward compares the reference
        against itself and reports a perfect match that means nothing. So prime the cache
        with the prompt, then measure the NEXT token -- which is the seq_len==1 path.

        The cache is rebuilt per arm on purpose. GLQ's kernel updates the recurrent state
        in place, so a cache shared between arms would carry one arm's mutation into the
        other; a fresh prefill each time is both correct and a check on that.
        """
        os.environ[args.flag] = flag
        with torch.no_grad():
            if is_moe:
                t = time.perf_counter()
                return model(**enc).logits[0, -1].float(), time.perf_counter() - t
            pre = model(**enc, use_cache=True)
            nxt = pre.logits[0, -1].argmax().view(1, 1)
            t = time.perf_counter()
            step = model(input_ids=nxt, past_key_values=pre.past_key_values,
                         use_cache=True)
            return step.logits[0, -1].float(), time.perf_counter() - t

    logits("0")                 # discard: the first forward pays cold-start page faults
    loop_before, t_loop = logits("0")
    # The FIRST fused forward also re-homes every expert into its stacked buffer, which
    # touches all 42 GiB of expert weights -- most of which the loop never faulted in,
    # since only top_k of num_experts are read per token. Timing that call would report
    # the one-time build as if it were the steady state.
    _, t_build = logits("1")
    fused, t_fused = logits("1")
    loop_after, _ = logits("0")

    # Did the fast path actually engage? Without this every number below could agree
    # perfectly and mean nothing.
    ok = True
    if is_moe:
        engaged = containers[0]._stacked_is_live()
        print(f"stacked buffers live: {engaged}", flush=True)
        same = torch.equal(loop_before, loop_after)
        print(f"re-home is value-preserving: loop_after == loop_before -> {same}")
        if not same:
            d = (loop_after - loop_before).abs().max().item()
            print(f"  max abs diff {d:.3e}  <-- the re-home changed the weights")
            ok = False
    else:
        # The GDN shim leaves no durable state, so engagement is proven by the numbers
        # MOVING at all: identical logits would mean the wrapper never fired.
        engaged = not torch.equal(fused, loop_before)
        print(f"gdn shim changed the logits (i.e. it ran): {engaged}", flush=True)
        same = torch.equal(loop_before, loop_after)
        print(f"reference is unperturbed: loop_after == loop_before -> {same}")
        if not same:
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

    # The threshold is dtype-dependent and the reason is documented, not guessed: the loop
    # rounds each expert's gate_up output to the activation dtype before the gated multiply
    # while the op stays fp32, so under bf16 (eps ~7.8e-3) the two paths genuinely differ.
    # float32 is therefore the CONTROL: same code, same weights, only the dtype moves. If
    # the divergence does not collapse there, it is not rounding.
    floor = {"float32": 0.99999, "float64": 0.99999}.get(args.dtype, 0.999)
    if cos < floor:
        print(f"  cosine below the {args.dtype} floor {floor} -- not rounding")
        ok = False
    elif args.dtype != "float32":
        print(f"  within the {args.dtype} floor {floor}; run --dtype float32 for the "
              f"control that says whether this is rounding or a bug")

    if is_moe:
        print(f"one-time re-home: {t_build - t_fused:.1f}s added to the first forward "
              f"(touches every expert; the loop only ever faults in the routed ones)")
    print(f"steady-state single forward: loop={t_loop:.2f}s fused={t_fused:.2f}s "
          f"({t_loop / t_fused:.2f}x)   # ONE prefill forward, not a decode throughput number")

    if args.tokens:
        for flag in ("0", "1"):
            os.environ[args.flag] = flag
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=args.tokens,
                                   min_new_tokens=args.tokens, do_sample=False)
            arm = "fused" if flag == "1" else "loop "
            print(f"{arm}: {tok.decode(o[0], skip_special_tokens=True)!r}", flush=True)

    print("AB_OK" if ok and engaged else "AB_FAIL")
    return 0 if (ok and engaged) else 1


if __name__ == "__main__":
    sys.exit(main())
