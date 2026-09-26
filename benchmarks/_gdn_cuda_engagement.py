"""Did the fused CUDA GDN decode kernel actually run, and what is it worth?

A B=1 A/B between `VLLM_GDN_DECODE_KERNEL=cuda` and `=triton` on a GLQ checkpoint measured
-0.26% and -0.09% on two different models. That is indistinguishable from a flag that does
nothing, and the only engagement evidence was vLLM's init-time line "GDN decode kernel: …",
which proves the flag was SET, not that different code executed under the timer.

`qwen_gdn_linear_attn.py:907` chooses between two distinct custom ops:

    use_fused_gdn_decode = (self.enable_fused_gdn_decode
        and hidden_states.dtype == torch.bfloat16
        and self.norm.weight.dtype in (torch.bfloat16, torch.float32))
    -> torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed   (fused)
    -> torch.ops.vllm.qwen_gdn_attention_core                     (not)

So the mechanism is observable: count the calls to each. This script wraps both ops and
reports the counts alongside per-op self time, which separates the two hypotheses:

    H1  the fused kernel runs, but the GDN recurrence is ~0% of a GLQ decode step
    H2  the flag changes nothing that executes, and both arms ran the same path

Under H1 the counts differ and the op's self time is a tiny share of the step. Under H2 the
counts are identical and the null was an artifact.

Runs IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0): the layers live in the EngineCore
subprocess otherwise, where a monkeypatched counter in this process would never be called --
and would silently report zero for both arms, which looks like evidence.

Eager, because a FULL cudagraph replay does not re-enter Python, so counts would reflect
capture only. Eager inflates non-kernel overhead, so it UNDERSTATES the kernel's share --
the honest direction for a null.

    python benchmarks/_gdn_cuda_engagement.py --model <repo> --kernel cuda|triton
"""
from __future__ import annotations

import argparse
import os

_OPS = ("qwen_gdn_attention_core", "qwen_gdn_attention_core_fused_norm_packed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kernel", choices=["cuda", "triton"], required=True)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--max-model-len", dest="max_model_len", type=int, default=2048)
    ap.add_argument("--gpu-mem", dest="gpu_mem", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    # Both must be set before vllm is imported: the first is read when the engine decides
    # whether to fork, the second when each GDN layer is constructed.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_GDN_DECODE_KERNEL"] = args.kernel

    import torch
    import glq_vllm  # noqa: F401  registers the GLQ quantization method

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, quantization="glq", dtype=args.dtype,
              trust_remote_code=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=1,
              enforce_eager=True, disable_log_stats=True)

    # Wrap AFTER load: construction-time calls are not decode calls, and counting them
    # would conflate "the op exists" with "the op ran under the timer".
    calls = {name: 0 for name in _OPS}

    def _wrap(name):
        orig = getattr(torch.ops.vllm, name)

        def counted(*a, **kw):
            calls[name] += 1
            return orig(*a, **kw)
        return orig, counted

    originals = {}
    for name in _OPS:
        orig, counted = _wrap(name)
        originals[name] = orig
        setattr(torch.ops.vllm, name, counted)

    prompt = "Explain in one paragraph why lattice quantization preserves model quality."
    sp_warm = SamplingParams(max_tokens=args.warmup, temperature=0.0, ignore_eos=True)
    llm.generate([prompt], sp_warm, use_tqdm=False)

    # Counts are reset after warmup so the reported number is the measured window only.
    for name in _OPS:
        calls[name] = 0

    from torch.profiler import ProfilerActivity, profile
    sp = SamplingParams(max_tokens=args.decode, temperature=0.0, ignore_eos=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = llm.generate([prompt], sp, use_tqdm=False)
    n_tok = len(out[0].outputs[0].token_ids)

    print(f"\nENGAGEMENT kernel={args.kernel} dtype={args.dtype} decode_tokens={n_tok}")
    for name in _OPS:
        print(f"  CALLS {name} = {calls[name]}")

    # The share that matters: this op's total device time against everything measured.
    tot = {}
    for ev in prof.key_averages():
        tot[ev.key] = (getattr(ev, "self_device_time_total", 0.0)
                       or getattr(ev, "self_cuda_time_total", 0.0))
    all_dev = sum(v for v in tot.values() if v > 0)

    def _attributed(name: str) -> float:
        """Sum only the events belonging to THIS op.

        `"qwen_gdn_attention_core" in "qwen_gdn_attention_core_fused_norm_packed"` is True, so
        a plain substring test credits the fused op's time to the unfused one as well — which
        reported both rows as an identical 15565 us and made the two paths look equal when one
        of them had run zero times.
        """
        other = [o for o in _OPS if o != name and name in o]
        return sum(v for k, v in tot.items()
                   if name in k and not any(o in k for o in other))

    for name in _OPS:
        us = _attributed(name)
        pct = (100.0 * us / all_dev) if all_dev else 0.0
        print(f"  DEVICE_US {name} = {us:.0f} ({pct:.2f}% of {all_dev:.0f} us total)")

    print("  TOP DEVICE OPS")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:8]:
        if v > 0:
            print(f"    {v / all_dev * 100:5.2f}%  {v:10.0f} us  {k[:70]}")

    for name, orig in originals.items():
        setattr(torch.ops.vllm, name, orig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
