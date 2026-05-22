"""v0.5 Phase 3 smoke test: inline-dequant attention produces coherent output.

Loads a model with E8 KV compression active and the v0.5 inline-dequant
env var enabled, sends "The capital of France is", expects "Paris" in
the response. Catches the obvious failure modes (kernel returns garbage,
crashes during prefill, infinite loop) early — Phase 4 will follow up
with quality + perf benchmarks.

Usage::

    GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 \\
        GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 \\
        GLQ_KV_E8_INLINE_DEQUANT=1 \\
        python benchmarks/smoke_inline_dequant.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--expected", default="Paris",
                   help="substring expected in the generation")
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    print(f"=== inline-dequant smoke — {args.model} ===", flush=True)
    print(f"  GLQ_KV_QUANT={os.environ.get('GLQ_KV_QUANT', '<unset>')}",
          flush=True)
    print(f"  GLQ_KV_E8_SIDECAR={os.environ.get('GLQ_KV_E8_SIDECAR', '<unset>')}",
          flush=True)
    print(f"  GLQ_KV_E8_SIDECAR_READ="
          f"{os.environ.get('GLQ_KV_E8_SIDECAR_READ', '<unset>')}",
          flush=True)
    print(f"  GLQ_KV_E8_INLINE_DEQUANT="
          f"{os.environ.get('GLQ_KV_E8_INLINE_DEQUANT', '<unset>')}",
          flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem)
    print(f"  load: {time.time() - t0:.1f}s", flush=True)

    # Gemma-4-E4B-it is an instruct model — use ``llm.chat`` so the
    # tokenizer applies its chat template. Raw ``llm.generate`` produces
    # degenerate output (the model just echoes the prompt).
    t0 = time.time()
    out = llm.chat(
        [[{"role": "user", "content": args.prompt}]],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    print(f"  generate: {time.time() - t0:.2f}s", flush=True)

    text = out[0].outputs[0].text
    print(f"\nPrompt: {args.prompt!r}", flush=True)
    print(f"Output: {text!r}", flush=True)

    if args.expected.lower() in text.lower():
        print(f"\n✓ smoke pass: {args.expected!r} found in output", flush=True)
        return 0
    print(f"\n✗ smoke fail: {args.expected!r} NOT in output", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
