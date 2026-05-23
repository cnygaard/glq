"""v0.5 Phase 4 perf bench — E8 KV decode tok/s at B=1, B=4.

Run twice (with and without ``GLQ_KV_E8_INLINE_DEQUANT=1``) to measure
the new path's speedup vs v0.3.5's pre-decompress workspace. Both
runs share the same model, sidecar setup, and max_model_len so
differences come solely from the attention dispatch.

Usage::

    # v0.3.5 baseline:
    GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 \\
        GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 \\
        python benchmarks/bench_e4b_kv_inline_dequant.py \\
        --label v035 --out /tmp/logs/bench_v035.json

    # Phase 3 inline-dequant:
    GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 \\
        GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 \\
        GLQ_KV_E8_INLINE_DEQUANT=1 \\
        python benchmarks/bench_e4b_kv_inline_dequant.py \\
        --label v05_inline --out /tmp/logs/bench_v05.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--prompt-len", type=int, default=32,
                   help="prompt token budget (kept short to isolate decode)")
    p.add_argument("--decode-len", type=int, default=256,
                   help="output tokens to time")
    p.add_argument("--warmup", type=int, default=8,
                   help="warmup decode tokens before timing")
    p.add_argument("--out", default="/tmp/logs/bench_e4b_kv.json")
    p.add_argument("--label", default="run")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    print(f"=== {args.label} — {args.model} ===", flush=True)
    for k in ("GLQ_KV_QUANT", "GLQ_KV_E8_SIDECAR", "GLQ_KV_E8_SIDECAR_READ",
              "GLQ_KV_E8_COMPRESSED_ALLOC", "GLQ_KV_E8_INLINE_DEQUANT"):
        print(f"  {k}={os.environ.get(k, '<unset>')}", flush=True)

    from vllm import LLM, SamplingParams

    t_load = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", quantization="glq",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem)
    t_load = time.time() - t_load
    print(f"  load: {t_load:.1f}s", flush=True)

    # Build prompts of roughly --prompt-len tokens. Using a chat template
    # via ``llm.chat`` since Gemma-4-E4B-it is instruct-tuned.
    prompt_user = (
        "Write a single Python function that computes the Fibonacci "
        "sequence. Be concise."
    )
    prompts_b1 = [[{"role": "user", "content": prompt_user}]]
    prompts_b4 = [[{"role": "user", "content": prompt_user}]] * 4

    # Warm-up — let CUDA-graph capture + autotuner settle.
    warm_sp = SamplingParams(temperature=0.0, max_tokens=args.warmup)
    llm.chat(prompts_b1, sampling_params=warm_sp)
    llm.chat(prompts_b4, sampling_params=warm_sp)

    def _time(prompts, label):
        sp = SamplingParams(temperature=0.0, max_tokens=args.decode_len)
        # Use ignore_eos so we always get exactly --decode-len tokens.
        sp.ignore_eos = True
        t0 = time.time()
        out = llm.chat(prompts, sampling_params=sp)
        wall = time.time() - t0
        tokens = sum(len(o.outputs[0].token_ids) for o in out)
        per_req = tokens / len(out)
        tps = tokens / wall
        print(f"  {label}: {tokens} tok in {wall:.2f}s = {tps:.1f} tok/s "
              f"(per-req {per_req:.0f} tok, wall {wall*1000/per_req:.1f}ms/tok)",
              flush=True)
        return dict(label=label, wall=wall, tokens=tokens,
                    per_req=per_req, tps=tps,
                    batch=len(prompts))

    runs = []
    for trial in range(2):
        runs.append(_time(prompts_b1, f"B=1 trial{trial}"))
        runs.append(_time(prompts_b4, f"B=4 trial{trial}"))

    result = dict(
        label=args.label,
        model=args.model,
        load_s=t_load,
        env={k: os.environ.get(k) for k in (
            "GLQ_KV_QUANT", "GLQ_KV_E8_SIDECAR", "GLQ_KV_E8_SIDECAR_READ",
            "GLQ_KV_E8_COMPRESSED_ALLOC", "GLQ_KV_E8_INLINE_DEQUANT",
        )},
        runs=runs,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
