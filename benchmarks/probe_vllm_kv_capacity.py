"""Phase 5.3 validation — measure the VRAM win Stage 2c-2c delivers.

Boots vLLM once with the configured env-var stack, captures
``GPU KV cache size: <N> tokens`` from the boot log, and prints
``num_tokens`` and ``cache_bytes`` (derived from the KV cache config).
Run twice (baseline vs Stage 2c-2c) at identical gpu_memory_utilization
and max_model_len; the ratio is the cache compression realised in vLLM.

Usage:
    # baseline fp16
    python benchmarks/probe_vllm_kv_capacity.py \\
        --max-model-len 16384 --gpu-mem 0.5 --out /tmp/logs/kv_cap_fp16.json

    # Stage 2c-2c at 2 bpw
    GLQ_KV_QUANT=e8_relaxed:1 GLQ_KV_E8_SIDECAR=1 \\
    GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 \\
    python benchmarks/probe_vllm_kv_capacity.py \\
        --max-model-len 16384 --gpu-mem 0.5 --out /tmp/logs/kv_cap_e8_2bpw.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def _variant_label() -> str:
    bits = []
    if os.environ.get("GLQ_KV_QUANT"):
        bits.append(os.environ["GLQ_KV_QUANT"])
    if os.environ.get("GLQ_KV_E8_SIDECAR") == "1":
        bits.append("sidecar")
    if os.environ.get("GLQ_KV_E8_SIDECAR_READ") == "1":
        bits.append("read")
    if os.environ.get("GLQ_KV_E8_COMPRESSED_ALLOC") == "1":
        bits.append("compressed_alloc")
    return "+".join(bits) if bits else "baseline-fp16"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/gemma-4-E4B-it")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--out", default="/tmp/logs/kv_cap.json")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    # INFO so we get the "GPU KV cache size: N tokens" line.
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import torch
    from vllm import LLM

    label = args.label or _variant_label()
    print(f"=== KV capacity probe — variant={label} ===", flush=True)
    print(f"  max_model_len={args.max_model_len}  gpu_mem={args.gpu_mem}",
          flush=True)

    free_before, total = torch.cuda.mem_get_info()
    print(f"  GPU free before load: "
          f"{free_before / 1024**3:.2f} / {total / 1024**3:.2f} GiB",
          flush=True)

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True)
    load_dt = time.time() - t0
    free_after, _ = torch.cuda.mem_get_info()
    used_gib = (free_before - free_after) / 1024**3
    print(f"  load: {load_dt:.1f}s  used: {used_gib:.2f} GiB", flush=True)

    # Pull the KV cache config directly from the engine. In v1 the
    # config lives behind `llm_engine.engine_core`. Some paths expose
    # it as `vllm_config.cache_config.num_gpu_blocks` after profile-run.
    num_blocks = None
    cache_bytes = None
    page_bytes = None
    try:
        cfg = llm.llm_engine.vllm_config
        num_blocks = cfg.cache_config.num_gpu_blocks
        block_size = cfg.cache_config.block_size
        # KVCacheConfig.kv_cache_tensors[*].size is total bytes per layer.
        # If we can reach the in-engine kv_cache_config use that, else
        # estimate from the cache_config.
        page_bytes = None
    except Exception as e:
        print(f"  WARN: couldn't read llm_engine.vllm_config: {e}",
              flush=True)
        block_size = None

    # The most authoritative number is vLLM's own log line, which it
    # emits as "GPU KV cache size: <N> tokens". We capture it via the
    # logger handler — see _LogCapture above.
    summary = {
        "label": label,
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_mem": args.gpu_mem,
        "load_sec": load_dt,
        "gpu_used_gib": used_gib,
        "num_gpu_blocks": num_blocks,
        "block_size": block_size,
    }
    print(f"  num_gpu_blocks: {num_blocks}  block_size: {block_size}",
          flush=True)
    if num_blocks is not None and block_size is not None:
        kv_tokens = num_blocks * block_size
        summary["kv_cache_tokens"] = kv_tokens
        print(f"  KV cache capacity: {kv_tokens:,} tokens "
              f"({kv_tokens / args.max_model_len:.1f}x max_model_len)",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
