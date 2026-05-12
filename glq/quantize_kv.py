"""CLI: build a per-layer KV bpw allocation map for a HF model.

Loads ``--model`` in bf16, profiles per-layer KV reconstruction MSE on a
calibration corpus, runs the greedy marginal-gain allocator, and writes
the resulting ``{layer_idx: bpw}`` map to a JSON file. The map plugs
directly into ``GLQQuantizedCache(bpw_map=...)``.

Example:
    python -m glq.quantize_kv \\
        --model unsloth/gemma-4-E4B-it \\
        --bpw 3.0 \\
        --e8-method e8_relaxed \\
        --output kv_bpw_map.json

    # Then at inference:
    import json
    from glq.kv_cache import GLQQuantizedCache
    bpw_map = {int(k): v for k, v in json.load(open("kv_bpw_map.json")).items()}
    cache = GLQQuantizedCache(model.config, quant_method="e8_relaxed",
                              bpw_map=bpw_map)
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser(description="GLQ KV bpw allocator")
    p.add_argument("--model", required=True,
                   help="HF model id or local path (bf16 weights)")
    p.add_argument("--output", required=True,
                   help="Path to write JSON allocation map")
    p.add_argument("--bpw", type=float, required=True,
                   help="Target average bpw across cache layers")
    p.add_argument("--e8-method", default="e8_relaxed",
                   choices=("e8_strict", "e8_relaxed"),
                   help="Which E8 codebook to use for 2/4 bpw entries")
    p.add_argument("--nsamples", type=int, default=32,
                   help="Number of calibration sequences")
    p.add_argument("--seqlen", type=int, default=512,
                   help="Tokens per calibration sequence")
    p.add_argument("--device", default="cuda",
                   help="Device for the forward pass")
    p.add_argument("--corpus", default="wikitext",
                   help="Calibration corpus: 'wikitext' or 'c4'")
    p.add_argument("--allowed-bpws", default="2,4,8,16",
                   help="Comma-separated candidate bpws "
                        "(must be subset of {2, 4, 8, 16})")
    args = p.parse_args()

    allowed = tuple(int(x) for x in args.allowed_bpws.split(","))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import glq.hf_integration  # noqa: F401
    from .kv_sensitivity import profile_and_allocate

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    print(f"  load: {time.time()-t0:.1f}s", flush=True)

    allocation = profile_and_allocate(
        model, tok,
        target_avg_bpw=args.bpw,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        candidate_bpws=allowed,
        e8_method=args.e8_method,
        calibration_corpus=args.corpus,
        device=args.device,
    )

    # Serialize with string keys so it round-trips via plain json.
    out = {str(k): int(v) for k, v in sorted(allocation.items())}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
