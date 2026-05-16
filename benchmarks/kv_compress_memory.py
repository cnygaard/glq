"""Phase 4.2: VRAM + decode tok/s vs context length at each KV bpw.

For every (compression recipe, context length) combo:
  1. Reset peak CUDA memory stats.
  2. Prefill ``ctx_len`` tokens of a long text into a fresh cache.
  3. Record standing memory (model weights + populated cache, after
     intermediate activations are freed).
  4. Generate ``--n-decode`` more tokens with the populated cache and
     time it.
  5. Report cache_bytes, peak_prefill_bytes, decode tok/s.

The HF ``GLQQuantizedCache`` path stores indices + scales for the
compressed tiers, so ``mem_with_cache - mem_baseline`` is the *real*
bytes-on-card cost of that recipe at that context — the number that
matters for "how many tokens fit on a GB".

Run:
    python benchmarks/kv_compress_memory.py \\
        --model google/gemma-4-E4B-it \\
        --ctx-lens 2048 4096 8192 16384 32768 \\
        --bpw-map /tmp/kv_bpw_e4b_4.0_full.json \\
        --out /tmp/logs/phase4_memory.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import torch


def read_gutenberg_text(url: str,
                        cache_dir: str = "/opt/dlami/nvme/hf_cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    s = text.find("*** START OF")
    if s != -1:
        text = text[text.find("\n", s) + 1:]
    e = text.find("*** END OF")
    if e != -1:
        text = text[:e]
    return text


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_one(model, ids_long: torch.Tensor, ctx_len: int,
                cache_factory, n_decode: int, mem_baseline: int) -> dict:
    """Measure VRAM + decode tok/s for one (ctx_len, cache) combo.

    ``mem_baseline`` is the standing memory before *any* cache exists —
    used to compute the cache's bytes-on-card contribution after
    intermediate activations are freed.
    """
    if ctx_len > ids_long.shape[-1]:
        raise ValueError(
            f"need at least {ctx_len} tokens, only have {ids_long.shape[-1]}")
    ids = ids_long[:, :ctx_len]
    from transformers.cache_utils import DynamicCache

    torch.cuda.empty_cache()
    _sync()
    torch.cuda.reset_peak_memory_stats()

    # ---- prefill ----
    cache = cache_factory() if cache_factory else DynamicCache()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    _sync()
    prefill_time = time.time() - t0
    peak_prefill = torch.cuda.max_memory_allocated()

    # Keep the last-token logits to seed the decode loop; free the rest.
    last_logits = out.logits[:, -1:, :].clone()
    del out
    torch.cuda.empty_cache()
    _sync()
    mem_with_cache = torch.cuda.memory_allocated()
    cache_bytes = mem_with_cache - mem_baseline

    # ---- decode n_decode tokens ----
    torch.cuda.reset_peak_memory_stats()
    cur = last_logits.argmax(dim=-1)  # [B, 1]
    _sync()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_decode):
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)
    _sync()
    decode_time = time.time() - t0
    peak_decode = torch.cuda.max_memory_allocated()
    tok_per_sec = n_decode / max(decode_time, 1e-9)

    return {
        "ctx_len": ctx_len,
        "n_decode": n_decode,
        "prefill_time_sec": prefill_time,
        "decode_time_sec": decode_time,
        "decode_tok_per_sec": tok_per_sec,
        "peak_prefill_bytes": peak_prefill,
        "peak_decode_bytes": peak_decode,
        "mem_with_cache_bytes": mem_with_cache,
        "cache_bytes": cache_bytes,
        "bytes_per_token": cache_bytes / ctx_len,
    }


def build_variants(model_config, *, e8_method: str, residual_length: int,
                   bpw_map_path: str | None,
                   bpws: tuple[int, ...] = (2, 3, 4, 6, 8)) -> list:
    """Return [(label, bpw_label, factory_or_None)] for the sweep.

    ``factory_or_None`` is ``None`` for the fp16 baseline (which uses
    HF's ``DynamicCache`` inside ``measure_one``)."""
    from glq.kv_cache import GLQQuantizedCache

    def factory_for(method, n_stages=1, bpw_map=None):
        def make():
            return GLQQuantizedCache(
                model_config, quant_method=method, n_stages=n_stages,
                bpw_map=bpw_map, residual_length=residual_length)
        return make

    cfg = (model_config.get_text_config(decoder=True)
           if hasattr(model_config, "get_text_config") else model_config)
    n_layers = getattr(cfg, "num_hidden_layers", 32)

    variants: list = [("baseline (fp16)", "16", None)]
    for b in bpws:
        if b == 8:
            variants.append((f"int8 absmax", "8", factory_for("int8")))
            continue
        # Uniform bpw via per-layer map (covers all rungs 2/3/4/5/6/7
        # uniformly without juggling n_stages/secondary_stages).
        uniform_map = {i: b for i in range(n_layers)}
        variants.append(
            (f"{e8_method} {b} bpw", str(b),
             factory_for(e8_method, bpw_map=uniform_map)))

    if bpw_map_path and os.path.exists(bpw_map_path):
        raw = json.load(open(bpw_map_path))
        bpw_map = {int(k): int(v) for k, v in raw.items()}
        avg = sum(bpw_map.values()) / len(bpw_map)
        variants.append(
            (f"{e8_method} mix avg {avg:.2f}", f"~{avg:.2f}",
             factory_for(e8_method, bpw_map=bpw_map)))
    return variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/cache/epub/6593/pg6593.txt")
    p.add_argument("--ctx-lens", type=int, nargs="+",
                   default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--n-decode", type=int, default=32)
    p.add_argument("--residual", type=int, default=8)
    p.add_argument("--e8-method", default="e8_relaxed",
                   choices=("e8_strict", "e8_relaxed"))
    p.add_argument("--bpws", type=int, nargs="+",
                   default=[2, 3, 4, 6, 8],
                   help="Uniform-bpw recipes to test (plus fp16 baseline)")
    p.add_argument("--bpw-map", default=None,
                   help="Optional per-layer mixed-precision JSON")
    p.add_argument("--out", default="/tmp/logs/phase4_memory.json")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"  load: {time.time()-t0:.1f}s", flush=True)
    _sync()
    mem_baseline = torch.cuda.memory_allocated()
    print(f"  baseline (model weights only): {mem_baseline / 1e9:.2f} GB",
          flush=True)

    text = read_gutenberg_text(args.text_url)
    max_ctx = max(args.ctx_lens)
    print(f"  tokenizing for max ctx {max_ctx} ...", flush=True)
    enc = tok(text[: max(max_ctx * 8, 1)], return_tensors="pt").input_ids
    if enc.shape[-1] < max_ctx:
        raise RuntimeError(
            f"text not long enough: got {enc.shape[-1]} tokens, need {max_ctx}")
    ids_long = enc[:, :max_ctx].to("cuda")
    print(f"  tokens prepared: {ids_long.shape[-1]}", flush=True)

    variants = build_variants(
        model.config, e8_method=args.e8_method,
        residual_length=args.residual, bpw_map_path=args.bpw_map,
        bpws=tuple(args.bpws))
    print(f"  variants: {[v[0] for v in variants]}", flush=True)

    results = {"args": vars(args), "mem_baseline_bytes": mem_baseline,
               "rows": []}
    print(f"\n{'variant':<28} {'bpw':>5} {'ctx':>6} "
          f"{'cache_GB':>9} {'B/tok':>7} {'peak_GB':>8} {'tok/s':>7}",
          flush=True)
    for label, bpw_label, factory in variants:
        for ctx_len in args.ctx_lens:
            try:
                res = measure_one(model, ids_long, ctx_len, factory,
                                  args.n_decode, mem_baseline)
            except torch.cuda.OutOfMemoryError:
                print(f"  {label:<26} {bpw_label:>5} {ctx_len:>6}  OOM",
                      flush=True)
                results["rows"].append({
                    "variant": label, "bpw_label": bpw_label,
                    "ctx_len": ctx_len, "oom": True})
                torch.cuda.empty_cache()
                continue
            row = {"variant": label, "bpw_label": bpw_label, **res}
            results["rows"].append(row)
            print(f"  {label:<26} {bpw_label:>5} {ctx_len:>6}  "
                  f"{res['cache_bytes']/1e9:>7.2f}   "
                  f"{res['bytes_per_token']:>5.0f}  "
                  f"{res['peak_prefill_bytes']/1e9:>6.2f}   "
                  f"{res['decode_tok_per_sec']:>5.1f}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
