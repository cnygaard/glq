"""Phase 4: sliding-window perplexity across KV bpw sweep.

For each variant in the bpw menu (16=baseline, 8=INT8, 6/4/2=E8, plus
mixed-precision from the sensitivity allocator) we compute true
sliding-window perplexity:

  - process the input in fixed-size chunks
  - each new chunk's attention reads from the accumulated cache
    (compressed for positions older than ``residual_length``)
  - NLL is summed across every predicted position, divided by total
    predictions → perplexity.

Distinguishes from the earlier "two-chunk PPL" probe because it
amortizes across many chunks: most positions sit deep in the past
where compression has been compounding, giving a properly weighted
read of compression-induced quality loss.

Run:
    python benchmarks/kv_compress_quality.py \\
        --model unsloth/gemma-4-E4B-it \\
        --text-url https://www.gutenberg.org/cache/epub/6593/pg6593.txt \\
        --tokens 4096 --residual 8 64 \\
        --bpw-map /tmp/kv_bpw_e4b_3.0.json \\
        --out /tmp/logs/phase4_ppl.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request

import torch


def read_gutenberg_text(url: str, cache_dir: str = "/opt/dlami/nvme/hf_cache") -> str:
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


def sliding_window_ppl(model, ids: torch.Tensor, cache_factory,
                       chunk: int, verbose: bool = False) -> dict:
    """Compute sliding-window perplexity by feeding ``ids`` chunk-by-chunk
    into ``model`` with a fresh cache from ``cache_factory()``.

    Baseline (``cache_factory`` is None) uses ``DynamicCache`` so it
    accumulates context across chunks just like the compressed variants
    — the comparison is apples-to-apples (same effective context window
    at every position; only the cache compression differs).

    Returns dict with ppl, total_tokens, total_time, peak VRAM.
    """
    if cache_factory is None:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
    else:
        cache = cache_factory()
    n = ids.shape[-1]
    total_nll = 0.0
    total_n = 0
    last_logit = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_ids = ids[:, start:end]
        with torch.no_grad():
            out = model(
                input_ids=chunk_ids,
                past_key_values=cache,
                use_cache=True,
            )
        cache = out.past_key_values
        # Predictions for positions [start:end] come from:
        #   pos start    : last_logit (from previous chunk) — None on first chunk
        #   pos start+1.. : out.logits[:, :-1]
        if start == 0:
            # First chunk: predictions for [1:end] from out.logits[:, :-1]
            targets = chunk_ids[0, 1:]
            target_logits = out.logits[0, :-1, :]
        else:
            # General chunk: predictions for [start:end] use
            #   first via last_logit, rest via out.logits[:, :-1]
            targets = chunk_ids[0]
            target_logits = torch.cat([last_logit[0], out.logits[0, :-1, :]], dim=0)
        log_probs = torch.log_softmax(target_logits.float(), dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()
        total_n += nll.numel()
        last_logit = out.logits[:, -1:, :]
        if verbose and start % (chunk * 8) == 0:
            cur = math.exp(total_nll / max(total_n, 1))
            print(f"    sliding pos {start+chunk}/{n} ppl_so_far={cur:.2f}",
                  flush=True)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {
        "ppl": math.exp(total_nll / total_n),
        "total_tokens": total_n,
        "elapsed_sec": dt,
        "peak_gb": peak,
    }


def build_variants(model_config, *, e8_method: str, bpw_map_path: str | None,
                   residual_length: int):
    """Return [(label, bpw_label, cache_factory)] tuples for the sweep."""
    import glq.hf_integration  # noqa
    from glq.kv_cache import GLQQuantizedCache

    variants = []

    def factory_for(method, n_stages=1, bpw_map=None):
        def make():
            return GLQQuantizedCache(
                model_config, quant_method=method, n_stages=n_stages,
                bpw_map=bpw_map, residual_length=residual_length)
        return make

    variants.append(("baseline (fp16)",   "16", None))
    variants.append(("int8 absmax",       "8",  factory_for("int8")))
    variants.append(("e8_relaxed 6 bpw",  "6",  factory_for("e8_relaxed", n_stages=3)))
    variants.append(("e8_relaxed 4 bpw",  "4",  factory_for("e8_relaxed", n_stages=2)))
    variants.append(("e8_relaxed 2 bpw",  "2",  factory_for("e8_relaxed", n_stages=1)))
    variants.append(("e8_strict  6 bpw",  "6*", factory_for("e8_strict",  n_stages=3)))
    variants.append(("e8_strict  4 bpw",  "4*", factory_for("e8_strict",  n_stages=2)))
    variants.append(("e8_strict  2 bpw",  "2*", factory_for("e8_strict",  n_stages=1)))

    if bpw_map_path and os.path.exists(bpw_map_path):
        raw = json.load(open(bpw_map_path))
        bpw_map = {int(k): int(v) for k, v in raw.items()}
        avg = sum(bpw_map.values()) / len(bpw_map)
        variants.append(
            (f"{e8_method} mix avg {avg:.2f} bpw", f"~{avg:.1f}",
             factory_for(e8_method, bpw_map=bpw_map)))
    return variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/gemma-4-E4B-it")
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/cache/epub/6593/pg6593.txt")
    p.add_argument("--tokens", type=int, default=4096,
                   help="Total tokens to evaluate")
    p.add_argument("--chunk", type=int, default=256,
                   help="Chunk size for the sliding-window forward")
    p.add_argument("--residual", type=int, nargs="+", default=[8, 64],
                   help="One or more residual_length values to sweep")
    p.add_argument("--e8-method", default="e8_relaxed",
                   choices=("e8_strict", "e8_relaxed"))
    p.add_argument("--bpw-map",
                   default="/tmp/kv_bpw_e4b_3.0.json",
                   help="Path to allocator JSON (skipped if missing)")
    p.add_argument("--out", default="/tmp/logs/phase4_ppl.json")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"  load: {time.time()-t0:.1f}s", flush=True)

    text = read_gutenberg_text(args.text_url)
    print(f"  text: {len(text):,} chars  ({args.text_url})", flush=True)
    enc = tok(text, return_tensors="pt").input_ids[:, :args.tokens]
    ids = enc.to("cuda")
    print(f"  tokens: {ids.shape[-1]}", flush=True)

    results = {"args": vars(args), "rows": []}
    for residual in args.residual:
        print(f"\n=== residual_length={residual} ===", flush=True)
        variants = build_variants(
            model.config, e8_method=args.e8_method,
            bpw_map_path=args.bpw_map, residual_length=residual)
        for label, bpw_label, factory in variants:
            res = sliding_window_ppl(model, ids, factory, args.chunk)
            row = {
                "residual_length": residual,
                "variant": label,
                "bpw_label": bpw_label,
                **res,
            }
            results["rows"].append(row)
            print(f"  {label:<26}  bpw={bpw_label:>4}  "
                  f"ppl={res['ppl']:7.3f}  "
                  f"time={res['elapsed_sec']:5.1f}s  "
                  f"peak={res['peak_gb']:5.2f} GB",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
