"""v0.5 Phase 5.7v — varlen-prefill + mixed/continuous-batch validation.

Exercises the two request regimes no existing GLQ harness covers, to
validate the FULL cudagraph path (``GLQ_KV_E8_ALLOW_FULL_CUDAGRAPH=1``)
before it becomes the v3 default:

  * ``--mode varlen``  — one ``llm.generate`` over prompts with very
    different lengths (32 / 512 / 2k / 6k / 12k tokens). Greedy, short
    decode. Deterministic, so the emitted text is compared
    token-for-token between FULL and PIECEWISE (run twice, diff the
    logs). Covers varlen PIECEWISE prefill + the prefill→pure-decode
    (FULL graph) handoff.

  * ``--mode mixed``   — staggered submission via the v1 ``AsyncLLM`` so
    new prefills interleave with in-flight decodes (true mixed
    scheduler steps, which ``FULL_AND_PIECEWISE`` routes to PIECEWISE).
    Timing-dependent batching → NOT token-for-token; gate is "coherent +
    no crash". Best-effort: if the AsyncLLM API is unavailable it logs
    and skips (mixed steps run the already-validated eager path anyway).

Run A/B by toggling ``GLQ_KV_E8_ALLOW_FULL_CUDAGRAPH`` (0 = PIECEWISE
control, 1 = FULL) with the rest of the E8-KV env set. Each prompt ends
with a checkable factual stem so coherence is easy to score.

Usage::

    GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 \\
      GLQ_KV_E8_COMPRESSED_ALLOC=1 GLQ_KV_E8_FUSED_GATHER=1 \\
      GLQ_KV_E8_FUSED_WRITE=1 GLQ_KV_E8_INLINE_DEQUANT_V3=1 \\
      GLQ_KV_E8_ALLOW_FULL_CUDAGRAPH=1 \\
      python benchmarks/bench_varlen_mixed.py --mode varlen \\
        --model xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw --out /tmp/logs/varlen_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Factual stems with a deterministic, easy-to-grep continuation. The
# long filler precedes the question so prefill length varies while the
# answerable part is identical. ``gold`` is a list of accepted
# lowercase substrings (a model may spell "4" as "four"). NOTE: the
# primary A/B gate is FULL-vs-PIECEWISE output *equality*, not this
# coherence flag — the flag is a secondary sanity signal.
_QUESTIONS = [
    ("The capital of France is", ["paris"]),
    ("Two plus two equals", ["4", "four"]),
    ("The chemical symbol for water is", ["h2o", "h₂o", "h 2 o"]),
    ("The largest planet in our solar system is", ["jupiter"]),
    ("The opposite of hot is", ["cold"]),
]


def _build_varlen_prompts(tok, filler_ids, ctx_lens):
    """One prompt per ctx_len: <filler truncated to ctx_len> + a factual
    stem. Returns (list[token_ids], list[(stem, gold)])."""
    prompts, meta = [], []
    for i, ctx in enumerate(ctx_lens):
        stem, gold = _QUESTIONS[i % len(_QUESTIONS)]
        stem_ids = tok(" " + stem, add_special_tokens=False).input_ids
        n_filler = max(0, ctx - len(stem_ids))
        reps = n_filler // max(1, len(filler_ids)) + 1
        ids = (filler_ids * reps)[:n_filler] + stem_ids
        prompts.append(ids)
        meta.append((ctx, stem, gold))
    return prompts, meta


def _run_varlen(args):
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    from kv_compress_niah import read_gutenberg_text

    tok = AutoTokenizer.from_pretrained(args.model)
    filler_ids = tok(read_gutenberg_text(args.text_url),
                     add_special_tokens=False).input_ids

    max_ctx = max(args.ctx_lens)
    llm = LLM(model=args.model, dtype="bfloat16", quantization="glq",
              max_model_len=max_ctx + 64,
              gpu_memory_utilization=args.gpu_mem)

    prompts, meta = _build_varlen_prompts(tok, filler_ids, args.ctx_lens)
    sp = SamplingParams(temperature=0.0, max_tokens=args.decode_len)
    out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                       sampling_params=sp, use_tqdm=False)

    rows = []
    for (ctx, stem, gold), o in zip(meta, out):
        text = o.outputs[0].text
        ok = any(g in text.lower() for g in gold)
        print(f"  ctx={ctx:>6} stem={stem!r:45} gold={'/'.join(gold):8} "
              f"ok={ok}  out={text!r}", flush=True)
        rows.append(dict(ctx=ctx, stem=stem, gold=gold, ok=ok, out=text))
    n_ok = sum(r["ok"] for r in rows)
    print(f"  VARLEN: {n_ok}/{len(rows)} coherent", flush=True)
    return dict(mode="varlen", rows=rows, n_ok=n_ok, n=len(rows))


def _run_mixed(args):
    """Staggered submission so prefills interleave with decodes."""
    import asyncio
    import time as _time
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm import SamplingParams
    except Exception as e:  # noqa: BLE001
        print(f"  MIXED: AsyncLLM unavailable ({e}); skipping "
              "(mixed steps route to the already-validated PIECEWISE path)",
              flush=True)
        return dict(mode="mixed", skipped=str(e))

    from transformers import AutoTokenizer
    from kv_compress_niah import read_gutenberg_text
    tok = AutoTokenizer.from_pretrained(args.model)
    filler_ids = tok(read_gutenberg_text(args.text_url),
                     add_special_tokens=False).input_ids
    prompts, meta = _build_varlen_prompts(tok, filler_ids, args.ctx_lens)

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, dtype="bfloat16", quantization="glq",
        max_model_len=max(args.ctx_lens) + 64,
        gpu_memory_utilization=args.gpu_mem,
    ))

    async def _one(rid, ids, max_tok, gold, stem, ctx):
        from vllm import TokensPrompt
        sp = SamplingParams(temperature=0.0, max_tokens=max_tok)
        final = None
        async for o in engine.generate(TokensPrompt(prompt_token_ids=ids),
                                       sp, request_id=str(rid)):
            final = o
        text = final.outputs[0].text
        ok = any(g in text.lower() for g in gold)
        print(f"  [mixed rid={rid} ctx={ctx} max_tok={max_tok}] "
              f"gold={'/'.join(gold)} ok={ok} out={text!r}", flush=True)
        return dict(rid=rid, ctx=ctx, gold=gold, ok=ok, out=text)

    async def _driver():
        tasks, max_toks = [], [8, 64, 256, 512, 128]
        for i, (ids, (ctx, stem, gold)) in enumerate(zip(prompts, meta)):
            tasks.append(asyncio.create_task(
                _one(i, ids, max_toks[i % len(max_toks)], gold, stem, ctx)))
            await asyncio.sleep(0.25)  # stagger arrivals → mixed steps
        return await asyncio.gather(*tasks)

    rows = asyncio.run(_driver())
    n_ok = sum(r["ok"] for r in rows)
    print(f"  MIXED: {n_ok}/{len(rows)} coherent", flush=True)
    return dict(mode="mixed", rows=rows, n_ok=n_ok, n=len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw")
    p.add_argument("--mode", choices=["varlen", "mixed", "both"],
                   default="both")
    p.add_argument("--ctx-lens", type=int, nargs="+",
                   default=[32, 512, 2048, 6000, 12000])
    p.add_argument("--decode-len", type=int, default=16)
    p.add_argument("--gpu-mem", type=float, default=0.86)
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/files/1342/1342-0.txt")
    p.add_argument("--out", default="/tmp/logs/varlen_mixed.json")
    p.add_argument("--label", default="run")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    print(f"=== varlen/mixed {args.label} — {args.model} "
          f"(mode={args.mode}) ===", flush=True)
    print(f"  ALLOW_FULL_CUDAGRAPH="
          f"{os.environ.get('GLQ_KV_E8_ALLOW_FULL_CUDAGRAPH', '<unset>')} "
          f"V3={os.environ.get('GLQ_KV_E8_INLINE_DEQUANT_V3', '<unset>')}",
          flush=True)

    results = {"label": args.label, "model": args.model, "runs": []}
    if args.mode in ("varlen", "both"):
        print("##### varlen prefill #####", flush=True)
        results["runs"].append(_run_varlen(args))
    if args.mode in ("mixed", "both"):
        print("##### mixed / continuous batch #####", flush=True)
        results["runs"].append(_run_mixed(args))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved {args.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
