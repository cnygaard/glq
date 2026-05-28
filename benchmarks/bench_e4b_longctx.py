"""v0.5 Phase 5.3a — long-context decode-throughput bench.

Measures decode tok/s as a function of context length, isolating the
decode phase from prefill via the two-point method:

    decode_tok_s = (N - 1) / (t_N - t_1)

where t_N is the wall time to generate N tokens and t_1 the wall time
to generate 1 token (= prefill + 1 decode step). The difference cancels
the one-time prefill cost, leaving pure decode throughput at the given
context length.

Run once per path (env selects workspace vs inline). The v0.3.5
workspace path re-decompresses the referenced K/V every decode token
(cost grows O(ctx)); the v3_fht inline path reads each K/V once fused
into attention. The gap should therefore WIDEN with context.

Usage::

    # workspace baseline:
    GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 \\
        GLQ_KV_E8_COMPRESSED_ALLOC=1 GLQ_KV_E8_FUSED_GATHER=1 \\
        GLQ_KV_E8_FUSED_WRITE=1 \\
        python benchmarks/bench_e4b_longctx.py --label workspace \\
            --ctx-lens 1024 4096 8192 16384 --out /tmp/logs/longctx_ws.json

    # v3_fht inline (add GLQ_KV_E8_INLINE_DEQUANT_V3=1):
    ... GLQ_KV_E8_INLINE_DEQUANT_V3=1 \\
        python benchmarks/bench_e4b_longctx.py --label v3fht ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw")
    p.add_argument("--ctx-lens", type=int, nargs="+",
                   default=[1024, 4096, 8192, 16384])
    p.add_argument("--decode-len", type=int, default=128,
                   help="N for the t_N timing point")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/files/1342/1342-0.txt")
    p.add_argument("--out", default="/tmp/logs/bench_longctx.json")
    p.add_argument("--label", default="run")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    print(f"=== longctx {args.label} — {args.model} ===", flush=True)
    for k in ("GLQ_KV_QUANT", "GLQ_KV_E8_SIDECAR_READ",
              "GLQ_KV_E8_COMPRESSED_ALLOC", "GLQ_KV_E8_FUSED_GATHER",
              "GLQ_KV_E8_FUSED_WRITE", "GLQ_KV_E8_INLINE_DEQUANT_V3"):
        print(f"  {k}={os.environ.get(k, '<unset>')}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from kv_compress_niah import read_gutenberg_text

    max_ctx = max(args.ctx_lens)
    max_model_len = max_ctx + args.decode_len + 64

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", quantization="glq",
              max_model_len=max_model_len,
              gpu_memory_utilization=args.gpu_mem)
    print(f"  load: {time.time()-t0:.1f}s  max_model_len={max_model_len}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    filler = read_gutenberg_text(args.text_url)
    filler_ids = tok(filler, add_special_tokens=False).input_ids

    def _prompt_ids(ctx_len):
        # Repeat filler to >= ctx_len, then truncate to exactly ctx_len.
        reps = ctx_len // len(filler_ids) + 2
        ids = (filler_ids * reps)[:ctx_len]
        return ids

    def _time_gen(prompt_ids, n_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=n_tokens)
        sp.ignore_eos = True
        from vllm import TokensPrompt
        t0 = time.time()
        llm.generate(TokensPrompt(prompt_token_ids=prompt_ids),
                     sampling_params=sp, use_tqdm=False)
        return time.time() - t0

    rows = []
    for ctx_len in args.ctx_lens:
        pids = _prompt_ids(ctx_len)
        # warm-up at this ctx (CUDA-graph capture for the bucket)
        _time_gen(pids, 4)
        t1 = _time_gen(pids, 1)
        tN = _time_gen(pids, args.decode_len)
        decode_s = max(tN - t1, 1e-6)
        decode_tok_s = (args.decode_len - 1) / decode_s
        ms_per_tok = decode_s * 1000.0 / (args.decode_len - 1)
        print(f"  ctx={ctx_len:>6}: t_1={t1:.3f}s t_N={tN:.3f}s "
              f"decode={decode_tok_s:6.1f} tok/s ({ms_per_tok:.2f} ms/tok)",
              flush=True)
        rows.append(dict(ctx_len=ctx_len, t_1=t1, t_N=tN,
                         decode_len=args.decode_len,
                         decode_tok_s=decode_tok_s, ms_per_tok=ms_per_tok))

    result = dict(label=args.label, model=args.model,
                  env={k: os.environ.get(k) for k in (
                      "GLQ_KV_QUANT", "GLQ_KV_E8_INLINE_DEQUANT_V3")},
                  rows=rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
