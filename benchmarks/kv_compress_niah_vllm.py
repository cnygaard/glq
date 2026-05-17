"""Phase 5.3 verification: needle-in-haystack via vLLM + Stage 2c-2c stack.

Mirrors ``benchmarks/kv_compress_niah.py`` but drives inference through
vLLM (with our E8 paged-cache sidecar active) instead of HF
transformers. The compression knobs are picked up at engine start via
env vars — one vLLM instance per variant, no per-trial swap.

Activate the full Stage 2c-2c stack via:
    GLQ_KV_QUANT=e8_relaxed:1                  # bpw recipe (1 = 2 bpw)
    GLQ_KV_E8_SIDECAR=1
    GLQ_KV_E8_SIDECAR_READ=1
    GLQ_KV_E8_COMPRESSED_ALLOC=1

Run baseline (fp16 cache) by leaving all of the above unset.

The prompt construction (passkey, needle template, depth splicing) is
imported from the existing HF NIAH harness so both backends test the
exact same questions.

Usage:
    python benchmarks/kv_compress_niah_vllm.py \\
        --ctx-lens 16384 \\
        --depths 0.05 0.25 0.5 0.75 0.95 \\
        --seeds 2 \\
        --out /tmp/logs/niah_vllm_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make the sibling benchmark module importable when invoked as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse prompt builders so vLLM and HF backends test identical questions.
from kv_compress_niah import (  # noqa: E402
    NEEDLE_TEMPLATE,
    QUESTION_TEMPLATE,
    extract_passkey,
    make_passkey,
    read_gutenberg_text,
)


def _build_chat_prompt(tok, distractor_ids, *, ctx_len, depth, seed):
    """Build the NIAH prompt as a chat-message body for vLLM.chat()."""
    import torch
    passkey = make_passkey(seed)
    needle = "\n" + NEEDLE_TEMPLATE.format(key=passkey) + "\n"
    question = "\n\n" + QUESTION_TEMPLATE
    needle_ids = tok(needle, add_special_tokens=False,
                     return_tensors="pt").input_ids[0]
    question_ids = tok(question, add_special_tokens=False,
                       return_tensors="pt").input_ids[0]
    overhead = 24
    budget = ctx_len - len(needle_ids) - len(question_ids) - overhead
    if budget < 16:
        raise ValueError(
            f"ctx_len={ctx_len} too short for needle+question (budget={budget})")
    if budget > distractor_ids.shape[-1]:
        raise ValueError(
            f"distractor pool only {distractor_ids.shape[-1]} tokens, "
            f"need {budget}")
    distractor = distractor_ids[:budget]
    split = max(1, min(budget - 1, int(round(depth * budget))))
    pre_text = tok.decode(distractor[:split], skip_special_tokens=True)
    post_text = tok.decode(distractor[split:], skip_special_tokens=True)
    body_text = pre_text + needle + post_text + question
    return [{"role": "user", "content": body_text}], passkey


def _variant_label() -> str:
    bits = []
    if os.environ.get("GLQ_KV_QUANT"):
        bits.append(os.environ["GLQ_KV_QUANT"])
    if os.environ.get("GLQ_KV_BPW_MAP"):
        bits.append(f"map={os.path.basename(os.environ['GLQ_KV_BPW_MAP'])}")
    if os.environ.get("GLQ_KV_E8_SIDECAR") == "1":
        bits.append("sidecar")
    if os.environ.get("GLQ_KV_E8_SIDECAR_READ") == "1":
        bits.append("read")
    if os.environ.get("GLQ_KV_E8_COMPRESSED_ALLOC") == "1":
        bits.append("compressed_alloc")
    return "+".join(bits) if bits else "baseline-fp16"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/cache/epub/6593/pg6593.txt")
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[16384])
    p.add_argument("--depths", type=float, nargs="+",
                   default=[0.05, 0.25, 0.5, 0.75, 0.95])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--n-decode", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="vLLM max_model_len; defaults to max(ctx_lens)+128")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-batched-tokens", type=int, default=None,
                   help="vLLM max_num_batched_tokens; controls chunked prefill chunk + profile_run activation peak")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="vLLM max_num_seqs; cap concurrent sequences (1 for single-seq NIAH)")
    p.add_argument("--enforce-eager", action="store_true",
                   help="disable vLLM's torch.compile + CUDA-graph capture (debugging only)")
    p.add_argument("--out", default="/tmp/logs/niah_vllm.json")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    label = args.label or _variant_label()
    print(f"=== NIAH vLLM probe — variant={label} ===", flush=True)

    max_ctx = max(args.ctx_lens)
    max_model_len = args.max_model_len or (max_ctx + 256)

    t0 = time.time()
    llm_kwargs = dict(
        model=args.model, dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_mem,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.max_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    print(f"  load: {time.time()-t0:.1f}s  max_model_len={max_model_len}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    text = read_gutenberg_text(args.text_url)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    distractor_ids = enc[0]
    if distractor_ids.shape[-1] < max_ctx:
        n_reps = max_ctx // distractor_ids.shape[-1] + 2
        distractor_ids = distractor_ids.repeat(n_reps)
    print(f"  distractor pool: {distractor_ids.shape[-1]:,} tokens",
          flush=True)

    total = len(args.ctx_lens) * len(args.depths) * args.seeds
    results = {"args": vars(args), "label": label, "rows": []}
    done = 0
    t_start = time.time()
    print(f"\n  total trials: {total}", flush=True)
    print(f"\n  {'ctx':>6} {'depth':>6} {'seed':>4} "
          f"{'pred':>7} {'gold':>6} {'ok':>3} {'sec':>6}", flush=True)
    for ctx_len in args.ctx_lens:
        for depth in args.depths:
            for seed in range(args.seeds):
                try:
                    msgs, gold = _build_chat_prompt(
                        tok, distractor_ids, ctx_len=ctx_len,
                        depth=depth, seed=seed)
                except Exception as e:
                    print(f"  {ctx_len:>6} {depth:>5.2f} {seed:>4}  "
                          f"build_err: {type(e).__name__}: {str(e)[:60]}",
                          flush=True)
                    continue

                t0 = time.time()
                out = llm.chat(
                    [msgs],
                    sampling_params=SamplingParams(
                        max_tokens=args.n_decode, temperature=0),
                    chat_template_kwargs={"enable_thinking": False},
                    use_tqdm=False,
                )
                dt = time.time() - t0
                text_out = out[0].outputs[0].text
                pred = extract_passkey(text_out)
                ok = pred == gold
                row = {
                    "ctx_len": ctx_len, "depth": depth, "seed": seed,
                    "gold": gold, "pred": pred, "ok": ok,
                    "elapsed_sec": dt, "output": text_out,
                    "input_tokens": len(out[0].prompt_token_ids),
                }
                results["rows"].append(row)
                done += 1
                print(f"  {ctx_len:>6} {depth:>5.2f} {seed:>4} "
                      f"{str(pred):>7} {gold:>6} "
                      f"{'Y' if ok else 'N':>3} {dt:>6.1f}  "
                      f"[{done}/{total}]", flush=True)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)

    dt = time.time() - t_start
    print(f"\n  total wall: {dt/60:.1f} min", flush=True)
    correct = sum(1 for r in results["rows"] if r.get("ok"))
    n = len(results["rows"])
    print(f"  accuracy: {correct}/{n} = {correct / max(n, 1):.3f}", flush=True)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
