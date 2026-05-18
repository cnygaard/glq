"""Phase 5.2: mmlu_pro 1% via vLLM + E8 KV-cache round-trip.

Drives one vLLM engine per invocation (the engine reads
``GLQ_KV_QUANT`` once at plugin load, so swapping configs mid-process
isn't possible). A wrapper script (``run_kv_mmlu_vllm.sh``) loops over
the four variants and aggregates the JSON outputs.

Usage:
    GLQ_KV_QUANT=e8_relaxed:2 python benchmarks/kv_compress_mmlu_vllm.py \\
        --n 120 --out /tmp/logs/mmlu_vllm_4bpw.json

Aggregating manually:
    for spec in "" e8_relaxed:1 e8_relaxed:2 e8_relaxed:3; do
        GLQ_KV_QUANT="$spec" python benchmarks/kv_compress_mmlu_vllm.py \\
            --n 120 --out /tmp/logs/mmlu_vllm_${spec:-baseline}.json
    done
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time


BOXED_RE = re.compile(r"\\boxed\{\s*([A-J])\s*\}")
ANSWER_RE = re.compile(r"\b([A-J])\b")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_letter(text: str) -> str | None:
    """Strip reasoning content, prefer \\boxed{X}, fall back to the last
    isolated A-J. None if nothing matches."""
    text = THINK_RE.sub("", text)
    m = BOXED_RE.search(text)
    if m:
        return m.group(1)
    letters = ANSWER_RE.findall(text)
    return letters[-1] if letters else None


def format_prompt(question: str, options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return (f"Question: {question}\n\n"
            f"Options:\n{opts}\n\n"
            f"Think step by step, then put your final answer (a single "
            f"letter A-J) within \\boxed{{}}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--n", type=int, default=120,
                   help="number of mmlu_pro questions (~1%% of full eval)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking",
                   action="store_false")
    p.add_argument("--out", default="/tmp/logs/mmlu_vllm.json")
    p.add_argument("--label",
                   default=os.environ.get("GLQ_KV_QUANT", "baseline-fp16"))
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print(f"=== mmlu_pro vLLM — {args.label} ===", flush=True)

    t0 = time.time()
    # glq_vllm v0.3.5+ auto-forces ``cudagraph_mode=PIECEWISE`` when E8
    # KV envs are set, so we no longer need ``enforce_eager=True`` here.
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem)
    print(f"  load: {time.time()-t0:.1f}s", flush=True)

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.shuffle(seed=args.seed).select(range(args.n))
    print(f"  dataset: {len(ds)} questions, "
          f"categories: {sorted(set(ds['category']))[:5]}...",
          flush=True)

    prompts = [
        [{"role": "user", "content": format_prompt(ex["question"], ex["options"])}]
        for ex in ds
    ]
    gold = [ex["answer"] for ex in ds]

    t0 = time.time()
    out = llm.chat(
        prompts,
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens, temperature=0),
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        use_tqdm=True,
    )
    dt = time.time() - t0

    rows = []
    total_in = total_out = correct = parsed = 0
    for ex, o in zip(ds, out):
        text = o.outputs[0].text
        pred = extract_letter(text)
        ok = pred == ex["answer"]
        rows.append({
            "category": ex["category"],
            "answer": ex["answer"],
            "pred": pred,
            "ok": ok,
            "n_in": len(o.prompt_token_ids),
            "n_out": len(o.outputs[0].token_ids),
        })
        total_in += len(o.prompt_token_ids)
        total_out += len(o.outputs[0].token_ids)
        if pred is not None:
            parsed += 1
        if ok:
            correct += 1

    acc = correct / len(ds)
    parse_rate = parsed / len(ds)
    in_per_sec = total_in / dt
    out_per_sec = total_out / dt
    print(f"\n  accuracy:   {correct}/{len(ds)} = {acc:.3f}", flush=True)
    print(f"  parse rate: {parsed}/{len(ds)} = {parse_rate:.3f}", flush=True)
    print(f"  prompt tok: {total_in:,}  ({in_per_sec:.0f} tok/s)", flush=True)
    print(f"  output tok: {total_out:,}  ({out_per_sec:.0f} tok/s)", flush=True)
    print(f"  wall time:  {dt:.1f}s", flush=True)

    summary = {
        "label": args.label,
        "model": args.model,
        "n": args.n,
        "accuracy": acc,
        "parse_rate": parse_rate,
        "correct": correct,
        "elapsed_sec": dt,
        "prompt_tokens": total_in,
        "output_tokens": total_out,
        "prompt_tok_per_sec": in_per_sec,
        "output_tok_per_sec": out_per_sec,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
