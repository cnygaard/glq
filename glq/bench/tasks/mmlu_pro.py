"""MMLU-Pro (thinking) accuracy adapter — lifted from
``benchmarks/_mmlu_think_sweep_glq.py`` so results stay comparable to the
hand-run session numbers. Uses the shared vLLM handle (``ctx.handle``).
"""
from __future__ import annotations

import time

from ..record import BenchmarkResult, ThroughputResult
from .parse import extract_mmlu_letter

_LETTERS = "ABCDEFGHIJKLMNOP"


def _build(n: int):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0).select(range(n))
    msgs, golds = [], []
    for it in ds:
        blk = "\n".join("%s. %s" % (_LETTERS[i], o) for i, o in enumerate(it["options"]))
        user = ("Question: %s\n\nOptions:\n%s\n\nThink step by step, then end your "
                "response with 'The answer is (X).' where X is the letter of the "
                "correct option." % (it["question"], blk))
        msgs.append([{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": user}])
        golds.append(it["answer"])
    return msgs, golds


def run(ctx, config: dict):
    from vllm import SamplingParams
    n = int(config.get("n", 60))
    budget = int(config.get("budget", 16384))
    thinking = bool(config.get("thinking", True))

    msgs, golds = _build(n)
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=budget, seed=0)
    chat_kw = {"enable_thinking": True} if thinking else {}
    t0 = time.time()
    outs = ctx.handle.llm.chat(msgs, sp, chat_template_kwargs=chat_kw,
                               use_tqdm=getattr(ctx, "pbar_factory", None) or True)
    dt = time.time() - t0

    correct = trunc = noans = 0
    gen = 0
    for i, o in enumerate(outs):
        c = o.outputs[0]
        gen += len(c.token_ids)
        if c.finish_reason == "length":
            trunc += 1
        pred = extract_mmlu_letter(c.text)
        if pred is None:
            noans += 1
        elif pred == golds[i].upper():
            correct += 1

    acc = correct / len(msgs)
    res = BenchmarkResult(
        task=config.get("task_name", "mmlu_pro"), metric="accuracy", value=acc,
        standardized=bool(config.get("standardized", True)),
        config={"n": len(msgs), "budget": budget, "thinking": thinking,
                "temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0,
                "dataset": "TIGER-Lab/MMLU-Pro"},
        extra={"correct": correct, "n": len(msgs), "truncated": trunc,
               "no_answer": noans, "mean_gen_tokens": round(gen / len(msgs), 1)})
    tp = ThroughputResult(output_tok_s=round(gen / dt, 1) if dt > 0 else None,
                          batch=len(msgs), measure="in_run_chat")
    return res, tp
