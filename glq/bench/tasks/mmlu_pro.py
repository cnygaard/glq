"""MMLU-Pro (thinking) accuracy adapter — lifted from
``benchmarks/_mmlu_think_sweep_glq.py`` so results stay comparable to the
hand-run session numbers. Uses the shared vLLM handle (``ctx.handle``).

Carries the same thinking-mode handling as the AIME adapter (``system`` defaults to None,
sampling comes from config, and generation length gates the run) — the silent-no-think
failure is a property of the chat template, not of the task. See ``.tasks.thinking``.
"""
from __future__ import annotations

import time

from ..record import BenchmarkResult, ThroughputResult
from .parse import extract_mmlu_letter
from .thinking import assert_engaged, build_turns, sampling

_LETTERS = "ABCDEFGHIJKLMNOP"

# Lower than AIME's floor: a multiple-choice question needs far less reasoning than a
# competition problem, so a genuine thinking answer here is hundreds-to-thousands of
# tokens. Still far above a no-think answer, which is a sentence.
_DEFAULT_MIN_MEAN_GEN = 600


def _build(n: int, system: str | None = None):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0).select(range(n))
    msgs, golds = [], []
    for it in ds:
        blk = "\n".join("%s. %s" % (_LETTERS[i], o) for i, o in enumerate(it["options"]))
        user = ("Question: %s\n\nOptions:\n%s\n\nThink step by step, then end your "
                "response with 'The answer is (X).' where X is the letter of the "
                "correct option." % (it["question"], blk))
        msgs.append(build_turns(user, system))
        golds.append(it["answer"])
    return msgs, golds


def run(ctx, config: dict):
    n = int(config.get("n", 60))
    budget = int(config.get("budget", 16384))
    thinking = bool(config.get("thinking", True))
    system = config.get("system")        # default None: NO system message

    msgs, golds = _build(n, system)
    sp = sampling(config, budget)
    chat_kw = {"enable_thinking": True} if thinking else {}
    t0 = time.time()
    outs = ctx.handle.llm.chat(msgs, sp, chat_template_kwargs=chat_kw,
                               use_tqdm=getattr(ctx, "pbar_factory", None) or True)
    dt = time.time() - t0

    correct = trunc = noans = 0
    gen = 0
    per_item = []                        # 1/0 per question, dataset order
    for i, o in enumerate(outs):
        c = o.outputs[0]
        gen += len(c.token_ids)
        if c.finish_reason == "length":
            trunc += 1
        pred = extract_mmlu_letter(c.text)
        hit = pred is not None and pred == golds[i].upper()
        if pred is None:
            noans += 1
        elif hit:
            correct += 1
        per_item.append(int(hit))

    mean_gen = gen / len(msgs) if msgs else 0.0
    assert_engaged(mean_gen, thinking=thinking, config=config,
                   floor=_DEFAULT_MIN_MEAN_GEN, system=system)

    acc = correct / len(msgs)
    res = BenchmarkResult(
        task=config.get("task_name", "mmlu_pro"), metric="accuracy", value=acc,
        standardized=bool(config.get("standardized", True)),
        config={"n": len(msgs), "budget": budget, "thinking": thinking,
                "temperature": sp.temperature, "top_p": sp.top_p,
                "top_k": getattr(sp, "top_k", None), "seed": sp.seed, "system": system,
                "min_mean_gen": float(config.get("min_mean_gen", _DEFAULT_MIN_MEAN_GEN)),
                "dataset": "TIGER-Lab/MMLU-Pro"},
        extra={"correct": correct, "n": len(msgs), "truncated": trunc,
               "no_answer": noans, "mean_gen_tokens": round(mean_gen, 1),
               # Per-item 1/0 in dataset order (the selection is deterministic:
               # shuffle(seed=0).select(range(n))), so two arms line up item-for-item and
               # can be compared with McNemar on the discordant pairs. Differencing two
               # marginal percentages throws that pairing away and needs far more items
               # for the same power.
               "per_item": per_item})
    tp = ThroughputResult(output_tok_s=round(gen / dt, 1) if dt > 0 else None,
                          batch=len(msgs), measure="in_run_chat")
    return res, tp
