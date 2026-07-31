"""AIME (2024/2025/2026) thinking accuracy adapter — lifted from
``benchmarks/_aime_multiyear.py``. ``config["sets"]`` selects the year(s); the
task name (aime_2024 / aime_2026) comes from the registry. Uses ``ctx.handle``.
"""
from __future__ import annotations

import time

from ..record import BenchmarkResult, ThroughputResult
from .parse import extract_boxed_int

_DATASETS = {
    "2024": ("Maxwell-Jia/AIME_2024", "train"),
    "2025": ("yentinglin/aime_2025", "train"),
    "2026": ("MathArena/aime_2026", "train"),
}


def _rows(year: str):
    from datasets import load_dataset
    repo, split = _DATASETS[year]
    ds = load_dataset(repo, split=split)
    out = []
    for i, it in enumerate(ds):
        prob = it.get("problem") or it.get("Problem")
        ans = it.get("answer", it.get("Answer"))
        rid = it.get("problem_idx", it.get("id", i))
        out.append(("%s-%s" % (year, rid), prob, int(str(ans).strip())))
    return out


def _build(sets, n: int | None):
    rows = []
    for y in sets:
        rows += _rows(y)
    if n:
        rows = rows[:n]
    msgs, golds = [], []
    for _rid, prob, ans in rows:
        user = ("Problem:\n%s\n\nThink step by step, then give the final answer as a "
                "non-negative integer in \\boxed{}." % prob)
        msgs.append([{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": user}])
        golds.append(ans)
    return msgs, golds


def run(ctx, config: dict):
    from vllm import SamplingParams
    sets = config.get("sets", ["2024"])
    budget = int(config.get("budget", 32768))
    n = config.get("n")
    thinking = bool(config.get("thinking", True))

    avg_k = max(1, int(config.get("avg_k", 1)))

    msgs, golds = _build(sets, int(n) if n else None)
    # n=avg_k draws avg_k distinct samples per problem (avg@k); seed makes the set
    # reproducible. avg@k accuracy = mean over problems of the fraction of its k
    # samples that are correct — the variance-reduced number AIME is reported at.
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=budget,
                        n=avg_k, seed=0)
    chat_kw = {"enable_thinking": True} if thinking else {}
    t0 = time.time()
    outs = ctx.handle.llm.chat(msgs, sp, chat_template_kwargs=chat_kw,
                               use_tqdm=getattr(ctx, "pbar_factory", None) or True)
    dt = time.time() - t0

    trunc = noans = nseq = gen = 0
    per_problem = []                     # fraction of the k samples correct
    for i, o in enumerate(outs):
        pc = 0
        for c in o.outputs:              # avg_k sampled completions for problem i
            nseq += 1
            gen += len(c.token_ids)
            if c.finish_reason == "length":
                trunc += 1
            pred = extract_boxed_int(c.text)
            if pred is None:
                noans += 1
            elif pred == golds[i]:
                pc += 1
        per_problem.append(pc / len(o.outputs))

    acc = sum(per_problem) / len(msgs)   # avg@k (== correct/n when avg_k==1)
    res = BenchmarkResult(
        task=config.get("task_name", "aime_%s" % "+".join(sets)),
        metric="accuracy", value=acc,
        standardized=bool(config.get("standardized", True)),
        config={"sets": sets, "n": len(msgs), "budget": budget, "thinking": thinking,
                "avg_k": avg_k, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0},
        extra={"avg_k": avg_k, "n": len(msgs), "samples": nseq,
               "solved_any": sum(1 for p in per_problem if p > 0),       # pass@k
               "solved_all": sum(1 for p in per_problem if p == 1.0),
               "truncated": trunc, "no_answer": noans,
               "mean_gen_tokens": round(gen / nseq, 1) if nseq else 0})
    tp = ThroughputResult(output_tok_s=round(gen / dt, 1) if dt > 0 else None,
                          batch=len(msgs) * avg_k, measure="in_run_chat")
    return res, tp
