"""AIME (2024/2025/2026) thinking accuracy adapter — lifted from
``benchmarks/_aime_multiyear.py``. ``config["sets"]`` selects the year(s); the
task name (aime_2024 / aime_2026) comes from the registry. Uses ``ctx.handle``.

**Thinking activation is model-family-specific and fails silently.** A SmolLM3 chat
template treats *any* custom system message as "Custom Instructions" mode, where
``Reasoning Mode`` defaults to ``/no_think`` — the model answers in ~1.8k tokens instead of
~14-15k and the run looks like a completed thinking eval. Appending ``/think`` to the system
message flips the rendered metadata line without engaging the reasoning preamble, so
inspecting the prompt does not catch it either. gemma-4 is the opposite: it reads thinking
from ``chat_template_kwargs`` and is unbothered by a system message.

Hence two things here. ``system`` defaults to **None** (no system message), which is what
works for both families. And because no prompt inspection is trustworthy, the guard is
**generation length**: below ``min_mean_gen`` tokens a ``thinking=True`` run RAISES, and the
runner records it as skipped-with-reason rather than as a number.
"""
from __future__ import annotations

import time

from ..record import BenchmarkResult, ThroughputResult
from .parse import extract_boxed_int
from .thinking import assert_engaged, build_turns, sampling

_DATASETS = {
    "2024": ("Maxwell-Jia/AIME_2024", "train"),
    "2025": ("yentinglin/aime_2025", "train"),
    "2026": ("MathArena/aime_2026", "train"),
}

# A thinking run at a >=32k budget means ~14-15k tokens on SmolLM3; a no-think run means
# ~1.8k. 4000 sits between them with room for a model that reasons more tersely, and is the
# floor rather than an estimate — set min_mean_gen=0 to opt out for a genuinely terse model.
_DEFAULT_MIN_MEAN_GEN = 4000


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


def _build(sets, n: int | None, system: str | None = None):
    rows = []
    for y in sets:
        rows += _rows(y)
    if n:
        rows = rows[:n]
    msgs, golds = [], []
    for _rid, prob, ans in rows:
        user = ("Problem:\n%s\n\nThink step by step, then give the final answer as a "
                "non-negative integer in \\boxed{}." % prob)
        msgs.append(build_turns(user, system))
        golds.append(ans)
    return msgs, golds


def run(ctx, config: dict):
    sets = config.get("sets", ["2024"])
    budget = int(config.get("budget", 32768))
    n = config.get("n")
    thinking = bool(config.get("thinking", True))

    avg_k = max(1, int(config.get("avg_k", 1)))
    system = config.get("system")        # default None: NO system message

    msgs, golds = _build(sets, int(n) if n else None, system)
    # n=avg_k draws avg_k distinct samples per problem (avg@k); seed makes the set
    # reproducible. avg@k accuracy = mean over problems of the fraction of its k
    # samples that are correct — the variance-reduced number AIME is reported at.
    sp = sampling(config, budget, n=avg_k)
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

    mean_gen = gen / nseq if nseq else 0.0
    assert_engaged(mean_gen, thinking=thinking, config=config,
                   floor=_DEFAULT_MIN_MEAN_GEN, system=system)

    acc = sum(per_problem) / len(msgs)   # avg@k (== correct/n when avg_k==1)
    res = BenchmarkResult(
        task=config.get("task_name", "aime_%s" % "+".join(sets)),
        metric="accuracy", value=acc,
        standardized=bool(config.get("standardized", True)),
        config={"sets": sets, "n": len(msgs), "budget": budget, "thinking": thinking,
                "avg_k": avg_k, "temperature": sp.temperature, "top_p": sp.top_p,
                "top_k": getattr(sp, "top_k", None), "seed": sp.seed, "system": system,
                "min_mean_gen": float(config.get("min_mean_gen", _DEFAULT_MIN_MEAN_GEN))},
        extra={"avg_k": avg_k, "n": len(msgs), "samples": nseq,
               "solved_any": sum(1 for p in per_problem if p > 0),       # pass@k
               "solved_all": sum(1 for p in per_problem if p == 1.0),
               # Fraction of the k samples correct, per problem, in dataset order — lets
               # two arms be compared paired instead of by differencing two avg@k numbers.
               "per_item": [round(p, 4) for p in per_problem],
               "truncated": trunc, "no_answer": noans,
               "mean_gen_tokens": round(mean_gen, 1)})
    tp = ThroughputResult(output_tok_s=round(gen / dt, 1) if dt > 0 else None,
                          batch=len(msgs) * avg_k, measure="in_run_chat")
    return res, tp
