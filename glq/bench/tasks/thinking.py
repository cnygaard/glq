"""Shared thinking-mode handling for the reasoning tasks.

One place, because the failure is identical across tasks and silent in all of them: a
model that never engaged its reasoning preamble still answers, still parses, and just
scores like a worse model. See ``.claude/skills/bench-quality/SKILL.md``.
"""
from __future__ import annotations


def build_turns(user: str, system: str | None):
    """Chat turns for one problem. ``system=None`` (the default everywhere) means
    user-only, which is the shape that engages thinking on BOTH families.

    A SmolLM3 template reads any custom system message as "Custom Instructions" mode,
    where ``Reasoning Mode`` falls back to ``/no_think``; appending ``/think`` moves the
    rendered metadata line without engaging the preamble, so the prompt looks right and
    the run is wrong. gemma-4 takes thinking from ``chat_template_kwargs`` and is
    indifferent to a system message — so omitting it costs gemma-4 nothing and saves
    SmolLM3 entirely.
    """
    turns = [{"role": "user", "content": user}]
    if system:
        turns.insert(0, {"role": "system", "content": system})
    return turns


def sampling(config: dict, budget: int, **extra):
    """SamplingParams from config. Sampling belongs to the model's card, not the task:
    gemma-4 is 1.0/0.95/64 (the defaults here), SmolLM3 is 0.6/0.95 with no top_k."""
    from vllm import SamplingParams
    top_k = config.get("top_k", 64)
    return SamplingParams(temperature=float(config.get("temperature", 1.0)),
                          top_p=float(config.get("top_p", 0.95)),
                          max_tokens=budget, seed=int(config.get("seed", 0)),
                          **({"top_k": int(top_k)} if top_k else {}), **extra)


def assert_engaged(mean_gen: float, *, thinking: bool, config: dict, floor: float,
                   system: str | None) -> None:
    """Raise unless generation length is consistent with reasoning having happened.

    Generation length is the ONLY reliable signal — not the printed flag, not the rendered
    prompt. The runner turns this into a skipped-with-reason record, which is the point:
    a wrong number that looks right reads as a catastrophic quantization regression and
    sends you debugging the kernel instead of the harness.
    """
    floor = float(config.get("min_mean_gen", floor))
    if not thinking or floor <= 0 or mean_gen >= floor:
        return
    raise RuntimeError(
        f"thinking=True but mean generation was {mean_gen:.0f} tokens (floor {floor:.0f}) "
        f"— the reasoning preamble almost certainly never engaged. system={system!r}; a "
        f"SmolLM3-family template silently falls back to /no_think whenever ANY system "
        f"message is set. Re-run with system=None, or lower min_mean_gen if this model is "
        f"genuinely this terse.")
