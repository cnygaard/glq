"""LiveCodeBench — reserved, not implemented.

The model-picker table carries a coding column, and a named-but-unimplemented task is
better than an absent one: `glq-bench run --tasks livecodebench` fails with the reason and
the record says "skipped", where a missing task name would just look like an oversight and
an empty column would look like a zero.

Why it is not a few lines: LiveCodeBench is **not an lm-eval task** (see
``benchmarks/bench_quality_lm_eval.py``), so it needs the standalone
https://github.com/LiveCodeBench/LiveCodeBench harness — which means a code-execution
sandbox for generated solutions, per-release problem sets (contamination windows move with
the release), and pass@k plumbing. That is its own piece of work, not a task adapter.
"""
from __future__ import annotations


def run(ctx, config: dict):
    raise NotImplementedError(
        "livecodebench has no harness yet — it is not an lm-eval task and needs the "
        "standalone LiveCodeBench repo plus a code-execution sandbox. The task name is "
        "reserved so the picker table's coding column has a stable identity; drop "
        "'livecodebench' from --tasks until the harness lands.")
