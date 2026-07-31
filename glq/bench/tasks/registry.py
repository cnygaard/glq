"""Task registry: name -> adapter spec.

Adapters are imported lazily (only when a task actually runs) so the registry —
and ``glq-bench`` introspection — stays import-light (no vllm/datasets at import
time). ``standardized=True`` tasks count toward the % -of-bf16 quality index.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    module: str                              # dotted path to the adapter module
    func: str = "run"                        # adapter callable: run(handle, config) -> (BenchmarkResult, Throughput|None)
    metric: str = "accuracy"
    standardized: bool = True                # counts toward the quality index
    kind: str = "quality"                    # "quality" (uses shared LLM) | "throughput" (own subprocess)
    weight: float = 1.0
    defaults: dict[str, Any] = field(default_factory=dict)

    def load(self) -> Callable:
        return getattr(importlib.import_module(self.module), self.func)


TASKS: dict[str, TaskSpec] = {
    "mmlu_pro": TaskSpec(
        "mmlu_pro", "glq.bench.tasks.mmlu_pro", metric="accuracy", standardized=True,
        defaults={"n": 60, "budget": 16384, "thinking": True}),
    "aime_2024": TaskSpec(
        "aime_2024", "glq.bench.tasks.aime", metric="accuracy", standardized=True,
        defaults={"sets": ["2024"], "n": 30, "budget": 32768, "thinking": True}),
    "aime_2025": TaskSpec(
        "aime_2025", "glq.bench.tasks.aime", metric="accuracy", standardized=True,
        defaults={"sets": ["2025"], "n": 30, "budget": 65536, "thinking": True}),
    "aime_2026": TaskSpec(
        "aime_2026", "glq.bench.tasks.aime", metric="accuracy", standardized=True,
        defaults={"sets": ["2026"], "n": 30, "budget": 65536, "thinking": True}),
    "wikitext2_ppl": TaskSpec(
        "wikitext2_ppl", "glq.bench.tasks.perplexity", metric="perplexity",
        standardized=True, defaults={"seqlen": 2048, "nsamples": 128}),
    "throughput": TaskSpec(
        "throughput", "glq.bench.tasks.throughput", metric="tokens_per_s",
        standardized=False, kind="throughput",
        defaults={"input_len": 1024, "output_len": 256, "num_prompts": 64}),
    # Per-concurrency decode speed via `vllm bench sweep serve` — the measure behind the
    # README's single-stream figures and benchmarks/cmp_results_l4/. Use this, not
    # `throughput`, for the model-picker table's tok/s columns; `throughput` is one
    # aggregate rate that includes prefill.
    "decode_sweep": TaskSpec(
        "decode_sweep", "glq.bench.tasks.decode_sweep", metric="tokens_per_s",
        standardized=False, kind="throughput",
        defaults={"concurrencies": [1, 32], "num_runs": 3, "max_model_len": 2048}),
    # Reserved so the picker table's coding column has a name before it has a harness:
    # LiveCodeBench is not an lm-eval task and needs the standalone LCB repo with a code
    # execution sandbox. Raises rather than silently reporting nothing.
    "livecodebench": TaskSpec(
        "livecodebench", "glq.bench.tasks.livecodebench", metric="pass_at_1",
        standardized=True, defaults={"release": "v5", "n": None}),
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task '{name}'. Available: {', '.join(sorted(TASKS))}")
    return TASKS[name]


def list_tasks() -> list[str]:
    return sorted(TASKS)
