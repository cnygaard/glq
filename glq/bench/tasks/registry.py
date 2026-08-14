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
    # n=500, sized from a power calculation rather than from the standard error alone.
    # Target effect is 3 pt — the published GLQ-vs-bf16 gaps are 1.6-3.4 pt, and 3 pt is
    # about where a checkpoint choice would actually change. At p~0.85, 80% power:
    #
    #                     unpaired      paired (McNemar, ~8% discordance)
    #     5 pt            ~900/arm      ~250
    #     3 pt            ~2500         ~700
    #     2 pt            ~5600         ~1600
    #
    # Pairing is worth ~3.5x the items, which is why extra["per_item"] matters more than n:
    # two quantizations of one model agree on most questions and only the discordant pairs
    # carry signal. 500 detects ~3.5 pt paired at ~25 min/arm (measured 2.9 s/item) — the
    # knee. Below ~250 even 5 pt is invisible; above ~1000 you pay an hour an arm chasing
    # gaps wikitext2_ppl resolves in five minutes.
    #
    # The 8% discordance is an assumption, and per_item makes it measurable: re-derive n
    # from the observed rate once two arms exist. Big models may want to stay lower — 500
    # is ~25 min on a 3B but 1.5-2 h on a 31B.
    "mmlu_pro": TaskSpec(
        "mmlu_pro", "glq.bench.tasks.mmlu_pro", metric="accuracy", standardized=True,
        defaults={"n": 500, "budget": 16384, "thinking": True}),
    "aime_2024": TaskSpec(
        "aime_2024", "glq.bench.tasks.aime", metric="accuracy", standardized=True,
        defaults={"sets": ["2024"], "n": 30, "budget": 32768, "thinking": True}),
    "aime_2025": TaskSpec(
        "aime_2025", "glq.bench.tasks.aime", metric="accuracy", standardized=True,
        defaults={"sets": ["2025"], "n": 30, "budget": 65536, "thinking": True}),
    "aime_2026": TaskSpec(
        "aime_2026", "glq.bench.tasks.aime", metric="accuracy", standardized=True,
        defaults={"sets": ["2026"], "n": 30, "budget": 65536, "thinking": True}),
    # kind="hf", matching the adapter's docstring: perplexity isn't natural through vLLM's
    # generate API, so it loads its own HF model. Registered as "quality" it landed in the
    # shared-engine group and the runner spun up a vLLM engine beside it that nothing used
    # — wasted minutes on a 3B, an OOM risk on a 31B.
    # `max_chunks`, not `nsamples`: the adapter reads max_chunks (the old key was dead
    # config, so the real sample count was silently 80 whatever you set).
    "wikitext2_ppl": TaskSpec(
        "wikitext2_ppl", "glq.bench.tasks.perplexity", metric="perplexity",
        standardized=True, kind="hf",
        defaults={"seqlen": 2048, "max_chunks": 128}),
    # Same quantity, measured through vLLM instead of HF. Needed because some quantized
    # formats only load in one of the two runtimes: Firworks/SmolLM3-3B-nvfp4 ships no
    # `input_scale`, so transformers newly-initializes 252 of them and the forward dies,
    # while vLLM serves it weight-only via Marlin. Same seqlen/max_chunks as the HF entry
    # so the two differ ONLY by runtime.
    # standardized=False: a different runtime's PPL must not share an index column with
    # the HF series. Compare vLLM-PPL only against vLLM-PPL.
    "wikitext2_ppl_vllm": TaskSpec(
        "wikitext2_ppl_vllm", "glq.bench.tasks.perplexity_vllm", metric="perplexity",
        standardized=False, kind="quality",
        defaults={"seqlen": 2048, "max_chunks": 128}),
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
    # Agentic terminal ability — the only task here that is multi-turn, so the only one that
    # can see errors compound. standardized=False until a bf16 arm exists: with no reference
    # the %-of-bf16 index cannot use it, and including it would distort the index.
    # kind="throughput" because it runs its own vllm server and harbor subprocess.
    "terminal_bench": TaskSpec(
        "terminal_bench", "glq.bench.tasks.terminal_bench", metric="reward_mean",
        standardized=False, kind="throughput",
        defaults={"dataset": "terminal-bench/terminal-bench-2", "n_attempts": 1}),
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
