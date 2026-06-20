"""Orchestrator: snapshot provenance + model metadata, load the model once for
the quality tasks (shared vLLM engine), run each task, and emit a fully
self-describing ``BenchRecord`` per task.

``kind`` drives execution:
  - "quality"    -> uses the shared vLLM handle (mmlu_pro, aime_*)
  - "hf"         -> the adapter loads its own HF model (perplexity)
  - "throughput" -> the adapter shells out to `vllm bench` (own engine)

Imports vllm/torch only through the runtime/adapters (lazily), so importing this
module is cheap; only ``run()`` needs a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .record import BenchmarkResult, BenchRecord, ServingMeta


@dataclass
class RunContext:
    """Shared state passed to every task adapter."""
    model: str
    quant: str | None
    arch: str | None
    hf_token: str | None
    gpu_mem_util: float
    max_model_len: int | None
    handle: Any = None               # runtime.LoadedModel for quality tasks
    standalone_serving: ServingMeta | None = None   # set by hf/throughput adapters
    pbar_factory: Any = None         # per-task vLLM use_tqdm= heartbeat factory


def _task_config(spec, *, n, budget, avg_k=1) -> dict:
    cfg = dict(spec.defaults)
    cfg["task_name"] = spec.name
    cfg["standardized"] = spec.standardized
    if n is not None:
        cfg["n"] = n
    if budget is not None:
        cfg["budget"] = budget
    if avg_k and avg_k > 1:
        cfg["avg_k"] = avg_k
    return cfg


def run(*, model: str, tasks: list[str], quant: str | None = None,
        runtime: str = "vllm", n: int | None = None, budget: int | None = None,
        avg_k: int = 1, gpu_mem_util: float = 0.9, max_model_len: int | None = None,
        hf_token: str | None = None) -> list[BenchRecord]:
    """Run ``tasks`` on ``model`` and return one ``BenchRecord`` per task."""
    from .hfmeta import model_meta
    from .provenance import env_snapshot, hardware_snapshot
    from .tasks.registry import get_task

    import time

    from .progress import _fmt, log_ts, pbar_factory

    env = env_snapshot()
    hw = hardware_snapshot()
    mm = model_meta(model, quant_override=quant, hf_token=hf_token)
    eff_quant = quant or mm.quant_method

    specs = [get_task(t) for t in tasks]
    quality = [s for s in specs if s.kind == "quality"]
    standalone = [s for s in specs if s.kind != "quality"]
    log_ts(f"run {model} (quant={eff_quant}) — tasks: {', '.join(t.name for t in specs)}")

    ctx = RunContext(model=model, quant=eff_quant, arch=mm.architecture,
                     hf_token=hf_token, gpu_mem_util=gpu_mem_util,
                     max_model_len=max_model_len)
    records: list[BenchRecord] = []

    def _record(s, idx, total, kind_serving):
        ctx.pbar_factory = pbar_factory(s.name)
        cfg = _task_config(s, n=n, budget=budget, avg_k=avg_k)
        log_ts(f"[task {idx}/{total}] {s.name} starting "
               f"(n={cfg.get('n', '?')}, budget={cfg.get('budget', '?')})")
        t0 = time.time()
        res, tp = _safe_run(s, ctx, cfg)
        status = (res.extra or {}).get("status")
        val = "skipped (" + str((res.extra or {}).get("error", "?"))[:80] + ")" \
            if status == "skipped" else f"{res.metric}={res.value}"
        log_ts(f"[task {idx}/{total}] {s.name} done in {_fmt(time.time() - t0)} → {val}")
        records.append(BenchRecord(model=mm, benchmark=res, env=env, hardware=hw,
                                   serving=kind_serving(), throughput=tp))

    total = len(specs)
    # ---- quality tasks share one vLLM engine --------------------------------
    if quality:
        budgets = [int(_task_config(s, n=n, budget=budget).get("budget", 16384))
                   for s in quality]
        mml = max_model_len or (max(budgets) + 4096)
        from . import runtime as rt
        log_ts(f"loading vLLM engine for {len(quality)} quality task(s) "
               f"(max_model_len={mml})…")
        t0 = time.time()
        ctx.handle = rt.load(model, quant=eff_quant, max_model_len=mml,
                             gpu_mem_util=gpu_mem_util, arch=mm.architecture)
        log_ts(f"engine ready: weights {ctx.handle.serving.load_gpu_mem_gib} GiB "
               f"loaded in {_fmt(time.time() - t0)}")
        for i, s in enumerate(quality, 1):
            _record(s, i, total, lambda: ctx.handle.serving)
        _free(ctx)

    # ---- standalone tasks (perplexity / throughput) -------------------------
    for j, s in enumerate(standalone, len(quality) + 1):
        ctx.standalone_serving = None
        _record(s, j, total,
                lambda: ctx.standalone_serving or ServingMeta(runtime=runtime))
    log_ts(f"run complete: {len(records)} record(s)")
    return records


def _safe_run(spec, ctx, cfg):
    """Run an adapter; on failure record a skipped result instead of aborting the
    whole batch (one bad task shouldn't lose the others)."""
    try:
        return spec.load()(ctx, cfg)
    except Exception as e:  # noqa: BLE001
        import traceback
        res = BenchmarkResult(
            task=spec.name, metric=spec.metric, value=None, standardized=False,
            config=dict(cfg),
            extra={"status": "skipped", "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-1500:]})
        return res, None


def _free(ctx: RunContext) -> None:
    try:
        import gc
        del ctx.handle.llm
        ctx.handle = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        ctx.handle = None
