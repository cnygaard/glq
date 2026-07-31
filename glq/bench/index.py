"""Weighted quality index = "% of the bf16 baseline" of the same base model.

For a model M (base B) over the standardized quality tasks it shares with B's
``quant_method=="none"`` baseline:

    retention_t = value_t(M) / baseline_t(B)          (higher-is-better metrics)
                = baseline_t(B) / value_t(M)          (perplexity etc.)
    index(M)    = Σ w_t · retention_t / Σ w_t

Throughput/VRAM are reported separately, never folded into the index (decision:
quality-only). Pure stdlib so it is CPU-unit-testable.
"""
from __future__ import annotations

from .record import BenchRecord

# Quality metrics where a SMALLER number is better (so retention inverts).
LOWER_IS_BETTER = {"perplexity", "loss", "bits_per_byte"}


def base_identity(rec: BenchRecord) -> str:
    """The base model a record measures against: its base_model, or itself if it
    IS the base (a bf16 ``quant_method=none`` record has no parent)."""
    return rec.model.base_model or rec.model.id


def _retention(metric: str, value: float, baseline: float) -> float | None:
    if not baseline or value is None:
        return None
    if metric in LOWER_IS_BETTER:
        return baseline / value if value else None
    return value / baseline


def _latest(records: list[BenchRecord]) -> BenchRecord:
    """Most recent record (records carry an ISO-8601 UTC timestamp that sorts)."""
    return max(records, key=lambda r: r.timestamp_utc or "")


def _dedup_latest(records: list[BenchRecord], key) -> dict:
    """Keep only the latest record per ``key(rec)``."""
    buckets: dict = {}
    for r in records:
        buckets.setdefault(key(r), []).append(r)
    return {k: _latest(v) for k, v in buckets.items()}


def baseline_table(records: list[BenchRecord]) -> dict[tuple[str, str], BenchRecord]:
    """(base_identity, task) -> latest bf16 baseline record."""
    bases = [r for r in records if (r.model.quant_method or "none") == "none"]
    return _dedup_latest(bases, lambda r: (base_identity(r), r.benchmark.task))


def compute_index(records: list[BenchRecord], *,
                  weights: dict[str, float] | None = None,
                  only_standardized: bool = True) -> dict[str, dict]:
    """Per-model % -of-bf16 index.

    Returns ``{model_id: {index, n_tasks, per_task:{task:{value,baseline,retention,
    weight,metric}}, missing_baselines:[task,...]}}``. Models with no usable task
    (no baseline) get ``index=None``.
    """
    weights = weights or {}
    base_tbl = baseline_table(records)

    # Latest record per (model_id, task), excluding non-standardized when asked.
    def keep(r: BenchRecord) -> bool:
        if only_standardized and not r.benchmark.standardized:
            return False
        return True

    latest = _dedup_latest([r for r in records if keep(r)],
                           lambda r: (r.model.id, r.benchmark.task))

    out: dict[str, dict] = {}
    for (model_id, task), rec in latest.items():
        b = base_tbl.get((base_identity(rec), task))
        entry = out.setdefault(model_id, {"per_task": {}, "missing_baselines": []})
        if b is None:
            entry["missing_baselines"].append(task)
            continue
        metric = rec.benchmark.metric
        ret = _retention(metric, rec.benchmark.value, b.benchmark.value)
        if ret is None:
            entry["missing_baselines"].append(task)
            continue
        entry["per_task"][task] = {
            "value": rec.benchmark.value,
            "baseline": b.benchmark.value,
            "retention": ret,
            "weight": float(weights.get(task, 1.0)),
            "metric": metric,
        }

    for model_id, entry in out.items():
        pt = entry["per_task"]
        wsum = sum(t["weight"] for t in pt.values())
        if wsum > 0:
            entry["index"] = sum(t["retention"] * t["weight"] for t in pt.values()) / wsum
        else:
            entry["index"] = None
        entry["n_tasks"] = len(pt)
    return out


def rank(records: list[BenchRecord], **kw) -> list[tuple[str, float | None, dict]]:
    """Models sorted by index descending (None last)."""
    idx = compute_index(records, **kw)
    items = [(m, e.get("index"), e) for m, e in idx.items()]
    items.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
    return items
