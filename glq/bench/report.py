"""Render markdown performance tables from records: the % -of-bf16 index table,
a flat per-task table, and a model-vs-model comparison. Pure stdlib (CPU-testable);
plots live in ``plot.py``.
"""
from __future__ import annotations

from .index import compute_index
from .record import BenchRecord


def _md_table(headers: list[str], rows: list[list]) -> str:
    cells = [headers] + [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    def fmt(row):
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([fmt(cells[0]), sep] + [fmt(r) for r in cells[1:]])


def _latest(records: list[BenchRecord], key) -> dict:
    buckets: dict = {}
    for r in records:
        buckets.setdefault(key(r), []).append(r)
    return {k: max(v, key=lambda r: r.timestamp_utc or "") for k, v in buckets.items()}


def model_perf(records: list[BenchRecord]) -> dict[str, dict]:
    """Per-model efficiency + identity: quant, bpw, base, VRAM-at-load, tok/s."""
    out: dict[str, dict] = {}
    for r in records:
        e = out.setdefault(r.model.id, {
            "quant": r.model.quant_method, "bpw": r.model.bpw,
            "base": r.model.base_model, "vram": None, "toks": None})
        if e["vram"] is None and r.serving and r.serving.load_gpu_mem_gib is not None:
            e["vram"] = r.serving.load_gpu_mem_gib
        # prefer the dedicated throughput task; else the best in-run decode tok/s
        tps = r.throughput.output_tok_s if r.throughput else None
        if r.benchmark.task == "throughput" and r.benchmark.value is not None:
            e["toks"] = r.benchmark.value
        elif tps is not None and (e["toks"] is None or e.get("_from") != "bench"):
            e["toks"] = tps if e["toks"] is None else max(e["toks"], tps)
        if r.benchmark.task == "throughput":
            e["_from"] = "bench"
    for e in out.values():
        e.pop("_from", None)
    return out


def _pct(x: float | None) -> str:
    return "" if x is None else f"{x * 100:.1f}%"


def index_table(records: list[BenchRecord], *, weights: dict | None = None,
                only_standardized: bool = True) -> str:
    """Index table: Model | Method | bpw | <per-task retention> | Index | VRAM | tok/s."""
    idx = compute_index(records, weights=weights, only_standardized=only_standardized)
    perf = model_perf(records)
    tasks = sorted({t for e in idx.values() for t in e["per_task"]})

    headers = (["Model", "Method", "bpw"] + tasks
               + ["Index", "n", "VRAM(GiB)", "tok/s"])
    ranked = sorted(idx.items(),
                    key=lambda kv: (kv[1].get("index") is None, -(kv[1].get("index") or 0)))
    rows = []
    for model_id, e in ranked:
        p = perf.get(model_id, {})
        row = [model_id, p.get("quant") or "", p.get("bpw") if p.get("bpw") is not None else ""]
        for t in tasks:
            row.append(_pct(e["per_task"].get(t, {}).get("retention")))
        row += [_pct(e.get("index")), e.get("n_tasks", 0),
                p.get("vram") if p.get("vram") is not None else "",
                p.get("toks") if p.get("toks") is not None else ""]
        rows.append(row)

    out = ["# Quality index (% of bf16 baseline)", "", _md_table(headers, rows)]
    missing = {m: e["missing_baselines"] for m, e in idx.items() if e["missing_baselines"]}
    if missing:
        out += ["", "_Missing bf16 baseline (excluded from index):_"]
        out += [f"- {m}: {', '.join(sorted(set(t)))}" for m, t in sorted(missing.items())]
    return "\n".join(out)


# friendly --filter keys -> ModelMeta attribute names
_FILTER_ALIAS = {"base": "base_model", "quant": "quant_method", "method": "quant_method",
                 "model": "id", "arch": "architecture"}


def perf_table(records: list[BenchRecord], *, filters: dict | None = None) -> str:
    """Flat latest-per-(model,task) table with VRAM + tok/s."""
    filters = filters or {}
    def keep(r: BenchRecord) -> bool:
        for k, v in filters.items():
            got = getattr(r.model, _FILTER_ALIAS.get(k, k), None)
            if str(got) != v:
                return False
        return True
    latest = _latest([r for r in records if keep(r)],
                     lambda r: (r.model.id, r.benchmark.task))
    headers = ["Model", "Method", "bpw", "Task", "Metric", "Value", "VRAM(GiB)", "tok/s", "GPU"]
    rows = []
    for r in sorted(latest.values(), key=lambda r: (r.model.id, r.benchmark.task)):
        val = r.benchmark.value
        valstr = "" if val is None else (f"{val * 100:.1f}%"
                 if r.benchmark.metric == "accuracy" else f"{val:.4g}")
        rows.append([
            r.model.id, r.model.quant_method or "",
            r.model.bpw if r.model.bpw is not None else "",
            r.benchmark.task, r.benchmark.metric, valstr,
            r.serving.load_gpu_mem_gib if r.serving else "",
            r.throughput.output_tok_s if r.throughput else "",
            r.hardware.gpu_model or "",
        ])
    return _md_table(headers, rows)


def compare_table(records: list[BenchRecord], *, models: list[str],
                  tasks: list[str] | None = None) -> str:
    """Pivot: one row per model, one column per task (+ VRAM + tok/s)."""
    latest = _latest(records, lambda r: (r.model.id, r.benchmark.task))
    if tasks is None:
        tasks = sorted({t for (_m, t) in latest})
    perf = model_perf(records)
    headers = ["Model", "Method", "bpw"] + tasks + ["VRAM(GiB)", "tok/s"]
    rows = []
    for m in models:
        p = perf.get(m, {})
        row = [m, p.get("quant") or "", p.get("bpw") if p.get("bpw") is not None else ""]
        for t in tasks:
            r = latest.get((m, t))
            if r is None or r.benchmark.value is None:
                row.append("")
            elif r.benchmark.metric == "accuracy":
                row.append(f"{r.benchmark.value * 100:.1f}%")
            else:
                row.append(f"{r.benchmark.value:.4g}")
        row += [p.get("vram") if p.get("vram") is not None else "",
                p.get("toks") if p.get("toks") is not None else ""]
        rows.append(row)
    return _md_table(headers, rows)


def leaderboard(records: list[BenchRecord], *, weights: dict | None = None,
                generated_utc: str | None = None) -> str:
    """Full auto-generated LEADERBOARD.md: the % -of-bf16 index across all bases,
    followed by one per-base comparison table (every quant vs that base's bf16)."""
    n_models = len({r.model.id for r in records})
    n_bases = len({(r.model.base_model or r.model.id) for r in records})
    head = ["# GLQ benchmark leaderboard", ""]
    if generated_utc:
        head.append(f"_Auto-generated by `glq-bench` from {len(records)} records "
                    f"({n_models} models, {n_bases} base families) on {generated_utc}._")
    else:
        head.append(f"_Auto-generated by `glq-bench` from {len(records)} records "
                    f"({n_models} models, {n_bases} base families)._")
    head += ["", "Quality is reported as **% of the bf16 baseline of the same base "
             "model**; tok/s and VRAM-at-load are shown beside it (GPU-dependent, not "
             "folded into the index). `n` = number of standardized tasks averaged.", ""]

    parts = ["\n".join(head), index_table(records, weights=weights)]

    # Per-base comparison tables (sorted by base id).
    by_base: dict[str, list[BenchRecord]] = {}
    for r in records:
        by_base.setdefault(r.model.base_model or r.model.id, []).append(r)
    for base in sorted(by_base):
        sub = by_base[base]
        models = sorted({r.model.id for r in sub},
                        key=lambda m: (m != base, m))   # base (bf16) first
        parts.append(f"## {base}\n\n" + compare_table(sub, models=models))
    return "\n\n".join(parts) + "\n"
