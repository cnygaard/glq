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
    """Per-model efficiency + identity: quant, bpw, base, VRAM-at-load, tok/s, GPU.

    tok/s and VRAM are GPU-specific and NOT by a constant factor — GLQ 4 bpw is 1.9x bf16
    single-stream on an L4 and at parity on an RTX PRO 6000. So each is reported with the
    GPU it came from, and ``gpu_mixed`` marks a model whose records span several GPUs
    (where taking the max would quietly advertise the fastest card).
    """
    out: dict[str, dict] = {}
    for r in records:
        e = out.setdefault(r.model.id, {
            "quant": r.model.quant_method, "bpw": r.model.bpw,
            "base": r.model.base_model, "vram": None, "toks": None,
            "vram_gpu": None, "toks_gpu": None, "gpus": set(), "gpu_mixed": False})
        gpu = r.hardware.gpu_model if r.hardware else None
        if gpu:
            e["gpus"].add(gpu)
        if e["vram"] is None and r.serving and r.serving.load_gpu_mem_gib is not None:
            e["vram"] = r.serving.load_gpu_mem_gib
            e["vram_gpu"] = gpu
        # prefer the dedicated throughput task; else the best in-run decode tok/s
        tps = r.throughput.output_tok_s if r.throughput else None
        if r.benchmark.task == "throughput" and r.benchmark.value is not None:
            e["toks"], e["toks_gpu"] = r.benchmark.value, gpu
        elif tps is not None and (e["toks"] is None or e.get("_from") != "bench"):
            if e["toks"] is None or tps > e["toks"]:
                e["toks"], e["toks_gpu"] = tps, gpu
        if r.benchmark.task == "throughput":
            e["_from"] = "bench"
    for e in out.values():
        e.pop("_from", None)
        e["gpu_mixed"] = len(e["gpus"]) > 1
        e["gpu"] = sorted(e["gpus"])[0] if len(e["gpus"]) == 1 else (
            " / ".join(sorted(e["gpus"])) if e["gpus"] else None)
    return out


def _pct(x: float | None) -> str:
    return "" if x is None else f"{x * 100:.1f}%"


def _raw_and_retention(cell: dict) -> str:
    """``93.3 (103.7%)`` — the score first, its ratio to bf16 in parentheses.

    Retention alone is unreadable and flatters small gaps: 103.7% on aime_2026 was one
    question at n=30. Showing the raw score lets the reader see that for themselves.
    """
    val, ret = cell.get("value"), cell.get("retention")
    if val is None:
        return _pct(ret)
    # accuracy-style metrics are stored as fractions; perplexity etc. are absolute.
    shown = f"{val * 100:.1f}" if 0.0 <= val <= 1.0 else f"{val:.4g}"
    out = shown if ret is None else f"{shown} ({_pct(ret)})"
    # One sample per problem has variance that swamps any quantization delta — the same
    # checkpoint has reproduced at 43.3% and 41.7%. Mark those so they are not read as
    # equals of an avg@8 cell; records too old to say leave avg_k None and get the mark too.
    if (cell.get("avg_k") or 1) < 2:
        out += " †"
    return out


def index_table(records: list[BenchRecord], *, weights: dict | None = None,
                only_standardized: bool = True) -> str:
    """Index table: Model | Method | bpw | <per-task retention> | Index | VRAM | tok/s."""
    idx = compute_index(records, weights=weights, only_standardized=only_standardized)
    perf = model_perf(records)
    tasks = sorted({t for e in idx.values() for t in e["per_task"]})

    headers = (["Model", "Method", "bpw"] + tasks
               + ["Index", "n", "VRAM(GiB)", "tok/s", "GPU"])
    ranked = sorted(idx.items(),
                    key=lambda kv: (kv[1].get("index") is None, -(kv[1].get("index") or 0)))
    rows = []
    mixed = []
    single_sample = False
    for model_id, e in ranked:
        p = perf.get(model_id, {})
        row = [model_id, p.get("quant") or "", p.get("bpw") if p.get("bpw") is not None else ""]
        for t in tasks:
            cell = e["per_task"].get(t, {})
            if cell and (cell.get("avg_k") or 1) < 2:
                single_sample = True
            row.append(_raw_and_retention(cell))
        gpu = p.get("gpu") or ""
        if p.get("gpu_mixed"):
            mixed.append((model_id, p.get("toks_gpu")))
        row += [_pct(e.get("index")), e.get("n_tasks", 0),
                p.get("vram") if p.get("vram") is not None else "",
                p.get("toks") if p.get("toks") is not None else "",
                gpu]
        rows.append(row)

    out = ["# Quality index (% of bf16 baseline)", "",
           "Quality cells are `raw (% of bf16)`. Quality is GPU-independent; **VRAM and "
           "tok/s are not** — read them only against the GPU named in the last column.", "",
           _md_table(headers, rows)]
    if single_sample:
        out += ["", "† **one sample per problem** (avg@1, or a record predating `avg_k` "
                "capture). At n=30 that variance swamps any quantization delta — the same "
                "checkpoint has reproduced at 43.3% and 41.7% on identical settings. Treat "
                "these as indicative only; avg@8 is required before a cell is comparable."]
    if mixed:
        out += ["", "_Records span several GPUs; the tok/s shown is the fastest single "
                "measurement, from:_"]
        out += [f"- {m}: {g or 'unknown GPU'}" for m, g in sorted(mixed)]
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
    headers = ["Model", "Method", "bpw"] + tasks + ["VRAM(GiB)", "tok/s", "GPU"]
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
                p.get("toks") if p.get("toks") is not None else "",
                p.get("gpu") or ""]
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
    head += ["", "Quality cells in the index are **`raw score (% of that base model's "
             "bf16)`** — the ratio alone flatters small gaps, so the raw number is shown "
             "with it. `n` = number of standardized tasks averaged. **tok/s and "
             "VRAM-at-load are GPU-specific** and are not folded into the index; each row "
             "names the GPU it was measured on, and figures from different GPUs are not "
             "comparable (GLQ 4 bpw is ~1.9x bf16 single-stream on an L4 and at parity on "
             "an RTX PRO 6000).", ""]

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
