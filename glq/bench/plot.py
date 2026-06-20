"""Performance plots (matplotlib). Renders PNGs:

  index           bar: model -> % -of-bf16 quality index
  quality-vs-bpw  scatter: bpw vs index (the quality/compression curve)
  pareto          scatter: VRAM-at-load vs index (quality/footprint frontier)
  tok-s-by-gpu    bar: tok/s per (model, GPU)

matplotlib is an optional [bench] dependency; imported lazily with the headless
Agg backend so this works over ssh/CI.
"""
from __future__ import annotations

from pathlib import Path

from .index import compute_index
from .record import BenchRecord
from .report import model_perf

_KINDS = ("index", "quality-vs-bpw", "pareto", "tok-s-by-gpu")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _short(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def render(records: list[BenchRecord], *, kind: str = "all", out_dir="plots") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = list(_KINDS) if kind == "all" else [kind]
    idx = compute_index(records)
    perf = model_perf(records)
    paths: list[Path] = []
    for k in kinds:
        fn = {
            "index": _plot_index,
            "quality-vs-bpw": _plot_quality_vs_bpw,
            "pareto": _plot_pareto,
            "tok-s-by-gpu": _plot_toks_by_gpu,
        }.get(k)
        if fn is None:
            raise ValueError(f"unknown plot kind '{k}'; choose from {_KINDS} or 'all'")
        p = fn(records, idx, perf, out)
        if p is not None:
            paths.append(p)
    return paths


def _plot_index(records, idx, perf, out: Path):
    items = [(m, e["index"]) for m, e in idx.items() if e.get("index") is not None]
    if not items:
        return None
    items.sort(key=lambda x: x[1])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(items))))
    ax.barh([_short(m) for m, _ in items], [v * 100 for _, v in items], color="#4c72b0")
    ax.axvline(100, color="k", ls="--", lw=1, label="bf16 = 100%")
    ax.set_xlabel("Quality index (% of bf16)")
    ax.set_title("Quality retention vs bf16")
    ax.legend()
    fig.tight_layout()
    p = out / "index.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_quality_vs_bpw(records, idx, perf, out: Path):
    pts = [(perf[m].get("bpw"), e["index"], m) for m, e in idx.items()
           if e.get("index") is not None and perf.get(m, {}).get("bpw") is not None]
    if not pts:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    for bpw, v, m in pts:
        ax.scatter(bpw, v * 100, s=60)
        ax.annotate(_short(m), (bpw, v * 100), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(100, color="k", ls="--", lw=1)
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("Quality index (% of bf16)")
    ax.set_title("Quality vs compression")
    fig.tight_layout()
    p = out / "quality_vs_bpw.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_pareto(records, idx, perf, out: Path):
    pts = [(perf[m].get("vram"), e["index"], m) for m, e in idx.items()
           if e.get("index") is not None and perf.get(m, {}).get("vram") is not None]
    if not pts:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    for vram, v, m in pts:
        ax.scatter(vram, v * 100, s=60)
        ax.annotate(_short(m), (vram, v * 100), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("VRAM at load (GiB)")
    ax.set_ylabel("Quality index (% of bf16)")
    ax.set_title("Quality vs footprint (Pareto)")
    fig.tight_layout()
    p = out / "pareto_quality_vs_vram.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_toks_by_gpu(records, idx, perf, out: Path):
    # bars: (model @ gpu) -> tok/s, using the latest throughput-bearing record
    rows = []
    for r in records:
        tps = (r.benchmark.value if r.benchmark.task == "throughput" else
               (r.throughput.output_tok_s if r.throughput else None))
        if tps is not None:
            rows.append((f"{_short(r.model.id)}\n{(r.hardware.gpu_model or '?')[:18]}", tps))
    if not rows:
        return None
    # keep the max per label (dedup re-runs)
    best: dict[str, float] = {}
    for label, tps in rows:
        best[label] = max(best.get(label, 0.0), tps)
    items = sorted(best.items(), key=lambda x: x[1])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#55a868")
    ax.set_xlabel("output tok/s")
    ax.set_title("Throughput by model / GPU")
    fig.tight_layout()
    p = out / "toks_by_gpu.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
