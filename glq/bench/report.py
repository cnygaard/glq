"""Render markdown performance tables from records: the % -of-bf16 index table,
a flat per-task table, and a model-vs-model comparison. Pure stdlib (CPU-testable);
plots live in ``plot.py``.
"""
from __future__ import annotations

import re

from .index import compute_index
from .record import BenchRecord
from .tasks.terminal_bench import _DATASET as TB_DATASET

#: The Terminal-Bench column shows only records of the version the task adapter pins: 4.0
#: dropped 8 of 3.0's 74 tasks and revised 20, so a score from another version is a different
#: number in the same column. Read from the adapter, so bumping the pin there moves the label
#: and the filter together.
TB_LABEL = "Terminal-Bench " + ".".join(TB_DATASET.rsplit("@", 1)[-1].split(".")[:2])

#: Batch sizes in the per-(model, GPU) table; each is one concurrency of `decode_sweep`.
GPU_TABLE_BATCHES = (1, 8, 32)

#: Markers around the generated section of a model card. Everything between them is
#: regenerated; everything outside is never touched.
CARD_BEGIN = "<!-- glq-bench:model-gpu-table:begin -->"
CARD_END = "<!-- glq-bench:model-gpu-table:end -->"


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


def gpu_short(name: str | None) -> str:
    """What people call the card: ``NVIDIA RTX PRO 6000 Blackwell Server Edition`` ->
    ``RTX PRO 6000``. The full name stays in every record; this is only the label."""
    if not name:
        return "unknown GPU"
    s = re.sub(r"^NVIDIA\s+", "", name.strip())
    s = re.sub(r"^GeForce\s+", "", s)
    s = re.sub(r"\s+(Blackwell|Ada|Ampere|Hopper|Lovelace)\b.*$", "", s)
    s = re.sub(r"\s+(Server|Workstation) Edition$", "", s)
    return s


_MISSING = "—"
#: Row order within a base family and GPU: GLQ builds, then the bf16 base, then the rest.
_METHOD_RANK = {"glq": 0, "none": 1, "bf16": 1}


def _avg_k(r: BenchRecord) -> int:
    return int(r.benchmark.config.get("avg_k") or r.benchmark.extra.get("avg_k") or 1)


def _gpu_cells(group: list[BenchRecord]) -> tuple[list[str], set[str]]:
    """The table cells for one (model, GPU), and the footnote marks they used."""
    def latest(rs):
        return max(rs, key=lambda r: r.timestamp_utc or "") if rs else None

    head = latest(group)
    m = head.model
    method = (m.quant_method or "none").lower()
    if m.bpw is not None:
        bpw = f"{m.bpw:g}"
    elif method in ("none", "bf16"):
        bpw = "16"
    else:
        bpw = _MISSING

    # VRAM: a full-footprint measurement beats one taken with weights offloaded to host,
    # whose number is only the resident share.
    measured = [r for r in group if r.serving and r.serving.load_gpu_mem_gib is not None]
    full = latest([r for r in measured if not r.serving.cpu_offload_gb])
    offloaded = False
    if full is not None:
        vram = f"{round(full.serving.load_gpu_mem_gib, 2):g}"
    elif measured:
        vram = f"{round(latest(measured).serving.load_gpu_mem_gib, 2):g} ‡"
        offloaded = True
    else:
        vram = _MISSING

    aime = latest([r for r in group if r.benchmark.task == "aime_2026" and _avg_k(r) == 8
                   and r.benchmark.value is not None])
    # Full-dataset runs only. `n_tasks` is None when harbor got no `-l` limit; a limited run
    # is a capability smoke, which belongs in no table — 1/3 would print as 33.3% beside
    # full-dataset scores.
    tb = latest([r for r in group if r.benchmark.task.startswith("terminal_bench")
                 and r.benchmark.config.get("dataset") == TB_DATASET
                 and not r.benchmark.config.get("n_tasks")
                 and r.benchmark.value is not None])

    toks, ttft, marks = [], [], set()
    sweeps = [r for r in group if r.benchmark.task == "decode_sweep"]
    for b in GPU_TABLE_BATCHES:
        hit = latest([r for r in sweeps
                      if str(b) in (r.benchmark.extra.get("per_concurrency") or {})])
        pc = hit.benchmark.extra["per_concurrency"][str(b)] if hit else {}
        t, f = pc.get("per_stream_tok_s"), pc.get("mean_ttft_ms")
        # `serve_extra` reaches `vllm serve`, so a sweep can run with weights offloaded.
        # That decode is bound by PCIe, not the card, and must not read as the card's speed.
        off = hit is not None and "cpu-offload" in (hit.benchmark.config.get("serve_extra") or "")
        if off and t is not None:
            marks.add("§")
        toks.append(_MISSING if t is None else f"{t:.1f}" + (" §" if off else ""))
        ttft.append(_MISSING if f is None else f"{f:.0f}")

    cells = [m.id, gpu_short(head.hardware.gpu_model if head.hardware else None), vram, bpw,
             _MISSING if aime is None else f"{aime.benchmark.value * 100:.1f}%",
             _MISSING if tb is None else f"{tb.benchmark.value * 100:.1f}%",
             *toks, *ttft]
    if offloaded:
        marks.add("‡")
    return cells, marks


def model_gpu_table(records: list[BenchRecord]) -> str:
    """One row per (model, GPU): VRAM, bpw, AIME 2026 avg@8, Terminal-Bench, and decode
    tok/s and TTFT at batch 1 / 8 / 32. Every cell comes from a record.

    Keyed on the GPU as well as the model because speed and VRAM do not transfer between
    cards, and not by a constant factor — the other tables here keep only the newest record
    per model and so mix GPUs. Rows run base family by base family, bigger card first,
    GLQ before the bf16 base before the other methods.
    """
    groups: dict[tuple[str, str], list[BenchRecord]] = {}
    for r in records:
        gpu = gpu_short(r.hardware.gpu_model if r.hardware else None)
        groups.setdefault((r.model.id, gpu), []).append(r)

    def order(item):
        (model_id, gpu), rs = item
        m = rs[0].model
        vram = max((r.hardware.gpu_total_vram_gib or 0) for r in rs if r.hardware) \
            if any(r.hardware for r in rs) else 0
        return ((m.base_model or m.id).lower(), -vram, gpu,
                _METHOD_RANK.get((m.quant_method or "none").lower(), 2), model_id)

    rows, marks = [], set()
    for _key, rs in sorted(groups.items(), key=order):
        cells, used = _gpu_cells(rs)
        rows.append(cells)
        marks |= used
    return _md_table(_GPU_HEADERS, rows) + "\n\n" + _gpu_legend(marks)


_GPU_HEADERS = (["Model", "GPU model", "VRAM GiB", "BPW", "AIME 2026 avg@8", TB_LABEL]
                + [f"Batch={b} tok/s" for b in GPU_TABLE_BATCHES]
                + [f"Batch={b} TTFT ms" for b in GPU_TABLE_BATCHES])


def _gpu_legend(marks: set[str]) -> str:
    """What the per-GPU columns mean, plus a line for each footnote mark actually used."""
    legend = [
        f"{_MISSING} = not measured on that GPU. **tok/s** is per-stream decode, "
        "`1000 / mean TPOT` from `vllm bench sweep serve` (128-token prompts, 256 output "
        "tokens, prefill excluded); **TTFT** is the mean time to first token at that "
        "concurrency. **AIME 2026** is avg@8; single-pass runs are not shown. **Terminal-Bench** is "
        "full-dataset runs only. **VRAM** is the "
        "weights resident after load (vLLM's `Model loading took` line); the KV-cache pool "
        "is not included. Every number belongs to the GPU in its row — figures from "
        "different GPUs do not compare."]
    if "‡" in marks:
        legend.append("‡ resident share only: the run offloaded weights to host memory "
                      "(`--cpu-offload-gb`), so this is not the model's full size.")
    if "§" in marks:
        legend.append("§ decoded with weights offloaded to host memory (`--cpu-offload-gb`): "
                      "bound by PCIe transfer, not by the GPU, so not comparable to rows "
                      "that fit on the card.")
    return "\n\n".join(legend)


def card_section(records: list[BenchRecord], model_id: str) -> str:
    """The generated block for ``model_id``'s card: the per-GPU table for its base family,
    so the card shows the bf16 and other references measured on the same cards."""
    own = [r for r in records if r.model.id == model_id]
    if not own:
        raise ValueError(f"no benchmark records for {model_id}: run `glq-bench run` and push "
                         "the records before updating its card")
    base = max(own, key=lambda r: r.timestamp_utc or "").model.base_model or model_id
    family = [r for r in records if (r.model.base_model or r.model.id) == base]
    body = ("## Speed and quality by GPU\n\n"
            f"_Generated by `glq-bench` from its benchmark records. Rows other than "
            f"`{model_id}` are reference builds of the same base model, measured on the same "
            "GPUs._\n\n" + model_gpu_table(family))
    return f"{CARD_BEGIN}\n{body}\n{CARD_END}"


def splice_card(card: str, section: str) -> str:
    """Put ``section`` into a model card: replace an existing generated block, or insert one
    just before ``## How GLQ works``.

    Every GLQ card ends with the base model's own card appended inside ``<details>``, and
    that text can carry any heading. So the anchor must sit before the first ``<details>``,
    and a card without one is refused rather than guessed at — a wrong guess edits somebody
    else's prose in a public repo.
    """
    if CARD_BEGIN in card or CARD_END in card:
        b, e = card.find(CARD_BEGIN), card.find(CARD_END)
        if b == -1 or e == -1 or e < b:
            raise ValueError("card has an unmatched glq-bench marker; fix it by hand first")
        return card[:b] + section + card[e + len(CARD_END):]
    anchor = card.find("\n## How GLQ works")
    details = card.find("<details>")
    if anchor == -1 or (details != -1 and anchor > details):
        raise ValueError("card not in expected state: no '## How GLQ works' heading before "
                         "the appended base-model card, so there is no safe place to insert")
    return card[:anchor + 1] + section + "\n\n" + card[anchor + 1:]


def _reference_ids(records: list[BenchRecord], base: str) -> list[str]:
    """The comparison rows for ``base``: the base itself (bf16), then its non-GLQ builds.

    Never a GLQ build. A GLQ checkpoint appears only as a member of a public collection, so
    a private experiment on the same base cannot slip in beside the published ones.
    """
    ids = {r.model.id for r in records
           if (r.model.quant_method or "none").lower() != "glq"
           and (r.model.id == base or r.model.base_model == base)}
    return sorted(ids, key=lambda i: (i != base, i.lower()))


def collection_section(records: list[BenchRecord], collection, *,
                       heading: str) -> tuple[str, set[str]]:
    """One LEADERBOARD.md section for a collection, and the footnote marks its rows used.

    Rows are grouped by base model, bases in the order their first benchmarked member appears
    in the collection. Within a base the table runs GPU by GPU, bigger card first: the
    collection's checkpoints in curator order, then the bf16 base, then other builds of the
    same base. References are shown only beside a measured member. Members with no records
    are listed under the table rather than given rows of dashes.
    """
    by_model: dict[str, list[BenchRecord]] = {}
    for r in records:
        by_model.setdefault(r.model.id, []).append(r)
    members = [m for m in collection.model_ids if m in by_model]
    unmeasured = [m for m in collection.model_ids if m not in by_model]

    def base_of(model_id: str) -> str:
        newest = max(by_model[model_id], key=lambda r: r.timestamp_utc or "")
        return newest.model.base_model or model_id

    bases: list[str] = []
    for m in members:
        if base_of(m) not in bases:
            bases.append(base_of(m))

    rows, marks = [], set()
    for base in bases:
        ids = [m for m in members if base_of(m) == base] + _reference_ids(records, base)
        by_gpu: dict[tuple[str, str], list[BenchRecord]] = {}
        card_vram: dict[str, float] = {}
        for mid in ids:
            for r in by_model.get(mid, []):
                gpu = gpu_short(r.hardware.gpu_model if r.hardware else None)
                by_gpu.setdefault((mid, gpu), []).append(r)
                vram = (r.hardware.gpu_total_vram_gib or 0) if r.hardware else 0
                card_vram[gpu] = max(card_vram.get(gpu, 0), vram)
        for gpu in sorted(card_vram, key=lambda g: (-card_vram[g], g)):
            for mid in ids:
                if (mid, gpu) in by_gpu:
                    cells, used = _gpu_cells(by_gpu[(mid, gpu)])
                    rows.append(cells)
                    marks |= used

    about = f"_[{collection.title}]({collection.url})"
    about += f" — {collection.description}_" if collection.description else "_"
    out = [heading, "", about]
    if rows:
        out += ["", _md_table(_GPU_HEADERS, rows)]
    if unmeasured:
        out += ["", "Not yet benchmarked: " + ", ".join(
            f"[{m}](https://huggingface.co/{m})" for m in unmeasured)]
    if not rows and not unmeasured:
        out += ["", "_No public checkpoints in this collection._"]
    return "\n".join(out), marks


def leaderboard(records: list[BenchRecord], *, start_here, families,
                generated_utc: str | None = None) -> str:
    """LEADERBOARD.md, structured by the Hugging Face collections: the recommended
    checkpoints first, then one section per model-family collection, by title.

    Only public collection members and their non-GLQ references appear — not every model
    with a record — so a private repo's id is never published here.
    """
    from .hfcollections import OWNER

    sections, marks = [], set()
    md, used = collection_section(records, start_here, heading="## Recommended checkpoints")
    sections.append(md)
    marks |= used
    for fam in sorted(families, key=lambda c: c.title.lower()):
        md, used = collection_section(records, fam, heading=f"## {fam.title}")
        sections.append(md)
        marks |= used

    when = f" on {generated_utc}" if generated_utc else ""
    head = ["# GLQ benchmark leaderboard", "",
            f"_Auto-generated by `glq-bench`{when} from the benchmark records and the "
            f"[GLQ collections on Hugging Face](https://huggingface.co/{OWNER}/collections)._",
            "",
            "The recommended checkpoints come first, then one section per model family. In "
            "each table rows are grouped by base model and run GPU by GPU, bigger card first: "
            "the GLQ checkpoints, then the bf16 base and other builds of the same base for "
            "reference.",
            "", _gpu_legend(marks)]
    return "\n".join(head) + "\n\n" + "\n\n".join(sections) + "\n"
