"""CPU-only tests for report (markdown tables) + store (local git roundtrip)."""
from __future__ import annotations

import subprocess

from glq.bench import report, store
from glq.bench import (
    BenchmarkResult,
    BenchRecord,
    HardwareMeta,
    ModelMeta,
    ServingMeta,
    ThroughputResult,
)


def _baseline():
    return BenchRecord(
        model=ModelMeta(id="org/M", quant_method="none"),
        benchmark=BenchmarkResult("mmlu_pro", "accuracy", 0.85, standardized=True),
        serving=ServingMeta(load_gpu_mem_gib=57.9),
        hardware=HardwareMeta(gpu_model="RTX PRO 6000"),
        timestamp_utc="2026-06-20T00:00:00Z")


def _glq():
    return BenchRecord(
        model=ModelMeta(id="xv/M-GLQ", base_model="org/M", quant_method="glq", bpw=5.0),
        benchmark=BenchmarkResult("mmlu_pro", "accuracy", 0.90, standardized=True),
        serving=ServingMeta(load_gpu_mem_gib=16.5),
        throughput=ThroughputResult(output_tok_s=69.0),
        hardware=HardwareMeta(gpu_model="RTX PRO 6000"),
        timestamp_utc="2026-06-20T00:00:00Z")


def test_index_table():
    t = report.index_table([_baseline(), _glq()])
    assert "Quality index (% of bf16 baseline)" in t
    assert "xv/M-GLQ" in t and "org/M" in t
    assert "105.9%" in t              # 0.90/0.85 retention
    assert "16.5" in t and "69" in t  # VRAM + tok/s columns


def test_index_table_shows_raw_value_next_to_retention():
    """A bare retention figure is unreadable and overstates tiny gaps.

    "103.7%" for aime_2026 was NVFP4's 93.3% over bf16's 90.0% — a ONE-question
    difference at n=30, rendered as a quality superiority. The raw score has to sit
    beside the ratio or the reader cannot judge that.
    """
    t = report.index_table([_baseline(), _glq()])
    assert "90.0 (105.9%)" in t       # raw accuracy, then the ratio
    assert "85.0 (100.0%)" in t       # the bf16 row shows its own raw score too


def test_index_table_names_the_gpu():
    """VRAM and tok/s are GPU-specific; a column of them with no GPU is not a number.

    GLQ 4bpw is 1.9x bf16 on an L4 and at parity on an RTX PRO 6000 — so an unlabelled
    tok/s column silently invites the wrong comparison.
    """
    t = report.index_table([_baseline(), _glq()])
    assert "GPU" in t
    assert "RTX PRO 6000" in t


def test_index_table_flags_tok_s_measured_on_mixed_gpus():
    """One model with records from two GPUs must not silently report the faster one."""
    a = _glq()
    b = _glq()
    b.hardware = HardwareMeta(gpu_model="NVIDIA L4")
    b.throughput = ThroughputResult(output_tok_s=180.0)
    b.timestamp_utc = "2026-06-21T00:00:00Z"
    perf = report.model_perf([a, b])["xv/M-GLQ"]
    assert perf["gpu_mixed"] is True
    # and whichever tok/s is reported must name the GPU it came from
    assert perf["toks_gpu"] in ("RTX PRO 6000", "NVIDIA L4")


def test_perf_table_and_filter():
    recs = [_baseline(), _glq()]
    pt = report.perf_table(recs)
    assert "85.0%" in pt and "90.0%" in pt and "mmlu_pro" in pt
    only_glq = report.perf_table(recs, filters={"quant_method": "glq"})
    assert "xv/M-GLQ" in only_glq and "org/M" not in only_glq


def test_compare_table():
    ct = report.compare_table([_baseline(), _glq()],
                              models=["org/M", "xv/M-GLQ"], tasks=["mmlu_pro"])
    assert "90.0%" in ct and "85.0%" in ct and "16.5" in ct and "57.9" in ct


def test_store_append_and_load_local(tmp_path):
    repo = tmp_path / "results"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    recs = [_glq(), _baseline()]
    store.append(recs, str(repo))
    # records land under records/<base>/<model>.jsonl
    assert (repo / "records" / "org__M" / "xv__M-GLQ.jsonl").exists()
    assert (repo / "records" / "org__M" / "org__M.jsonl").exists()
    got = store.load_all(str(repo), sync=False)
    assert {g.model.id for g in got} == {"xv/M-GLQ", "org/M"}


def _glq_task(task, value):
    return BenchRecord(
        model=ModelMeta(id="xv/M-GLQ", base_model="org/M", quant_method="glq", bpw=5.0),
        benchmark=BenchmarkResult(task, "accuracy", value, standardized=True),
        timestamp_utc="2026-06-20T00:00:00Z")


def test_store_append_multi_task_per_model_and_idempotent(tmp_path):
    # Regression: multiple records for ONE model.id (different tasks) must all
    # survive — they share a destination file and must not overwrite each other.
    repo = tmp_path / "results"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    recs = [_glq_task("mmlu_pro", 0.90), _glq_task("aime_2024", 0.83),
            _glq_task("aime_2026", 0.80)]
    store.append(recs, str(repo))
    got = store.load_all(str(repo), sync=False)
    assert {g.benchmark.task for g in got} == {"mmlu_pro", "aime_2024", "aime_2026"}
    # Re-appending the same records is idempotent (dedup on exact JSON).
    store.append(recs, str(repo))
    got2 = store.load_all(str(repo), sync=False)
    assert len(got2) == 3


def test_leaderboard_renders():
    md = report.leaderboard([_baseline(), _glq()], generated_utc="2026-06-20T00:00:00Z")
    assert md.startswith("# GLQ benchmark leaderboard")
    assert "Auto-generated by `glq-bench`" in md
    assert "Quality index (% of bf16 baseline)" in md   # embedded index table
    assert "## org/M" in md                              # per-base section
    assert "105.9%" in md                                # retention carried through
