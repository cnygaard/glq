"""The per-(model, GPU) table published to LEADERBOARD.md and to GLQ model cards.

One row per model per GPU, because decode speed and VRAM do not transfer between cards and not
by a constant factor: GLQ 4 bpw is ~1.9x bf16 single-stream on an L4 and at parity on an
RTX PRO 6000. A table keyed on the model alone keeps whichever record is newest and silently
mixes GPUs — which is what the per-base tables still do, and why this one exists.

Every cell is computed from records; nobody types a number. That is the point of the two
filters tested hardest here: an avg@8 column must never show a single-pass AIME score, and a
Terminal-Bench 4.0 column must never show a 3.0 result. Both would be plausible numbers in the
right column, and neither is visible as wrong once rendered.
"""
from __future__ import annotations

import pytest

from glq.bench import (
    BenchmarkResult,
    BenchRecord,
    HardwareMeta,
    ModelMeta,
    ServingMeta,
)
from glq.bench import report

PRO6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
L4 = "NVIDIA L4"

GLQ = ModelMeta(id="xv0y5ncu/gemma-4-31B-it-GLQ-trellis-3inst-4bpw",
                base_model="google/gemma-4-31B-it", quant_method="glq", bpw=4.0)
BF16 = ModelMeta(id="google/gemma-4-31B-it", quant_method="none")
NVFP4 = ModelMeta(id="nvidia/Gemma-4-31B-IT-NVFP4", base_model="google/gemma-4-31B-it",
                  quant_method="modelopt", bpw=4.0)
OTHER = ModelMeta(id="xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw",
                  base_model="HuggingFaceTB/SmolLM3-3B", quant_method="glq", bpw=4.0)


def _hw(gpu):
    vram = {PRO6000: 95.6, L4: 22.0}.get(gpu, 32.0)
    return HardwareMeta(gpu_model=gpu, gpu_count=1, gpu_total_vram_gib=vram)


def aime(model, gpu, acc, avg_k=8, ts="2026-10-28T00:00:00Z", vram=None, offload=None):
    return BenchRecord(
        model=model,
        benchmark=BenchmarkResult("aime_2026", "accuracy", acc, standardized=True,
                                  config={"avg_k": avg_k, "budget": 32768, "thinking": True},
                                  extra={"avg_k": avg_k}),
        hardware=_hw(gpu),
        serving=ServingMeta(load_gpu_mem_gib=vram, cpu_offload_gb=offload),
        timestamp_utc=ts)


def tbench(model, gpu, reward, dataset="terminal-bench/terminal-bench@4.0.0",
           ts="2026-10-28T00:00:00Z"):
    return BenchRecord(
        model=model,
        benchmark=BenchmarkResult("terminal_bench", "reward_mean", reward,
                                  config={"dataset": dataset, "n_tasks": 66}),
        hardware=_hw(gpu), timestamp_utc=ts)


def sweep(model, gpu, per_conc, ts="2026-10-28T00:00:00Z"):
    """`per_conc` = {batch: (per_stream_tok_s, mean_ttft_ms)} — decode_sweep's own shape."""
    pc = {str(b): {"per_stream_tok_s": t, "mean_ttft_ms": ttft, "output_throughput": t * b}
          for b, (t, ttft) in per_conc.items()}
    head = per_conc.get(1, next(iter(per_conc.values())))[0]
    return BenchRecord(
        model=model,
        benchmark=BenchmarkResult("decode_sweep", "tokens_per_s", head,
                                  config={"concurrencies": sorted(per_conc)},
                                  extra={"per_concurrency": pc}),
        hardware=_hw(gpu), timestamp_utc=ts)


def rows_of(table: str) -> list[list[str]]:
    """Body rows of the first markdown table, as stripped cells."""
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]


def header_of(table: str) -> list[str]:
    first = next(ln for ln in table.splitlines() if ln.startswith("|"))
    return [c.strip() for c in first.strip("|").split("|")]


def row(table, model_id, gpu_short):
    for r in rows_of(table):
        if r[0] == model_id and r[1] == gpu_short:
            return dict(zip(header_of(table), r))
    raise AssertionError(f"no row for {model_id} on {gpu_short}:\n{table}")


# --------------------------------------------------------------------------- GPU names

@pytest.mark.parametrize("raw,short", [
    (PRO6000, "RTX PRO 6000"),
    ("NVIDIA RTX PRO 4500 Blackwell Server Edition", "RTX PRO 4500"),
    (L4, "L4"),
    ("NVIDIA GeForce RTX 5090", "RTX 5090"),
    ("NVIDIA H100 80GB HBM3", "H100 80GB HBM3"),
    ("RTX PRO 6000", "RTX PRO 6000"),           # already short: unchanged
])
def test_gpu_names_are_shortened_to_what_people_call_the_card(raw, short):
    assert report.gpu_short(raw) == short


def test_an_unknown_gpu_is_named_as_unknown_not_dropped():
    assert report.gpu_short(None) == "unknown GPU"


# --------------------------------------------------------------------------- shape

def test_the_columns_are_the_published_set_in_order():
    t = report.model_gpu_table([aime(GLQ, PRO6000, 0.9)])
    assert header_of(t) == [
        "Model", "GPU model", "VRAM GiB", "BPW", "AIME 2026 avg@8", "Terminal-Bench 4.0",
        "Batch=1 tok/s", "Batch=8 tok/s", "Batch=32 tok/s",
        "Batch=1 TTFT ms", "Batch=8 TTFT ms", "Batch=32 TTFT ms"]


def test_one_row_per_model_per_gpu_never_mixed():
    """The same model on two cards is two rows, each with its own card's numbers."""
    recs = [sweep(GLQ, PRO6000, {1: (130.9, 42.0)}),
            sweep(GLQ, L4, {1: (21.5, 180.0)}, ts="2026-10-29T00:00:00Z")]
    t = report.model_gpu_table(recs)
    assert row(t, GLQ.id, "RTX PRO 6000")["Batch=1 tok/s"] == "130.9"
    assert row(t, GLQ.id, "L4")["Batch=1 tok/s"] == "21.5"


def test_decode_and_ttft_come_from_each_batch_of_the_sweep():
    t = report.model_gpu_table([sweep(GLQ, PRO6000, {1: (130.9, 42.0), 8: (114.2, 90.4),
                                                      32: (60.3, 210.6)})])
    r = row(t, GLQ.id, "RTX PRO 6000")
    assert (r["Batch=1 tok/s"], r["Batch=8 tok/s"], r["Batch=32 tok/s"]) == \
        ("130.9", "114.2", "60.3")
    assert (r["Batch=1 TTFT ms"], r["Batch=8 TTFT ms"], r["Batch=32 TTFT ms"]) == \
        ("42", "90", "211")


def test_an_unmeasured_cell_is_a_dash_never_blank_or_zero():
    t = report.model_gpu_table([sweep(GLQ, PRO6000, {1: (130.9, 42.0)})])
    r = row(t, GLQ.id, "RTX PRO 6000")
    for col in ("VRAM GiB", "AIME 2026 avg@8", "Terminal-Bench 4.0",
                "Batch=8 tok/s", "Batch=32 TTFT ms"):
        assert r[col] == "—", (col, r[col])
    assert "— = not measured" in t


# --------------------------------------------------------------------------- the filters

def test_an_avg_at_1_aime_never_fills_the_avg_at_8_column():
    """The newest record is a single pass; it must lose to the older avg@8 one."""
    recs = [aime(GLQ, PRO6000, 0.904, avg_k=8, ts="2026-10-01T00:00:00Z"),
            aime(GLQ, PRO6000, 0.967, avg_k=1, ts="2026-10-20T00:00:00Z")]
    assert row(report.model_gpu_table(recs), GLQ.id, "RTX PRO 6000")[
        "AIME 2026 avg@8"] == "90.4%"


def test_only_single_pass_aime_leaves_the_avg_at_8_cell_empty():
    recs = [aime(GLQ, PRO6000, 0.967, avg_k=1)]
    assert row(report.model_gpu_table(recs), GLQ.id, "RTX PRO 6000")[
        "AIME 2026 avg@8"] == "—"


def test_a_terminal_bench_3_result_never_fills_the_4_0_column():
    """4.0 dropped 8 of 3.0's 74 tasks and revised 20: the scores are not comparable."""
    recs = [tbench(GLQ, PRO6000, 0.30, dataset="terminal-bench/terminal-bench@3.0.0")]
    assert row(report.model_gpu_table(recs), GLQ.id, "RTX PRO 6000")[
        "Terminal-Bench 4.0"] == "—"
    recs.append(tbench(GLQ, PRO6000, 0.25))
    assert row(report.model_gpu_table(recs), GLQ.id, "RTX PRO 6000")[
        "Terminal-Bench 4.0"] == "25.0%"


def test_the_terminal_bench_column_follows_the_adapters_pin():
    """The label and the filter come from the task adapter, so bumping the pin there moves
    both — a hand-kept '4.0' here would keep advertising the old version."""
    from glq.bench.tasks.terminal_bench import _DATASET
    assert _DATASET.endswith("@4.0.0")
    assert report.TB_DATASET == _DATASET
    assert report.TB_LABEL == "Terminal-Bench 4.0"


# --------------------------------------------------------------------------- VRAM / BPW

def test_vram_comes_from_whatever_run_measured_it_on_that_gpu():
    recs = [sweep(GLQ, PRO6000, {1: (130.9, 42.0)}),
            aime(GLQ, PRO6000, 0.9, vram=17.84)]
    assert row(report.model_gpu_table(recs), GLQ.id, "RTX PRO 6000")["VRAM GiB"] == "17.84"


def test_an_offloaded_vram_figure_is_marked_as_the_resident_share():
    """With --cpu-offload-gb the number is what stayed on the card, not the model's size."""
    recs = [aime(BF16, L4, 0.88, vram=19.1, offload=42.0)]
    t = report.model_gpu_table(recs)
    assert row(t, BF16.id, "L4")["VRAM GiB"] == "19.1 ‡"
    assert "‡" in t.split("\n\n", 1)[-1]          # the footnote explains it


def test_bf16_reports_16_bits_and_a_quant_reports_its_bpw():
    recs = [aime(BF16, PRO6000, 0.87), aime(NVFP4, PRO6000, 0.93), aime(GLQ, PRO6000, 0.9)]
    t = report.model_gpu_table(recs)
    assert row(t, BF16.id, "RTX PRO 6000")["BPW"] == "16"
    assert row(t, NVFP4.id, "RTX PRO 6000")["BPW"] == "4"
    assert row(t, GLQ.id, "RTX PRO 6000")["BPW"] == "4"


# --------------------------------------------------------------------------- order

def test_rows_group_by_base_then_bigger_gpu_first_then_glq_bf16_others():
    """The order of the table in the request: GLQ, bf16, NVFP4 per GPU, RTX PRO 6000
    before L4, and each base family together."""
    recs = [aime(m, g, 0.9) for g in (L4, PRO6000) for m in (NVFP4, BF16, GLQ)]
    recs.append(aime(OTHER, PRO6000, 0.4))
    order = [(r[0], r[1]) for r in rows_of(report.model_gpu_table(recs))]
    assert order[:6] == [(GLQ.id, "RTX PRO 6000"), (BF16.id, "RTX PRO 6000"),
                         (NVFP4.id, "RTX PRO 6000"), (GLQ.id, "L4"), (BF16.id, "L4"),
                         (NVFP4.id, "L4")]
    assert order[6] == (OTHER.id, "RTX PRO 6000")   # another base family follows


# --------------------------------------------------------------------------- leaderboard

def test_the_leaderboard_leads_with_the_table():
    recs = [aime(BF16, PRO6000, 0.87), aime(GLQ, PRO6000, 0.9),
            sweep(GLQ, PRO6000, {1: (130.9, 42.0)})]
    md = report.leaderboard(recs, generated_utc="2026-10-28T00:00:00Z")
    assert "## Speed and quality by GPU" in md
    assert md.index("## Speed and quality by GPU") < md.index("# Quality index")


# --------------------------------------------------------------------------- model card

CARD = """---
license: apache-2.0
---

# gemma-4-31B-it — GLQ trellis (3INST) 4 bpw

## Install

pip install 'glq[vllm]'

## How GLQ works

GLQ text.

<details>
<summary>Original model card</summary>

## How GLQ works

(a heading inside the appended base card, which must never be used as the anchor)

</details>
"""


def test_the_card_section_holds_only_the_models_base_family():
    recs = [aime(GLQ, PRO6000, 0.9), aime(BF16, PRO6000, 0.87),
            aime(NVFP4, PRO6000, 0.93), aime(OTHER, PRO6000, 0.4)]
    sec = report.card_section(recs, GLQ.id)
    assert GLQ.id in sec and BF16.id in sec and NVFP4.id in sec
    assert OTHER.id not in sec
    assert sec.startswith(report.CARD_BEGIN) and sec.rstrip().endswith(report.CARD_END)


def test_a_model_with_no_records_is_refused_rather_than_given_an_empty_table():
    with pytest.raises(ValueError, match="no benchmark records"):
        report.card_section([aime(OTHER, PRO6000, 0.4)], GLQ.id)


def test_the_section_goes_before_how_glq_works_and_nothing_else_moves():
    sec = report.card_section([aime(GLQ, PRO6000, 0.9)], GLQ.id)
    new = report.splice_card(CARD, sec)
    assert new.index(report.CARD_BEGIN) < new.index("## How GLQ works")
    # everything outside the section is byte-for-byte what was there
    assert new.replace(sec + "\n\n", "") == CARD


def test_splicing_twice_replaces_rather_than_duplicates():
    first = report.splice_card(CARD, report.card_section([aime(GLQ, PRO6000, 0.9)], GLQ.id))
    newer = report.card_section([aime(GLQ, PRO6000, 0.95, ts="2026-11-01T00:00:00Z")],
                                GLQ.id)
    second = report.splice_card(first, newer)
    assert second.count(report.CARD_BEGIN) == 1
    assert "95.0%" in second and "90.0%" not in second
    assert report.splice_card(second, newer) == second          # idempotent


def test_a_card_without_the_glq_anchor_is_refused_not_guessed():
    """Only a heading INSIDE the appended base card: inserting there would edit somebody
    else's prose, which is the failure the update-model-cards skill exists to prevent."""
    only_in_base = CARD.replace("## How GLQ works\n\nGLQ text.\n\n", "")
    sec = report.card_section([aime(GLQ, PRO6000, 0.9)], GLQ.id)
    with pytest.raises(ValueError, match="How GLQ works"):
        report.splice_card(only_in_base, sec)


def test_the_card_command_writes_the_spliced_card_offline(tmp_path):
    from glq.bench import cli
    from glq.bench.record import write_jsonl
    recs = tmp_path / "r.jsonl"
    write_jsonl([aime(GLQ, PRO6000, 0.9), aime(BF16, PRO6000, 0.87)], recs)
    card = tmp_path / "README.md"
    card.write_text(CARD, encoding="utf-8")
    out = tmp_path / "new.md"
    rc = cli.main(["card", "--model", GLQ.id, "--records", str(recs),
                   "--card-file", str(card), "--out", str(out)])
    assert rc == 0
    new = out.read_text(encoding="utf-8")
    assert report.CARD_BEGIN in new and "90.0%" in new
    assert card.read_text(encoding="utf-8") == CARD        # the source is not modified
