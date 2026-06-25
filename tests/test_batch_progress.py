"""Batch-quantizer resume + smoke-gate logic (scripts/batch_quantization.py).

Pure-logic tests (no GPU / no network): the smoke verdict (`judge_smoke_output`), the
public-aware skip decision (`is_done`), and the atomic progress ledger
(`load_ledger`/`record_progress`). The vLLM smoke worker + HF upload are exercised
end-to-end on the GPU box, not here.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Import the script (lives under scripts/, not an installed package). Register in
# sys.modules before exec so its @dataclass can resolve its own module annotations.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "batch_quantization.py"
_spec = importlib.util.spec_from_file_location("batchq", _SCRIPT)
batchq = importlib.util.module_from_spec(_spec)
sys.modules["batchq"] = batchq
_spec.loader.exec_module(batchq)


# --------------------------------------------------------------------------- #
# judge_smoke_output — Balanced criterion
# --------------------------------------------------------------------------- #


def test_smoke_pass_on_known_answer():
    samples = [("The capital of France is", "paris", " Paris, a beautiful city."),
               ("The opposite of hot is", "cold", " cold and chilly weather.")]
    ok, reason = batchq.judge_smoke_output(samples)
    assert ok, reason


def test_smoke_pass_with_short_correct_answer():
    # short answers ('4') must not trip the empty/degenerate guards
    samples = [("The capital of France is", "paris", " Paris."),
               ("Q: What is 2+2?\nA:", "4", " 4")]
    ok, reason = batchq.judge_smoke_output(samples)
    assert ok, reason


def test_smoke_fail_all_empty():
    samples = [("The capital of France is", "paris", "   "),
               ("Q: What is 2+2?\nA:", "4", "")]
    ok, reason = batchq.judge_smoke_output(samples)
    assert not ok
    assert "empty" in reason.lower()


def test_smoke_fail_degenerate_repetition():
    loop = " the the the the the the the the the the the"  # 11x same word
    samples = [("The capital of France is", "paris", loop)]
    ok, reason = batchq.judge_smoke_output(samples)
    assert not ok
    assert "repetition" in reason.lower()


def test_smoke_fail_fluent_but_wrong():
    # coherent, non-repetitive, but the known answer never appears
    samples = [("The capital of France is", "paris",
                " London is a lovely town in the north of Italy near the sea.")]
    ok, reason = batchq.judge_smoke_output(samples)
    assert not ok
    assert "known-answer" in reason.lower()


# --------------------------------------------------------------------------- #
# is_done — public-aware skip
# --------------------------------------------------------------------------- #

NAME = "Gemma-4-E4B-it-GLQ-4bpw"
JOB = "gemma-4-e4b@4bpw"


def test_done_public_on_hf_regardless_of_target():
    pub = {NAME.lower(): False}  # public
    assert batchq.is_done(NAME, JOB, False, pub, {}) is True   # target public
    assert batchq.is_done(NAME, JOB, True, pub, {}) is True    # target private


def test_not_done_private_on_hf_when_target_public():
    pub = {NAME.lower(): True}  # private
    assert batchq.is_done(NAME, JOB, False, pub, {}) is False


def test_done_private_on_hf_when_target_private():
    pub = {NAME.lower(): True}  # private
    assert batchq.is_done(NAME, JOB, True, pub, {}) is True


def test_not_done_when_absent():
    assert batchq.is_done(NAME, JOB, False, {}, {}) is False


def test_done_via_ledger_when_absent_on_hf():
    ledger = {JOB: {"status": "done"}}
    assert batchq.is_done(NAME, JOB, False, {}, ledger) is True


def test_not_done_via_ledger_uploaded_private():
    ledger = {JOB: {"status": "uploaded-private"}}
    assert batchq.is_done(NAME, JOB, False, {}, ledger) is False


# --------------------------------------------------------------------------- #
# progress ledger — atomic round-trip + incremental merge
# --------------------------------------------------------------------------- #


def test_ledger_roundtrip_and_merge(tmp_path):
    p = tmp_path / "batch_progress.json"
    assert batchq.load_ledger(p) == {}        # absent -> empty

    ledger = {}
    batchq.record_progress(p, "xv0y5ncu", ledger, JOB,
                           model="google/gemma-4-e4b-it", output_name=NAME,
                           repo_id=f"xv0y5ncu/{NAME}", quantized=True, status="quantized")
    assert p.exists()
    doc = json.loads(p.read_text())
    assert doc["namespace"] == "xv0y5ncu"
    assert doc["jobs"][JOB]["status"] == "quantized"
    assert doc["jobs"][JOB]["quantized"] is True
    assert "updated" in doc["jobs"][JOB]

    # incremental update merges into the same record, preserves prior fields
    batchq.record_progress(p, "xv0y5ncu", ledger, JOB,
                           uploaded=True, visibility="public", status="done", smoke="passed")
    reloaded = batchq.load_ledger(p)
    rec = reloaded[JOB]
    assert rec["status"] == "done"
    assert rec["quantized"] is True            # preserved from first write
    assert rec["uploaded"] is True
    assert rec["visibility"] == "public"
    assert rec["smoke"] == "passed"


def test_ledger_load_corrupt_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    assert batchq.load_ledger(p) == {}


def test_ledger_no_tmp_left_behind(tmp_path):
    p = tmp_path / "batch_progress.json"
    batchq.record_progress(p, "ns", {}, JOB, status="done")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert not leftovers, f"temp file not cleaned up: {leftovers}"


# --------------------------------------------------------------------------- #
# codebook routing — e8p is first-class (forwarded flag, tagged name, guarded)
# --------------------------------------------------------------------------- #


def _uniform_argv(job):
    cmds = batchq.commands_for(job, Path("/tmp/out"), "glq-quantize", "cuda")
    assert len(cmds) == 1, "uniform job should be a single pass"
    return cmds[0][1]


def test_e8p_job_forwards_codebook_flag():
    job = batchq.QuantJob(model="HuggingFaceTB/SmolLM3-3B", bpw=2, codebook="e8p")
    argv = _uniform_argv(job)
    assert "--codebook" in argv and argv[argv.index("--codebook") + 1] == "e8p"


def test_default_codebook_omits_flag():
    # e8_shell is the default -> no --codebook flag (keeps existing commands unchanged)
    assert "--codebook" not in _uniform_argv(batchq.QuantJob(model="m", bpw=4))


def test_e8p_output_name_tagged():
    job = batchq.QuantJob(model="HuggingFaceTB/SmolLM3-3B", bpw=3, codebook="e8p")
    assert batchq.output_name(job) == "SmolLM3-3B-GLQ-3bpw-e8p"


def test_e8p_rejects_mixed_precision():
    with pytest.raises(ValueError, match="uniform-only"):
        batchq.QuantJob(model="m", bpw=3.5, codebook="e8p")          # fractional
    with pytest.raises(ValueError, match="uniform-only"):
        batchq.QuantJob(model="m", bpw=3, min_bpw=2, max_bpw=4, codebook="e8p")  # min/max


def test_e8p_accepts_5_to_8_bpw():
    # e8p now supports the full 2-8 RVQ depth (E8P=+2bpw, E81B=+1bpw final stage).
    for b in (5, 6, 7, 8):
        batchq.QuantJob(model="m", bpw=b, codebook="e8p")  # no raise


def test_e8p_rejects_bad_bpw():
    with pytest.raises(ValueError, match="bpw 2-8 only"):
        batchq.QuantJob(model="m", bpw=9, codebook="e8p")


def test_e8p_rejects_codebook_size():
    with pytest.raises(ValueError, match="codebook_size"):
        batchq.QuantJob(model="m", bpw=2, codebook="e8p", codebook_size=4096)


def test_invalid_codebook_rejected():
    with pytest.raises(ValueError, match="codebook must be one of"):
        batchq.QuantJob(model="m", bpw=2, codebook="bogus")
