"""CPU-only tests for trellis support in scripts/batch_quantization.py.

The batch driver predates the trellis codebook, and the ways it went wrong are all silent:
a job that runs at the wrong variant produces a checkpoint that quantizes fine and only
fails at serving time, hours later. So these tests pin the *mechanism* — the flag reaching
the child, the env var reaching the child, the directory name that keeps two codebooks
apart — rather than just "the job was accepted".
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Register in sys.modules before exec so the module's @dataclass can resolve its own
# annotations (see tests/test_batch_progress.py).
_SPEC = importlib.util.spec_from_file_location(
    "batchq_trellis",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "batch_quantization.py")
bq = importlib.util.module_from_spec(_SPEC)
sys.modules["batchq_trellis"] = bq
_SPEC.loader.exec_module(bq)


def _job(**kw):
    base = dict(model="HuggingFaceTB/SmolLM3-3B", bpw=4, codebook="trellis")
    return bq.QuantJob(**{**base, **kw})


# ---- #1 the codebook is accepted at all --------------------------------------
def test_trellis_is_a_valid_codebook():
    job = _job()
    assert job.codebook == "trellis"
    assert "--codebook" in bq._shared_flags(job, "cuda")


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7, 8])
def test_trellis_accepts_every_supported_rate(bpw):
    """2-4 are native trellis rates, 5-8 are stacked RVQ (K=4 primary + K=bpw-4)."""
    assert _job(bpw=bpw).bpw == bpw


# ---- #4 validation happens here, not hours into the child --------------------
def test_trellis_rejects_mixed_precision():
    """quantize_model.py refuses mixed-bpw trellis, but only after loading the model.
    Catching it at job construction is the difference between an instant error and a
    wasted multi-hour run."""
    with pytest.raises(ValueError, match="uniform"):
        _job(bpw=3.5)
    with pytest.raises(ValueError, match="uniform"):
        _job(bpw=5, min_bpw=3, max_bpw=8)


def test_trellis_rejects_out_of_range_and_shell_only_knobs():
    with pytest.raises(ValueError, match="2-8"):
        _job(bpw=9)
    with pytest.raises(ValueError, match="codebook_size"):
        _job(codebook_size=4096)


# ---- #2 the variant must reach the child process -----------------------------
def test_variant_defaults_to_3inst():
    """This tool produces NEW checkpoints, and 3inst is the variant for those: lookup-free
    kernels, and the only one that decodes 5-8 bpw. quantize_model.py still defaults to hyb
    for back-compat, so the default here is a deliberate divergence."""
    assert _job().variant == "3inst"
    with pytest.raises(ValueError, match="variant"):
        _job(variant="nonsense")


@pytest.mark.parametrize("bpw", [5, 6, 7, 8])
def test_hyb_is_refused_at_stacked_rvq_rates(bpw):
    """glq_vllm refuses hyb at bpw>=5 — stacked RVQ has no 2-stage HYB kernel. But
    quantize_model.py will happily WRITE one, so without this guard the batch tool spends
    hours producing a checkpoint that cannot be loaded at all."""
    with pytest.raises(ValueError, match="3inst"):
        _job(bpw=bpw, variant="hyb")


@pytest.mark.parametrize("bpw", [2, 3, 4])
def test_hyb_still_allowed_at_native_rates(bpw):
    """hyb has real 1-stage kernels at 2-4; refusing it there would break reproduction of
    the existing hyb checkpoints."""
    assert _job(bpw=bpw, variant="hyb").variant == "hyb"


def test_variant_is_exported_to_the_child_env():
    """GLQ_TRELLIS_VARIANT is read by quantize_model.py from the environment. If the batch
    driver does not set it per job, an ambient shell value silently decides the variant —
    and a `hyb` checkpoint cannot be served at 5-8 bpw on vLLM at all."""
    env = bq.job_env(_job(variant="3inst"), {"PATH": "/usr/bin"})
    assert env["GLQ_TRELLIS_VARIANT"] == "3inst"
    assert env["PATH"] == "/usr/bin"          # inherits, does not replace

    # An ambient value must not leak into a job that asked for something else.
    env2 = bq.job_env(_job(variant="hyb"), {"GLQ_TRELLIS_VARIANT": "3inst"})
    assert env2["GLQ_TRELLIS_VARIANT"] == "hyb"


def test_non_trellis_jobs_do_not_set_the_variant():
    env = bq.job_env(bq.QuantJob(model="m", bpw=4, codebook="e8p"), {})
    assert "GLQ_TRELLIS_VARIANT" not in env


# ---- #3 output naming keeps codebooks apart ----------------------------------
def test_output_name_encodes_codebook_and_variant():
    """Matches the published convention (SmolLM3-3B-trellis-3inst-4bpw). Without the
    suffix a trellis run overwrites the e8_shell directory of the same rate."""
    assert bq.output_name(_job(variant="3inst")) == "SmolLM3-3B-trellis-3inst-4bpw"
    assert bq.output_name(_job(variant="hyb")) == "SmolLM3-3B-trellis-hyb-4bpw"
    assert bq.output_name(_job(bpw=6)) == "SmolLM3-3B-trellis-3inst-6bpw"
    # other codebooks keep their existing names — this must not be a rename in disguise
    assert bq.output_name(bq.QuantJob(model="x/SmolLM3-3B", bpw=4, codebook="e8p")) \
        == "SmolLM3-3B-GLQ-4bpw-e8p"
    assert bq.output_name(bq.QuantJob(model="x/SmolLM3-3B", bpw=4)) == "SmolLM3-3B-GLQ-4bpw"


def test_explicit_output_still_wins():
    assert bq.output_name(_job(output="custom-dir")) == "custom-dir"


# ---- the assembled command ---------------------------------------------------
# ---- #5 the card the batch tool uploads --------------------------------------
def _render_card(bpw, variant):
    jinja2 = pytest.importorskip("jinja2")
    tpl = pathlib.Path(__file__).resolve().parents[1] / "glq/templates/model_card.md.j2"
    return jinja2.Template(tpl.read_text()).render(
        is_trellis=True, variant=variant, avg_bpw=bpw, is_mixed=False,
        base_model="HuggingFaceTB/SmolLM3-3B", model_name="X", nsamples=128, seqlen=2048,
        glq_repo="https://github.com/cnygaard/glq", stages_blurb="", benchmarks=[],
        avg_sqnr_db=None, n_quantized_layers=None, min_bpw=None, max_bpw=None)


def test_card_demands_0_8_0_for_stacked_rvq_rates():
    """A 5-8 bpw card that says 0.7.0 sends users to the build that silently decodes the
    primary stage only — 4 bpw quality, no error. The version in the card is a correctness
    claim, not decoration."""
    card = _render_card(6, "3inst")
    assert "glq >= 0.8.0" in card and "glq >= 0.7.0" not in card
    assert "stacked residual VQ" in card and "no residual stacking" not in card
    assert "K=2 residual" in card                      # 6 bpw = K=4 primary + K=2


def test_card_keeps_the_native_rate_wording_below_5_bpw():
    card = _render_card(4, "3inst")
    assert "glq >= 0.7.0" in card and "0.8.0" not in card
    assert "no residual stacking" in card


def test_commands_for_trellis_is_a_single_uniform_pass():
    cmds = bq.commands_for(_job(bpw=6, variant="3inst"), pathlib.Path("/tmp/out"),
                           "glq-quantize", "cuda")
    assert len(cmds) == 1 and cmds[0][0] == "quant"
    argv = cmds[0][1]
    assert argv[argv.index("--codebook") + 1] == "trellis"
    assert argv[argv.index("--bpw") + 1] == "6"
    assert "--bpw-map" not in argv and "--min-bpw" not in argv
