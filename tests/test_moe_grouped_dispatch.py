"""Unit tests for the GLQ MoE grouped-GEMM dispatch policy (GLQ_MOE_GROUPED).

Pure-logic tests for ``glq_vllm._dispatch._grouped_enabled`` — the tri-state gate
that picks the Stage-3 grouped-GEMM path vs the bit-exact block-diag matvec. No
torch / vLLM / GPU needed, so this locks the default-on-for-batched policy in CI.
"""
import pytest

from glq_vllm import _dispatch as D
from glq_vllm._dispatch import _grouped_enabled


# --- default ("auto"): grouped for batched MoE, block-diag for b1 ------------

@pytest.mark.parametrize("env", [None, "", "auto", "AUTO", "  auto  ", "bogus"])
def test_auto_keeps_b1_on_blockdiag(env):
    # b1 (num_tokens < grouped_min) must NOT take the grouped path.
    assert _grouped_enabled(env, grouped_min=2, num_tokens=1) is False


@pytest.mark.parametrize("env", [None, "auto"])
@pytest.mark.parametrize("nt", [2, 4, 8, 16, 32, 64, 256])
def test_auto_uses_grouped_for_batched(env, nt):
    assert _grouped_enabled(env, grouped_min=2, num_tokens=nt) is True


def test_auto_respects_custom_min():
    # With a higher floor, mid batches fall back to block-diag until the floor.
    assert _grouped_enabled("auto", grouped_min=8, num_tokens=4) is False
    assert _grouped_enabled("auto", grouped_min=8, num_tokens=8) is True


# --- "1"/on: force grouped everywhere (incl. b1) -----------------------------

@pytest.mark.parametrize("env", ["1", "on", "true", "yes", "TRUE", " On "])
@pytest.mark.parametrize("nt", [1, 2, 32])
def test_force_on(env, nt):
    assert _grouped_enabled(env, grouped_min=2, num_tokens=nt) is True


# --- "0"/off: never grouped (force block-diag; A/B isolation) -----------------

@pytest.mark.parametrize("env", ["0", "off", "false", "no", "FALSE", " Off "])
@pytest.mark.parametrize("nt", [1, 2, 32, 1000])
def test_force_off(env, nt):
    assert _grouped_enabled(env, grouped_min=2, num_tokens=nt) is False


def test_b1_default_off_but_forceable():
    # The shipped default: b1 -> block-diag (bit-exact), batched -> grouped.
    assert _grouped_enabled(None, 2, 1) is False     # b1 decode
    assert _grouped_enabled(None, 2, 32) is True      # b32 decode
    # ...but a user can force grouped on b1 for testing.
    assert _grouped_enabled("1", 2, 1) is True


# ---- which MoE checkpoints can serve on the CPU platform ---------------------------------
#
# Until now every GLQ MoE was refused on CPU because the expert kernels are CUDA-only. That
# is still true of the FUSED paths, but the trellis per-expert fallback
# (fused_moe_method._apply_trellis) reaches E8RHTLinear._trellis_linear_apply, which has had
# a CPU branch since the fused CPU decode shipped — so a trellis MoE can serve on CPU today,
# unfused. e8p and shell have no such branch: their per-expert dequant asserts sv.is_cuda.

def test_trellis_moe_is_allowed_on_cpu():
    assert D.moe_cpu_refusal("trellis") is None


@pytest.mark.parametrize("codebook", ["e8p", "e8_shell"])
def test_other_codebooks_are_refused_with_the_reason_and_the_fix(codebook):
    why = D.moe_cpu_refusal(codebook)
    assert why, f"{codebook} MoE has no CPU expert path and must be refused"
    assert codebook in why, "the message must name the codebook the user actually has"
    assert "trellis" in why, "and point at the thing that does work on CPU"


def test_an_unknown_codebook_is_refused_rather_than_assumed_to_work():
    """A new codebook must opt in explicitly: serving garbage beats nothing only never."""
    assert D.moe_cpu_refusal("some_future_codebook")


# ---- which layers the FUSED CPU MoE op can take ------------------------------------------
#
# `glq_fused_moe_trellis_3inst_cpu` is faster than the per-expert loop but narrower: it has
# no stage-2 (5-8 bpw RVQ) inputs, assumes trellis' unpadded shapes, and needs the same
# m % 32 / n % 64 alignment the CPU kernel does. Everything it cannot take must land on the
# loop, which is correct on every shape — so each entry here is a real limit of the op.

FUSED_OK = dict(fused_shape_ok=True, has_stage2=False, unpadded=True, activation_type=1,
                ext_has_entry=True, force_fallback=False, cpu_fused_enabled=True)


def test_a_healthy_trellis_moe_takes_the_fused_cpu_op():
    assert D.moe_cpu_fused_refusal(**FUSED_OK) is None


@pytest.mark.parametrize("override, must_mention", [
    (dict(has_stage2=True), "5-8"),               # RVQ stage 2: no CPU op takes packed2
    (dict(unpadded=False), "pad"),                # the op passes logical dims to the bracket
    (dict(fused_shape_ok=False), "shape"),        # m % 32 / n % 64 / R out of range
    (dict(activation_type=3), "activation"),      # *_no_mul: the op is gated-only
    (dict(ext_has_entry=False), "extension"),     # wheel older than the symbol
    (dict(force_fallback=True), "GLQ_MOE_FORCE_FALLBACK"),
    (dict(cpu_fused_enabled=False), "GLQ_FUSED_TRELLIS_CPU"),
])
def test_each_limit_falls_back_with_a_reason_naming_it(override, must_mention):
    why = D.moe_cpu_fused_refusal(**{**FUSED_OK, **override})
    assert why, f"{override} must not take the fused CPU op"
    assert must_mention.lower() in why.lower(), (
        f"the reason must name what stopped it; got {why!r}")


def test_the_two_switches_are_independent():
    """Turning off CPU fused decode must not read as 'force the MoE fallback', and vice
    versa — they are separate A/B levers and a shared one cannot isolate either path."""
    assert D.moe_cpu_fused_refusal(**{**FUSED_OK, "force_fallback": True}) \
        != D.moe_cpu_fused_refusal(**{**FUSED_OK, "cpu_fused_enabled": False})
