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


# ---- which activations the FUSED CUDA trellis MoE op can evaluate -------------------------
#
# The fused op exists to keep the MoE decode step CUDA-graph capturable: the per-expert
# fallback (`fused_moe_method._apply_trellis`) iterates `topk_ids.unique()`, a host sync, so
# capture dies with `cudaErrorStreamCaptureUnsupported` at EVERY batch size — not just above
# the token cap. So an activation wrongly excluded here does not merely cost throughput, it
# makes the model unservable without `--enforce-eager`. That is what happened to NemotronH:
# the guard read `< 3` and its non-gated relu2 is id 5.


def test_the_gated_three_are_accepted():
    for i in (0, 1, 2):                       # silu / gelu-tanh / relu2, gated
        assert D.trellis_moe_activation_ok(i), i


def test_the_non_gated_three_are_accepted():
    """3/4/5 are the same three functions without the gate multiply. NemotronH is 5."""
    for i in (3, 4, 5):
        assert D.trellis_moe_activation_ok(i), i


@pytest.mark.parametrize("bad", [6, 7, -1, None])
def test_an_id_outside_the_table_is_refused(bad):
    """Not merely tidiness: the CUDA activation kernel picks the function with `% 3`, so an
    unknown id would silently compute silu rather than fail."""
    assert not D.trellis_moe_activation_ok(bad)


def test_the_predicate_covers_exactly_the_kernels_documented_range():
    """Pinned against the kernel's own TORCH_CHECK (`>= 0 && < 6`). If one side moves, this
    is the test that notices — the two live in different languages and cannot share a
    constant."""
    assert {i for i in range(-2, 9) if D.trellis_moe_activation_ok(i)} == set(range(6))


# --- the activation NAME -> id mapping, and its unknown-name default ---------
#
# `activation_id` ends the lookup with a default, and which default is safe depends on what
# the fused path accepts. While the fused trellis op refused ids 3-5, defaulting an
# unrecognised name to 5 was harmless: every guard refused it and the per-expert loop served
# the layer. Once the op accepted the non-gated ids, that same default became silent
# miscomputation -- and not for hypothetical names. vLLM 0.32.0 ships five activations this
# table does not list, verified from `MoEActivation`:
#
#   swigluoai, situ, swigluoai_uninterleave, swiglustep   <- all is_gated=True
#   gelu_tanh_no_mul                                      <- non-gated, but not relu^2
#
# Computing a GATED activation as non-gated relu^2 also reads the wrong number of w13
# halves. Nothing raises, nothing slows down, and no downstream check can see it.

#: Names GLQ's kernels implement, paired with the id the kernel dispatches on.
NAMED_ACTIVATIONS = [
    ("silu", 0),
    ("gelu", 1), ("gelu_tanh", 1), ("gelu_pytorch_tanh", 1),
    ("relu2", 2),
    ("silu_no_mul", 3),
    ("gelu_no_mul", 4),
    ("relu2_no_mul", 5),
]

#: vLLM activations GLQ does NOT implement. The first four are gated.
UNNAMED_VLLM_ACTIVATIONS = [
    "swigluoai", "situ", "swigluoai_uninterleave", "swiglustep", "gelu_tanh_no_mul",
]


@pytest.mark.parametrize("name,act_id", NAMED_ACTIVATIONS)
def test_the_named_activations_map_to_their_kernel_id(name, act_id):
    assert D.activation_id(name) == act_id
    assert D.trellis_moe_activation_ok(act_id)


def test_an_absent_activation_is_silu():
    """A layer that declared nothing is the overwhelmingly common gated case."""
    assert D.activation_id(None) == 0


def test_an_enum_is_read_through_its_value():
    """vLLM passes a `MoEActivation`, not a string. Duck-typed on `.value` so this stays
    testable without vLLM installed — which is the whole reason the table lives here."""
    class _Fake:
        value = "relu2_no_mul"
    assert D.activation_id(_Fake()) == 5


@pytest.mark.parametrize("name", UNNAMED_VLLM_ACTIVATIONS)
def test_an_unnamed_activation_is_refused_not_guessed(name):
    """It must decline the fused path, not become relu^2. Declining costs a fall back to
    the per-expert loop, which is correct on every shape."""
    act_id = D.activation_id(name)
    assert act_id == D.ACTIVATION_UNSUPPORTED
    assert not D.trellis_moe_activation_ok(act_id), (
        f"{name!r} -> id {act_id}, which the fused trellis gate ACCEPTS")


@pytest.mark.parametrize("name", UNNAMED_VLLM_ACTIVATIONS)
def test_the_sentinel_fails_every_guard_not_just_the_trellis_one(name):
    """Three guards have to refuse it, which is what forces the sentinel to be >= 6.

    A negative sentinel would pass the e8p/shell `< 3` guard and reach a kernel whose
    `act_type % 3` is -1 in C++ (not 2), falling through to relu^2 *with* the gated
    up-multiply. This test is what pins the sign down.
    """
    act_id = D.activation_id(name)
    assert not act_id < 3, f"{name!r} -> {act_id} passes the e8p/shell `< 3` guard"
    assert D.moe_cpu_fused_refusal(
        fused_shape_ok=True, has_stage2=False, unpadded=True, activation_type=act_id,
        ext_has_entry=True, force_fallback=False, cpu_fused_enabled=True) is not None, (
        f"{name!r} -> {act_id} is accepted by the fused CPU MoE gate")
