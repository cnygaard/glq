"""GLM-5.3's experts clamp before SwiGLU, and every GLQ MoE path must apply the same clamp.

transformers' ``Glm5NextTextExperts._apply_gate`` and vLLM's ``SiluAndMulWithClamp`` both
compute ``silu(min(gate, L)) * clamp(up, -L, L)`` with ``L = swiglu_limit`` (10 on
GLM-5.3-Flash). vLLM hands the limit to FusedMoE but keeps the activation named ``silu``,
so a path that maps the name alone serves the unclamped function: no error, finite output,
a different model.

This file pins the decision helpers and the HF container, which need neither vLLM nor a
GPU. The vLLM method and the CUDA kernel are covered by tests/test_glq_vllm_swiglu_limit.py.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.moe_cpu_gate import moe_cpu_fused_refusal  # noqa: E402
from glq_vllm._dispatch import (ACTIVATION_UNSUPPORTED, moe_swiglu_limit,  # noqa: E402
                                swiglu_limit_refusal)

LIMIT = 10.0


# ---- where the limit is read from --------------------------------------------------------

def test_the_limit_comes_from_the_moe_config():
    assert moe_swiglu_limit(SimpleNamespace(swiglu_limit=LIMIT), None) == LIMIT


def test_the_limit_falls_back_to_the_layer():
    """vLLM 0.32 stores it twice: on ``FusedMoEConfig`` and on ``RoutedExperts``."""
    layer = SimpleNamespace(swiglu_limit=7.0)
    assert moe_swiglu_limit(SimpleNamespace(), layer) == 7.0
    assert moe_swiglu_limit(None, layer) == 7.0


@pytest.mark.parametrize("unset", [None, 0, 0.0, -1.0])
def test_no_positive_limit_means_no_clamp(unset):
    """vLLM clamps only when the limit is > 0 (``_swiglu_limit_torch``)."""
    assert moe_swiglu_limit(SimpleNamespace(swiglu_limit=unset),
                            SimpleNamespace(swiglu_limit=unset)) == 0.0


def test_the_moe_config_wins_over_the_layer():
    assert moe_swiglu_limit(SimpleNamespace(swiglu_limit=LIMIT),
                            SimpleNamespace(swiglu_limit=3.0)) == LIMIT


# ---- which paths can apply it ------------------------------------------------------------

@pytest.mark.parametrize("codebook,act", [("trellis", 0), ("e8p", 0), ("e8_shell", 1),
                                          ("trellis", ACTIVATION_UNSUPPORTED)])
def test_no_limit_refuses_nothing(codebook, act):
    assert swiglu_limit_refusal(codebook, act, 0.0) is None


def test_trellis_silu_takes_a_limit():
    assert swiglu_limit_refusal("trellis", 0, LIMIT) is None


@pytest.mark.parametrize("codebook", ["e8p", "e8_shell"])
def test_other_codebooks_refuse_a_limit(codebook):
    """Their fused kernels and loops have no clamp. Refusing at load is the only safe
    answer: falling back would still compute the unclamped function."""
    why = swiglu_limit_refusal(codebook, 0, LIMIT)
    assert why is not None and codebook in why and "swiglu_limit" in why


@pytest.mark.parametrize("act", [1, 2, 5, ACTIVATION_UNSUPPORTED])
def test_a_limit_on_anything_but_gated_silu_is_refused(act):
    why = swiglu_limit_refusal("trellis", act, LIMIT)
    assert why is not None and "silu" in why


# ---- the fused CPU op has no clamp -------------------------------------------------------

HEALTHY_CPU = dict(fused_shape_ok=True, has_stage2=False, unpadded=True, activation_type=0,
                   ext_has_entry=True, force_fallback=False, cpu_fused_enabled=True)


def test_the_fused_cpu_op_takes_an_unclamped_layer():
    assert moe_cpu_fused_refusal(**HEALTHY_CPU) is None
    assert moe_cpu_fused_refusal(**HEALTHY_CPU, swiglu_limit=0.0) is None


def test_the_fused_cpu_op_refuses_a_clamped_layer():
    """The refusal lands on the per-expert loop, which applies the clamp."""
    why = moe_cpu_fused_refusal(**HEALTHY_CPU, swiglu_limit=LIMIT)
    assert why is not None and "swiglu_limit" in why


def test_the_clamp_reason_does_not_depend_on_the_extension():
    """A wheel without the CPU entry would otherwise report the extension, hiding that the
    layer could not take the op in any build."""
    why = moe_cpu_fused_refusal(**{**HEALTHY_CPU, "ext_has_entry": False},
                                swiglu_limit=LIMIT)
    assert "swiglu_limit" in why


# ---- HF: GLQStackedGatedExperts ----------------------------------------------------------

def _glm_experts_cls():
    pytest.importorskip("transformers", minversion="5.17")
    mod = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    return mod.Glm5NextTextExperts


def _glm_experts(limit=LIMIT, n=4, hidden=64, inter=32):
    """transformers' module, built from the four config fields its __init__ reads."""
    cfg = SimpleNamespace(num_local_experts=n, hidden_size=hidden,
                          moe_intermediate_size=inter, swiglu_limit=limit)
    return _glm_experts_cls()(cfg)


def _replaced(experts):
    from glq.fused_experts import _replace_stacked_gated_experts

    parent = torch.nn.Module()
    parent.experts = experts
    assert _replace_stacked_gated_experts(parent, codebook_type="trellis") == 1
    return parent.experts


def test_the_hf_container_inherits_the_limit():
    assert _replaced(_glm_experts()).swiglu_limit == LIMIT


def _gate_up(rows=64, inter=32, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    # ±40: well past the limit on both sides, so both clamps engage.
    return (torch.rand(rows, 2 * inter, generator=g) * 80 - 40).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_hf_gate_matches_transformers(dtype):
    """The oracle is transformers' own ``_apply_gate``, not a paraphrase of it."""
    native = _glm_experts()
    glq = _replaced(_glm_experts())
    gu = _gate_up(dtype=dtype)
    assert torch.equal(glq._activate(*gu.chunk(2, dim=-1)), native._apply_gate(gu))


def test_the_clamp_engages_on_these_inputs():
    """Guards the test above: inputs inside the limit would pass with the clamp missing."""
    glq = _replaced(_glm_experts())
    gate, up = _gate_up().chunk(2, dim=-1)
    assert not torch.equal(glq._activate(gate, up), torch.nn.functional.silu(gate) * up)


def test_a_container_without_a_limit_is_unchanged():
    """Every other stacked-expert model: the activation must be exactly ``act_fn(g) * u``."""
    from glq.fused_experts import GLQStackedGatedExperts

    with torch.device("meta"):
        glq = GLQStackedGatedExperts(2, 64, 32, torch.nn.functional.silu,
                                     codebook_type="trellis")
    assert glq.swiglu_limit is None
    gate, up = _gate_up().chunk(2, dim=-1)
    assert torch.equal(glq._activate(gate, up), torch.nn.functional.silu(gate) * up)


def test_the_hf_fused_cpu_op_refuses_a_clamped_container(monkeypatch):
    """With the weight facts healthy, the clamp alone must keep the container on the loop."""
    glq = _replaced(_glm_experts())
    facts = {"loaded": True, "has_s2": False, "is_3inst": True, "sv_shared": True,
             "shape_ok": True, "unpadded": True, "activation": 0}
    monkeypatch.setattr(type(glq), "_glq_moe_cpu_weight_facts", lambda self: facts)
    why = glq._glq_moe_cpu_refusal()
    assert why is not None and "swiglu_limit" in why
