"""Qwen4Exp (Qwen/Qwen3.8-Flash-Next) support: the profile, and the stacked-expert opt-in.

Why this file exists: 67% of that model's 335 GiB is fused MoE experts stored as 3-D
nn.Parameters (`experts.gate_up_proj` [512,1280,2560], `experts.down_proj` [512,2560,640]),
which the nn.Linear walk cannot see. The detection for those existed but was gated on
`is_gemma4`, a substring match on the architecture name — so a Qwen4Exp run would have
walked past the experts, completed without error, and produced a "4 bpw" checkpoint that
was still ~290 GiB and mostly bf16. A silent wrong answer, which is the failure mode this
repo cares most about.

The fix follows the pattern already established for `multimodal_text`: new architectures
opt in through their profile rather than accreting name substrings.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from glq import quantize_model as qm

ARCH = "Qwen4ExpForConditionalGeneration"


class _Experts(nn.Module):
    """Stands in for the fused expert container: 3-D Parameters, no nn.Linear."""
    def __init__(self, e=4, i=8, h=16):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.zeros(e, h, i))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _Experts()
        # Shapes taken from the real model (meta-device walk), not invented.
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_b = nn.Linear(2560, 48, bias=False)
        self.linear_attn.in_proj_a = nn.Linear(2560, 48, bias=False)
        self.linear_attn.out_proj = nn.Linear(6144, 2560, bias=False)
        self.mlp.shared_expert_gate = nn.Linear(2560, 1, bias=False)
        self.attn_hyper_connection = nn.Module()
        self.attn_hyper_connection.block_inject_weight = nn.Linear(10240, 4, bias=False)


def test_qwen4exp_profile_is_registered():
    assert ARCH in qm._MODEL_PROFILES, "no profile: the walk falls back to the default"


def test_profile_points_at_the_real_module_tree():
    """Verified on the meta-device model: the decoder is under model.language_model."""
    p = qm._MODEL_PROFILES[ARCH]
    assert p["layers_attr"] == "model.language_model.layers"
    assert p["embed_attr"] == "model.language_model.embed_tokens"
    assert p["sd_prefix"] == "model.language_model.layers"


def test_streaming_is_mandatory():
    """`mtp` is absent from the meta-instantiated class — transformers drops it — so a
    non-streaming save would silently lose the MTP head, exactly as for Qwen3.5."""
    p = qm._MODEL_PROFILES[ARCH]
    assert p.get("multimodal_text") is True
    with pytest.raises(ValueError, match="streaming"):
        qm._require_streaming_for_wrapper(ARCH, p, streaming=False)
    qm._require_streaming_for_wrapper(ARCH, p, streaming=True)   # must not raise


def test_unservable_linears_are_skipped():
    """Every one of these fails the trellis out%32 serving gate, measured on the real
    model: in_proj_a/b are 48 rows (not Qwen3.5's 16), shared_expert_gate is 1,
    block_inject_weight is 4."""
    p = qm._MODEL_PROFILES[ARCH]
    kept = qm._collect_linears(_Layer(), p)
    for banned in ("in_proj_b", "in_proj_a", "shared_expert_gate", "block_inject_weight"):
        assert not any(banned in n for n in kept), f"{banned} would be quantized: {list(kept)}"
    assert any("out_proj" in n for n in kept), "out_proj [2560,6144] must still be quantized"


def test_stacked_experts_are_detected_for_this_arch():
    """The load-bearing one. 67% of the model is here, and the old gate was a substring
    match on 'Gemma4' — under which this returns nothing and the run silently under-quantizes."""
    p = qm._MODEL_PROFILES[ARCH]
    assert qm._stacked_experts_enabled(ARCH, p) is True
    found = qm._collect_stacked_experts(_Layer())
    assert [n for n, _ in found] == ["mlp.experts"], f"experts not found: {found}"


def test_default_profile_does_not_opt_in():
    """A plain Llama-shaped model has no stacked experts; the opt-in must stay explicit."""
    assert qm._stacked_experts_enabled("LlamaForCausalLM", {}) is False
    assert qm._stacked_experts_enabled("LlamaForCausalLM", None) is False


def test_gemma4_still_opts_in_without_a_flag():
    """Back-compat: gemma-4 was detected by name and must keep working unchanged."""
    assert qm._stacked_experts_enabled("Gemma4ForConditionalGeneration", {}) is True
