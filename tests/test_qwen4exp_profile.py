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


# ---- mRoPE: position_ids must carry the 3 rope axes ------------------------------------

class _Cfg:
    """Minimal stand-in: only what _build_forward_kwargs reads."""
    def __init__(self, mrope):
        self.text_config = self
        self.rope_parameters = {"mrope_section": [11, 11, 10]} if mrope else {}
        self.layer_types = ["linear_attention"]
        self.hidden_size_per_layer_input = 0
        self.num_kv_shared_layers = 0


class _Rotary:
    """Reproduces the real failure: qwen4_exp indexes position_ids[:, :, None, :]."""
    def __init__(self):
        self.seen = None

    def __call__(self, x, position_ids=None, **kw):
        self.seen = tuple(position_ids.shape)
        if position_ids.dim() != 3:
            raise IndexError("too many indices for tensor of dimension 2")
        return (torch.zeros(1), torch.zeros(1))


def test_mrope_models_get_three_axis_position_ids():
    """Qwen4Exp uses mRoPE (mrope_section [11,11,10]), so modeling_qwen4_exp does
    `position_ids[:, :, None, :]` and a 2-D tensor raises IndexError. Measured live: the
    335 GiB run died here 46 s into quantization, after a 4-minute download."""
    rot = _Rotary()
    h = torch.zeros(1, 8, 4)
    kw = qm._build_forward_kwargs({}, h, rot, layer_idx=0, cfg=_Cfg(mrope=True))
    assert kw["position_ids"].dim() == 3, f"position_ids is {kw['position_ids'].shape}"
    assert kw["position_ids"].shape[0] == 3, "one row per rope axis (t/h/w)"
    assert rot.seen[0] == 3


def test_non_mrope_models_keep_two_axis_position_ids():
    """Standard rope must be untouched — every other architecture depends on it."""
    class _PlainRotary:
        def __call__(self, x, position_ids=None, **kw):
            assert position_ids.dim() == 2, "standard rope expects 2-D"
            return (torch.zeros(1), torch.zeros(1))
    h = torch.zeros(1, 8, 4)
    kw = qm._build_forward_kwargs({}, h, _PlainRotary(), layer_idx=0, cfg=_Cfg(mrope=False))
    assert kw["position_ids"].dim() == 2


# ---- hyper-connections: the decoder consumes hc_count parallel residual streams --------

def test_hc_expansion_widens_the_calibration_states():
    """Qwen4Exp keeps hc_count=4 parallel residual streams. transformers does
    `hidden_states.repeat(1, 1, config.hc_count)` between the embedding and the layer loop
    (modeling_qwen4_exp.py:1480), so the layers see 4*hidden_size.

    Feeding them plain embeddings fails with

        ValueError: Expected 10240 hyper-connection features, got 2560.

    which is what a 335 GiB run hit 22 s in, on the second attempt.
    """
    class _HC:
        hc_count = 4
    h = torch.zeros(2, 8, 2560)
    out = qm._apply_hc_expansion(h, _HC())
    assert out.shape == (2, 8, 10240)
    # It must be a repeat of the same stream, not zeros or a broadcast view.
    ref = torch.arange(4.0).reshape(1, 1, 4)
    got = qm._apply_hc_expansion(ref, _HC())
    assert torch.equal(got, ref.repeat(1, 1, 4))


def test_hc_expansion_is_a_no_op_without_hc_count():
    """Every other architecture must be untouched."""
    class _Plain:
        pass
    h = torch.zeros(2, 8, 2560)
    assert qm._apply_hc_expansion(h, _Plain()).shape == (2, 8, 2560)
    assert qm._apply_hc_expansion(h, None).shape == (2, 8, 2560)


def test_hc_expansion_ignores_hc_count_of_one():
    """hc_count == 1 means a single stream; repeating would still be a no-op but the
    guard keeps the intent explicit."""
    class _One:
        hc_count = 1
    h = torch.zeros(1, 4, 16)
    assert qm._apply_hc_expansion(h, _One()).shape == (1, 4, 16)
