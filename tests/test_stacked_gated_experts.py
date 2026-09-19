"""Loading a stacked-expert MoE checkpoint under HF transformers.

`_replace_nemotron_h_experts` (glq/fused_experts.py) exists because NemotronH's native
integration packs experts into 3-D Parameters that the `nn.Linear` walk cannot see. Every
other stacked-expert architecture has the same problem and no such handler, so
`_process_model_before_weight_loading` replaces **nothing**, logs

    GLQ: no nn.Linear or NemotronHExperts modules found to replace

and transformers builds a dense bf16 model. For Qwen3.8-Flash-Next that is **335 GiB**
against a 77.5 GiB checkpoint — measured: the HF arm reported every GLQ tensor as
UNEXPECTED and exhausted a 96 GiB card.

NemotronH's MoE is **non-gated** (`up -> act -> down`). Qwen4Exp and Gemma-4 are **gated**,
storing `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]`, so `_ExpertPair` cannot be
reused. The native forward chunks the fused projection:

    gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)
    down(act(gate) * up)

so **gate is rows 0:I and up is rows I:2I** — matching `_split_gate_up_arts(arts, inter,
inter)` in the quantizer (glq/quantize_model.py:2617), which writes per-expert
`gate_proj`/`up_proj` keys in that order.

Detection is structural — a 3-D `gate_up_proj` Parameter — mirroring `_collect_stacked_experts`
(glq/quantize_model.py:667), so one implementation covers both families without importing
either transformers class.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn  # noqa: E402

from glq.fused_experts import (  # noqa: E402
    GLQStackedGatedExperts,
    _replace_stacked_gated_experts,
)

E, I, H = 4, 8, 6


class _NativeStacked(nn.Module):
    """The shape of Qwen4ExpTextExperts / Gemma4TextExperts, and its exact forward."""

    def __init__(self, num_experts=E, inter=I, hidden=H):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = inter
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for eid in hit:
            eid = eid[0]
            if eid == self.num_experts:
                continue
            pos, tok = torch.where(mask[eid])
            cur = hidden_states[tok]
            gate, up = nn.functional.linear(cur, self.gate_up_proj[eid]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = nn.functional.linear(h, self.down_proj[eid])
            h = h * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, h.to(final.dtype))
        return final


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _NativeStacked()
        self.other = nn.Linear(H, H)


# ---- structural detection ------------------------------------------------------------

def test_a_stacked_gated_container_is_replaced():
    m = _Wrapper()
    n = _replace_stacked_gated_experts(m)
    assert n == 1
    assert isinstance(m.mlp.experts, GLQStackedGatedExperts)


def test_a_model_without_stacked_experts_is_untouched():
    """The default path must not acquire an expert replacement by accident."""
    class _Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(H, H)
    m = _Plain()
    assert _replace_stacked_gated_experts(m) == 0
    assert isinstance(m.fc, nn.Linear)


def test_a_two_dimensional_gate_up_is_not_a_stacked_container():
    """A fused gate_up_proj on an ordinary MLP is 2-D. Only the 3-D (per-expert) form is
    a stacked container; matching 2-D would replace every SwiGLU MLP in the model."""
    class _FusedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2 * I, H))
    m = _FusedMLP()
    assert _replace_stacked_gated_experts(m) == 0


def test_dimensions_come_from_the_parameter_not_config_names():
    """E8RHTFusedExperts reads config.n_routed_experts / moe_intermediate_size — NemotronH
    spellings that do not exist on a Qwen4Exp or Gemma-4 config. Shapes are the only
    portable source."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m)
    ex = m.mlp.experts
    assert (ex.num_experts, ex.intermediate_dim, ex.hidden_dim) == (E, I, H)


# ---- checkpoint key layout -----------------------------------------------------------

def test_per_expert_children_match_checkpoint_keys():
    """The quantizer writes ``...experts.{e}.gate_proj.*`` (glq/quantize_model.py:2478).
    Children must sit under integer-string keys so HF's loader finds them with no rename."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m)
    names = dict(m.named_modules())
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"mlp.experts.{e}.{proj}" in names, f"missing experts.{e}.{proj}"


def test_the_codebook_type_reaches_every_expert_linear():
    """``_ExpertPair`` never forwards codebook_type, so it silently builds e8_shell
    buffers. A trellis checkpoint loaded into shell buffers fails on shapes, deep in HF's
    loader, with nothing naming GLQ."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m, codebook_type="trellis")
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(m.mlp.experts[e], proj)
            assert getattr(lin, "_is_trellis", False), f"experts.{e}.{proj} is not trellis"


def test_expert_linear_shapes_match_the_split():
    """gate/up are [I, H] each (the 2I fused projection halved); down is [H, I]."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m)
    ex = m.mlp.experts
    assert (ex[0].gate_proj.in_features, ex[0].gate_proj.out_features) == (H, I)
    assert (ex[0].up_proj.in_features, ex[0].up_proj.out_features) == (H, I)
    assert (ex[0].down_proj.in_features, ex[0].down_proj.out_features) == (I, H)


# ---- the forward ---------------------------------------------------------------------

def _swap_in_dense(container, native):
    """Replace the GLQ linears with plain nn.Linear carrying the native weights, so the
    routing and gating math can be compared without a real quantized payload."""
    for e in range(container.num_experts):
        gate_w, up_w = native.gate_up_proj[e].chunk(2, dim=0)
        g = nn.Linear(H, I, bias=False); g.weight.data = gate_w.clone()
        u = nn.Linear(H, I, bias=False); u.weight.data = up_w.clone()
        d = nn.Linear(I, H, bias=False); d.weight.data = native.down_proj[e].clone()
        container[e].gate_proj, container[e].up_proj, container[e].down_proj = g, u, d


def test_forward_matches_the_native_gated_implementation():
    torch.manual_seed(0)
    m = _Wrapper()
    native = m.mlp.experts
    ref_out = None
    tokens = 5
    x = torch.randn(tokens, H)
    idx = torch.randint(0, E, (tokens, 2))
    w = torch.rand(tokens, 2)
    with torch.no_grad():
        ref_out = native(x, idx, w)

    _replace_stacked_gated_experts(m)
    _swap_in_dense(m.mlp.experts, native)
    with torch.no_grad():
        got = m.mlp.experts(x, idx, w)
    assert torch.allclose(got, ref_out, atol=1e-5), (got - ref_out).abs().max()


def test_forward_handles_an_unrouted_expert():
    """Only hit experts run. A container that ran all E would still be numerically right
    but would cost 512x the work per token on this architecture."""
    torch.manual_seed(1)
    m = _Wrapper()
    native = m.mlp.experts
    x = torch.randn(3, H)
    idx = torch.zeros(3, 1, dtype=torch.long)     # every token -> expert 0
    w = torch.ones(3, 1)
    with torch.no_grad():
        ref = native(x, idx, w)
    _replace_stacked_gated_experts(m)
    _swap_in_dense(m.mlp.experts, native)
    with torch.no_grad():
        got = m.mlp.experts(x, idx, w)
    assert torch.allclose(got, ref, atol=1e-5)
