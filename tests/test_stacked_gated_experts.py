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

def test_per_expert_children_sit_under_integer_keys():
    """Children must sit under integer-string keys so HF's loader reaches them with no
    rename. gate/up are fused, so the modules are gate_up_proj and down_proj — the
    checkpoint's separate halves are stitched by the load pre-hook."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m)
    names = dict(m.named_modules())
    for e in range(E):
        for proj in ("gate_up_proj", "down_proj"):
            assert f"mlp.experts.{e}.{proj}" in names, f"missing experts.{e}.{proj}"


def test_the_codebook_type_reaches_every_expert_linear():
    """``_ExpertPair`` never forwards codebook_type, so it silently builds e8_shell
    buffers. A trellis checkpoint loaded into shell buffers fails on shapes, deep in HF's
    loader, with nothing naming GLQ."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m, codebook_type="trellis")
    for e in range(E):
        for proj in ("gate_up_proj", "down_proj"):
            lin = getattr(m.mlp.experts[e], proj)
            assert getattr(lin, "_is_trellis", False), f"experts.{e}.{proj} is not trellis"


def test_expert_linear_shapes():
    """gate_up is the fused [2I, H]; down is [H, I]."""
    m = _Wrapper()
    _replace_stacked_gated_experts(m)
    ex = m.mlp.experts
    assert (ex[0].gate_up_proj.in_features, ex[0].gate_up_proj.out_features) == (H, 2 * I)
    assert (ex[0].down_proj.in_features, ex[0].down_proj.out_features) == (I, H)


# ---- the forward ---------------------------------------------------------------------

def _swap_in_dense(container, native):
    """Replace the GLQ linears with plain nn.Linear carrying the native weights, so the
    routing and gating math can be compared without a real quantized payload."""
    for e in range(container.num_experts):
        gu = nn.Linear(H, 2 * I, bias=False)
        gu.weight.data = native.gate_up_proj[e].clone()
        d = nn.Linear(I, H, bias=False); d.weight.data = native.down_proj[e].clone()
        container[e].gate_up_proj, container[e].down_proj = gu, d


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


# ---- the fused gate_up requirement ---------------------------------------------------
#
# The decode applies a ROW-direction block-diagonal Hadamard (quantized_linear.py:74):
#
#     y = block_diagonal_fht(y_rht, self.blocks_m) * self.SU
#
# blocks_m comes from out_features. The quantizer quantizes the FUSED [2I, H] matrix, so
# the codes live in a 2I-row RHT basis: _block_decompose(1408) = [1024, 256, 128]. Decoding
# a 704-row half applies [512, 128, 64] instead -- a different transform, so the weights
# come out wrong while everything still loads. Measured on gemma-4-26B-A4B: max weight
# error 0.264 against a weight std of 0.026, and the model emitted pure garbage.
#
# vLLM avoids this by concatenating gate+up into one w13 buffer (fused_moe_method.py
# allocates w13_SU at 2 * intermediate_size) and decoding them together. The container has
# to do the same: hold ONE fused [2I, H] linear and chunk its output, exactly as the native
# forward does with `linear(x, gate_up_proj[e]).chunk(2, -1)`.

def test_the_pair_holds_one_fused_gate_up_projection():
    m = _Wrapper()
    _replace_stacked_gated_experts(m)
    pair = m.mlp.experts[0]
    assert hasattr(pair, "gate_up_proj"), "gate and up must share one fused linear"
    assert pair.gate_up_proj.out_features == 2 * I
    assert pair.gate_up_proj.in_features == H
    # gate_proj/up_proj exist only as landing pads for the checkpoint's separate halves
    # (transformers assigns by key and never calls load hooks); fuse_gate_up() drops them.
    assert pair.gate_proj.out_features == I and pair.up_proj.out_features == I


def test_checkpoint_halves_are_merged_into_the_fused_buffers():
    """The checkpoint stores per-expert ``gate_proj.*`` / ``up_proj.*`` because the
    quantizer split the fused artifacts. Loading must put them back together, row-wise,
    or the row-Hadamard basis is wrong."""
    # Built directly (not via the replacement) so the buffers are real rather than meta:
    # E8RHTLinear resizes its 0-size trellis buffers during a normal load, and the meta
    # skeleton takes HF's assign path instead.
    experts = GLQStackedGatedExperts(1, H, I, nn.SiLU(), codebook_type="trellis")

    gate_packed = torch.arange(4 * 8, dtype=torch.int16).reshape(4, 8)
    up_packed = (gate_packed + 100).to(torch.int16)
    sd = {
        "0.gate_proj.trellis_packed": gate_packed,
        "0.up_proj.trellis_packed": up_packed,
        "0.gate_proj.SU": torch.arange(I, dtype=torch.float16),
        "0.up_proj.SU": torch.arange(I, dtype=torch.float16) + 1000,
        "0.gate_proj.SV": torch.ones(H, dtype=torch.float16),
        "0.up_proj.SV": torch.ones(H, dtype=torch.float16),
        "0.gate_proj.Wscale": torch.tensor(0.5),
        "0.up_proj.Wscale": torch.tensor(0.5),
        "0.down_proj.SU": torch.ones(H, dtype=torch.float16),
        "0.down_proj.SV": torch.ones(I, dtype=torch.float16),
        "0.down_proj.Wscale": torch.tensor(0.25),
        "0.down_proj.trellis_packed": torch.zeros(4, 8, dtype=torch.int16),
    }
    experts.load_state_dict(sd, strict=False)
    assert experts[0].fuse_gate_up() is True
    assert experts[0].gate_proj is None, "halves should be dropped after fusing"
    fused = experts[0].gate_up_proj
    assert torch.equal(fused.trellis_packed,
                       torch.cat([gate_packed, up_packed], dim=0)), "packed not row-merged"
    assert torch.equal(fused.SU, torch.cat([sd["0.gate_proj.SU"],
                                            sd["0.up_proj.SU"]], dim=0)), "SU not merged"
    # SV and Wscale are shared artifacts: both halves carry identical copies.
    assert torch.equal(fused.SV, sd["0.gate_proj.SV"])
    assert float(fused.Wscale) == 0.5


def test_forward_chunks_the_fused_projection():
    """Output parity again, now through the fused linear."""
    torch.manual_seed(2)
    m = _Wrapper()
    native = m.mlp.experts
    x = torch.randn(4, H)
    idx = torch.randint(0, E, (4, 2))
    w = torch.rand(4, 2)
    with torch.no_grad():
        ref = native(x, idx, w)
    _replace_stacked_gated_experts(m)
    for e in range(E):
        gu = nn.Linear(H, 2 * I, bias=False)
        gu.weight.data = native.gate_up_proj[e].clone()
        d = nn.Linear(I, H, bias=False)
        d.weight.data = native.down_proj[e].clone()
        m.mlp.experts[e].gate_up_proj, m.mlp.experts[e].down_proj = gu, d
    with torch.no_grad():
        got = m.mlp.experts(x, idx, w)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
