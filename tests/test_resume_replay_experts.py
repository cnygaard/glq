"""Resume replay must restore STACKED routed experts, not just nn.Linear.

Replay is not bookkeeping: layer N's Hessians come from activations that passed through the
quantized layers 0..N-1, so a replayed layer has to carry its quantized weights into the
calibration forward. The restore walk was `isinstance(mod, nn.Linear)`, which on a
stacked-expert MoE (Flash-Next, gemma-4 26B-A4B, Qwen4Exp) matches attention and the dense
MLP but **not** the routed experts — those are 3-D nn.Parameters. Those experts are the
overwhelming majority of a MoE layer's weights.

The failure is silent in the worst way: `n_restored` prints a plausible non-zero count from
attention alone, the output checkpoint's already-done layers are correct, and only the layers
quantized AFTER the resume point are calibrated against a distribution a fresh run never
produces. Nothing raises.

A dense model cannot catch this — which is why these tests build a stacked layer explicitly.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _restore_replayed_weights  # noqa: E402

N_EXP, HID, INTER = 4, 8, 6


class _Experts(nn.Module):
    """The shape Gemma4TextExperts / Qwen4Exp's container actually have: stacked 3-D
    Parameters, no nn.Linear children for the walk to find."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(N_EXP, 2 * INTER, HID))
        self.down_proj = nn.Parameter(torch.zeros(N_EXP, HID, INTER))


class _MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(HID, HID, bias=False)
        self.self_attn.q_proj.weight.data.zero_()
        self.mlp = nn.Module()
        self.mlp.experts = _Experts()


def _tag(value, rows):
    """A stub artifact set: one row artifact (SU) carrying ``value`` in every row, so a
    decode of joined halves shows which rows came from which half."""
    return {"SU": torch.full((rows,), float(value))}


def _ones_dequantize(arts, in_features, out_features, codebook_type, **kw):
    """Stub decode: row r of W is SU[r]. It also checks the shape it was asked for, so a
    half decoded on its own (I rows) instead of the joined [gate; up] (2I) is caught."""
    su = arts["SU"].float()
    assert su.numel() == out_features, (su.numel(), out_features)
    return su[:, None].expand(out_features, in_features).clone()


def _arts_map(prefix):
    """Artifacts under the prefixes the quantizer actually writes, confirmed against a
    published checkpoint: per-expert `experts.{e}.{gate,up,down}_proj`."""
    m = {f"{prefix}.self_attn.q_proj": _tag(1, HID)}
    for e in range(N_EXP):
        m[f"{prefix}.mlp.experts.{e}.gate_proj"] = _tag(2, INTER)
        m[f"{prefix}.mlp.experts.{e}.up_proj"] = _tag(3, INTER)
        m[f"{prefix}.mlp.experts.{e}.down_proj"] = _tag(4, HID)
    return m


def test_stacked_experts_are_restored():
    """The bug. Without this the experts stay at their loaded bf16 values through the
    replay forward, and every later layer is calibrated against the wrong distribution."""
    layer = _MoELayer()
    n = _restore_replayed_weights(layer, _arts_map("model.layers.0"), "model.layers.0",
                                  "trellis", None, dequantize_fn=_ones_dequantize)
    gup, dwn = layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj
    assert not torch.any(gup == 0), "gate_up_proj left at its loaded value"
    assert not torch.any(dwn == 0), "down_proj left at its loaded value"
    # q_proj + 2 (gate_up counts once per expert) + down, per expert
    assert n == 1 + 2 * N_EXP


def test_gate_and_up_land_in_the_right_halves():
    """`gate_up_proj[e]` is the row-concatenation [gate; up] — the layout the forward reads
    and the same one the quantizer split on save. Swapping the halves would produce a model
    that runs, generates text, and is wrong."""
    layer = _MoELayer()
    _restore_replayed_weights(layer, _arts_map("model.layers.0"), "model.layers.0",
                              "trellis", None, dequantize_fn=_ones_dequantize)
    gup = layer.mlp.experts.gate_up_proj
    assert torch.all(gup[:, :INTER] == 2), "gate half"
    assert torch.all(gup[:, INTER:] == 3), "up half"


def test_a_dense_layer_still_restores_its_linears():
    """The pre-existing behaviour, unchanged: this is the path every dense checkpoint's
    resume already depends on."""
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(HID, HID, bias=False)
    layer.self_attn.q_proj.weight.data.zero_()
    n = _restore_replayed_weights(layer, {"model.layers.0.self_attn.q_proj": _tag(7, HID)},
                                  "model.layers.0", "trellis", None,
                                  dequantize_fn=_ones_dequantize)
    assert n == 1
    assert torch.all(layer.self_attn.q_proj.weight.data == 7)


def test_a_whitelisted_bf16_linear_without_artifacts_is_left_alone():
    """Absent artifacts means the layer was deliberately not quantized (e.g. Qwen3.5's
    16-row GDN projections). Restoring a zero or raising would both be wrong."""
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(HID, HID, bias=False)
    layer.self_attn.q_proj.weight.data.fill_(9)
    n = _restore_replayed_weights(layer, {}, "model.layers.0", "trellis", None,
                                  dequantize_fn=_ones_dequantize)
    assert n == 0
    assert torch.all(layer.self_attn.q_proj.weight.data == 9)


# ---- real decodes: gate and up were quantized as ONE matrix ---------------------------------
#
# The quantizer quantizes an expert's [gate; up] as one [2I, H] matrix and stores the result
# split (`_split_gate_up_arts`). The decode applies a row Hadamard over blocks of 2I rows, so
# each half decoded on its own inverts a different transform. Measured on random trellis
# weights: max error 0.33-0.38 against a weight std of 0.05, while the joined decode matches
# the quantizer's own W_hat exactly. The stubs above cannot see this; these use real codes.

def _joint_trellis(inter=32, hidden=64, seed=0):
    """``(W_hat, gate_arts, up_arts, codebook)`` for one jointly quantized [gate; up]."""
    import glq.trellis as gt
    from glq.quantize_model import _split_gate_up_arts
    torch.manual_seed(seed)
    W = torch.randn(2 * inter, hidden) * 0.05
    X = torch.randn(256, hidden)
    cb = gt.TrellisCodebook(variant="3inst", K=2, device="cpu")
    W_hat, arts = gt.quantize_layer_trellis_rht(W, X.T @ X / 256, cb)
    arts = {k: v if torch.is_tensor(v) else torch.tensor(v) for k, v in arts.items()}
    g, u = _split_gate_up_arts(arts, inter, inter)
    return W_hat.float(), g, u, cb


def test_replay_decodes_stacked_gate_up_as_the_matrix_that_was_quantized():
    inter, hidden = 32, 64
    W_hat, g, u, cb = _joint_trellis(inter, hidden)
    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.zeros(1, 2 * inter, hidden))
    experts.down_proj = nn.Parameter(torch.zeros(1, hidden, inter))
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = experts
    arts = {"L.mlp.experts.0.gate_proj": g, "L.mlp.experts.0.up_proj": u}
    _restore_replayed_weights(layer, arts, "L", "trellis", cb)
    err = (experts.gate_up_proj.data[0] - W_hat).abs().max().item()
    assert err == 0.0, f"replayed gate_up differs from the quantized weight by {err:.3e}"


def test_replay_decodes_per_expert_linear_gate_up_jointly_when_the_profile_fused_them():
    """Sarvam's experts are nn.Linear, quantized jointly under `fuse_expert_gate_up`."""
    inter, hidden = 32, 64
    W_hat, g, u, cb = _joint_trellis(inter, hidden, seed=1)
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.ModuleList([nn.Module()])
    ex = layer.mlp.experts[0]
    ex.gate_proj = nn.Linear(hidden, inter, bias=False)
    ex.up_proj = nn.Linear(hidden, inter, bias=False)
    arts = {"L.mlp.experts.0.gate_proj": g, "L.mlp.experts.0.up_proj": u}
    n = _restore_replayed_weights(layer, arts, "L", "trellis", cb, joint_expert_gate_up=True)
    assert n == 2
    assert (ex.gate_proj.weight.data - W_hat[:inter]).abs().max().item() == 0.0
    assert (ex.up_proj.weight.data - W_hat[inter:]).abs().max().item() == 0.0
