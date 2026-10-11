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


def _ones_dequantize(arts, in_features, out_features, codebook_type, **kw):
    """Stub decode: a recognisable non-zero block, so "was it written back" is visible."""
    return torch.full((out_features, in_features), float(arts["tag"]))


def _arts_map(prefix):
    """Artifacts under the prefixes the quantizer actually writes, confirmed against a
    published checkpoint: per-expert `experts.{e}.{gate,up,down}_proj`."""
    m = {f"{prefix}.self_attn.q_proj": {"tag": 1}}
    for e in range(N_EXP):
        m[f"{prefix}.mlp.experts.{e}.gate_proj"] = {"tag": 2}
        m[f"{prefix}.mlp.experts.{e}.up_proj"] = {"tag": 3}
        m[f"{prefix}.mlp.experts.{e}.down_proj"] = {"tag": 4}
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
    n = _restore_replayed_weights(layer, {"model.layers.0.self_attn.q_proj": {"tag": 7}},
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
