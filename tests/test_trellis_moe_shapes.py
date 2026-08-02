"""Trellis MoE buffer sizing (glq_vllm.fused_moe_method).

vLLM's FusedMoE loader `copy_`s each expert into a pre-registered slot, so a wrong shape
here is not a loud error — it is a resize that silently lands the weights somewhere else.
These tests pin the sizing math on its own, without needing vLLM installed, by exercising
the pure helper the registration path calls.

The numbers come from the storage layout: `pack_layer` emits
`[(m//16)*(n//16), ceil(256*K/16)]` int16 per matrix, and 5-8 bpw adds a second stage at
K = bpw-4 over the same tile grid.
"""
from __future__ import annotations

import pytest

# The helper is pure (ints in, shapes out) and must import without vllm present.
from glq_vllm.moe_shapes import trellis_moe_shapes

from glq.trellis import trellis_rvq_recipe


def test_single_stage_4bpw_gemma4_expert():
    """gemma-4-26B-A4B: hidden 2816, inter 2112, gated -> w13 out 4224."""
    s = trellis_moe_shapes(num_experts=128, hidden_size=2816, inter=2112,
                           w13_out=4224, bpw=4)
    # w13: (4224//16) * (2816//16) = 264 * 176 tiles, width 16*K = 64
    assert s["w13_trellis_packed"] == (128, 264 * 176, 64)
    # w2:  (2816//16) * (2112//16) = 176 * 132
    assert s["w2_trellis_packed"] == (128, 176 * 132, 64)
    assert s["w13_SU"] == (128, 4224) and s["w2_SU"] == (128, 2816)
    assert s["w13_SV"] == (2816,) and s["w2_SV"] == (2112,)
    assert s["w13_Wscale"] == (128,) and s["w2_Wscale"] == (128,)


def test_stage2_sentinel_below_5bpw():
    """Below 5 bpw there is no residual. Register a numel-tiny sentinel rather than
    omitting the buffer: the loader and the numel-gated decode both key off numel, and a
    missing attribute would be an AttributeError deep in apply()."""
    s = trellis_moe_shapes(num_experts=8, hidden_size=512, inter=256, w13_out=512, bpw=4)
    assert s["w13_trellis_packed2"] == (8, 1, 1)
    assert s["w2_trellis_packed2"] == (8, 1, 1)


@pytest.mark.parametrize("bpw,k1,k2", [(5, 4, 1), (6, 4, 2), (7, 4, 3), (8, 4, 4)])
def test_stacked_rvq_second_stage_width_follows_the_recipe(bpw, k1, k2):
    """5-8 bpw is [4, bpw-4] — stage widths must come from trellis_rvq_recipe, not from a
    second hand-written table that can drift out of step with it."""
    assert trellis_rvq_recipe(bpw) == [k1, k2]
    s = trellis_moe_shapes(num_experts=4, hidden_size=512, inter=256, w13_out=512, bpw=bpw)
    tiles13 = (512 // 16) * (512 // 16)
    assert s["w13_trellis_packed"] == (4, tiles13, 16 * k1)
    assert s["w13_trellis_packed2"] == (4, tiles13, 16 * k2)
    assert s["w13_inv_resid_scale2"] == (4,)


@pytest.mark.parametrize("bpw", [2, 3, 4])
def test_native_rates(bpw):
    assert trellis_rvq_recipe(bpw) == [bpw]
    s = trellis_moe_shapes(num_experts=2, hidden_size=256, inter=128, w13_out=256, bpw=bpw)
    assert s["w13_trellis_packed"][-1] == 16 * bpw


def test_gate_up_halves_are_32_row_aligned():
    """vLLM's loader splits w13 by halving axis 0, which for the row-block-major packed
    layout means splitting on a 16-row-block boundary. The MMA byte flip pairs those blocks,
    so the half must be 32-row aligned or gate and up bytes interleave (see
    split_trellis_packed). Refuse at registration, where the message can still be useful."""
    with pytest.raises(ValueError, match="32"):
        trellis_moe_shapes(num_experts=2, hidden_size=256, inter=144,  # 144 = 4.5*32
                           w13_out=288, bpw=4)


def test_dims_must_be_tileable():
    with pytest.raises(ValueError, match="16"):
        trellis_moe_shapes(num_experts=2, hidden_size=250, inter=128, w13_out=256, bpw=4)
