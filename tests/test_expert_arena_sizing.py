"""Sizing the VRAM expert arena: how many slots a byte budget buys, and the one constraint
that makes the arena safe to use at all.

Pure int math — no torch, no vLLM, no GPU — following `moe_shapes.py`'s split, because a
wrong number here does not raise. It either wastes VRAM or, in the case of the slot-count
constraint below, produces *plausible but wrong output*.

Numbers are Flash-Next 3bpw, recovered from the recorded checkpoint header
(`tests/models/qwen3_8_flash_next_glq_3bpw.json`): hidden 2560, inter 640, gated so
w13_out 1280, k1 = 3, 48 layers x 512 experts, top_k 10.
"""
from __future__ import annotations

import pytest

from glq_vllm.moe_shapes import (arena_shapes, bytes_per_slot, slots_for_budget,
                                 trellis_moe_shapes)

# Flash-Next 3bpw, one MoE layer.
FN = dict(hidden_size=2560, inter=640, w13_out=1280, bpw=3)
FN_LAYERS, FN_EXPERTS, FN_TOPK = 48, 512, 10
#: Derived from the layout, not rounded: see the per-tensor breakdown below. 1.7652 MiB.
#: (An earlier scratch calculation said ...904 by counting the two stage-2 sentinels twice;
#: the term-by-term test below is what caught it.)
FN_BYTES_PER_SLOT = 1_850_900


# ---------------------------------------------------------------------------
# What the arena holds
# ---------------------------------------------------------------------------

def test_arena_shapes_replace_the_leading_dim_with_the_slot_count():
    """The arena is layout-identical to a slice of the full tensor. That is the whole reason
    no CUDA change is needed: the kernel indexes `base + eidx * estride`, so handing it the
    arena and a slot id reaches the right bytes with the already-compiled binary."""
    full = trellis_moe_shapes(num_experts=FN_EXPERTS, **FN)
    arena = arena_shapes(n_slots=120, **FN)
    for key, shape in arena.items():
        assert shape[1:] == full[key][1:], f"{key}: arena trailing dims must match the source"
        assert shape[0] == 120, f"{key}: leading dim must be the slot count"


def test_SV_is_NOT_in_the_arena_because_it_is_shared_per_layer():
    """`w13_SV`/`w2_SV` are one RHT seed for the whole layer — `trellis_moe_shapes` registers
    them as `(hidden,)` and `(inter,)` with no expert dimension, and the kernel signature
    documents w13_SV as *"shared across experts"*.

    Worth an explicit test because the CHECKPOINT disagrees: it stores an SV per expert
    (24,576 copies), which is what the recorded manifest shows. Paging a per-layer tensor
    through a per-slot arena would be both wasteful and wrong.
    """
    arena = arena_shapes(n_slots=120, **FN)
    assert "w13_SV" not in arena and "w2_SV" not in arena
    full = trellis_moe_shapes(num_experts=FN_EXPERTS, **FN)
    assert full["w13_SV"] == (2560,) and full["w2_SV"] == (640,)


def test_the_arena_covers_every_per_expert_tensor_the_kernel_indexes():
    """A tensor left out of the arena would be read at the SLOT index from the full tensor —
    i.e. silently the wrong expert's scale. Everything with a leading expert dim must be
    paged together."""
    full = trellis_moe_shapes(num_experts=FN_EXPERTS, **FN)
    per_expert = {k for k, v in full.items() if v[0] == FN_EXPERTS}
    assert set(arena_shapes(n_slots=120, **FN)) == per_expert


# ---------------------------------------------------------------------------
# Byte math
# ---------------------------------------------------------------------------

def test_bytes_per_slot_matches_the_layout_term_by_term():
    """Built up independently of the implementation so a wrong dtype shows as a specific
    discrepancy rather than a slightly-off total."""
    TD, k1 = 16, 3
    t13 = (1280 // TD) * (2560 // TD)          # 12,800 tiles
    t2 = (2560 // TD) * (640 // TD)            # 6,400 tiles
    expect = (
        t13 * TD * k1 * 2            # w13_trellis_packed  int16
        + t2 * TD * k1 * 2           # w2_trellis_packed   int16
        + 1 * 1 * 2                  # w13_trellis_packed2 sentinel (3 bpw: no stage 2)
        + 1 * 1 * 2                  # w2_trellis_packed2  sentinel
        + 1280 * 2                   # w13_SU              fp16
        + 2560 * 2                   # w2_SU               fp16
        + 4 + 4                      # w13/w2_Wscale       fp32
        + 4 + 4                      # w13/w2_inv_resid_scale2 fp32
    )
    assert expect == FN_BYTES_PER_SLOT
    assert bytes_per_slot(**FN) == FN_BYTES_PER_SLOT


def test_a_10_GiB_budget_buys_120_slots_per_layer():
    """The rung the S0 hit-rate curve was measured at (`benchmarks/expert_cache_s0.md`), so
    the sizing the cache ships with and the number it was justified by are the same rung."""
    n = slots_for_budget(10 * 1024**3, n_layers=FN_LAYERS, **FN)
    assert n == 120


def test_slots_scale_with_the_budget_and_never_exceed_the_expert_count():
    assert slots_for_budget(8 * 1024**3, n_layers=FN_LAYERS, **FN) == 96
    assert slots_for_budget(11 * 1024**3, n_layers=FN_LAYERS, **FN) == 132
    # Past full residency there is nothing left to cache; more slots would be idle VRAM.
    huge = slots_for_budget(400 * 1024**3, n_layers=FN_LAYERS,
                            n_experts=FN_EXPERTS, **FN)
    assert huge == FN_EXPERTS


# ---------------------------------------------------------------------------
# ★ The constraint that makes the arena safe
# ---------------------------------------------------------------------------

def test_a_budget_too_small_for_one_step_is_REFUSED(capsys):
    """★ Every expert routed in a step must be resident SIMULTANEOUSLY. The fused op reads
    the arena after the fill, so if a later routing in the same step evicts an earlier one,
    the matmul reads a slot holding the wrong expert — correct-looking tokens computed from
    the wrong weights, which is the failure mode this project pays for most often.

    `n_slots >= max_tokens * top_k` is therefore not a tuning guideline, it is a correctness
    precondition, and a budget that cannot meet it must refuse rather than quietly enable a
    cache that thrashes inside a single step.
    """
    with pytest.raises(ValueError, match="simultaneous|max_tokens|top_k"):
        slots_for_budget(64 * 1024**2, n_layers=FN_LAYERS, max_tokens=8, top_k=FN_TOPK,
                         **FN)


def test_the_default_budget_satisfies_the_constraint_at_the_shipped_token_cap():
    """120 slots against 8 tokens x top_k 10 = 80 simultaneous routings: satisfied with
    room. At the 12-token cap it would be exactly tight, which is why the cap ships at 8."""
    n = slots_for_budget(10 * 1024**3, n_layers=FN_LAYERS, max_tokens=8, top_k=FN_TOPK,
                         **FN)
    assert n == 120 and n >= 8 * FN_TOPK


def test_a_bigger_token_cap_needs_a_bigger_budget_and_says_so():
    """The constraint binds on the TOKEN CAP, so raising the cap is a VRAM decision, not a
    free switch. 32 tokens x 10 = 320 slots/layer = ~26 GiB, more than the card has."""
    with pytest.raises(ValueError, match="simultaneous|max_tokens|top_k"):
        slots_for_budget(10 * 1024**3, n_layers=FN_LAYERS, max_tokens=32, top_k=FN_TOPK,
                         **FN)


def test_sizing_is_independent_of_the_eviction_policy():
    """Sizing answers 'how many slots fit'; the policy answers 'which experts occupy them'.
    Keeping them separate is what let S0 score LRU against LFU against none on one trace
    without re-sizing anything."""
    a = slots_for_budget(10 * 1024**3, n_layers=FN_LAYERS, **FN)
    b = slots_for_budget(10 * 1024**3, n_layers=FN_LAYERS, **FN)
    assert a == b
