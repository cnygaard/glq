"""The arena's device buffers: what gets allocated, what is refused, and that a hit moves
no bytes.

Construction runs on CPU (`torch.empty` follows the source's device), so the allocation
contract is checked without a GPU. `fill` needs Triton and is skipped without CUDA — the
bit-exactness gate against the real fused op lives with the GPU suites on the box.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from glq_vllm.expert_arena import (ArenaDecision, ExpertArena,  # noqa: E402
                                   arena_decision, plan_step)

FN = dict(hidden_size=2560, inter=640, w13_out=1280, bpw=3)
LAYER = dict(n_layers=48, n_experts=512, top_k=10, **FN)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for Triton")


def _sources(n_experts=16, device="cpu"):
    """A miniature stand-in for one layer's per-expert buffers, with the real dtypes."""
    return {
        "w13_trellis_packed": torch.arange(n_experts * 8 * 4, dtype=torch.int16,
                                           device=device).reshape(n_experts, 8, 4),
        "w13_SU": torch.arange(n_experts * 6, dtype=torch.float16,
                               device=device).reshape(n_experts, 6),
        "w13_Wscale": torch.arange(n_experts, dtype=torch.float32, device=device),
    }


def _decision(slots=8):
    return ArenaDecision(True, slots=slots, bytes_per_slot=1_850_900, n_layers=1,
                         reason="test")


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def test_each_buffer_mirrors_its_source_with_the_slot_count_as_the_leading_dim():
    """Layout identity is the reason no CUDA change is needed: the kernel indexes
    `base + eidx * estride`, so an arena shaped like a slice of the source is readable with
    a slot id in place of the expert id."""
    src = _sources()
    a = ExpertArena(_decision(8), src, n_experts=16)
    for name, s in src.items():
        buf = a.buffers[name]
        assert buf.shape == (8,) + tuple(s.shape[1:])
        assert buf.dtype == s.dtype, f"{name}: dtype must match or the bytes are not verbatim"


def test_row_elems_matches_the_per_expert_element_count():
    """The kernel walks `eidx * row_elems`; a wrong stride would read a neighbouring
    expert's bytes, which decodes to plausible noise rather than failing."""
    a = ExpertArena(_decision(8), _sources(), n_experts=16)
    assert a.row_elems["w13_trellis_packed"] == 8 * 4
    assert a.row_elems["w13_SU"] == 6
    assert a.row_elems["w13_Wscale"] == 1


def test_a_PER_LAYER_tensor_is_refused():
    """★ `w13_SV` is one RHT seed for the whole layer, shape `(hidden,)`, with no expert
    dimension. Paging it would read row `slot` of a tensor whose rows are not experts —
    silently the wrong data, and no shape error to catch it. (The checkpoint's per-expert SV
    copies are a separate thing; the serving layout shares one.)"""
    src = _sources()
    src["w13_SV"] = torch.ones(2560, dtype=torch.float16)
    with pytest.raises(ValueError, match="per-expert|leading dim"):
        ExpertArena(_decision(8), src, n_experts=16)


def test_building_a_DISABLED_arena_is_refused():
    """Allocating against a refusal would hold VRAM for a cache that is off."""
    d = arena_decision(budget="", offloaded_bytes=40 * 1024**3, **LAYER)
    with pytest.raises(ValueError, match="disabled"):
        ExpertArena(d, _sources(), n_experts=512)


def test_the_arena_reports_its_own_hit_accounting():
    """A cache at a 0% hit rate still emits correct tokens, so the counters are the only
    evidence it engaged — a benchmark asserts on these, not on the output."""
    a = ExpertArena(_decision(8), _sources(), n_experts=16)
    assert "hit_rate=" in a.report() and "fills=" in a.report()


# ---------------------------------------------------------------------------
# The fill — needs Triton
# ---------------------------------------------------------------------------

@cuda_only
def test_a_fill_copies_the_missing_experts_byte_for_byte():
    """Verbatim bytes are what makes the arena path bit-exact: the fused op then runs the
    same kernel over the same values and only the address differs."""
    src = _sources(device="cuda")
    a = ExpertArena(_decision(8), src, n_experts=16)
    plan = plan_step(a.state, torch.tensor([3, 7, 1], dtype=torch.int32, device="cuda"))
    a.fill(plan)
    for name, s in src.items():
        for r, e in enumerate([3, 7, 1]):
            slot = int(plan.slot_of[r])
            assert torch.equal(a.buffers[name][slot], s[e]), f"{name} expert {e}"


@cuda_only
def test_a_HIT_leaves_the_slot_untouched_and_moves_no_bytes():
    """★ The saving. A hit must issue no load from the source — that is why the kernel early
    returns rather than copying unconditionally. Shown by overwriting the source with a
    sentinel after the first fill: if a hit re-copied, the sentinel would appear in the
    arena."""
    src = _sources(device="cuda")
    a = ExpertArena(_decision(8), src, n_experts=16)
    a.fill(plan_step(a.state, torch.tensor([3], dtype=torch.int32, device="cuda")))
    before = a.buffers["w13_trellis_packed"].clone()

    src["w13_trellis_packed"].fill_(-123)          # overwrite the SOURCE only
    plan = plan_step(a.state, torch.tensor([3], dtype=torch.int32, device="cuda"))
    assert int(plan.fill_src[0]) < 0, "expert 3 should be resident now"
    a.fill(plan)
    assert torch.equal(a.buffers["w13_trellis_packed"], before), (
        "a hit re-read the source; the arena would move the same bytes as today and win "
        "nothing")


@cuda_only
def test_warmup_compiles_the_kernel_without_touching_residency():
    """Triton JITs on first launch, and that must not happen inside `cudaStreamCapture`.
    Warmup pays it at load time and must not disturb the cache state it will later serve."""
    a = ExpertArena(_decision(8), _sources(device="cuda"), n_experts=16)
    a.warmup()
    assert a.state.hits == 0 and a.state.misses == 0 and a.state.fills == 0


# ---------------------------------------------------------------------------
# The weight selector used at the op call site
# ---------------------------------------------------------------------------

def test_arena_weights_prefers_the_arena_and_falls_back_to_the_layer():
    """★ One call site, two weight sources. `w13_SV` has no arena buffer (it is per-layer),
    so it must come from the layer — reading a per-layer tensor at a slot index would be
    silently the wrong data, with no shape error to catch it."""
    from glq_vllm.expert_arena import ArenaWeights

    class _Layer:
        w13_trellis_packed = "layer-packed"
        w13_SV = "layer-sv"
        w13_Wscale = "layer-wscale"

    buffers = {"w13_trellis_packed": "arena-packed", "w13_Wscale": "arena-wscale"}
    W = ArenaWeights(buffers, _Layer())
    assert W.w13_trellis_packed == "arena-packed"
    assert W.w13_Wscale == "arena-wscale"
    assert W.w13_SV == "layer-sv", "a per-layer tensor must not be taken from the arena"


def test_arena_weights_raises_for_a_name_neither_side_has():
    """A typo'd buffer name must fail loudly rather than resolve to None and be passed into
    the fused op as a null pointer."""
    from glq_vllm.expert_arena import ArenaWeights

    class _Layer:
        pass

    with pytest.raises(AttributeError):
        _ = ArenaWeights({}, _Layer()).w13_trellis_packed
