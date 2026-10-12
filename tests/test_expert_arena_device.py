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


# ---------------------------------------------------------------------------
# ★ CUDA-graph capturability
# ---------------------------------------------------------------------------

@cuda_only
def test_the_whole_arena_step_is_CUDA_GRAPH_CAPTURABLE():
    """★ The arena must not cost the CUDA graph, because the graph is worth far more.

    `torch.ops.glq.fused_moe_trellis_3inst` exists so the MoE decode step stays capturable
    under FULL cudagraph — its own comment says the fallback loop's `topk_ids.unique()` is
    what forces `--enforce-eager`, and trellis MoE under graph measured **91.6 vs 8.5 tok/s**.
    An arena that reintroduces a dynamic shape or a host sync would trade a 10x for a 1.6x.

    Measured on an L4: `torch.unique`, `bool(t.any())`, `int(t.sum())` and `nonzero` each
    invalidate capture; `argsort` and `index_copy_` are fine. So the plan has to be built from
    fixed-shape ops with every state update in place.
    """
    from glq_vllm.expert_arena import ArenaDecision, ExpertArena, plan_step

    src = {"w13_trellis_packed": torch.randint(-100, 100, (64, 16, 4), dtype=torch.int16,
                                               device="cuda")}
    arena = ExpertArena(ArenaDecision(True, slots=32, bytes_per_slot=1, n_layers=1,
                                      reason="t"), src, n_experts=64)
    arena.warmup()
    ids = torch.arange(10, dtype=torch.int32, device="cuda")
    plan_step(arena.state, ids)            # eager once, so any lazy init is done
    arena.fill(plan_step(arena.state, ids))

    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):              # raises if anything is dynamic or syncs
        arena.fill(plan_step(arena.state, ids))
    torch.cuda.synchronize()


@cuda_only
def test_a_captured_graph_REPLAYS_the_same_decisions_as_eager():
    """Capturing is not enough: the replay must still resolve residency correctly. A graph
    that records stale addresses would replay a fixed answer and look fast while serving the
    wrong slots."""
    from glq_vllm.expert_arena import ArenaDecision, ExpertArena, plan_step

    src = {"w13_trellis_packed": torch.arange(64 * 16 * 4, dtype=torch.int16,
                                              device="cuda").reshape(64, 16, 4)}

    def _fresh():
        a = ExpertArena(ArenaDecision(True, slots=16, bytes_per_slot=1, n_layers=1,
                                      reason="t"), src, n_experts=64)
        a.warmup()
        return a

    steps = [torch.arange(s, s + 4, dtype=torch.int32, device="cuda")
             for s in (0, 4, 0, 8, 4)]

    eager = _fresh()
    want = []
    for ids in steps:
        p = plan_step(eager.state, ids)
        eager.fill(p)
        want.append(p.slot_of.tolist())

    graphed = _fresh()
    ids_buf = torch.zeros(4, dtype=torch.int32, device="cuda")
    p0 = plan_step(graphed.state, ids_buf)      # warm the shapes
    graphed.fill(p0)
    graphed.state.residency.fill_(-1)
    graphed.state.slot_owner.fill_(-1)
    graphed.state.slot_last_use.zero_()
    graphed.state.clock.zero_()
    graphed.state.counters.zero_()

    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        plan = plan_step(graphed.state, ids_buf)
        graphed.fill(plan)
    torch.cuda.synchronize()

    got = []
    for ids in steps:
        ids_buf.copy_(ids)
        g.replay()
        torch.cuda.synchronize()
        got.append(plan.slot_of.tolist())
    assert got == want, f"captured replay diverged from eager: {got} != {want}"


# ---------------------------------------------------------------------------
# ★ Allocation and source binding have opposite timing constraints
# ---------------------------------------------------------------------------

def test_sources_can_be_REBOUND_without_reallocating():
    """★ The buffers must exist before vLLM profiles memory and captures graphs; the sources
    can only be read after vLLM's UVA offload, which happens later. So they are separate
    steps, and rebinding must not touch the allocation.

    Measured consequence of getting this wrong: allocating at first apply put a 9.97 GiB
    arena inside cudaStreamCapture, vLLM reported "Graph capturing finished in 3 secs, took
    10.06 GiB" against the usual 0.3, charged it against the KV pool, and refused to start
    with "Available KV cache memory: -0.82 GiB".
    """
    early = _sources()
    a = ExpertArena(_decision(8), early, n_experts=16)
    ptrs = {k: v.data_ptr() for k, v in a.buffers.items()}

    late = _sources()                      # same shapes/dtypes, different storage
    a.rebind_sources(late)
    assert {k: v.data_ptr() for k, v in a.buffers.items()} == ptrs, "buffers were reallocated"
    for k in late:
        assert a.sources[k] is late[k], f"{k} still points at the pre-offload tensor"


def test_rebinding_a_CHANGED_shape_is_refused():
    """Layout identity is what makes the arena readable by the unmodified kernel. If a source
    came back a different shape, filling it into the arena would produce plausible bytes at
    the wrong stride."""
    a = ExpertArena(_decision(8), _sources(), n_experts=16)
    bad = _sources()
    bad["w13_SU"] = torch.ones(16, 7, dtype=torch.float16)      # was (16, 6)
    with pytest.raises(ValueError, match="shape/dtype"):
        a.rebind_sources(bad)


def test_rebinding_an_UNKNOWN_tensor_is_refused():
    a = ExpertArena(_decision(8), _sources(), n_experts=16)
    extra = _sources()
    extra["w2_trellis_packed"] = torch.zeros(16, 8, 4, dtype=torch.int16)
    with pytest.raises(KeyError, match="no arena buffer"):
        a.rebind_sources(extra)
