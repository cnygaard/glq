"""The arena's per-step plan: which slot each routing reads, and which slots need filling.

★ **This is the mechanism test for S1.** The device plan must agree *exactly* with
`ResidencyPolicy` — the same object S0 used to size the arena and to choose LRU over LFU
(`benchmarks/expert_cache_s0.md`). If the two drift, the hit rate the arena was justified by
describes a policy that is not running.

The device plan is **vectorized** where `ResidencyPolicy` is serial, so the agreement is a
real claim and not a tautology. It holds because after a serial admit evicts the
least-recently-used slot it *touches* it, so that slot is no longer the least recent and the
next admit takes the second-least — i.e. serial admission of M misses selects exactly the M
least-recently-used slots, which is what the vectorized form takes in one shot.

Runs on CPU: `plan_step` is device-agnostic torch, so the algorithm is checked without a GPU
and the Triton kernel is later checked against it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from glq_vllm.expert_arena import ArenaState, plan_step          # noqa: E402
from glq_vllm.expert_cache import NOT_RESIDENT, ResidencyPolicy  # noqa: E402


def _policy_slots(steps, n_experts, slots):
    """What the reference assigns, STEP by step.

    `plan_step`, not `lookup_or_admit`: the per-routing form can evict a slot that a later
    routing of the same step reads, which offline costs a hit but live would make the matmul
    read the wrong expert. See `ResidencyPolicy.plan_step`.
    """
    pol = ResidencyPolicy(n_layers=1, n_experts=n_experts, slots=slots,
                          per_layer=True, eviction="lru")
    out = []
    for step in steps:
        got, _fills = pol.plan_step(0, [int(e) for e in step])
        out.extend(got)
    return out, pol


def _device_slots(steps, n_experts, slots):
    """What the vectorized plan assigns, step by step."""
    st = ArenaState(n_experts=n_experts, n_slots=slots, device="cpu")
    out = []
    for step in steps:
        ids = torch.tensor(step, dtype=torch.int32)
        p = plan_step(st, ids)
        out.extend(int(s) for s in p.slot_of)
    return out, st


# ---------------------------------------------------------------------------
# ★ Agreement with the policy S0 sized the arena with
# ---------------------------------------------------------------------------

def test_one_routing_per_step_matches_the_serial_policy_exactly():
    """With one routing per step the vectorized form degenerates to the serial one, so any
    disagreement here is a plain bug rather than a batching subtlety."""
    seq = [[e] for e in [3, 7, 3, 11, 7, 2, 9, 3, 11, 40, 41, 42, 3]]
    want, _ = _policy_slots(seq, n_experts=64, slots=4)
    got, _ = _device_slots(seq, n_experts=64, slots=4)
    assert got == want


def test_a_multi_routing_step_matches_the_serial_policy():
    """★ The real claim: M misses in one step take the same M slots the serial policy would
    hand out one at a time."""
    steps = [[0, 1, 2, 3], [4, 5, 0, 1], [6, 7, 8, 9], [0, 4, 6, 2], [10, 11, 12, 13]]
    want, _ = _policy_slots(steps, n_experts=64, slots=8)
    got, _ = _device_slots(steps, n_experts=64, slots=8)
    assert got == want


def test_agreement_holds_over_a_long_pseudo_random_sequence():
    """Thrashing is where a candidate-selection difference would show, so drive the cache well
    past its capacity."""
    g = torch.Generator().manual_seed(1234)
    steps = [[int(v) for v in torch.randint(0, 48, (5,), generator=g)] for _ in range(120)]
    want, _ = _policy_slots(steps, n_experts=48, slots=10)
    got, _ = _device_slots(steps, n_experts=48, slots=10)
    assert got == want


def test_hit_and_miss_accounting_matches_too():
    """The counters are what a run reports to prove the cache engaged; if they disagree with
    the policy, the reported hit rate is not the policy's hit rate."""
    steps = [[0, 1, 2], [0, 1, 3], [4, 5, 6], [0, 4, 2]]
    _, pol = _policy_slots(steps, n_experts=32, slots=6)
    _, st = _device_slots(steps, n_experts=32, slots=6)
    assert (int(st.hits), int(st.misses)) == (pol.hits, pol.misses)


# ---------------------------------------------------------------------------
# What the plan has to guarantee for the fused op to be correct
# ---------------------------------------------------------------------------

def test_every_routing_in_a_step_gets_a_DISTINCT_slot_when_experts_are_distinct():
    """★ The correctness precondition. The fused op reads the arena *after* the fill, so if
    two routings of one step shared a slot, or one evicted another, the matmul would read a
    slot holding a different expert — plausible output from the wrong weights."""
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    ids = torch.arange(10, dtype=torch.int32)
    p = plan_step(st, ids)
    assert len(set(int(s) for s in p.slot_of)) == 10
    assert all(int(s) >= 0 for s in p.slot_of)


def test_a_slot_used_earlier_in_the_SAME_step_is_never_evicted_later_in_it():
    """Drive a step that needs every slot, then another that needs them all again. Nothing in
    a step may evict a routing from that same step."""
    st = ArenaState(n_experts=64, n_slots=10, device="cpu")
    plan_step(st, torch.arange(10, dtype=torch.int32))
    p = plan_step(st, torch.arange(10, 20, dtype=torch.int32))
    assert len(set(int(s) for s in p.slot_of)) == 10


def test_a_step_larger_than_the_arena_is_REFUSED_not_served_wrongly():
    """Serving it would silently evict a routing before the matmul read it. The precondition
    is asserted in sizing, and enforced again here where the actual count is known."""
    st = ArenaState(n_experts=64, n_slots=8, device="cpu")
    with pytest.raises(ValueError, match="simultaneous|slots"):
        plan_step(st, torch.arange(12, dtype=torch.int32))


def test_repeated_experts_within_a_step_are_served_without_extra_fills():
    """Two tokens routing to the same expert must share one slot: distinct slots would be two
    verbatim copies of identical bytes (still bit-exact, but wasted PCIe and wasted slots)."""
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    p = plan_step(st, torch.tensor([5, 5, 6, 5, 6], dtype=torch.int32))
    assert int(p.slot_of[0]) == int(p.slot_of[1]) == int(p.slot_of[3])
    assert int(p.slot_of[2]) == int(p.slot_of[4])
    assert int((p.fill_src >= 0).sum()) == 2, "one fill per DISTINCT missing expert"


def test_fill_src_marks_exactly_the_misses():
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    p1 = plan_step(st, torch.tensor([1, 2, 3], dtype=torch.int32))
    assert [int(v) for v in p1.fill_src] == [1, 2, 3]          # all cold
    p2 = plan_step(st, torch.tensor([1, 2, 9], dtype=torch.int32))
    assert [int(v) for v in p2.fill_src] == [NOT_RESIDENT, NOT_RESIDENT, 9]


def test_a_hit_issues_no_fill_which_is_where_the_saving_comes_from():
    """The whole gain is that a hit reads device memory instead of crossing PCIe. If a hit
    ever produced a fill, the arena would move the same bytes as today and win nothing."""
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    plan_step(st, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    p = plan_step(st, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert int((p.fill_src >= 0).sum()) == 0
    assert int(st.hits) == 4 and int(st.fills) == 4           # 4 fills from the cold step


# ---------------------------------------------------------------------------
# ★ Out-of-range expert ids (vLLM captures graphs with dummy inputs)
# ---------------------------------------------------------------------------

def test_an_expert_id_past_the_end_does_not_index_out_of_bounds():
    """★ vLLM captures CUDA graphs with DUMMY inputs, so the router can emit ids outside
    [0, n_experts). Measured: this took down the graphed arena arm with
    `RuntimeError: Triton Error [CUDA]: device-side assert triggered`, after the memory
    accounting was already healthy (capture 0.15 GiB, KV 1.16 GiB).

    `residency[ids]` is a gather, so an id past the end is an out-of-bounds read — an
    IndexError on CPU and a device-side assert on CUDA.
    """
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    p = plan_step(st, torch.tensor([70, 1, 2], dtype=torch.int32))
    assert all(0 <= int(s) < 16 for s in p.slot_of)


def test_a_NEGATIVE_expert_id_is_not_served_from_an_unfilled_slot():
    """★ The dangerous direction, because it does not raise.

    A negative id reaches `residency[-1]` through Python's negative indexing — the scratch
    element — reads -1, is treated as a miss, is "admitted" as expert -1, and emits
    `fill_src = -1`, which the kernel reads as a hit and skips. The op would then read a slot
    nothing ever filled: stale weights, no error, plausible output.

    So every routing that is handed a slot must also be either resident or filled.
    """
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    p = plan_step(st, torch.tensor([-1, 5, 6], dtype=torch.int32))
    for r in range(3):
        slot, src = int(p.slot_of[r]), int(p.fill_src[r])
        assert 0 <= slot < 16, f"routing {r} got slot {slot}"
        # cold arena: every routing is a first touch, so each must carry a real fill
        assert src >= 0, (f"routing {r} was given slot {slot} with no fill, so the op would "
                          f"read a slot that was never written")


def test_clamping_does_not_disturb_valid_ids():
    """The guard must be a no-op on real routing, or it would change the policy the hit rate
    was measured with."""
    st_a = ArenaState(n_experts=64, n_slots=16, device="cpu")
    st_b = ArenaState(n_experts=64, n_slots=16, device="cpu")
    ids = torch.tensor([3, 17, 3, 40], dtype=torch.int32)
    a = plan_step(st_a, ids)
    b = plan_step(st_b, ids.to(torch.int64))
    assert [int(v) for v in a.slot_of] == [int(v) for v in b.slot_of]


def test_the_callers_topk_ids_tensor_is_not_mutated():
    """`plan_step` receives vLLM's own topk_ids. Clamping it in place would corrupt the
    tensor the fused op is about to read for its own routing."""
    st = ArenaState(n_experts=64, n_slots=16, device="cpu")
    ids = torch.tensor([70, -1, 5], dtype=torch.int64)
    before = ids.clone()
    plan_step(st, ids)
    assert torch.equal(ids, before), "plan_step modified the caller's topk_ids"
