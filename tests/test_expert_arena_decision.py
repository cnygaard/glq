"""Whether to build an expert arena for a given layer, and why not when not.

Mirrors `glq/quantize_model.py:ple_reuse_decision`: a decision function that returns its
reason, so a refusal is visible in a log instead of showing up as a feature that quietly did
nothing. That matters more than usual here — a cache that is silently off produces completely
correct output, just slowly, which is exactly how Strata's global-counter bug survived
(`2.97%` hit rate, correct tokens throughout).

Pure logic: no torch allocation, no GPU.
"""
from __future__ import annotations

import pytest

from glq_vllm.expert_arena import ArenaDecision, arena_decision

FN = dict(hidden_size=2560, inter=640, w13_out=1280, bpw=3)
LAYER = dict(n_layers=48, n_experts=512, top_k=10, **FN)


# ---------------------------------------------------------------------------
# Default off
# ---------------------------------------------------------------------------

def test_the_arena_is_OFF_unless_asked_for():
    """The standing rule for a new feature here: the existing path stays byte-identical
    until someone opts in."""
    d = arena_decision(budget="", offloaded_bytes=40 * 1024**3, **LAYER)
    assert not d.enabled
    assert "GLQ_MOE_EXPERT_CACHE" in d.reason


def test_an_explicit_budget_enables_it_with_the_sized_slot_count():
    d = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, **LAYER)
    assert d.enabled and d.slots == 120
    assert d.bytes_per_slot == 1_850_900


@pytest.mark.parametrize("budget", ["10", "10GiB", "10 GiB", "10.0"])
def test_the_budget_accepts_the_forms_a_person_would_type(budget):
    d = arena_decision(budget=budget, offloaded_bytes=40 * 1024**3, **LAYER)
    assert d.enabled and d.slots == 120


def test_a_budget_that_is_not_a_number_is_refused_not_ignored():
    """Silently treating a typo as 'off' would make a measurement arm report the baseline
    while claiming to test the cache."""
    with pytest.raises(ValueError, match="GLQ_MOE_EXPERT_CACHE"):
        arena_decision(budget="yes", offloaded_bytes=40 * 1024**3, **LAYER)


# ---------------------------------------------------------------------------
# ★ No offload means no reason to exist
# ---------------------------------------------------------------------------

def test_with_nothing_offloaded_the_arena_is_REFUSED():
    """★ The arena's entire job is to stop re-reading experts over PCIe. If the weights are
    already resident there is no PCIe traffic to remove, and the arena would be a second
    copy of weights the card already holds — pure waste, and it would push the model back
    into offload to pay for itself.

    Measured context: the L4 gain comes from offload being 45.1% of the step. On a 96 GiB
    card running this model at zero offload that term is 0, which is why the same feature
    must read ~1.00x there and should not allocate at all.
    """
    d = arena_decision(budget="10", offloaded_bytes=0, **LAYER)
    assert not d.enabled
    assert "offload" in d.reason.lower()


def test_a_trivial_amount_of_offload_is_also_refused():
    """A few MiB offloaded is rounding, not a workload; allocating 10 GiB to chase it would
    itself force more offload."""
    d = arena_decision(budget="10", offloaded_bytes=64 * 1024**2, **LAYER)
    assert not d.enabled and "offload" in d.reason.lower()


# ---------------------------------------------------------------------------
# ★ The simultaneity precondition
# ---------------------------------------------------------------------------

def test_a_budget_too_small_for_even_ONE_token_is_REFUSED_with_the_arithmetic():
    """★ Correctness, not tuning: every routing of a step must be resident at once, or the
    fill evicts a routing the matmul has not read yet.

    The *floor* is one token's `top_k` routings. Above that the cap is lowered rather than
    refused (see the derivation tests below); below it there is no narrower arena to fall back
    to. Either way the reason must name the numbers so the fix is obvious."""
    d = arena_decision(budget="0.5", offloaded_bytes=40 * 1024**3, max_tokens=8, **LAYER)
    assert not d.enabled
    assert "top_k 10" in d.reason and "resident at once" in d.reason
    assert "6 slots" in d.reason      # what 0.5 GiB actually bought


def test_the_shipped_defaults_satisfy_the_precondition():
    d = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, max_tokens=8, **LAYER)
    assert d.enabled and d.slots >= 8 * 10


def test_raising_the_token_cap_past_the_budget_LOWERS_it_and_says_so():
    """SUPERSEDED BEHAVIOUR, deliberately. This test previously asserted that a cap the budget
    cannot meet is *refused*, on the reasoning that silently clamping it would serve larger
    batches through the slow path while the log claimed a larger cap.

    The refusal turned out to be the worse failure: it cost three measured arms, where a 6 GiB
    budget declined entirely and the runs silently measured no arena at all — which is far
    harder to notice than a narrower one. Lowering the cap is safe (steps above it take
    today's exact path), so it is lowered.

    The original concern is answered by reporting it rather than by refusing: the chosen cap is
    in `reason` and in `max_tokens`, so nothing claims a cap it does not serve."""
    d = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, max_tokens=32, **LAYER)
    assert d.enabled
    assert d.max_tokens == d.slots // 10 < 32
    assert f"{d.max_tokens} token" in d.reason and "cap lowered from 32" in d.reason


# ---------------------------------------------------------------------------
# The decision is reportable
# ---------------------------------------------------------------------------

def test_the_decision_reports_the_footprint_it_will_take():
    """So a run can state what the arena cost before it allocates, and a benchmark can
    assert the footprint matches the bpw budget (CLAUDE.md requires the memory figure beside
    any perf claim)."""
    d = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, **LAYER)
    assert d.total_bytes == d.slots * d.bytes_per_slot * 48
    assert 9.0 < d.total_bytes / 1024**3 < 10.0


def test_a_refusal_never_reports_a_slot_count_that_would_be_used():
    d = arena_decision(budget="", offloaded_bytes=40 * 1024**3, **LAYER)
    assert isinstance(d, ArenaDecision) and d.slots == 0 and d.total_bytes == 0


# ---------------------------------------------------------------------------
# ★ The offload facts must be read where they are readable
# ---------------------------------------------------------------------------

def test_a_refusal_reason_names_the_offload_figure_it_saw():
    """★ Both arena bugs on the L4 were SILENT declines that looked like healthy runs: the
    two A/B arms landed within 0.4% of each other and only the per-layer decision line showed
    why ("only 0.00 GiB of experts are offloaded" against vLLM's own "Total CPU offloaded
    parameters: 41.48").

    So the reason has to carry the number the decision was made on. A refusal that says only
    "disabled" is indistinguishable from a feature that is working.
    """
    d = arena_decision(budget="10", offloaded_bytes=0, **LAYER)
    assert not d.enabled
    assert "0.00 GiB" in d.reason, f"reason does not name what it measured: {d.reason!r}"


def test_the_decision_is_driven_by_the_offload_SIZE_not_by_tensor_state():
    """A UVA view reports `is_cuda=True` with host storage behind it, so is_cuda/is_pinned
    cannot tell an offloaded expert tensor from a resident one — which is why the decision
    takes a byte count from vLLM's config rather than inspecting tensors. Pinning that here
    keeps a future refactor from reaching for introspection again.
    """
    on = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, **LAYER)
    off = arena_decision(budget="10", offloaded_bytes=0, **LAYER)
    assert on.enabled and not off.enabled
    # Nothing about the decision depends on a tensor; it is pure arithmetic over the config.
    assert on.slots == 120 and off.slots == 0


# ---------------------------------------------------------------------------
# ★ The token cap is DERIVED, not demanded
# ---------------------------------------------------------------------------

def test_a_budget_short_of_the_cap_DOWNGRADES_rather_than_refusing():
    """★ Lowering the cap is always safe: fewer token counts take the arena path and the rest
    take today's exact path. Refusing instead is what cost three measured arms — 6 GiB over 49
    containers is 71 slots/layer, a cap of 8 demands 8 x top_k 10 = 80, so the arena declined
    and three runs silently measured no arena at all while looking healthy.

    So the cap is derived from what the budget affords, `min(requested, n_slots // top_k)`,
    and only a budget that cannot serve a SINGLE token's routings is refused.
    """
    d = arena_decision(budget="6", offloaded_bytes=40 * 1024**3, max_tokens=8,
                       n_layers=49, n_experts=512, top_k=10, **FN)
    assert d.enabled, f"should downgrade the cap, not refuse: {d.reason}"
    assert d.slots == 71
    assert d.max_tokens == 7, f"71 // 10 = 7, got {d.max_tokens}"


def test_the_derived_cap_never_exceeds_what_was_requested():
    """A generous budget must not widen the arena path past the caller's ceiling: the caller
    knows the real step width (max_num_seqs), and serving wider would be guessing."""
    d = arena_decision(budget="16", offloaded_bytes=40 * 1024**3, max_tokens=4,
                       n_layers=49, n_experts=512, top_k=10, **FN)
    assert d.enabled and d.max_tokens == 4


def test_the_chosen_cap_is_reported():
    """The reason line is what makes a downgrade legible. Three arms were only recognisable as
    no-arena runs because the refusal named its arithmetic."""
    d = arena_decision(budget="6", offloaded_bytes=40 * 1024**3, max_tokens=8,
                       n_layers=49, n_experts=512, top_k=10, **FN)
    assert "7" in d.reason and "token" in d.reason.lower()


def test_a_budget_too_small_for_ONE_token_is_still_refused():
    """The floor stays a refusal: below top_k slots the arena cannot serve even a single
    token's routings, and serving anyway would evict a routing before the matmul read it."""
    d = arena_decision(budget="0.05", offloaded_bytes=40 * 1024**3, max_tokens=8,
                       n_layers=49, n_experts=512, top_k=10, **FN)
    assert not d.enabled
    assert "top_k" in d.reason or "simultaneous" in d.reason.lower()
