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

def test_a_budget_too_small_for_the_token_cap_is_REFUSED_with_the_arithmetic():
    """★ Correctness, not tuning: all `max_tokens * top_k` routings of a step must be
    resident at once, or the fill evicts a routing the matmul has not read yet. The refusal
    must name the numbers so the fix is obvious."""
    d = arena_decision(budget="0.5", offloaded_bytes=40 * 1024**3, max_tokens=8, **LAYER)
    assert not d.enabled
    assert "simultaneous" in d.reason.lower()
    assert "80" in d.reason          # 8 tokens x top_k 10


def test_the_shipped_defaults_satisfy_the_precondition():
    d = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, max_tokens=8, **LAYER)
    assert d.enabled and d.slots >= 8 * 10


def test_raising_the_token_cap_past_the_budget_is_refused_rather_than_silently_capped():
    """Silently clamping the cap would be worse than refusing: the run would serve larger
    batches through the slow path while the log claimed a larger cap."""
    d = arena_decision(budget="10", offloaded_bytes=40 * 1024**3, max_tokens=32, **LAYER)
    assert not d.enabled and "simultaneous" in d.reason.lower()


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
