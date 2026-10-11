"""Residency policy for the GLQ expert cache: admission, eviction, hit accounting.

This is the logic S0 scores a recorded routing trace with, and it is deliberately the SAME
code the live cache will use for admission — a simulator that reimplements the policy is a
simulator that will disagree with the cache it was used to size.

The policy is pure (no torch device, no GPU), so these run anywhere.

Why per-layer admission is the first thing tested: Strata measured a globally-shared slot
counter at **2.97%** hit rate (1781 of 60000) where the same routing scored **21.4%** at 8
slots/layer and **70.4%** at 64 — because a shared counter fills entirely inside the first
token position it sees (*"about 26 layers of POSITION 0"*) and never changes again.

**S0 has since settled this on the real trace, and the answer is not Strata's.** On
Flash-Next's own decode routing (`benchmarks/expert_cache_s0.md`) global and per-layer tie:
**0.5236 vs 0.5300** at 115 slots/layer. The pathology requires `slots < n_layers * top_k` —
Strata had 256 slots against 480 lookups per position; a 10 GiB arena has 5,565, i.e. 11.6
positions, so no single position can monopolise it. The *mechanism* tested below is real and
worth keeping (it is what a sub-1-GiB arena would hit), but it is not the lever at the sizes
GLQ would ship.

**Eviction turned out to be the lever instead — and in the opposite direction to this
fixture.** Real decode: LRU 0.5263 -> **0.8233** at the same slots, beating LFU everywhere
above 16 slots/layer. Here, at 2 slots/layer, LRU is **12x worse** than no eviction. Both are
true; they are different regimes. That is precisely why the policy was measured rather than
chosen, and why `eviction=None` remains the baseline a policy has to beat.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq_vllm.expert_cache import ResidencyPolicy, score_trace  # noqa: E402


# ---------------------------------------------------------------------------
# Per-layer slot ranges
# ---------------------------------------------------------------------------

def test_each_layer_owns_a_disjoint_slot_range():
    """Strata's `layer_slot_range`: q = slots // n_layers, layer l owns [l*q, (l+1)*q),
    and the LAST layer takes the remainder so no slot is unreachable."""
    p = ResidencyPolicy(n_layers=4, n_experts=10, slots=10, per_layer=True)
    ranges = [p.layer_slot_range(li) for li in range(4)]
    assert ranges[0] == (0, 2)
    assert ranges[1] == (2, 4)
    assert ranges[2] == (4, 6)
    assert ranges[3] == (6, 10), "the last layer must absorb the remainder"
    # disjoint and covering
    covered = [s for lo, hi in ranges for s in range(lo, hi)]
    assert sorted(covered) == list(range(10))


def test_a_layer_cannot_admit_into_another_layers_range():
    p = ResidencyPolicy(n_layers=2, n_experts=8, slots=4, per_layer=True)
    for e in range(2):
        assert p.admit(0, e) is not None
    # layer 0's quota is 2 slots; the third distinct expert must be refused
    assert p.admit(0, 7) is None
    # ...while layer 1's own range is untouched
    s = p.admit(1, 7)
    assert s is not None and s >= 2


# ---------------------------------------------------------------------------
# The pathology per-layer admission exists to prevent
# ---------------------------------------------------------------------------

def _layered_trace(n_layers, top_k, n_tokens, n_experts, seed=0):
    """A trace shaped like real MoE routing: every token visits EVERY layer, and each
    (layer, token) routes top_k experts drawn from a layer-specific skewed distribution.
    The skew matters — a uniform draw would make any cache look equally bad."""
    import random
    rng = random.Random(seed)
    # each layer has its own "hot" experts, which is what makes a cache possible at all
    hot = {li: [rng.randrange(n_experts) for _ in range(top_k * 2)] for li in range(n_layers)}
    recs = []
    for _ in range(n_tokens):
        for li in range(n_layers):
            ids = set()
            while len(ids) < top_k:
                ids.add(rng.choice(hot[li]) if rng.random() < 0.8 else rng.randrange(n_experts))
            recs.append((li, sorted(ids)))
    return recs


def test_a_global_counter_starves_later_layers():
    """THE structural finding, asserted on the MECHANISM rather than on hit rate.

    With one shared counter the earliest records consume every slot, so later layers can
    never admit and their lookups miss for the whole run. Needs
    `slots < n_layers * top_k` — the cache cannot hold even ONE token's lookups, which is
    Strata's situation (256 slots against 48 x 10 = 480 per token). With more slots than
    that, token 0 alone hands every layer a share and the bug hides; an earlier version of
    this test used slots=64 against 8 x 4 = 32 and saw nothing.

    Deliberately NOT asserting that per-layer wins on hit rate here — see
    `test_per_layer_only_wins_when_the_quota_can_hold_a_routed_set` for why a synthetic
    fixture cannot settle that.
    """
    trace = _layered_trace(n_layers=8, top_k=4, n_tokens=40, n_experts=32)
    assert 20 < 8 * 4, "slots must be below one token's lookups or the pathology hides"
    glob = score_trace(trace, n_layers=8, n_experts=32, slots=20, per_layer=False)
    per = score_trace(trace, n_layers=8, n_experts=32, slots=20, per_layer=True)

    assert min(glob.admitted_per_layer) == 0, (
        "the global counter should starve at least one layer entirely: "
        f"{glob.admitted_per_layer}")
    assert min(per.admitted_per_layer) > 0, (
        f"every layer must get a share under per-layer: {per.admitted_per_layer}")


def test_the_synthetic_fixture_CANNOT_settle_which_admission_policy_wins():
    """An honest limit on this fixture, asserted so nobody later reads a cherry-picked point
    as a law.

    Strata measured a **7x** gap (global 2.97% vs 21.4% at 8 slots/layer). On this synthetic
    trace the two policies land within a couple of points of each other and the winner is not
    even monotone in slots — per-layer at 8, global at 16-20, tie at 32, global at 48,
    per-layer at 96. The reason is that this fixture gives every layer an equally-sized,
    equally-hot expert set, while the benefit of partitioning comes from layers differing in
    how concentrated their routing is.

    So the mechanism test above (a global counter starves whole layers) is structural and
    worth keeping; the hit-rate comparison is **S0's job on the real trace**. A previous
    version of this file asserted a quota-vs-top_k crossover using two hand-picked slot
    counts, which passed for the wrong reason.
    """
    trace = _layered_trace(n_layers=8, top_k=4, n_tokens=40, n_experts=32)
    kw = dict(n_layers=8, n_experts=32)
    gaps, winners = [], set()
    for slots in (8, 16, 20, 32, 48, 96, 160):
        a = score_trace(trace, slots=slots, per_layer=True, **kw).hit_rate
        b = score_trace(trace, slots=slots, per_layer=False, **kw).hit_rate
        gaps.append(abs(a - b))
        winners.add("per" if a > b else ("glob" if b > a else "tie"))

    assert max(gaps) < 0.05, (
        f"this fixture now separates the policies by {max(gaps):.3f} — if it genuinely "
        "reproduces Strata's 7x gap, replace this test with the real comparison")
    assert len(winners) > 1, (
        "the winner is expected to be unstable on synthetic routing; a stable winner here "
        "would mean the fixture has acquired real structure and should be trusted more")


def test_hit_rate_rises_with_slots_per_layer():
    """Strata's curve (8/layer -> 21.4%, 64/layer -> 70.4%) is monotone; ours must be too,
    or the simulator is not measuring what it claims."""
    trace = _layered_trace(n_layers=6, top_k=4, n_tokens=50, n_experts=48)
    rates = [score_trace(trace, n_layers=6, n_experts=48, slots=6 * q,
                         per_layer=True).hit_rate
             for q in (1, 2, 4, 8, 16)]
    assert rates == sorted(rates), f"not monotone in slots/layer: {rates}"
    assert rates[-1] > rates[0]


# ---------------------------------------------------------------------------
# Hit accounting
# ---------------------------------------------------------------------------

def test_hit_rate_counts_every_expert_lookup():
    """Strata's denominator is expert LOOKUPS, not tokens: "1781 of 60000" for 125 tokens
    x 48 layers x 10 experts. Getting this wrong rescales every number we would compare."""
    trace = [(0, [1, 2]), (0, [1, 2])]
    r = score_trace(trace, n_layers=1, n_experts=4, slots=8, per_layer=True)
    assert r.lookups == 4
    # first record: both compulsory misses, then admitted. second: both hit.
    assert r.hits == 2
    assert r.misses == 2
    assert r.hit_rate == pytest.approx(0.5)


def test_an_unlimited_cache_hits_everything_after_the_compulsory_misses():
    trace = _layered_trace(n_layers=3, top_k=2, n_tokens=30, n_experts=8)
    r = score_trace(trace, n_layers=3, n_experts=8, slots=3 * 8, per_layer=True)
    # every (layer, expert) fits, so misses are exactly the distinct pairs touched
    distinct = {(li, e) for li, ids in trace for e in ids}
    assert r.misses == len(distinct)


def test_a_zero_slot_cache_misses_everything():
    """The default-off path must be representable, and must not divide by zero."""
    trace = _layered_trace(n_layers=2, top_k=2, n_tokens=5, n_experts=8)
    r = score_trace(trace, n_layers=2, n_experts=8, slots=0, per_layer=True)
    assert r.hits == 0
    assert r.hit_rate == 0.0


# ---------------------------------------------------------------------------
# Eviction — mandatory for GLQ, because bit-exactness forces fill-on-miss
# ---------------------------------------------------------------------------

def test_without_eviction_a_full_layer_misses_forever():
    """Strata's `admit` deliberately never evicts; it sends misses to a CPU tier. GLQ cannot
    (fp32-vs-fp16 and a different reduction order), so this behaviour is the baseline that
    eviction has to beat, not the shipping policy."""
    p = ResidencyPolicy(n_layers=1, n_experts=8, slots=2, per_layer=True, eviction=None)
    assert p.admit(0, 0) == 0
    assert p.admit(0, 1) == 1
    assert p.admit(0, 2) is None          # full, and stays full
    assert p.slot_of(0, 0) == 0           # the incumbents are untouched
    assert p.slot_of(0, 2) == -1


def test_lru_eviction_replaces_the_least_recently_used_within_the_layer():
    p = ResidencyPolicy(n_layers=1, n_experts=8, slots=2, per_layer=True, eviction="lru")
    p.admit(0, 0)
    p.admit(0, 1)
    p.touch(0, 0)                          # 0 is now more recent than 1
    s = p.admit(0, 2)
    assert s is not None
    assert p.slot_of(0, 1) == -1, "expert 1 was least recently used and should be gone"
    assert p.slot_of(0, 0) != -1, "expert 0 was touched and must survive"
    assert p.slot_of(0, 2) == s


def test_eviction_never_crosses_a_layer_boundary():
    """A layer evicting from another layer's range would corrupt that layer's residency."""
    p = ResidencyPolicy(n_layers=2, n_experts=8, slots=4, per_layer=True, eviction="lru")
    for e in range(2):
        p.admit(0, e)
        p.admit(1, e)
    before = [p.slot_of(1, e) for e in range(2)]
    p.admit(0, 5)                          # forces an eviction inside layer 0 only
    after = [p.slot_of(1, e) for e in range(2)]
    assert before == after, "layer 1's residency changed when layer 0 evicted"


def test_lru_can_LOSE_to_no_eviction_when_the_cache_is_far_too_small():
    """Measured here, and it is why Strata refused to pick an eviction policy by reasoning.

    With 4 slots/layer against 40 experts and top_k=3, every record touches most of the
    layer's slots, so LRU evicts an entry that is about to be needed: 429 evictions for 445
    misses. No-eviction accidentally does better because it keeps the FIRST experts it sees,
    and in a skewed routing distribution those are probably hot.

    The lesson for S2.5 is that the eviction policy must be scored on a real trace, not
    assumed — an LRU shipped on intuition here would have cost hit rate outright."""
    trace = _layered_trace(n_layers=4, top_k=3, n_tokens=60, n_experts=40)
    kw = dict(n_layers=4, n_experts=40, slots=4 * 4, per_layer=True)
    none = score_trace(trace, eviction=None, **kw)
    lru = score_trace(trace, eviction="lru", **kw)
    assert lru.evictions > lru.hits, "expected thrashing for this fixture to mean anything"
    assert lru.hit_rate < none.hit_rate


def test_lfu_beats_no_eviction_where_lru_does_not():
    """Frequency, not recency, is what matches a skewed routing distribution — so LFU is the
    policy that should actually earn its keep. Asserted against the same fixture LRU loses
    on, so the comparison is like-for-like."""
    trace = _layered_trace(n_layers=4, top_k=3, n_tokens=60, n_experts=40)
    kw = dict(n_layers=4, n_experts=40, slots=4 * 4, per_layer=True)
    none = score_trace(trace, eviction=None, **kw)
    lfu = score_trace(trace, eviction="lfu", **kw)
    assert lfu.hit_rate >= none.hit_rate, (
        f"LFU {lfu.hit_rate:.4f} < no-evict {none.hit_rate:.4f}")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_an_out_of_range_lookup_returns_not_resident_rather_than_reading_adjacent():
    """Strata bounds-checks `slot_of` for exactly this reason."""
    p = ResidencyPolicy(n_layers=2, n_experts=4, slots=4, per_layer=True)
    assert p.slot_of(-1, 0) == -1
    assert p.slot_of(99, 0) == -1
    assert p.slot_of(0, 99) == -1
    assert p.admit(99, 0) is None


def test_an_unknown_eviction_policy_is_refused_not_ignored():
    with pytest.raises(ValueError, match="eviction"):
        ResidencyPolicy(n_layers=1, n_experts=4, slots=4, eviction="magic")
