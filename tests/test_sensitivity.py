"""Tests for the GLQ mixed-precision allocator (glq.sensitivity)."""
from __future__ import annotations

import collections

import pytest

from glq.sensitivity import allocate_bpw


def _uniform_layers(n: int, weights_each: int = 1_000_000):
    """Return ``n`` layers of equal size with sensitivity = layer_idx + 1."""
    sens = {f"layer.{i}": float(i + 1) for i in range(n)}
    sizes = {f"layer.{i}": weights_each for i in range(n)}
    return sens, sizes


def test_allocate_bpw_respects_budget_and_bounds():
    sens, sizes = _uniform_layers(20)
    alloc = allocate_bpw(sens, sizes, target_avg_bpw=3.5, min_bpw=3, max_bpw=5)

    assert set(alloc.keys()) == set(sens.keys())
    assert all(3 <= v <= 5 for v in alloc.values())

    total_w = sum(sizes.values())
    avg = sum(alloc[n] * sizes[n] for n in alloc) / total_w
    assert avg <= 3.5 + 1e-9


def test_allocator_distributes_across_intermediate_bpws():
    """Mixed-precision targeting 3.5 with [3, 5] should produce a graduated
    3/4/5 mix — not a bimodal split where almost all layers are 3 or 5 with
    just one borderline at 4.

    The previous static-gain formula collapsed to a bimodal distribution
    (Gemma-4-31B: 205@3, 1@4, 204@5; Gemma-4-E2B: 88@3, 1@4, 186@5). Under the
    decreasing-returns formula each added bit is worth ~4× less proxy
    reduction than the previous bit, so the second-best layer wins before the
    same layer climbs twice.
    """
    sens, sizes = _uniform_layers(60)
    alloc = allocate_bpw(sens, sizes, target_avg_bpw=3.5, min_bpw=3, max_bpw=5)
    counts = collections.Counter(alloc.values())

    n_intermediate = counts.get(4, 0)
    n_layers = len(alloc)
    # At least 10% of layers should land at the intermediate bpw on a
    # uniform-size benchmark; a bimodal allocator gives < 2%.
    assert n_intermediate / n_layers >= 0.1, (
        f"intermediate-bpw fraction too low: {counts} "
        f"(expected graduated distribution, got bimodal)"
    )


def test_allocator_uniform_when_target_equals_min():
    sens, sizes = _uniform_layers(10)
    alloc = allocate_bpw(sens, sizes, target_avg_bpw=3.0, min_bpw=3, max_bpw=5)
    assert all(v == 3 for v in alloc.values())


def test_allocator_uniform_when_target_equals_max():
    sens, sizes = _uniform_layers(10)
    alloc = allocate_bpw(sens, sizes, target_avg_bpw=5.0, min_bpw=3, max_bpw=5)
    assert all(v == 5 for v in alloc.values())


def test_allocator_higher_sensitivity_gets_more_bits():
    """A layer with much higher sensitivity should reach max before others."""
    sens = {"hot": 100.0, "cold0": 1.0, "cold1": 1.0, "cold2": 1.0}
    sizes = {n: 1_000_000 for n in sens}
    alloc = allocate_bpw(sens, sizes, target_avg_bpw=3.5, min_bpw=3, max_bpw=5)
    assert alloc["hot"] >= max(alloc[n] for n in ("cold0", "cold1", "cold2"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
