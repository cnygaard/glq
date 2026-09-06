"""MRCR scoring and aggregation.

The metric is the dataset card's reference implementation, so these tests pin it
against that definition rather than against our own reading of it — including the
part that trips people up: a missing prefix scores zero however good the recall
was, because the prefix is what distinguishes following the instruction from
pattern-matching the most recent similar answer.

No vLLM, no dataset, no GPU: the graded quantities are pure functions, and they
are worth having fast because the surrounding run costs hours.
"""
from __future__ import annotations

import math

import pytest

from glq.bench.tasks.mrcr import auc, grade


PREFIX = "a1b2c3d4e5"


def test_exact_match_with_prefix_scores_one():
    answer = PREFIX + "The quick brown fox."
    assert grade(answer, answer, PREFIX) == 1.0


def test_missing_prefix_scores_zero_even_when_perfect():
    """The trap the prefix exists for: byte-perfect recall, no prefix, zero."""
    answer = PREFIX + "The quick brown fox."
    assert grade("The quick brown fox.", answer, PREFIX) == 0.0


def test_partial_recall_scores_between():
    answer = PREFIX + "The quick brown fox jumps over the lazy dog."
    got = grade(PREFIX + "The quick brown fox sat down.", answer, PREFIX)
    assert 0.0 < got < 1.0


def test_wrong_needle_scores_low_but_is_not_special_cased():
    """Reproducing a *different* assistant response is the characteristic failure;
    it still scores by similarity, which is why the noise floor is ~1/n_needles
    rather than 0."""
    answer = PREFIX + "A poem about the sea."
    got = grade(PREFIX + "A poem about the mountains.", answer, PREFIX)
    assert got < 0.9


def test_auc_of_a_flat_curve_is_that_value():
    scores = {(4096, 8192): 0.8, (8192, 16384): 0.8, (16384, 32768): 0.8}
    assert auc(scores) == pytest.approx(0.8)


def test_auc_weights_each_doubling_equally_not_each_bucket():
    """Two buckets one doubling apart, two buckets three doublings apart: the
    aggregate must not treat those spans as equal, or a method that dies at long
    context is flattered by however many short buckets happened to be run."""
    near = auc({(4096, 8192): 1.0, (8192, 16384): 0.0})
    far = auc({(4096, 8192): 1.0, (65536, 131072): 0.0})
    assert near == pytest.approx(0.5)
    assert far == pytest.approx(0.5), "trapezoid over log2 is symmetric in span"


def test_auc_collapse_at_long_context_is_visible():
    """The case the aggregate exists to expose: perfect until 32k, zero beyond."""
    holds = {(4096, 8192): 1.0, (8192, 16384): 1.0, (16384, 32768): 1.0}
    collapses = dict(holds)
    collapses[(32768, 65536)] = 0.0
    collapses[(65536, 131072)] = 0.0
    assert auc(collapses) < auc(holds)


def test_auc_of_one_bucket_is_its_score():
    """A partial run still reports something interpretable rather than 0 area."""
    assert auc({(4096, 8192): 0.73}) == pytest.approx(0.73)


def test_auc_is_empty_safe():
    assert auc({}) == 0.0


def test_auc_matches_a_hand_computed_trapezoid():
    scores = {(4096, 8192): 1.0, (8192, 16384): 0.5, (16384, 32768): 0.0}
    # log2 upper edges 13, 14, 15: trapezoids (1.0+0.5)/2 and (0.5+0.0)/2 over
    # unit spans, divided by the span of 2.
    expected = ((1.0 + 0.5) / 2 + (0.5 + 0.0) / 2) / 2
    assert auc(scores) == pytest.approx(expected)
    assert math.isclose(expected, 0.5)
