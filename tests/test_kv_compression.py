"""Offering GLQ's E8 KV-cache compression as a choice, not a research incantation.

The stack it turns on already exists — `glq_vllm/__init__.py` reads six environment
variables and monkey-patches vLLM's attention. What did not exist is any way for a user to
find it: it lived in a benchmark command line. This makes it a flag, with the trade-off
stated where the choice is made.

The trap this file guards is the one that makes the option worse than useless: an env var
that no longer engages. `glq_vllm` announces each stage it activates on stderr, so
engagement is checkable — and must be checked, because the last confirmed activation was on
vLLM 0.25.1 and users are on newer.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import kv_compression as kv  # noqa: E402


def test_the_offered_flags_are_vllms_own():
    """Verified against vLLM 0.27.1: `cache_dtype` is a Literal that includes 'fp8', and
    `kv_cache_dtype_skip_layers` is a list[str]. Being upstream flags is the point — they do
    not rot when vLLM moves, which is exactly how the E8 path broke."""
    assert kv.FP8_KV_ARGS == ("--kv-cache-dtype", "fp8",
                              "--kv-cache-dtype-skip-layers", "sliding_window")


def test_serve_args_are_empty_unless_chosen():
    assert kv.serve_args(False) == ()
    assert "fp8" in kv.serve_args(True)


def test_the_printed_suffix_can_be_appended_to_a_command():
    assert kv.shell_suffix(False) == ""
    suffix = kv.shell_suffix(True)
    assert suffix.startswith(" ") and "\n" not in suffix
    assert "--kv-cache-dtype fp8" in suffix


def test_the_tradeoff_states_the_gain_and_the_cost():
    text = kv.TRADEOFF.lower()
    assert "context" in text, "no statement of what it buys"
    assert "precision" in text, "no statement of what it costs"
    assert "not measured" in text or "has not measured" in text, (
        "GLQ has not measured fp8 KV on its own checkpoints; the text must not imply it has")


def test_the_e8_definition_survives_but_is_not_an_offer():
    """Keep the knowledge, do not wire it: it announces all six stages on vLLM 0.27.1 and
    then dies in EngineCore."""
    assert kv.E8_KV_ENV["GLQ_KV_QUANT"] == "e8_relaxed:2"
    assert not hasattr(kv, "KV_COMPRESSION_ENV"), (
        "the old name implied it was the offered path")
