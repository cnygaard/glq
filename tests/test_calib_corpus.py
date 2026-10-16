"""Calibration corpus selection, and the per-expert routing report.

Both exist because of one measured property of fine-grained MoE routing: with top-6 of 384
experts, each expert sees ~1.6% of the calibration tokens, so the corpus does not merely
shift the Hessians — it decides which of the 19,200 routed-expert matrices get a real
Hessian and which fall back to `torch.eye` (plain nearest-neighbour). On a bilingual model
an English-only corpus can starve a whole family of experts, and before these changes that
outcome was indistinguishable from a healthy run in the logs.

The default corpus must not move: every published checkpoint was calibrated on English
WikiText-2, and the `benchmarks/_ckpt_bytecmp.py` exactness gate only means something if
the default path is byte-identical. That is asserted here rather than assumed.
"""
from __future__ import annotations

import pytest
import torch

from glq import quantize_model as qm


class _Tokenizer:
    """One token per character — enough to make chunk arithmetic exact and legible."""

    def __call__(self, text, return_tensors=None):
        ids = torch.arange(len(text), dtype=torch.long).unsqueeze(0)
        return type("Enc", (), {"input_ids": ids})()


# ------------------------------------------------------------------ the default

def test_default_corpus_is_still_english_wikitext():
    """Changing this silently re-calibrates every future checkpoint."""
    assert qm.DEFAULT_CALIB_CORPUS == "wikitext"
    assert qm.build_parser().parse_args(
        ["--model", "m", "--output", "o"]).calib_corpus == "wikitext"


def test_wikitext_is_a_single_segment_so_the_historical_path_is_unchanged(monkeypatch):
    """One segment takes the original branch: concatenate, tokenize, slice from the front.

    No interleaving, no per-language split — the shipped path must stay bit-for-bit what it
    was, or existing checkpoints stop being reproducible.
    """
    monkeypatch.setitem(qm._CALIB_CORPORA, "wikitext",
                        lambda _needed: [("wikitext-en", "x" * 1000)])
    ids, counts = qm._load_calib_ids("wikitext", _Tokenizer(), nsamples=3, seqlen=10)
    assert ids.shape == (3, 10)
    assert counts == {"wikitext-en": 3}
    # Contiguous from token 0 — the historical slicing, not a shuffle.
    assert torch.equal(ids[0], torch.arange(10))
    assert torch.equal(ids[1], torch.arange(10, 20))


def test_short_single_segment_truncates_rather_than_raising(monkeypatch):
    """Historical behaviour: `min(nsamples, len // seqlen)`. Preserved deliberately.

    The multi-segment path raises on a shortfall instead, because there the shortfall means
    an unbalanced mix. For one segment, fewer chunks is just a smaller corpus.
    """
    monkeypatch.setitem(qm._CALIB_CORPORA, "wikitext",
                        lambda _n: [("short", "x" * 25)])
    ids, counts = qm._load_calib_ids("wikitext", _Tokenizer(), nsamples=99, seqlen=10)
    assert ids.shape == (2, 10)          # 25 chars -> 2 full chunks, not an error
    assert counts == {"short": 2}


# ------------------------------------------------------------- the bilingual mix

def test_bilingual_corpus_is_registered_and_offered_by_the_cli():
    assert "wikipedia-de-en" in qm._CALIB_CORPORA
    args = qm.build_parser().parse_args(
        ["--model", "m", "--output", "o", "--calib-corpus", "wikipedia-de-en"])
    assert args.calib_corpus == "wikipedia-de-en"


def test_two_segments_split_the_budget_evenly_and_interleave(monkeypatch):
    """Interleaved, so a run cut short still saw both languages.

    Concatenating instead would put every German chunk after every English one, and a
    calibration pass that stopped early — or any future per-chunk subsetting — would see
    one language only. That is the failure this corpus exists to prevent, so the ordering
    is part of the contract.
    """
    monkeypatch.setitem(
        qm._CALIB_CORPORA, "pair",
        lambda _needed: [("de", "d" * 500), ("en", "e" * 500)])
    ids, counts = qm._load_calib_ids("pair", _Tokenizer(), nsamples=6, seqlen=10)

    assert counts == {"de": 3, "en": 3}
    assert ids.shape == (6, 10)
    # de, en, de, en, ... — each segment restarts its own token numbering at 0.
    assert torch.equal(ids[0], torch.arange(0, 10))      # de chunk 0
    assert torch.equal(ids[1], torch.arange(0, 10))      # en chunk 0
    assert torch.equal(ids[2], torch.arange(10, 20))     # de chunk 1
    assert torch.equal(ids[3], torch.arange(10, 20))     # en chunk 1


def test_an_under_filled_language_raises_rather_than_skewing_the_mix(monkeypatch):
    """A silently unbalanced mix is the exact thing this corpus is for; refuse it."""
    monkeypatch.setitem(
        qm._CALIB_CORPORA, "lopsided",
        lambda _needed: [("de", "d" * 15), ("en", "e" * 500)])
    with pytest.raises(ValueError, match="wanted|yielded"):
        qm._load_calib_ids("lopsided", _Tokenizer(), nsamples=6, seqlen=10)


def test_unknown_corpus_names_the_valid_choices():
    with pytest.raises(ValueError, match="unknown --calib-corpus"):
        qm._load_calib_ids("nope", _Tokenizer(), 4, 8)


# ------------------------------------------------------- the routing report

class _Experts:
    def __init__(self, width=2560, inter=512, n=4):
        self.gate_up_proj = torch.empty(n, 2 * inter, width, device="meta")


def test_routing_report_names_experts_that_saw_nothing(capsys):
    """A never-routed expert gets an identity Hessian. That is the single most useful fact
    about a calibration run and it had no summary line before."""
    counts = {("mlp.experts", 0): 4096, ("mlp.experts", 1): 0,
              ("mlp.experts", 2): 3000, ("mlp.experts", 3): 5000}
    qm._report_expert_routing(counts, [("mlp.experts", _Experts())])
    out = capsys.readouterr().out
    assert "saw NOTHING" in out
    assert "min 0" in out


def test_routing_report_flags_an_under_determined_median(capsys):
    """4096 rows against a 2560-wide Hessian is 1.6x — the real figure at nsamples 128."""
    counts = {("mlp.experts", e): 1000 for e in range(4)}      # < 2560
    qm._report_expert_routing(counts, [("mlp.experts", _Experts())])
    out = capsys.readouterr().out
    assert "under-determined" in out
    assert "nsamples" in out, "the note should say what to do about it"


def test_routing_report_is_quiet_when_every_expert_is_well_fed(capsys):
    counts = {("mlp.experts", e): 20000 for e in range(4)}
    qm._report_expert_routing(counts, [("mlp.experts", _Experts())])
    out = capsys.readouterr().out
    assert "under-determined" not in out
    assert "saw NOTHING" not in out
    assert "routed rows/expert" in out, "the distribution is still worth printing"


def test_routing_report_does_nothing_without_counts(capsys):
    """Dense models and the non-stacked path must not grow a spurious line."""
    qm._report_expert_routing({}, [])
    assert capsys.readouterr().out == ""
