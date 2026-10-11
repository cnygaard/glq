"""`--streaming` must supply position_embeddings for default-profile architectures too.

The calibration forward passes `position_embeddings` only when a rotary module was resolved.
Under `--streaming` the model is never fully built, so the rotary is re-instantiated by class
from a meta model — but the capture was gated on `profile['rotary_attr']`, which only the
NAMED arch profiles (gemma-4, mistral3, sarvam_moe, ...) set. A default-profile arch
(Llama / SmolLM2 / SmolLM3) therefore got no rotary, no `position_embeddings`, and transformers
5.x attention needs it.

That makes `--streaming` unusable on exactly the small dense models a resume dry-run wants,
which is how this was found.
"""
from __future__ import annotations

import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _discover_rotary_cls  # noqa: E402


class _Rotary(nn.Module):
    pass


def test_an_explicit_rotary_attr_still_wins():
    """The named profiles' behaviour, unchanged — this path is what gemma-4 and sarvam use."""
    inner = nn.Module()
    inner.rotary_emb = _Rotary()
    model = nn.Module()
    model.language_model = inner
    assert _discover_rotary_cls(
        model, {"rotary_attr": "language_model.rotary_emb"}) is _Rotary


def test_a_default_profile_falls_back_to_the_standard_layout():
    """The fix. `model.model.rotary_emb` is where every Llama-family decoder puts it, and
    `get_rotary_emb` already knew both standard layouts — the streaming path just never
    asked."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.rotary_emb = _Rotary()
    assert _discover_rotary_cls(model, {}) is _Rotary


def test_a_top_level_rotary_is_found():
    model = nn.Module()
    model.rotary_emb = _Rotary()
    assert _discover_rotary_cls(model, {}) is _Rotary


def test_an_architecture_with_no_rotary_yields_none():
    """Not every decoder has one (ALiBi, learned positions). None must stay None rather than
    raising, so those archs keep streaming exactly as before."""
    assert _discover_rotary_cls(nn.Module(), {}) is None


def test_a_wrong_rotary_attr_does_not_raise():
    """A profile can name an attribute a newer transformers moved. Falling back beats
    crashing a multi-hour quantize on a lookup."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.rotary_emb = _Rotary()
    assert _discover_rotary_cls(model, {"rotary_attr": "nope.not_here"}) is _Rotary


class _OtherRotary(nn.Module):
    """A DIFFERENT class, so "which one was picked" is observable rather than inferred."""


def test_a_profile_that_declares_no_rotary_keeps_getting_none():
    """THE regression case. NemotronH sets `rotary_attr: None` WITH `layers_attr`, which
    `get_rotary_emb` reads as "this architecture has no global rotary" — deliberately, not as
    a gap to fill. The danger in falling back to the heuristic is handing such an arch a real
    rotary it never had, which changes its forward and breaks byte-identical checkpoints.

    The discriminator is `layers_attr`: present means the profile spoke deliberately, absent
    (the default profile, whose comment literally reads "use heuristic") means fall through.
    A rotary IS reachable here, so this fails if the fallback ignores the declaration."""
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.layers = nn.ModuleList([nn.Module()])
    model.model = nn.Module()
    model.model.rotary_emb = _Rotary()          # reachable by the heuristic — must be ignored
    profile = {"layers_attr": "backbone.layers", "rotary_attr": None}
    assert _discover_rotary_cls(model, profile) is None


def test_an_explicit_rotary_attr_beats_a_reachable_heuristic_one():
    """For the named profiles (gemma-4 x5, sarvam) the explicit path must win, or a streaming
    arch that already worked would silently switch rotary class — the one regression that
    could not be caught by a non-streaming byte gate, since this code is only reached under
    --streaming."""
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.rotary_emb = _OtherRotary()
    model.rotary_emb = _Rotary()                 # the heuristic would find THIS one
    got = _discover_rotary_cls(model, {"rotary_attr": "language_model.rotary_emb"})
    assert got is _OtherRotary, f"explicit attr lost to the heuristic: {got}"
