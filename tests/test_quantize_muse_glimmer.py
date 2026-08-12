"""Streaming-quantize support for MuseGlimmerForConditionalGeneration.

Muse-Glimmer-30B is a vision-language model whose text half is an ordinary dense causal
transformer (52 layers, GQA 32/2, hidden 6656, intermediate 19968 — every linear
16-divisible, so trellis tiles fit). Two things stopped `glq-quantize --streaming`:

  1. quantize() instantiates the model on meta with AutoModelForCausalLM.from_config,
     which raises ValueError for a multimodal config — the run dies in 2s.
  2. No _MODEL_PROFILES entry, so it fell back to sd_prefix='model.layers'. That prefix
     matches ZERO tensors in this checkpoint (the decoder is at
     model.language_model.layers), which is the dangerous failure: _load_layer_state
     would find nothing per layer rather than erroring, i.e. a silently empty quantize.

The fix mirrors the existing gemma-4 path. The streaming branch already derives sd_prefix
generically by looking for "language_model" in the weight-map keys and already skips
vision/audio towers; it was only gated behind a hardcoded `is_mistral3 or is_gemma4`.
These tests pin the profile, the loader fallback, and that gate — all on CPU, no model.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import quantize_model as QM  # noqa: E402


class _Cfg:
    def __init__(self, arch):
        self.architectures = [arch]


MUSE = "MuseGlimmerForConditionalGeneration"


def test_profile_points_at_the_language_model_decoder():
    """sd_prefix='model.layers' (the default) matches nothing in this checkpoint, so a
    wrong profile yields an empty quantize rather than an error."""
    prof = QM._detect_profile(_Cfg(MUSE))
    assert prof is not QM._DEFAULT_PROFILE, "arch must have its own profile"
    assert prof["sd_prefix"] == "model.language_model.layers"
    assert prof["layers_attr"] == "model.language_model.layers"
    assert prof["embed_attr"] == "model.language_model.embed_tokens"
    assert prof["trust_remote_code"] is False   # repo ships no auto_map; policy forbids it


def test_profile_marks_it_multimodal_so_streaming_derives_the_prefix():
    """The streaming branch skips vision/audio towers and derives sd_prefix from keys
    containing 'language_model' — but only for archs flagged multimodal."""
    prof = QM._detect_profile(_Cfg(MUSE))
    assert QM._is_multimodal_text(MUSE, prof) is True


@pytest.mark.parametrize("arch,expected", [
    ("Gemma4ForConditionalGeneration", True),
    ("Gemma4UnifiedForConditionalGeneration", True),
    ("Mistral3ForConditionalGeneration", True),
    ("SmolLM3ForCausalLM", False),
    ("SarvamMoEForCausalLM", False),
])
def test_multimodal_gate_is_unchanged_for_existing_archs(arch, expected):
    """Regression: generalising the gate must not change routing for anything already
    supported — a text-only model taking the multimodal path would mis-derive sd_prefix."""
    assert QM._is_multimodal_text(arch, QM._detect_profile(_Cfg(arch))) is expected


class _TextCfg:
    max_position_embeddings = 131072
    rope_theta = 500000.0


class _WrapperCfg:
    """Multimodal wrapper: rope params live on .text_config, NOT at the top level."""
    architectures = [MUSE]
    text_config = _TextCfg()


def test_text_config_is_the_subconfig_for_multimodal():
    """The rotary constructor reads max_position_embeddings. On a multimodal wrapper that
    attribute only exists on cfg.text_config — passing the wrapper raises AttributeError
    and kills the run after layer discovery has already succeeded. The gemma-4 branch
    already does this (quantize_model.py:1163); the generic branch must too."""
    cfg = _WrapperCfg()
    assert QM._text_config(cfg, is_multimodal=True) is cfg.text_config


def test_text_config_is_the_config_itself_for_plain_causal():
    """Regression: text-only archs (sarvam_moe et al) must keep getting the top-level
    config — they have no .text_config and their rope params live there."""
    class _Plain:
        architectures = ["SarvamMoEForCausalLM"]
        max_position_embeddings = 8192
    cfg = _Plain()
    assert QM._text_config(cfg, is_multimodal=False) is cfg


def test_text_config_falls_back_when_multimodal_lacks_text_config():
    """Defensive: a multimodal arch without .text_config must not crash on attribute
    lookup — fall back to the wrapper rather than raising inside a helper."""
    class _NoText:
        architectures = [MUSE]
    cfg = _NoText()
    assert QM._text_config(cfg, is_multimodal=True) is cfg


def _fake_transformers(monkeypatch, causal_ok, itt_ok):
    """Fake the two auto-classes; from_config records which one was used."""
    used = {}

    class _Causal:
        @staticmethod
        def from_config(cfg, **kw):
            if not causal_ok:
                raise ValueError(
                    "Unrecognized configuration class ... for this kind of AutoModel: "
                    "AutoModelForCausalLM.")
            used["cls"] = "causal"
            return types.SimpleNamespace(kind="causal")

    class _ITT:
        @staticmethod
        def from_config(cfg, **kw):
            if not itt_ok:
                raise ValueError("Unrecognized configuration class ...")
            used["cls"] = "itt"
            return types.SimpleNamespace(kind="itt")

    mod = types.ModuleType("transformers")
    mod.AutoModelForCausalLM = _Causal
    mod.AutoModelForImageTextToText = _ITT
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return used


def test_meta_loader_falls_back_to_image_text_to_text(monkeypatch):
    """The Muse-Glimmer case: AutoModelForCausalLM rejects the config, so quantize()
    must fall back rather than abort the whole run."""
    used = _fake_transformers(monkeypatch, causal_ok=False, itt_ok=True)
    m = QM._meta_model_from_config(_Cfg(MUSE), trust_remote_code=False, dtype=None)
    assert m.kind == "itt"
    assert used["cls"] == "itt"


def test_meta_loader_prefers_causal_when_available(monkeypatch):
    """Regression: ordinary causal models must keep using the causal class."""
    used = _fake_transformers(monkeypatch, causal_ok=True, itt_ok=True)
    m = QM._meta_model_from_config(_Cfg("SmolLM3ForCausalLM"),
                                   trust_remote_code=False, dtype=None)
    assert m.kind == "causal"
    assert used["cls"] == "causal"


def test_meta_loader_reraises_when_neither_class_accepts(monkeypatch):
    """An arch nothing can load must fail loudly, not silently produce an empty model."""
    _fake_transformers(monkeypatch, causal_ok=False, itt_ok=False)
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        QM._meta_model_from_config(_Cfg("NoSuchArch"), trust_remote_code=False, dtype=None)
