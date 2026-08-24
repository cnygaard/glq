"""Streaming-quantize support for Qwen3_5ForConditionalGeneration (Qwen3.5-0.8B).

First hybrid-attention arch in GLQ: per 4 layers, 3 GatedDeltaNet (linear-attention)
blocks + 1 full-attention block, wrapped in a multimodal shell (vision ViT + MTP head).
Three things stand between it and the muse-glimmer path:

  1. transformers maps model_type ``qwen3_5`` to ``Qwen3_5ForCausalLM`` in the CausalLM
     auto-mapping ("VLM compatibility"), a text-only class whose __init__ reads
     ``config.vocab_size`` — absent on the wrapper config — so ``from_config`` raises
     **AttributeError**, not the ValueError ``_meta_model_from_config`` catches. The
     fallback never runs and the quantize dies before layer discovery.
  2. GDN blocks carry two 16-row projections (``in_proj_b``/``in_proj_a``) that fail the
     trellis serving gate (out % 32) and encode the SSM dynamics; the generic nn.Linear
     walk would quantize them. Profiles gain a declarative ``skip_linears``.
  3. A multimodal wrapper cannot quantize non-streaming: the save path iterates
     ``named_parameters()`` on a class that has no ``mtp.*`` modules, silently dropping
     them. Refuse with instructions instead of an AttributeError deep in discovery.

All CPU, no model download.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("transformers")  # noqa: E402
import torch.nn as nn  # noqa: E402

from glq import quantize_model as QM  # noqa: E402


class _Cfg:
    def __init__(self, arch):
        self.architectures = [arch]


QWEN = "Qwen3_5ForConditionalGeneration"


def test_profile_points_at_the_language_model_decoder():
    """Default sd_prefix='model.layers' matches zero tensors here — the silent-empty-
    quantize failure the muse-glimmer profile exists to prevent."""
    prof = QM._detect_profile(_Cfg(QWEN))
    assert prof is not QM._DEFAULT_PROFILE, "arch must have its own profile"
    assert prof["sd_prefix"] == "model.language_model.layers"
    assert prof["layers_attr"] == "model.language_model.layers"
    assert prof["embed_attr"] == "model.language_model.embed_tokens"
    assert prof["rotary_attr"] == "model.language_model.rotary_emb"
    assert prof["trust_remote_code"] is False
    assert prof["forward_kwargs"] == "default"


def test_profile_marks_it_multimodal_so_streaming_derives_the_prefix():
    prof = QM._detect_profile(_Cfg(QWEN))
    assert QM._is_multimodal_text(QWEN, prof) is True


def test_profile_skips_the_ssm_gate_projections():
    """in_proj_b / in_proj_a are 16 rows: below the trellis kernel's out%32 floor, and
    they parameterize the delta-rule dynamics — quantizing them is both unservable and
    the likeliest place to wreck the recurrence."""
    prof = QM._detect_profile(_Cfg(QWEN))
    assert tuple(prof["skip_linears"]) == (".in_proj_b", ".in_proj_a")


# ---------------------------------------------------------------- skip_linears walk

class _GDNBlock(nn.Module):
    """Module tree shaped like Qwen3_5GatedDeltaNet + MLP, tiny dims."""

    def __init__(self):
        super().__init__()
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(16, 48, bias=False)
        self.linear_attn.in_proj_z = nn.Linear(16, 16, bias=False)
        self.linear_attn.in_proj_b = nn.Linear(16, 2, bias=False)
        self.linear_attn.in_proj_a = nn.Linear(16, 2, bias=False)
        self.linear_attn.out_proj = nn.Linear(16, 16, bias=False)
        self.linear_attn.conv1d = nn.Conv1d(48, 48, 4, groups=48)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(16, 32, bias=False)
        self.mlp.up_proj = nn.Linear(16, 32, bias=False)
        self.mlp.down_proj = nn.Linear(32, 16, bias=False)


def test_skip_linears_drops_b_a_but_keeps_qkv_z():
    prof = QM._detect_profile(_Cfg(QWEN))
    linears = QM._collect_linears(_GDNBlock(), prof)
    names = set(linears)
    assert "linear_attn.in_proj_qkv" in names
    assert "linear_attn.in_proj_z" in names
    assert "linear_attn.out_proj" in names
    assert "mlp.gate_proj" in names and "mlp.down_proj" in names
    assert "linear_attn.in_proj_b" not in names
    assert "linear_attn.in_proj_a" not in names
    # conv1d is not nn.Linear — never collected, no skip entry needed
    assert not any("conv1d" in n for n in names)


def test_collect_linears_without_skip_matches_the_plain_walk():
    """Regression: every existing arch (no skip_linears key) must see the identical set
    the inline isinstance walk produced."""
    block = _GDNBlock()
    got = QM._collect_linears(block, QM._DEFAULT_PROFILE)
    expected = {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    assert got.keys() == expected.keys()


# ---------------------------------------------------------------- meta-model fallback

def _fake_transformers(monkeypatch, causal_exc):
    used = {}

    class _Causal:
        @staticmethod
        def from_config(cfg, **kw):
            raise causal_exc

    class _ITT:
        @staticmethod
        def from_config(cfg, **kw):
            used["cls"] = "itt"
            return types.SimpleNamespace(kind="itt")

    mod = types.ModuleType("transformers")
    mod.AutoModelForCausalLM = _Causal
    mod.AutoModelForImageTextToText = _ITT
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return used


def test_meta_loader_falls_back_on_attribute_error(monkeypatch):
    """The qwen3_5 case: the CausalLM auto-mapping resolves (to Qwen3_5ForCausalLM) and
    then raises AttributeError reading vocab_size off the wrapper config — a different
    exception class than muse-glimmer's ValueError, same required outcome."""
    used = _fake_transformers(
        monkeypatch, AttributeError("'Qwen3_5Config' object has no attribute 'vocab_size'"))
    m = QM._meta_model_from_config(_Cfg(QWEN), trust_remote_code=False, dtype=None)
    assert m.kind == "itt"
    assert used["cls"] == "itt"


def test_real_meta_build_reaches_the_language_model():
    """With real transformers (>=5.2 ships qwen3_5): the wrapper builds on meta via the
    fallback, and the profile's dotted paths resolve."""
    pytest.importorskip("transformers.models.qwen3_5")
    import torch
    from transformers import AutoConfig
    cfg = AutoConfig.for_model("qwen3_5")
    cfg.architectures = [QWEN]
    with torch.device("meta"):
        model = QM._meta_model_from_config(cfg, trust_remote_code=False, dtype=None)
    prof = QM._detect_profile(_Cfg(QWEN))
    layers = QM._resolve_attr(model, prof["layers_attr"])
    assert len(layers) > 0
    assert QM._resolve_attr(model, prof["rotary_attr"]) is not None


# ---------------------------------------------------------------- streaming guard

def test_non_streaming_refused_for_multimodal_wrapper():
    """Non-streaming save iterates named_parameters() on a class without mtp.* modules —
    it would silently drop them. The refusal must say what to do instead."""
    prof = QM._detect_profile(_Cfg(QWEN))
    with pytest.raises(ValueError, match="--streaming"):
        QM._require_streaming_for_wrapper(QWEN, prof, streaming=False)


def test_streaming_wrapper_passes_the_guard():
    prof = QM._detect_profile(_Cfg(QWEN))
    QM._require_streaming_for_wrapper(QWEN, prof, streaming=True)


def test_plain_causal_never_hits_the_guard():
    QM._require_streaming_for_wrapper(
        "SmolLM3ForCausalLM", QM._DEFAULT_PROFILE, streaming=False)
