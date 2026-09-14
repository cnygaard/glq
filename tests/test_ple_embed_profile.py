"""Which per-layer-embedding table a model quantizes, and how — as a profile capability.

The PLE hook was gated on `is_gemma4` plus one hardcoded tensor name. That is the same
substring-sniffing that `stacked_experts` replaced: it means a new architecture with a PLE
table silently gets none of this, and the table is not a rounding error — Qwen3.8-Flash-Next's
n-gram table is 95.4 GiB, **60% of its own 4 bpw checkpoint**. Left bf16 the model needs two
96 GiB cards; quantized it fits one.

These pin the descriptor rather than the plumbing, because getting the descriptor wrong is
silent: a missing entry quantizes nothing and the footprint regression only shows up at load.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _MODEL_PROFILES, _ple_embed_spec  # noqa: E402


def _cfg(ple_dim=256):
    return SimpleNamespace(text_config=SimpleNamespace(
        hidden_size_per_layer_input=ple_dim))


GEMMA4 = "Gemma4ForConditionalGeneration"
QWEN4EXP = "Qwen4ExpForConditionalGeneration"


def test_gemma4_keeps_exactly_what_the_hardcoded_path_did():
    """The byte-identical guard, at the descriptor level.

    Before this was a profile entry it was four literals inline: the tensor name,
    ple_embed_bpw=4, the shell codebook, and block_diagonal=False (full Hadamard). Any drift
    here changes published gemma-4 checkpoints, which must not happen silently.
    """
    spec = _ple_embed_spec(GEMMA4, _MODEL_PROFILES[GEMMA4], _cfg())
    assert spec is not None
    assert spec["prefix"] == "model.language_model.embed_tokens_per_layer"
    assert spec["bpw"] == 4
    assert spec["codebook"] == "shell"
    assert spec["block_diagonal"] is False
    assert spec.get("shards") is None


def test_gemma4_without_a_ple_dim_has_no_table():
    """gemma4_unified (12B) is dense: hidden_size_per_layer_input == 0. Sharing the gemma-4
    module layout must not make it quantize a table it does not have."""
    assert _ple_embed_spec(GEMMA4, _MODEL_PROFILES[GEMMA4], _cfg(ple_dim=0)) is None


def test_qwen4exp_quantizes_its_ngram_table_with_trellis_block_diagonal():
    """160 is not a power of two, and that decides the codebook.

    The shell path's full Hadamard pads a 160-wide row to 256 — a 1.6x footprint tax, and the
    reason trellis was excluded from the PLE in 0.7.2 (it asserts on the padded layout).
    Block-diagonal RHT leaves 160 alone, so trellis stores 60 B/row against shell's 64 while
    decoding a gathered row faster.
    """
    spec = _ple_embed_spec(QWEN4EXP, _MODEL_PROFILES[QWEN4EXP], _cfg(ple_dim=160))
    assert spec is not None
    assert spec["codebook"] == "trellis"
    assert spec["block_diagonal"] is True, "full Hadamard would pad 160 -> 256"
    assert spec["bpw"] == 3
    assert spec["shards"] == 128, "the table is stored as 128 shard_N tensors"
    assert "ngram_embedding" in spec["prefix"]


def test_an_architecture_without_a_ple_table_gets_none():
    """The default path must not acquire a PLE step by accident."""
    assert _ple_embed_spec("LlamaForCausalLM", {}, SimpleNamespace()) is None


def test_a_missing_text_config_is_not_an_error():
    """Profiles are consulted for every architecture, including ones whose config has no
    text_config at all. This runs before any weights load, so raising here would refuse the
    model outright."""
    assert _ple_embed_spec("LlamaForCausalLM", {}, SimpleNamespace()) is None
    assert _ple_embed_spec(GEMMA4, _MODEL_PROFILES[GEMMA4], SimpleNamespace()) is None


@pytest.mark.parametrize("arch", [GEMMA4, QWEN4EXP])
def test_every_ple_spec_is_complete(arch):
    """A half-filled descriptor fails at quantize time, hours in. Fail at import instead."""
    spec = _ple_embed_spec(arch, _MODEL_PROFILES[arch],
                           _cfg(160 if arch == QWEN4EXP else 256))
    for key in ("prefix", "codebook", "bpw", "block_diagonal"):
        assert key in spec, f"{arch} missing {key}"
    assert spec["codebook"] in ("shell", "trellis")
    assert isinstance(spec["bpw"], int) and 2 <= spec["bpw"] <= 8


# ---- the serving marker ------------------------------------------------------------------

def test_the_ple_codebook_round_trips_through_the_config():
    """vLLM must know the table's codebook BEFORE weights load.

    `create_weights` registers buffers up front, and shell and trellis need different ones
    (Qidxs [vocab, n_pad/8] vs trellis_packed [vocab, ceil(width*K/16)]). The checkpoint's
    tensor keys settle it, but they are not available at that point — so the marker rides in
    config.json -> quantization_config, the same place `variant` and `trellis_layout` do.
    """
    from glq.hf_integration import GLQConfig
    cfg = GLQConfig(codebook="trellis", variant="3inst", ple_codebook="trellis")
    assert cfg.to_dict()["ple_codebook"] == "trellis"


def test_a_shell_ple_does_not_emit_the_marker():
    """Absent means shell, so existing gemma-4 checkpoints keep byte-identical config.json
    rather than gaining a key that changes their hash."""
    from glq.hf_integration import GLQConfig
    assert "ple_codebook" not in GLQConfig(codebook="trellis", variant="3inst").to_dict()
    assert "ple_codebook" not in GLQConfig(codebook="e8_shell").to_dict()
