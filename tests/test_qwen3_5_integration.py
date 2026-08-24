"""End-to-end streaming quantize of a tiny Qwen3_5ForConditionalGeneration on CPU.

The unit tests pin the profile, the skip filter and the meta-loader fallback in
isolation; this is the one place they run together through the REAL `quantize()` —
hybrid GDN + full-attention layers, the multimodal wrapper, and the streaming
byte-for-byte copy of everything unquantized (vision tower, b/a, conv1d).

A shrunken config (378k params, 3 GDN + 1 full-attention layers) built and saved
locally — sharded with a tiny max_shard_size so the streaming loader gets the
weight-map index it keys on. vocab_size matches the SmolLM2 tokenizer used for
calibration so token ids stay inside the embedding.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("transformers.models.qwen3_5",
                    reason="transformers too old for qwen3_5")  # noqa: E402

from pytest import MonkeyPatch  # noqa: E402

pytestmark = pytest.mark.slow

QWEN = "Qwen3_5ForConditionalGeneration"

#: Layers 0-2 are GDN, layer 3 full attention — one of each kind inside the 2-layer
#: truncation below would miss the full-attn block, so the truncation keeps 0 and 3.
TEXT = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=32,
    vocab_size=49152, max_position_embeddings=256, tie_word_embeddings=True,
    layer_types=["linear_attention", "linear_attention", "linear_attention",
                 "full_attention"],
    linear_num_key_heads=2, linear_num_value_heads=2,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    mtp_num_hidden_layers=0,
)
VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
              patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
              out_hidden_size=64)


@pytest.fixture(scope="class")
def _monkeypatch_class():
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="class")
def quantize_output(tmp_path_factory, _monkeypatch_class):
    import torch
    from transformers import AutoTokenizer, Qwen3_5Config
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    from glq.quantize_model import get_decoder_layers, quantize

    torch.manual_seed(0)
    cfg = Qwen3_5Config(text_config=dict(TEXT), vision_config=dict(VISION))
    cfg.architectures = [QWEN]
    # bf16, matching a real published checkpoint: the streaming path runs calibration in
    # the checkpoint dtype, and a mixed fp32/bf16 tree dies in F.linear on CPU.
    model = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)

    src = str(tmp_path_factory.mktemp("qwen35_tiny_src"))
    # Tiny shard size forces a model.safetensors.index.json — the weight map the
    # streaming branch derives sd_prefix and the embed key from.
    model.save_pretrained(src, max_shard_size="200KB")
    AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M").save_pretrained(src)

    # Keep one GDN layer and the full-attention layer: [:2] would test only GDN.
    _orig = get_decoder_layers

    def _limited(text_model, profile=None):
        layers = _orig(text_model, profile)
        return [layers[0], layers[3]]

    _monkeypatch_class.setattr("glq.quantize_model.get_decoder_layers", _limited)

    out = str(tmp_path_factory.mktemp("qwen35_glq"))
    # codebook_type explicitly: the CLI default is trellis since 0.8.8, but the
    # FUNCTION default is still e8_shell — and trellis/3inst is the path Qwen3.5
    # ships on (and the one the serving-side shard-group code assumes).
    os.environ.setdefault("GLQ_TRELLIS_VARIANT", "3inst")
    avg_sqnr = quantize(model_name=src, output_dir=out, bpw=2, nsamples=1,
                        seqlen=64, device="cpu", workers=1, streaming=True,
                        codebook_type="trellis")
    return src, out, avg_sqnr


class TestQwen35StreamingQuantize:
    def _keys(self, out):
        from safetensors import safe_open
        import glob
        keys = set()
        for f in glob.glob(os.path.join(out, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                keys |= set(sf.keys())
        return keys

    def test_positive_sqnr(self, quantize_output):
        _, _, avg_sqnr = quantize_output
        assert avg_sqnr > 0

    def test_quantized_set_is_exactly_the_supported_linears(self, quantize_output):
        _, out, _ = quantize_output
        keys = self._keys(out)
        pfx = "model.language_model.layers"
        # GDN layer 0: qkv/z/out quantized...
        for lin in ("in_proj_qkv", "in_proj_z", "out_proj"):
            assert any(k.startswith(f"{pfx}.0.linear_attn.{lin}.") for k in keys), lin
            assert f"{pfx}.0.linear_attn.{lin}.weight" not in keys
        # ...b/a skipped: plain weights present, no artifacts
        for lin in ("in_proj_b", "in_proj_a"):
            assert f"{pfx}.0.linear_attn.{lin}.weight" in keys
            assert not any(k.startswith(f"{pfx}.0.linear_attn.{lin}.")
                           and not k.endswith(".weight") for k in keys), lin
        # conv1d streams through untouched
        assert f"{pfx}.0.linear_attn.conv1d.weight" in keys
        # full-attention layer 3 quantized normally
        assert any(k.startswith(f"{pfx}.3.self_attn.q_proj.") for k in keys)
        assert f"{pfx}.3.self_attn.q_proj.weight" not in keys

    def test_vision_tower_copied_verbatim(self, quantize_output):
        src, out, _ = quantize_output
        src_vis = {k for k in self._keys(src) if "visual" in k}
        out_vis = {k for k in self._keys(out) if "visual" in k}
        assert src_vis and src_vis == out_vis

    def test_layer_bpw_covers_exactly_the_artifact_set(self, quantize_output):
        _, out, _ = quantize_output
        cfg = json.load(open(os.path.join(out, "config.json")))
        q = cfg["quantization_config"]
        assert cfg["architectures"] == [QWEN]
        listed = set(q["layer_bpw"])
        # Identify quantized linears by the trellis payload itself — plain non-weight
        # keys (A_log, dt_bias, conv1d.bias) would otherwise pollute the root set.
        art_roots = {k[: -len(".trellis_packed")] for k in self._keys(out)
                     if k.endswith(".trellis_packed")}
        assert listed == art_roots
        assert not any(b.endswith(("in_proj_b", "in_proj_a")) for b in listed)
