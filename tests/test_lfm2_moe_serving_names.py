"""LFM2-MoE on vLLM: every layer vLLM builds must find its GLQ entry, under vLLM's own names.

vLLM 0.33's `lfm2_moe.py` renames on the way in, while `quantization_config.layer_bpw` keeps
the checkpoint's spelling (the quantizer writes it from the safetensors keys):

* `.conv.` -> `.short_conv.` (`hf_to_vllm_mapper`, to keep LoRA's regex off the inner
  `conv.conv`), so `get_quant_method` is asked about `short_conv.in_proj` while `layer_bpw`
  holds `conv.in_proj`. `short_conv.in_proj` is a 3-partition merged linear built from ONE
  checkpoint matrix, so it must also resolve as pre-fused, or it gets three shard slots and a
  whole-matrix load with nowhere to land — the NemotronH `mixer.in_proj` failure, which served
  zero weights with every footprint check passing.
* the dense MLP's `w1`/`w3` load as shards 0/1 of `feed_forward.w13`.
* routed experts are read by `ckpt_names=("w1", "w2", "w3")`; a name outside that set is
  skipped by vLLM's loader without an error, so the experts would serve uninitialised memory.

`layer_bpw` here is derived from the recorded manifest `tests/models/lfm2_8b_a1b.json`: every
2-D `.weight` under `model.layers` is an `nn.Linear` the quantizer takes, except the router
(`feed_forward.gate`, a bare Parameter). `tests/test_lfm2_moe_profile.py`'s end-to-end run
checks that same set against a real quantize.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from glq.name_aliases import backbone_aliases, with_aliases  # noqa: E402
from model_shapes import load_or_skip as load_shapes  # noqa: E402

try:
    from vllm.model_executor.layers.quantization import get_quantization_config  # noqa: F401
    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False

requires_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")


# ------------------------------------------------ the alias, no dependencies at all

def test_vllm_short_conv_resolves_to_the_checkpoint_conv():
    assert "model.layers.0.conv.in_proj" in backbone_aliases(
        "model.layers.0.short_conv.in_proj")


def test_and_the_other_way_round():
    assert "model.layers.0.short_conv.out_proj" in backbone_aliases(
        "model.layers.0.conv.out_proj")


def test_the_lookup_order_tries_the_name_itself_first():
    forms = with_aliases("model.layers.0.short_conv.in_proj")
    assert forms[0] == "model.layers.0.short_conv.in_proj"
    assert "model.layers.0.conv.in_proj" in forms


@pytest.mark.parametrize("name,want", [
    # the prefix aliases these always had, and nothing more: `conv1d` and `_conv` are not
    # the `conv` segment
    ("model.layers.0.self_attn.q_proj", {"backbone.layers.0.self_attn.q_proj"}),
    ("backbone.layers.0.mixer.conv1d", {"model.layers.0.mixer.conv1d"}),
    ("model.layers.0.linear_attn.conv1d", {"backbone.layers.0.linear_attn.conv1d"}),
    ("visual.blocks.0.patch_conv.weight", set()),
])
def test_names_without_a_conv_segment_keep_their_old_aliases(name, want):
    assert backbone_aliases(name) == want


def test_the_nemotron_prefix_alias_is_unchanged():
    assert backbone_aliases("backbone.layers.0.mixer.in_proj") == {
        "model.layers.0.mixer.in_proj"}
    assert backbone_aliases("language_model.backbone.layers.3.mixer.out_proj") == {
        "language_model.model.layers.3.mixer.out_proj"}


# ----------------------------------------------------------- the plugin, with vLLM

def _layer_bpw(bpw=4):
    ms = load_shapes("lfm2_8b_a1b")
    lb = {}
    for key in ms.keys():
        if not (key.startswith("model.layers.") and key.endswith(".weight")):
            continue
        if len(ms.shape_of(key)) != 2 or key.endswith("feed_forward.gate.weight"):
            continue
        lb[key[: -len(".weight")]] = bpw
    return lb


def _cfg():
    from glq_vllm.config import GLQvLLMConfig
    return GLQvLLMConfig.from_config({
        "bpw": 4, "codebook": "trellis", "variant": "3inst", "trellis_layout": "kernel",
        "layer_bpw": _layer_bpw()})


def test_the_derived_layer_bpw_has_the_published_shape():
    """24 layers: 18 conv (in/out), 6 attention (q/k/v/out), 2 dense (w1/w2/w3), 22 MoE
    layers x 32 experts x 3."""
    lb = _layer_bpw()
    assert len(lb) == 18 * 2 + 6 * 4 + 2 * 3 + 22 * 32 * 3
    assert "model.layers.2.feed_forward.experts.31.w2" in lb
    assert "model.layers.0.conv.in_proj" in lb


@requires_vllm
@pytest.mark.parametrize("prefix,prefused", [
    ("model.layers.0.short_conv.in_proj", True),     # one matrix, three partitions
    ("model.layers.0.short_conv.out_proj", True),
    ("model.layers.0.feed_forward.w13", False),      # w1 + w3, per-shard
    ("model.layers.0.feed_forward.w2", True),
    ("model.layers.2.self_attn.qkv_proj", False),    # q + k + v, per-shard
    ("model.layers.2.self_attn.out_proj", True),
    ("model.layers.23.short_conv.in_proj", True),
])
def test_every_quantized_linear_resolves_under_the_vllm_name(prefix, prefused):
    cfg = _cfg()
    assert cfg._lookup_bpw(prefix) == 4, prefix
    assert cfg._is_prefused(prefix) is prefused, prefix


@requires_vllm
@pytest.mark.parametrize("prefix", [
    "model.layers.0.short_conv.conv1d",    # the depthwise kernel stays bf16
    "model.layers.2.feed_forward.gate",    # the router
    "model.layers.3.self_attn.qkv_proj",   # layer 3 is a conv layer: no attention
])
def test_layers_the_checkpoint_keeps_bf16_are_declined(prefix):
    assert _cfg()._lookup_bpw(prefix) is None


def _routed_experts(names):
    """A vLLM experts layer as `get_quant_method` receives it, with LFM2's ckpt names."""
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts as cls
    except ImportError:                                             # pragma: no cover
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE as cls
    layer = cls.__new__(cls)
    layer.moe_config = object()
    layer.ckpt_gate_proj_name, layer.ckpt_down_proj_name, layer.ckpt_up_proj_name = names
    return layer


@requires_vllm
def test_the_moe_container_is_claimed_with_lfm2s_names():
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod
    method = _cfg().get_quant_method(_routed_experts(("w1", "w2", "w3")),
                                     "model.layers.2.feed_forward.experts")
    assert isinstance(method, GLQFusedMoEMethod)


@requires_vllm
def test_a_checkpoint_spelling_the_layer_will_not_load_is_refused():
    """A GLQ build that saved LFM2's experts as gate_proj/up_proj/down_proj would serve
    uninitialised experts: vLLM looks for w1/w2/w3 and skips the rest. Refuse by name."""
    from glq_vllm.config import GLQvLLMConfig
    lb = {k.replace(".w1", ".gate_proj").replace(".w3", ".up_proj").replace(".w2", ".down_proj")
          if ".experts." in k else k: v for k, v in _layer_bpw().items()}
    cfg = GLQvLLMConfig.from_config({
        "bpw": 4, "codebook": "trellis", "variant": "3inst", "trellis_layout": "kernel",
        "layer_bpw": lb})
    with pytest.raises(ValueError, match="w1") as exc:
        cfg.get_quant_method(_routed_experts(("w1", "w2", "w3")),
                             "model.layers.2.feed_forward.experts")
    assert "gate_proj" in str(exc.value)


@requires_vllm
def test_the_default_names_still_serve_the_default_checkpoints():
    """gemma-4 / Flash-Next / GLM: gate_proj/up_proj/down_proj on both sides."""
    from glq_vllm.config import GLQvLLMConfig
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod
    lb = {f"model.layers.0.mlp.experts.{e}.{p}": 3
          for e in range(4) for p in ("gate_proj", "up_proj", "down_proj")}
    cfg = GLQvLLMConfig.from_config({"bpw": 3, "codebook": "trellis", "variant": "3inst",
                                     "trellis_layout": "kernel", "layer_bpw": lb})
    method = cfg.get_quant_method(
        _routed_experts(("gate_proj", "down_proj", "up_proj")), "model.layers.0.mlp.experts")
    assert isinstance(method, GLQFusedMoEMethod)


@requires_vllm
def test_nemotrons_non_gated_names_still_serve():
    """vLLM's NemotronH passes `ckpt_names=("up_proj", "down_proj", None)`."""
    from glq_vllm.config import GLQvLLMConfig
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod
    lb = {f"backbone.layers.1.mixer.experts.{e}.{p}": 3
          for e in range(4) for p in ("up_proj", "down_proj")}
    cfg = GLQvLLMConfig.from_config({"bpw": 3, "codebook": "trellis", "variant": "3inst",
                                     "trellis_layout": "kernel", "layer_bpw": lb})
    method = cfg.get_quant_method(
        _routed_experts(("up_proj", "down_proj", None)), "model.layers.1.mixer.experts")
    assert isinstance(method, GLQFusedMoEMethod)
