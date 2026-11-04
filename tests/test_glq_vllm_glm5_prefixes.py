"""A GLM-5.3 GLQ checkpoint resolves in vLLM 0.32 exactly where it was quantized.

The quantizer writes ``layer_bpw`` in checkpoint spelling (``model.language_model.layers.N``)
and quantizes only the MLPs: the dense MLPs of layers 0-2, and each MoE layer's shared
expert and routed experts. vLLM asks with its own prefix (``language_model.model.layers.N``)
and merged names (``gate_up_proj``). Both directions matter:

* an MLP that fails to resolve is built bf16, and the GLQ buffers in the checkpoint have
  nowhere to load;
* an attention projection that resolves would be claimed by GLQ, but vLLM builds GLM's
  MLA, KDA and indexer with ``quant_config=None`` and the checkpoint stores them bf16.

``layer_bpw`` here is built from the recorded checkpoint keys, keeping what the profile
quantizes, so it carries the real expert count and the real MTP layer (45).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("vllm.model_executor.layers.fused_moe",
                    reason="no installed vllm (wheel needed)")

from model_shapes import load  # noqa: E402

N_LAYERS = 45                   # decoder layers; the MTP layer is 45
VLLM = "language_model.model.layers"
CKPT = "model.language_model.layers."


def _layer_bpw():
    """What the quantizer writes: every MLP linear in the decoder layers, minus the router.

    Mirrors the GLM profile (``skip_linears = ('.self_attn.',)``, MLPs only) and the
    streaming loop, which never visits ``layers.45``.
    """
    out = {}
    for key in load("glm_5_3_flash_bf16").keys():
        if not (key.startswith(CKPT) and key.endswith("_proj.weight") and ".mlp." in key):
            continue
        if int(key[len(CKPT):].split(".", 1)[0]) >= N_LAYERS:
            continue
        out[key[:-len(".weight")]] = 4
    return out


@pytest.fixture(scope="module")
def cfg():
    from glq_vllm.config import GLQvLLMConfig
    return GLQvLLMConfig(bpw=4, layer_bpw=_layer_bpw(), codebook="trellis",
                         variant="3inst", trellis_layout="kernel")


def test_the_fixture_is_what_the_profile_quantizes():
    """3 dense MLPs x 3 + 42 MoE layers x (3 shared + 288 x 3 routed)."""
    assert len(_layer_bpw()) == 3 * 3 + 42 * (3 + 288 * 3)


@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("proj", ["gate_up_proj", "down_proj"])
def test_dense_mlps_resolve(cfg, layer, proj):
    assert cfg._lookup_bpw(f"{VLLM}.{layer}.mlp.{proj}") == 4


@pytest.mark.parametrize("layer", [3, 44])
@pytest.mark.parametrize("proj", ["gate_up_proj", "down_proj"])
def test_shared_experts_resolve(cfg, layer, proj):
    assert cfg._lookup_bpw(f"{VLLM}.{layer}.mlp.shared_experts.{proj}") == 4


@pytest.mark.parametrize("layer", [3, 7, 44])
def test_routed_experts_are_claimed(cfg, layer):
    assert cfg._moe_is_quantized(f"{VLLM}.{layer}.mlp.experts")


def test_the_dense_layers_have_no_moe(cfg):
    assert not cfg._moe_is_quantized(f"{VLLM}.0.mlp.experts")


def test_the_mtp_layer_stays_unquantized(cfg):
    """Copied bf16 from the source. Under speculative decoding vLLM must build it with its
    own unquantized methods, which only happens when GLQ declines it."""
    assert not cfg._moe_is_quantized(f"{VLLM}.45.mlp.experts")
    assert cfg._lookup_bpw(f"{VLLM}.45.mlp.shared_experts.gate_up_proj") is None


# KDA layers (0, 4, ...) and MLA + indexer layers (3, 7, ..., 43), in vLLM's module names.
ATTENTION = ["self_attn.o_proj", "self_attn.in_proj_qkvbfg_a", "self_attn.fused_qkv_a_proj",
             "self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.indexer.wq_b",
             "self_attn.indexer.wk_weights_proj"]


@pytest.mark.parametrize("layer", [0, 3, 4, 43])
@pytest.mark.parametrize("name", ATTENTION)
def test_attention_is_never_claimed(cfg, layer, name):
    assert cfg._lookup_bpw(f"{VLLM}.{layer}.{name}") is None


def test_the_router_and_vision_tower_are_not_claimed(cfg):
    assert cfg._lookup_bpw(f"{VLLM}.3.mlp.gate") is None
    assert cfg._lookup_bpw("visual.blocks.0.mlp.gate_up_proj") is None
    assert cfg._lookup_bpw("visual.blocks.0.attn.qkv") is None


def test_the_shared_expert_is_not_fused_into_the_moe(cfg):
    """If vLLM fused the shared expert into FusedMoE, its weights would load as expert 288
    of a GLQ MoE built for 288. That fusion is AITER-only (ROCm); on CUDA it stays off."""
    from vllm.model_executor.layers.fused_moe.utils import resolve_layer_fused_shared_expert
    assert not resolve_layer_fused_shared_expert(cfg, f"{VLLM}.3.mlp")
