"""vLLM CPU-platform integration (phase 2 of the CPU fused-decode work).

These tests need an importable vllm (any install, including the +cpu wheel or a source
checkout) but NO GPU and NO CUDA extension — that is the point: they pin the behaviors
that make `--quantization glq` viable on vLLM's CPU backend:

* `embedding_dequant` registers under the CPU dispatch key even when the CUDA extension
  is absent (its real impl is pure torch) — the gemma-4 PLE path.
* The MoE quant method refuses loudly on the CPU platform (every GLQ MoE path bottoms
  out in CUDA kernels; a mid-load failure would be far harder to read).
* fp32 joins the supported activation dtypes (the CPU platform's list is bf16-first).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# importorskip on the SUBMODULE: a bare source checkout at the repo root shadows `vllm`
# as an empty namespace package, so `import vllm` succeeding proves nothing.
pytest.importorskip("vllm.platforms", reason="no installed vllm (wheel needed)")

from vllm.platforms import current_platform  # noqa: E402

pytestmark = pytest.mark.skipif(
    getattr(current_platform, "device_type", "") != "cpu",
    reason="these gates target the CPU platform (dispatch key CPU)")


def test_embedding_dequant_registers_without_the_cuda_ext():
    from glq_vllm import custom_ops
    custom_ops._ensure_registered()
    assert hasattr(torch.ops, "glq") and hasattr(torch.ops.glq, "embedding_dequant"), \
        "embedding_dequant must register on the CPU platform without the CUDA extension"


def test_embedding_dequant_cpu_matches_the_pure_torch_impl():
    from glq.quantized_linear import _dequant_embedding_rows
    from glq_vllm import custom_ops
    custom_ops._ensure_registered()

    torch.manual_seed(0)
    vocab, n_pad, dim = 32, 64, 64
    qidxs = torch.randint(0, 255, (vocab, n_pad // 8), dtype=torch.uint8)
    sv = torch.where(torch.rand(n_pad) < 0.5, -1.0, 1.0).half()
    wscale = torch.tensor(0.5)
    codebook = torch.randn(256, 8).half()
    ids = torch.tensor([3, 7, 1])

    ref = _dequant_embedding_rows(ids, qidxs, sv, wscale, codebook, None, None, None,
                                  n_pad, dim, 1.0, torch.float16)
    out = torch.ops.glq.embedding_dequant(ids, qidxs, sv, wscale, codebook, None, None,
                                          None, n_pad, dim, 1.0, torch.float16)
    assert torch.equal(out, ref)


def test_moe_quant_method_refuses_on_cpu():
    from glq_vllm.config import GLQvLLMConfig
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts as _MoELayer)
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _MoELayer
        except ImportError:
            pytest.skip("no MoE layer class in this vllm")

    cfg = GLQvLLMConfig(bpw=4, codebook="trellis", variant="3inst")
    layer = _MoELayer.__new__(_MoELayer)          # isinstance without __init__ plumbing
    with pytest.raises(NotImplementedError, match="not servable on the CPU platform"):
        GLQvLLMConfig.get_quant_method(cfg, layer, prefix="model.layers.0.mlp.experts")


def test_fp32_is_a_supported_act_dtype():
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig(bpw=4, codebook="trellis", variant="3inst")
    assert torch.float32 in cfg.get_supported_act_dtypes()
