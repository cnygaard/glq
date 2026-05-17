"""Regression: torch.compile(fullgraph=True) must not break on GLQ ops.

vLLM 0.20's piecewise CUDA-graph capture (default since v0.3.2 for
weight-only GLQ) requires every GLQ op to be a registered
``torch.library`` custom op so dynamo can trace through it as an opaque
kernel boundary. This test captures a toy module exercising the
linear-path ops under ``fullgraph=True``; if a future change
re-introduces a raw pybind / Triton call that dynamo can't trace, the
test raises ``BackendCompilerFailed`` at CI time.

KV-side ops (``gather_kv_paged_dequant`` / ``scatter_kv_paged_quant``)
are exercised by ``test_glq_compile_fullgraph_kv.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import pytest
import torch

try:
    import glq_vllm  # registers torch.ops.glq.* via _ensure_registered()
    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False

requires_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for compile test"
)


@pytest.fixture(scope="module", autouse=True)
def _init_vllm_tp():
    """vLLM's ``BasevLLMParameter`` (and the custom_op registration code
    that lives in ``glq_vllm.custom_ops``) requires a TP group + a
    current ``VllmConfig`` context to be set."""
    if not _HAS_VLLM:
        yield
        return
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    with set_current_vllm_config(VllmConfig()):
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method="tcp://127.0.0.1:29611",
                local_rank=0, backend="gloo",
            )
        ensure_model_parallel_initialized(1, 1)
        yield


@requires_vllm
@requires_gpu
def test_input_rht_fullgraph_capture():
    """``torch.ops.glq.input_rht`` (C++ path, n_pad ≤ 16384) must trace
    cleanly under ``fullgraph=True``. Catches regressions like
    re-introducing a raw pybind call inside ``_glq_apply_shard``."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "input_rht"):
        pytest.skip("input_rht not registered (CUDA ext missing)")

    n_pad, log_n = 1024, 10
    B, in_features = 2, n_pad
    device = "cuda"
    x = torch.randn(B, in_features, dtype=torch.float16, device=device)
    sv = torch.randn(n_pad, dtype=torch.float16, device=device).sign()
    out = torch.empty(B, n_pad, dtype=torch.float32, device=device)
    rsqrt_n = 1.0 / (n_pad ** 0.5)

    def call_op(x, sv, out):
        torch.ops.glq.input_rht(
            x, sv, out, in_features, in_features, rsqrt_n, n_pad, log_n)
        return out

    compiled = torch.compile(call_op, fullgraph=True)
    # If anything in the path breaks dynamo, this raises BackendCompilerFailed.
    y = compiled(x, sv, out)
    assert y.shape == (B, n_pad)


@requires_vllm
@requires_gpu
def test_input_rht_triton_fullgraph_capture():
    """``torch.ops.glq.input_rht_triton`` (Triton path, n_pad > 16384).
    Test exercises a smaller pow2 shape since the dispatch is shape-only;
    the op surface is what we're validating, not the kernel's optimised
    range."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "input_rht_triton"):
        pytest.skip("input_rht_triton not registered")

    n_pad, log_n = 1024, 10
    B, in_features = 2, n_pad
    device = "cuda"
    x = torch.randn(B, in_features, dtype=torch.float16, device=device)
    sv = torch.randn(n_pad, dtype=torch.float16, device=device).sign()
    out = torch.empty(B, n_pad, dtype=torch.float32, device=device)
    rsqrt_n = 1.0 / (n_pad ** 0.5)

    def call_op(x, sv, out):
        torch.ops.glq.input_rht_triton(
            x, sv, out, in_features, in_features, rsqrt_n, n_pad, log_n)
        return out

    compiled = torch.compile(call_op, fullgraph=True)
    y = compiled(x, sv, out)
    assert y.shape == (B, n_pad)


@requires_vllm
@requires_gpu
def test_output_rht_triton_fullgraph_capture():
    """Symmetric test for ``output_rht_triton``."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "output_rht_triton"):
        pytest.skip("output_rht_triton not registered")

    m_pad, log_m = 1024, 10
    B, out_features = 2, m_pad
    device = "cuda"
    y_rht = torch.randn(B, m_pad, dtype=torch.float32, device=device)
    su = torch.randn(m_pad, dtype=torch.float16, device=device).sign()
    y = torch.empty(B, out_features, dtype=torch.float16, device=device)
    rsqrt_m = 1.0 / (m_pad ** 0.5)

    def call_op(y_rht, su, y):
        torch.ops.glq.output_rht_triton(
            y_rht, su, y, out_features, m_pad, log_m, rsqrt_m)
        return y

    compiled = torch.compile(call_op, fullgraph=True)
    result = compiled(y_rht, su, y)
    assert result.shape == (B, out_features)
    assert result.dtype == torch.float16
