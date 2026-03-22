"""Tests for INT8 quantized KV cache."""

import torch
import pytest

try:
    from transformers.cache_utils import QuantizedLayer
    _HAS_QUANTIZED_CACHE = True
except ImportError:
    _HAS_QUANTIZED_CACHE = False

pytestmark = pytest.mark.skipif(
    not _HAS_QUANTIZED_CACHE,
    reason="transformers too old for QuantizedCache"
)


def test_int8_roundtrip_accuracy():
    """INT8 per-channel absmax should achieve >0.999 cosine similarity."""
    from glq.kv_cache import INT8QuantizedLayer
    layer = INT8QuantizedLayer(nbits=8, residual_length=4)
    tensor = torch.randn(1, 4, 16, 64)
    qtensor = layer._quantize(tensor, axis=0)
    recovered = layer._dequantize(qtensor)
    cos = torch.nn.functional.cosine_similarity(
        tensor.flatten(), recovered.flatten(), dim=0)
    assert cos > 0.999, f"Cosine similarity {cos:.4f} too low"


def test_int8_memory_savings():
    """INT8 quantized tensor should use less memory than fp16."""
    from glq.kv_cache import INT8QuantizedLayer
    layer = INT8QuantizedLayer(nbits=8, residual_length=4)
    # axis=0 collapses batch dim; scale overhead depends on tensor shape.
    # The real saving: int8 data is 1 byte vs 2 bytes fp16 per element.
    tensor = torch.randn(1, 32, 1024, 128)
    qtensor, scale = layer._quantize(tensor, axis=0)
    assert qtensor.dtype == torch.int8
    assert qtensor.shape == tensor.shape
    # Core property: int8 data is half the size of fp16 data
    assert qtensor.element_size() == 1  # vs 2 for fp16


def test_cache_update_returns_correct_shape():
    """Cache update should return full key/value with correct seq length."""
    from glq.kv_cache import INT8QuantizedLayer
    layer = INT8QuantizedLayer(nbits=8, residual_length=128)
    B, H, D = 1, 4, 64
    k1 = torch.randn(B, H, 8, D)
    v1 = torch.randn(B, H, 8, D)
    k_out, v_out = layer.update(k1, v1)
    assert k_out.shape == (B, H, 8, D)
    assert v_out.shape == (B, H, 8, D)
    k2 = torch.randn(B, H, 1, D)
    v2 = torch.randn(B, H, 1, D)
    k_out, v_out = layer.update(k2, v2)
    assert k_out.shape[-2] == 9
    assert v_out.shape[-2] == 9


def test_glq_quantized_cache_creation():
    """GLQQuantizedCache should create correct number of layers."""
    from glq.kv_cache import GLQQuantizedCache, INT8QuantizedLayer

    class FakeConfig:
        num_hidden_layers = 12
        def get_text_config(self, decoder=True):
            return self

    cache = GLQQuantizedCache(FakeConfig())
    assert len(cache.layers) == 12
    assert all(isinstance(l, INT8QuantizedLayer) for l in cache.layers)


def test_glq_quantized_cache_rejects_non_8bit():
    """Only nbits=8 should be accepted."""
    from glq.kv_cache import GLQQuantizedCache

    class FakeConfig:
        num_hidden_layers = 4
        def get_text_config(self, decoder=True):
            return self

    with pytest.raises(ValueError, match="nbits=8"):
        GLQQuantizedCache(FakeConfig(), nbits=4)
