"""INT8 quantized KV cache for GLQ inference — no external dependencies.

Plugs into HuggingFace's QuantizedCache framework with a pure-PyTorch
INT8 per-channel absmax quantization backend. Halves KV cache memory
vs fp16, critical for long-context serving.

Requires transformers >= 4.45 (QuantizedCache API).

Usage:
    from glq.kv_cache import GLQQuantizedCache
    cache = GLQQuantizedCache(model.config)
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch

try:
    from transformers.cache_utils import QuantizedCache, QuantizedLayer
    _HAS_QUANTIZED_CACHE = True
except ImportError:
    _HAS_QUANTIZED_CACHE = False


def _check_available():
    if not _HAS_QUANTIZED_CACHE:
        raise ImportError(
            "GLQQuantizedCache requires transformers >= 4.45 with "
            "QuantizedCache support. Please upgrade: pip install -U transformers"
        )


class INT8QuantizedLayer(QuantizedLayer if _HAS_QUANTIZED_CACHE else object):
    """Per-channel INT8 absmax quantization for a single cache layer."""

    def __init__(self, **kwargs):
        _check_available()
        super().__init__(**kwargs)

    def _quantize(self, tensor, axis):
        scale = tensor.abs().amax(dim=axis, keepdim=True).clamp(min=1e-12) / 127
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def _dequantize(self, qtensor):
        quantized, scale = qtensor
        return quantized.to(scale.dtype) * scale


class GLQQuantizedCache:
    """INT8 quantized KV cache using pure PyTorch (no quanto/hqq needed).

    Args:
        config: Model's PreTrainedConfig.
        nbits: Quantization bits (8 only for now).
        q_group_size: Group size for quantization (64 default).
        residual_length: Tokens kept in full precision before quantizing (128 default).
    """

    def __new__(cls, config, nbits=8, q_group_size=64, residual_length=128):
        _check_available()

        if nbits != 8:
            raise ValueError(f"GLQQuantizedCache only supports nbits=8, got {nbits}")

        cfg = config.get_text_config(decoder=True) if hasattr(config, 'get_text_config') else config
        n_layers = getattr(cfg, 'num_hidden_layers', 32)

        layers = [
            INT8QuantizedLayer(
                nbits=nbits, axis_key=0, axis_value=0,
                q_group_size=q_group_size, residual_length=residual_length,
            )
            for _ in range(n_layers)
        ]
        # Construct a real QuantizedCache, bypassing backend string requirement
        from transformers.cache_utils import Cache
        obj = QuantizedCache.__new__(QuantizedCache)
        Cache.__init__(obj, layers=layers)
        return obj
