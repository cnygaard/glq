"""Quantized KV cache for GLQ inference — no external dependencies.

Plugs into HuggingFace's QuantizedCache framework. Three backends:

  * ``int8``       — per-channel INT8 absmax (KIVI-style, default;
                     the original GLQQuantizedCache behavior). 8 bpw.
  * ``e8_strict``  — E8 lattice (Conway-Sloane parity) per group of 8,
                     with optional RVQ second stage. 2 or 4 bpw.
  * ``e8_relaxed`` — Relaxed E8 (D̃8, no parity) per group of 8. Wins on
                     sub-Gaussian Hadamard-rotated KV per microbench in
                     ``tests/test_kv_e8.py``. 2 or 4 bpw.

Requires transformers >= 4.45 (QuantizedCache API).

Usage:
    from glq.kv_cache import GLQQuantizedCache

    # Default INT8 (8 bpw):
    cache = GLQQuantizedCache(model.config)

    # E8 relaxed at 4 bpw (recommended for long context):
    cache = GLQQuantizedCache(model.config, quant_method="e8_relaxed",
                              n_stages=2)

    output = model.generate(**inputs, past_key_values=cache)
"""

import torch

try:
    from transformers.cache_utils import QuantizedCache, QuantizedLayer
    _HAS_QUANTIZED_CACHE = True
except ImportError:
    _HAS_QUANTIZED_CACHE = False

# Lazy codebook + E8KVQuantizer imports — these pull in
# ``glq.codebook_kernel`` which optionally imports triton; we don't want
# to force that import just for the INT8 path.
_E8_STRICT_CB = None
_E8_RELAXED_CB = None


def _check_available():
    if not _HAS_QUANTIZED_CACHE:
        raise ImportError(
            "GLQQuantizedCache requires transformers >= 4.45 with "
            "QuantizedCache support. Please upgrade: pip install -U transformers"
        )


def _get_codebook(quant_method: str, device: str = "cpu"):
    """Lazily build (and cache) the strict / relaxed codebook."""
    global _E8_STRICT_CB, _E8_RELAXED_CB
    if quant_method == "e8_strict":
        if _E8_STRICT_CB is None or str(_E8_STRICT_CB.device) != str(device):
            from glq.codebook import E8ShellCodebook
            _E8_STRICT_CB = E8ShellCodebook(device=device, verbose=False)
        return _E8_STRICT_CB
    if quant_method == "e8_relaxed":
        if _E8_RELAXED_CB is None or str(_E8_RELAXED_CB.device) != str(device):
            from glq.codebook_relaxed import E8RelaxedCodebook
            _E8_RELAXED_CB = E8RelaxedCodebook(device=device, verbose=False)
        return _E8_RELAXED_CB
    raise ValueError(f"unknown quant_method: {quant_method!r}")


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


class E8QuantizedLayer(QuantizedLayer if _HAS_QUANTIZED_CACHE else object):
    """E8 (strict or relaxed) lattice quantization for a single cache layer.

    The HF QuantizedLayer API passes an ``axis`` argument intended for
    per-channel absmax scaling; E8 always scales per group of 8 along the
    last dim, so the argument is accepted but ignored. The codebook is
    shared across all layers (singleton via ``_get_codebook``).
    """

    def __init__(self, quant_method: str, n_stages: int = 1, **kwargs):
        _check_available()
        super().__init__(**kwargs)
        from glq.kv_e8 import E8KVQuantizer
        cb = _get_codebook(quant_method, device="cpu")
        self._quantizer = E8KVQuantizer(cb, n_stages=n_stages)
        self._quant_method = quant_method
        self._n_stages = n_stages
        # Codebook is moved to the input device lazily on first _quantize.
        self._cb_device_synced = False

    def _maybe_move_codebook(self, target_device):
        if self._cb_device_synced:
            return
        cb = self._quantizer.codebook
        # Move codebook buffers onto the device where data lives.
        if cb.codebook.device != target_device:
            cb.codebook = cb.codebook.to(target_device)
            cb.codebook_norms = cb.codebook_norms.to(target_device)
            cb.codebook_half = cb.codebook_half.to(target_device)
            cb.codebook_half_t = cb.codebook_half_t.to(target_device)
            cb.codebook_norms_half = cb.codebook_norms_half.to(target_device)
            cb.codebook_packed = cb.codebook_packed.to(target_device)
            cb.device = target_device
        self._cb_device_synced = True

    def _quantize(self, tensor, axis):
        self._maybe_move_codebook(tensor.device)
        return self._quantizer.quantize(tensor)

    def _dequantize(self, qtensor):
        return self._quantizer.dequantize(qtensor)


def attach_kv_cache(model, *, quant_method: str = "int8", n_stages: int = 1,
                    residual_length: int = 128):
    """Attach a quantized-KV-cache factory to ``model``.

    Replaces (or installs) ``model._glq_kv_cache_factory`` so subsequent
    calls to ``cache = model._glq_kv_cache_factory()`` build a fresh
    ``GLQQuantizedCache`` with the chosen method. Use at inference time
    when the same checkpoint should run with different KV compression
    strategies (e.g. bf16 vs INT8 vs E8-relaxed at 4 bpw).

    Example:
        from glq.kv_cache import attach_kv_cache
        attach_kv_cache(model, quant_method="e8_relaxed", n_stages=2)
        cache = model._glq_kv_cache_factory()
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)

    Args:
        model: A HuggingFace ``PreTrainedModel`` (any architecture).
        quant_method: ``"int8"``, ``"e8_strict"``, or ``"e8_relaxed"``.
        n_stages: For E8 paths, 1 = 2 bpw, 2 = 4 bpw (RVQ). Ignored for
            INT8 (always 8 bpw).
        residual_length: Tokens kept in fp16 before compressing
            (KIVI-style sliding window). 128 default.
    """
    config = model.config
    model._glq_kv_cache_factory = (
        lambda: GLQQuantizedCache(
            config, quant_method=quant_method, n_stages=n_stages,
            residual_length=residual_length))
    return model


class GLQQuantizedCache:
    """Quantized KV cache using pure PyTorch (no quanto/hqq needed).

    Args:
        config: Model's ``PreTrainedConfig``.
        quant_method: ``"int8"`` (default), ``"e8_strict"``, ``"e8_relaxed"``.
        nbits: Bits per coordinate for INT8 path (must be 8 there).
            Ignored for E8 paths — use ``n_stages`` instead.
        n_stages: For E8 paths, 1 = 2 bpw, 2 = 4 bpw (RVQ). Default 1.
        q_group_size: Group size for INT8 (64 default). Ignored for E8
            (always 8).
        residual_length: Tokens kept in full precision before quantizing
            (128 default). Same meaning across all methods.
    """

    def __new__(cls, config, *, quant_method: str = "int8", nbits=8,
                n_stages: int = 1, q_group_size=64, residual_length=128):
        _check_available()

        if quant_method == "int8":
            if nbits != 8:
                raise ValueError(
                    f"int8 path only supports nbits=8, got {nbits}")
        elif quant_method not in ("e8_strict", "e8_relaxed"):
            raise ValueError(
                f"unknown quant_method {quant_method!r}; expected one of "
                "'int8', 'e8_strict', 'e8_relaxed'")

        cfg = config.get_text_config(decoder=True) if hasattr(
            config, 'get_text_config') else config
        n_layers = getattr(cfg, 'num_hidden_layers', 32)

        if quant_method == "int8":
            layers = [
                INT8QuantizedLayer(
                    nbits=nbits, axis_key=0, axis_value=0,
                    q_group_size=q_group_size, residual_length=residual_length,
                )
                for _ in range(n_layers)
            ]
        else:
            # E8 path: nbits placeholder (QuantizedLayer requires it).
            layers = [
                E8QuantizedLayer(
                    quant_method=quant_method, n_stages=n_stages,
                    nbits=4 if n_stages == 2 else 2,
                    axis_key=-1, axis_value=-1,
                    q_group_size=8, residual_length=residual_length,
                )
                for _ in range(n_layers)
            ]
        # Construct a real QuantizedCache, bypassing backend string requirement
        from transformers.cache_utils import Cache
        obj = QuantizedCache.__new__(QuantizedCache)
        Cache.__init__(obj, layers=layers)
        return obj
