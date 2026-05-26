"""Quantized KV cache for GLQ inference — no external dependencies.

Plugs into HuggingFace's QuantizedCache framework. Backends:

  * ``int8``       — per-channel INT8 absmax (KIVI-style, default;
                     the original GLQQuantizedCache behavior). 8 bpw.
  * ``e8_strict``  — E8 lattice (Conway-Sloane parity) per group of 8,
                     with optional RVQ second stage. 2 or 4 bpw.
  * ``e8_relaxed`` — Relaxed E8 (D̃8, no parity) per group of 8. Wins on
                     sub-Gaussian Hadamard-rotated KV per microbench in
                     ``tests/test_kv_e8.py``. 2 or 4 bpw.

Mixed-precision per layer is supported via ``bpw_map``: pass a dict
``{layer_idx: bpw}`` to ``GLQQuantizedCache`` and each cache layer gets
the indicated bpw. Allowed bpw values: 16 (no compression, kept as
fp16/bf16), 8 (INT8), 4 (E8 RVQ), 2 (E8 stage-1). Layers absent from
the map fall back to the ``quant_method`` / ``n_stages`` defaults.

Requires transformers >= 4.45 (QuantizedCache API).

Usage:
    from glq.kv_cache import GLQQuantizedCache

    # Default INT8 (8 bpw):
    cache = GLQQuantizedCache(model.config)

    # Uniform E8 relaxed at 4 bpw:
    cache = GLQQuantizedCache(model.config, quant_method="e8_relaxed",
                              n_stages=2)

    # Mixed: sensitive layers at 4 bpw, the rest at 2 bpw (E8 relaxed),
    # using the JSON map from ``glq.cli.quantize_kv``:
    import json
    bpw_map = {int(k): v for k, v in json.load(open("kv_bpw_map.json")).items()}
    cache = GLQQuantizedCache(model.config, quant_method="e8_relaxed",
                              bpw_map=bpw_map)

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
# Cache keyed by (quant_method, device, target_size). v0.5 Phase 5.2
# Branch B needs the parametric ``target_size`` for the smem-resident
# attention kernel (typically 4096); the legacy paths leave it None for
# the default 65 536-entry codebook.
_CODEBOOK_CACHE: dict = {}


def _check_available():
    if not _HAS_QUANTIZED_CACHE:
        raise ImportError(
            "GLQQuantizedCache requires transformers >= 4.45 with "
            "QuantizedCache support. Please upgrade: pip install -U transformers"
        )


def _get_codebook(quant_method: str, device: str | None = None,
                  target_size: int | None = None):
    """Lazily build (and cache) a strict / relaxed codebook.

    Defaults to ``cuda`` when available so construction (especially
    ``_compute_opt_scale`` / ``_compute_resid_scale``, which do hundreds
    of millions of 8D NN distance computations) runs on the GPU. On
    CPU these take ~3 min for the 65,536-entry codebook; on GPU it's
    sub-second.

    ``target_size``: codebook entry count. ``None`` means default
    (65 536). v0.5 Phase 5.2 Branch B passes ``target_size=4096`` on
    the KV side for the smem-resident attention kernel — see
    ``glq/codebook.py:E8ShellCodebook.__init__`` for the shell-sorted
    truncation contract.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (quant_method, str(device), target_size)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    if quant_method == "e8_strict":
        from glq.codebook import E8ShellCodebook
        cb = E8ShellCodebook(device=device, verbose=False,
                             target_size=target_size)
    elif quant_method == "e8_relaxed":
        from glq.codebook_relaxed import E8RelaxedCodebook
        cb = E8RelaxedCodebook(device=device, verbose=False,
                               target_size=target_size)
    else:
        raise ValueError(f"unknown quant_method: {quant_method!r}")
    _CODEBOOK_CACHE[key] = cb
    return cb


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


class FP16PassthroughLayer(QuantizedLayer if _HAS_QUANTIZED_CACHE else object):
    """No-op quantized layer for layers we want to keep at full precision.

    Used as the bpw=16 entry in a mixed-precision ``bpw_map`` so the
    sensitivity allocator can leave the most-sensitive layers uncompressed.
    Encoded form is the input tensor itself.
    """

    def __init__(self, **kwargs):
        _check_available()
        super().__init__(**kwargs)

    def _quantize(self, tensor, axis):
        return tensor

    def _dequantize(self, qtensor):
        return qtensor


class E8QuantizedLayer(QuantizedLayer if _HAS_QUANTIZED_CACHE else object):
    """E8 (strict or relaxed) lattice quantization for a single cache layer.

    The HF QuantizedLayer API passes an ``axis`` argument intended for
    per-channel absmax scaling; E8 always scales per group of 8 along the
    last dim, so the argument is accepted but ignored. The codebook is
    shared across all layers (singleton via ``_get_codebook``).
    """

    def __init__(self, quant_method: str, n_stages: int = 1,
                 secondary_stages: int = 0, **kwargs):
        _check_available()
        super().__init__(**kwargs)
        from glq.kv_e8 import E8KVQuantizer
        # Build the codebook on cuda when available — saves ~3 minutes
        # per fresh process vs CPU construction (opt_scale/resid_scale
        # NN sweeps).
        cb = _get_codebook(quant_method)
        # 3-bpw and 5-bpw recipes need the 256-entry small codebook on
        # the same device as the primary.
        small_cb = cb.make_small() if secondary_stages > 0 else None
        self._quantizer = E8KVQuantizer(
            cb, n_stages=n_stages,
            secondary_codebook=small_cb,
            secondary_stages=secondary_stages)
        self._quant_method = quant_method
        self._n_stages = n_stages
        self._secondary_stages = secondary_stages
        # Codebook may still need a per-call move if the input device
        # differs from the build device (e.g. multi-GPU).
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


# Allowed bpw values for the ``bpw_map`` mixed-precision API and the menu
# used by ``glq.kv_sensitivity.allocate_kv_bpw``.
#
# 2/4/6 are pure E8 (1/2/3 primary 16-bit stages). 3/5/7 mix in one
# 8-bit secondary stage on top of 1/2/3 primary stages (n_primary*16+8
# bits / 8 dims). 8 is INT8 absmax, 16 is fp16 passthrough.
VALID_KV_BPW = (2, 3, 4, 5, 6, 7, 8, 16)


def _build_layer_for_bpw(bpw: int, *, e8_method: str, residual_length: int,
                         q_group_size: int):
    """Pick the right ``QuantizedLayer`` subclass for the requested bpw.

    bpw=16 → FP16 passthrough; bpw=8 → INT8 absmax; bpw=6/4/2 → pure
    E8 RVQ (3/2/1 primary stages); bpw=3 → E8 primary + one 8-bit
    secondary stage. ``e8_method`` selects strict vs relaxed for the
    E8 paths.
    """
    if bpw == 16:
        return FP16PassthroughLayer(
            nbits=16, axis_key=0, axis_value=0,
            q_group_size=q_group_size, residual_length=residual_length)
    if bpw == 8:
        return INT8QuantizedLayer(
            nbits=8, axis_key=0, axis_value=0,
            q_group_size=q_group_size, residual_length=residual_length)
    if bpw in (2, 3, 4, 5, 6, 7):
        if e8_method not in ("e8_strict", "e8_relaxed"):
            raise ValueError(
                f"e8_method must be 'e8_strict' or 'e8_relaxed' for "
                f"bpw={bpw}, got {e8_method!r}")
        # (n_primary, n_secondary) — pure-primary stages give 2/4/6 bpw,
        # appending one 8-bit secondary stage gives 3/5/7 bpw.
        n_primary, n_secondary = {
            2: (1, 0), 3: (1, 1),
            4: (2, 0), 5: (2, 1),
            6: (3, 0), 7: (3, 1),
        }[bpw]
        return E8QuantizedLayer(
            quant_method=e8_method, n_stages=n_primary,
            secondary_stages=n_secondary,
            nbits=bpw, axis_key=-1, axis_value=-1,
            q_group_size=8, residual_length=residual_length)
    raise ValueError(
        f"bpw={bpw} is not supported; allowed: {VALID_KV_BPW}")


class GLQQuantizedCache:
    """Quantized KV cache using pure PyTorch (no quanto/hqq needed).

    Two ways to specify the per-layer bpw:

    1. **Uniform** (``quant_method`` + ``n_stages``) — all layers use
       the same backend. Default ``"int8"`` → 8 bpw across all layers.
    2. **Mixed-precision** (``bpw_map``) — ``{layer_idx: bpw}`` selects
       a different layer backend per cache layer. Allowed bpw:
       ``{2, 4, 8, 16}``. ``quant_method`` controls which E8 variant
       (strict vs relaxed) is used for the 2/4 bpw entries.

    Args:
        config: Model's ``PreTrainedConfig``.
        quant_method: For uniform: backend selector. For mixed: E8
            variant. ``"int8"`` (default), ``"e8_strict"``, ``"e8_relaxed"``.
        nbits: Bits per coordinate for the uniform INT8 path (must be 8).
            Ignored for E8 paths — use ``n_stages``.
        n_stages: For uniform E8 paths, 1 = 2 bpw, 2 = 4 bpw (RVQ).
        bpw_map: ``{int layer_idx: int bpw}`` for mixed-precision. Layers
            not present in the map fall back to the uniform setting.
        q_group_size: Group size for INT8 (64 default). Ignored for E8.
        residual_length: Tokens kept in full precision before quantizing.
    """

    def __new__(cls, config, *, quant_method: str = "int8", nbits=8,
                n_stages: int = 1, bpw_map: dict | None = None,
                q_group_size=64, residual_length=128):
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

        # Default bpw for layers absent from bpw_map (or all layers when
        # bpw_map is None): derived from quant_method + n_stages.
        if quant_method == "int8":
            default_bpw = 8
        else:
            # E8: n_stages → bpw mapping. 1→2 (single codebook lookup),
            # 2→4 (RVQ), 3→6 (3-stage RVQ).
            try:
                default_bpw = {1: 2, 2: 4, 3: 6}[n_stages]
            except KeyError:
                raise ValueError(
                    f"n_stages must be 1, 2, or 3 for E8 paths, "
                    f"got {n_stages}")
        # For the E8 branch, e8_method is what we requested; for INT8
        # uniform, force e8_method to "e8_relaxed" as a placeholder
        # (won't be consulted since default_bpw=8 routes through INT8).
        e8_method = quant_method if quant_method != "int8" else "e8_relaxed"

        bpw_map = dict(bpw_map) if bpw_map else {}
        for idx, bpw in bpw_map.items():
            if bpw not in VALID_KV_BPW:
                raise ValueError(
                    f"bpw_map[{idx}]={bpw} not in {VALID_KV_BPW}")

        layers = []
        for i in range(n_layers):
            bpw = bpw_map.get(i, default_bpw)
            layers.append(_build_layer_for_bpw(
                bpw, e8_method=e8_method, residual_length=residual_length,
                q_group_size=q_group_size))

        # Construct a real QuantizedCache, bypassing backend string requirement
        from transformers.cache_utils import Cache
        obj = QuantizedCache.__new__(QuantizedCache)
        Cache.__init__(obj, layers=layers)
        return obj
