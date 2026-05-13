"""Monkey-patch vLLM 0.20's KV cache write to round-trip through E8 quantizer.

Phase 5.1a: quality-only probe. The cache memory stays fp16 (no VRAM win
— that needs C++ kernel work, Phase 5.3), but every K/V tensor written
to the paged cache gets compressed → decompressed first. Lets us
measure the quality impact of E8 KV compression at vLLM speed
(continuous batching, paged attention, fast prefill).

Usage (recommended — env var works across the parent + engine subprocess):
    GLQ_KV_QUANT=e8_relaxed:2 python my_script.py     # 4 bpw RVQ

Usage (programmatic, parent process only — won't reach the v1 engine):
    import glq_vllm.kv_compression as glqkv
    glqkv.enable(quant_method="e8_relaxed", n_stages=2)
    glqkv.disable()  # restore original behavior

vLLM 0.20 has three distinct call paths for K/V cache writes. We patch
all of them so the round-trip catches every backend:

  1. ``vllm._custom_ops.reshape_and_cache_flash`` — Python wrapper used
     by flash_attn / tree_attn / rocm_aiter_unified_attn backends.
  2. ``torch.ops._C_cache_ops.reshape_and_cache_flash`` — the underlying
     C++ op, called directly by flashinfer / flex_attention backends
     (they skip the Python wrapper).
  3. ``vllm.v1.attention.ops.triton_reshape_and_cache_flash`` — Triton
     implementation used by triton_attn / rocm_attn. vLLM forces
     ``TRITON_ATTN`` for Gemma-4 (heterogeneous head_dim 256/512) so
     this is the path Gemma-4 actually exercises.

All cache reads (paged_attention, flash_attn_varlen_func) see the lossy
fp16 values transparently — no decompression wrapper needed.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Any

import torch


_state_lock = threading.Lock()
_original_reshape: Any = None
_original_torch_op: Any = None
_original_triton: Any = None
_quantizer: Any = None
_bpw_quantizers: dict[int, Any] = {}     # bpw -> E8KVQuantizer (or None for passthrough)
_bpw_list: list[int] = []                # bpw per layer, in vLLM's call order
_ptr_to_layer_idx: dict[int, int] = {}   # kv_cache.data_ptr() -> position in _bpw_list
_active_config: dict[str, Any] = {}
_call_counter: int = 0  # debug: counts how many times the patched fn fires


_E8_BPW_RECIPES = {
    2: (1, 0),   # 1 primary, 0 secondary
    3: (1, 1),   # 1 primary, 1 secondary (8-bit) → 16+8=24 bits/8 dims
    4: (2, 0),
    6: (3, 0),
}


def _get_quantizer(quant_method: str, n_stages: int,
                   secondary_stages: int = 0):
    """Lazily build an E8KVQuantizer with the given primary/secondary stages."""
    from glq.kv_cache import _get_codebook
    from glq.kv_e8 import E8KVQuantizer
    cb = _get_codebook(quant_method)  # cuda-default after 666ffcd
    small = cb.make_small() if secondary_stages > 0 else None
    return E8KVQuantizer(cb, n_stages=n_stages,
                        secondary_codebook=small,
                        secondary_stages=secondary_stages)


def _build_bpw_quantizers(quant_method: str, bpws: set[int]):
    """One quantizer per distinct E8 bpw in the map.

    bpw → (n_primary, n_secondary): see ``_E8_BPW_RECIPES``. bpw=8 uses
    inline INT8 absmax (no persistent state). bpw=16 is passthrough."""
    out = {}
    for b in bpws:
        if b in _E8_BPW_RECIPES:
            n_p, n_s = _E8_BPW_RECIPES[b]
            out[b] = _get_quantizer(quant_method, n_p, secondary_stages=n_s)
        elif b in (8, 16):
            out[b] = None
        else:
            raise ValueError(
                f"bpw={b} not supported in vLLM KV patch; "
                f"allowed: 2/3/4/6 (E8 RVQ), 8 (INT8), 16 (passthrough)")
    return out


def _int8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """KIVI-style per-channel INT8 absmax round-trip along the last dim.

    Matches the math of ``glq.kv_cache.INT8QuantizedLayer`` but
    streaming: no persistent scale buffers needed since the cache
    memory is still fp16 — we just want the lossy round-trip on top
    of whatever K/V was about to be written."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = (x / scale).round().clamp(-127, 127)
    return q * scale


def _quantizer_for(kv_cache: torch.Tensor):
    """Pick the right (quantizer, bpw) for this layer based on the paged
    cache's data pointer. Builds the ptr→layer-idx mapping lazily by
    first-seen order — vLLM walks layers in HF topology order, so the
    n-th distinct pointer we see corresponds to ``_bpw_list[n]``."""
    if not _bpw_list:
        return _quantizer, None  # uniform mode (legacy)
    ptr = kv_cache.data_ptr()
    idx = _ptr_to_layer_idx.get(ptr)
    if idx is None:
        idx = len(_ptr_to_layer_idx)
        if idx >= len(_bpw_list):
            # More layers than the map covers — fall back to no compression.
            return None, 16
        _ptr_to_layer_idx[ptr] = idx
    bpw = _bpw_list[idx]
    return _bpw_quantizers.get(bpw), bpw


def _roundtrip_kv(key: torch.Tensor, value: torch.Tensor,
                  kv_cache: torch.Tensor):
    """Round-trip K, V through the quantizer chosen for this layer.

    Uniform mode: uses ``_quantizer`` for every call. Mixed-bpw mode:
    looks up per-layer quantizer via ``_quantizer_for``."""
    head_size = kv_cache.shape[-1]
    quant, bpw = _quantizer_for(kv_cache)
    if quant is None and bpw in (16, None):
        # 16 bpw or no compression configured for this layer.
        if _quantizer is None and not _bpw_list:
            return key, value  # patch not initialised yet
        if bpw == 16:
            return key, value
    orig_shape_k = key.shape
    orig_shape_v = value.shape
    if key.dim() == 2:
        # [num_tokens, num_kv_heads * head_size] — unflatten the heads
        # so the codebook grouping is per-head.
        num_kv_heads = key.shape[-1] // head_size
        key = key.reshape(-1, num_kv_heads, head_size)
        value = value.reshape(-1, num_kv_heads, head_size)
    if bpw == 8:
        # INT8 absmax round-trip (per-channel along head dim).
        key_rt = _int8_roundtrip(key)
        val_rt = _int8_roundtrip(value)
    else:
        q = quant if quant is not None else _quantizer
        key_rt = q.dequantize(q.quantize(key))
        val_rt = q.dequantize(q.quantize(value))
    return key_rt.reshape(orig_shape_k), val_rt.reshape(orig_shape_v)


def _maybe_log(key, value):
    """Print on the very first call and every 500 calls thereafter, so the
    engine log shows that the patch is exercised without being noisy."""
    if _call_counter == 1 or _call_counter % 500 == 0:
        try:
            print(f"[glq_vllm.kv_compression] hook fire "
                  f"call #{_call_counter} key.shape={tuple(key.shape)} "
                  f"value.shape={tuple(value.shape)} dtype={key.dtype}",
                  flush=True)
        except Exception:
            pass


def _patched_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Compress→decompress K, V then delegate to the original Python wrapper."""
    global _call_counter
    _call_counter += 1
    _maybe_log(key, value)
    if _quantizer is not None or _bpw_list:
        key, value = _roundtrip_kv(key, value, key_cache)
    return _original_reshape(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )


def _patched_torch_op(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Replacement for ``torch.ops._C_cache_ops.reshape_and_cache_flash``.

    vLLM's flashinfer / flex_attention backends call the C++ op directly
    (``torch.ops._C_cache_ops.reshape_and_cache_flash(...)``), bypassing
    the Python wrapper in ``vllm._custom_ops``. We patch the op namespace
    too so those code paths also see the E8 round-trip.
    """
    global _call_counter
    _call_counter += 1
    _maybe_log(key, value)
    if _quantizer is not None or _bpw_list:
        key, value = _roundtrip_kv(key, value, key_cache)
    return _original_torch_op(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )


def _patched_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    """Replacement for ``triton_reshape_and_cache_flash``.

    vLLM auto-selects TRITON_ATTN for Gemma-4 (heterogeneous head dims).
    The triton backend lives at
    ``vllm.v1.attention.ops.triton_reshape_and_cache_flash`` and is
    imported by name in ``triton_attn.py`` / ``rocm_attn.py``. We patch
    the source module AND the importing modules.
    """
    global _call_counter
    _call_counter += 1
    _maybe_log(key, value)
    if _quantizer is not None or _bpw_list:
        key, value = _roundtrip_kv(key, value, key_cache)
    return _original_triton(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )


def _backend_modules_to_patch():
    """Return any vllm.v1 attention backend modules that may have
    `from vllm._custom_ops import reshape_and_cache_flash` style imports
    binding the original name at module load."""
    import importlib
    mods = []
    for name in (
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.backends.flashinfer",
        "vllm.v1.attention.backends.rocm_attn",
        "vllm.v1.attention.backends.tree_attn",
    ):
        try:
            mods.append(importlib.import_module(name))
        except Exception:
            pass
    return mods


def _triton_modules_to_patch():
    """vllm.v1 backends that ``from vllm.v1.attention.ops.triton_reshape_and_cache_flash
    import triton_reshape_and_cache_flash``. Patch the source module AND
    the importing modules so direct bindings are caught."""
    import importlib
    mods = []
    for name in (
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash",  # source
        "vllm.v1.attention.backends.triton_attn",
        "vllm.v1.attention.backends.rocm_attn",
        "vllm.v1.attention.backends.flash_attn_diffkv",
    ):
        try:
            mods.append(importlib.import_module(name))
        except Exception:
            pass
    return mods


def enable(quant_method: str = "e8_relaxed", n_stages: int = 1,
           bpw_map: dict[int, int] | None = None) -> None:
    """Activate E8 KV-cache compression (round-trip mode).

    Two modes:

    * **Uniform** (``bpw_map=None``) — every layer uses the same
      ``quant_method`` and ``n_stages``. ``n_stages`` 1/2/3 → 2/4/6 bpw.
    * **Mixed-precision** (``bpw_map={layer_idx: bpw}``) — each layer
      gets its own bpw, exactly mirroring ``glq.kv_cache.bpw_map``.
      Bpw values: 2/4/6 (E8 RVQ via ``quant_method``), 8 (INT8 absmax),
      16 (passthrough). ``n_stages`` is ignored in this mode.

    Layer-index resolution is by call order: the n-th distinct paged
    cache pointer we observe corresponds to ``sorted(bpw_map.keys())[n]``.
    This is robust under vLLM's continuous batching because the
    per-layer cache buffers are stable across the run.
    """
    global _original_reshape, _original_torch_op, _original_triton
    global _quantizer, _bpw_quantizers, _bpw_list, _ptr_to_layer_idx
    global _active_config
    with _state_lock:
        import vllm._custom_ops as ops
        if _original_reshape is None:
            _original_reshape = ops.reshape_and_cache_flash
        if _original_torch_op is None:
            _original_torch_op = torch.ops._C_cache_ops.reshape_and_cache_flash

        # Reset per-layer state before reinitialising.
        _ptr_to_layer_idx.clear()
        if bpw_map is not None:
            # Mixed-precision mode: build one quantizer per distinct E8 bpw,
            # remember the per-layer bpw list in layer-key order.
            _bpw_list = [bpw_map[k] for k in sorted(bpw_map.keys())]
            _bpw_quantizers = _build_bpw_quantizers(
                quant_method, set(_bpw_list))
            _quantizer = None
        else:
            _bpw_list = []
            _bpw_quantizers = {}
            _quantizer = _get_quantizer(quant_method, n_stages)

        # 1) Python wrapper used by flash_attn backend.
        ops.reshape_and_cache_flash = _patched_reshape_and_cache_flash
        for mod in _backend_modules_to_patch():
            if hasattr(mod, "reshape_and_cache_flash"):
                mod.reshape_and_cache_flash = _patched_reshape_and_cache_flash

        # 2) torch.ops C++ op directly, used by flashinfer / flex_attention.
        try:
            torch.ops._C_cache_ops.reshape_and_cache_flash = _patched_torch_op
        except Exception as e:
            print(f"[glq_vllm.kv_compression] could not patch torch.ops: {e}",
                  flush=True)

        # 3) Triton-backed op used by TRITON_ATTN backend (Gemma-4, sliding
        #    window). vLLM forces this for heterogeneous head dims.
        try:
            from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
                triton_reshape_and_cache_flash as _orig_tri,
            )
            if _original_triton is None:
                _original_triton = _orig_tri
            for mod in _triton_modules_to_patch():
                if hasattr(mod, "triton_reshape_and_cache_flash"):
                    mod.triton_reshape_and_cache_flash = _patched_triton
        except Exception as e:
            print(f"[glq_vllm.kv_compression] could not patch triton: {e}",
                  flush=True)

        if bpw_map is not None:
            avg_bpw = sum(_bpw_list) / len(_bpw_list)
            _active_config = {
                "mode": "mixed",
                "quant_method": quant_method,
                "n_layers": len(_bpw_list),
                "bpw_avg": avg_bpw,
                "bpw_hist": {b: _bpw_list.count(b)
                             for b in sorted(set(_bpw_list))},
            }
        else:
            _active_config = {
                "mode": "uniform",
                "quant_method": quant_method,
                "n_stages": n_stages,
                "bpw": _quantizer.bpw,
            }


def disable() -> None:
    """Restore vLLM's original ``reshape_and_cache_flash``."""
    global _original_reshape, _original_torch_op, _original_triton
    global _quantizer, _bpw_quantizers, _bpw_list, _ptr_to_layer_idx
    global _active_config
    with _state_lock:
        if _original_reshape is not None:
            import vllm._custom_ops as ops
            ops.reshape_and_cache_flash = _original_reshape
        if _original_torch_op is not None:
            try:
                torch.ops._C_cache_ops.reshape_and_cache_flash = _original_torch_op
            except Exception:
                pass
        if _original_triton is not None:
            for mod in _triton_modules_to_patch():
                if hasattr(mod, "triton_reshape_and_cache_flash"):
                    mod.triton_reshape_and_cache_flash = _original_triton
        _quantizer = None
        _bpw_quantizers.clear()
        _bpw_list.clear()
        _ptr_to_layer_idx.clear()
        _active_config = {}


@contextlib.contextmanager
def compressed(quant_method: str = "e8_relaxed", n_stages: int = 1):
    """Context manager: ``with compressed(...): generate(...)``."""
    enable(quant_method=quant_method, n_stages=n_stages)
    try:
        yield
    finally:
        disable()


def is_active() -> bool:
    return _quantizer is not None


def active_config() -> dict[str, Any]:
    return dict(_active_config)


def call_count() -> int:
    """How many times has the patched reshape_and_cache_flash been called?

    Useful to verify the monkey-patch is actually hit (vLLM v1 backends
    sometimes call ``torch.ops._C_cache_ops.*`` directly, bypassing the
    Python wrapper).
    """
    return _call_counter


def reset_call_count() -> None:
    global _call_counter
    _call_counter = 0
