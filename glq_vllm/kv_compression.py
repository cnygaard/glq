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
import os
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

# --- Phase 5.3 Stage 2b/2c sidecar (one E8PagedKVCache per layer, by ptr) ---
# Stage 2b runs alongside vLLM's fp16 paged cache (validation only).
# Stage 2c-1 adds an attention-forward hook so attention READS go
# through our sidecar instead of vLLM's roundtripped fp16 cache. With
# 2c-1 active, attention output is bit-exact to the 2b path (same
# E8 round-trip math, just routed through a different storage layout).
# Stage 2c key: (data_ptr, head_size). vLLM v1's KVCacheTensor design
# shares one paged buffer across multiple logical layers (e.g. Gemma-4
# full+sliding pairs share a ptr but have different head_size views).
# Keying by (ptr, head_size) gives us one sidecar per layer-type-per-buffer
# without losing data when shape changes between layers.
_e8_sidecar: dict[tuple[int, int], Any] = {}
# Stage 2c-2b: when active, vLLM allocates its paged buffer at the
# compressed size (via the E8 spec override). The original triton
# reshape_and_cache_flash assumes fp16 layout — calling it on the
# compressed buffer crashes — so the patched fn skips it.
_compressed_allocation_active: bool = False
_original_get_kv_cache_spec: Any = None
_original_triton_get_kv_cache_shape: Any = None
_sidecar_enabled: bool = False
_sidecar_read_enabled: bool = False      # Stage 2c-1: route attention reads
_sidecar_write_counter: int = 0
_sidecar_read_counter: int = 0
_original_unified: Any = None            # vllm.v1.attention.backends.triton_attn.unified_attention


_E8_BPW_RECIPES = {
    # bpw : (n_primary 16-bit stages, n_secondary 8-bit stages)
    2: (1, 0),
    3: (1, 1),   # 16+8 = 24 bits / 8 dims
    4: (2, 0),
    5: (2, 1),   # 16+16+8 = 40 bits / 8 dims
    6: (3, 0),
    7: (3, 1),   # 16+16+16+8 = 56 bits / 8 dims
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
                f"allowed: 2/3/4/5/6/7 (E8 RVQ), 8 (INT8), 16 (passthrough)")
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


def _sidecar_write(key: torch.Tensor, value: torch.Tensor,
                   key_cache: torch.Tensor, slot_mapping: torch.Tensor,
                   value_cache: torch.Tensor | None = None) -> None:
    """Phase 5.3 Stage 2b: also push the same K/V into our E8PagedKVCache.

    Allocates the sidecar lazily the first time we see a given
    ``key_cache.data_ptr()`` — vLLM allocates one paged buffer per
    layer at engine start, and the pointer is stable thereafter.

    Layers at bpw=8 (INT8) and bpw=16 (passthrough) are skipped — the
    E8 paged layout only covers 2..7 bpw recipes.
    """
    global _sidecar_write_counter
    if not _sidecar_enabled:
        return
    quant, bpw = _quantizer_for(key_cache)
    # Uniform mode: _quantizer_for returns bpw=None. Recover it from
    # the active config so every layer uses the same uniform bpw.
    if bpw is None:
        bpw = _active_config.get("bpw")
    if bpw not in {2, 3, 4, 5, 6, 7}:
        return
    if quant is None:
        quant = _quantizer
    if quant is None:
        return
    from glq_vllm.e8_paged_cache import (
        E8PagedKVCache,
        write_kv,
    )

    if key_cache.dim() != 4:
        return  # unexpected layout — skip silently
    num_blocks, block_size, num_kv_heads, cache_last_dim = key_cache.shape
    # Under Stage 2c-2b the cache's last dim is the *compressed*
    # element count, not head_size. Always derive head_size from the
    # input K tensor (which the model produces at the model's real
    # head_size, regardless of how vLLM stores it).
    if key.dim() == 3:
        head_size = key.shape[-1]
    elif key.dim() == 2:
        # Flattened [tokens, num_kv_heads * head_size]. Without a
        # compressed-aware fallback we'd need an out-of-band source
        # of head_size; for now skip this case (rare in the v1 path).
        return
    else:
        return
    if head_size % 8 != 0:
        return

    ptr = key_cache.data_ptr()
    bin_key = (ptr, head_size)
    cache = _e8_sidecar.get(bin_key)
    if cache is None:
        if _compressed_allocation_active and value_cache is not None:
            # Stage 2c-2c: build views into vLLM's already-allocated
            # buffer. No new memory — same bytes interpreted as our
            # E8 layout (idx1 / idx_s1 / scale / etc.).
            cache = E8PagedKVCache.from_paged_storage(
                key_cache, value_cache,
                head_size=head_size, bpw=bpw, dtype=key_cache.dtype)
        else:
            cache = E8PagedKVCache.alloc(
                num_blocks=num_blocks, block_size=block_size,
                num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
                device=key_cache.device, dtype=key_cache.dtype)
        _e8_sidecar[bin_key] = cache
        try:
            shared = " (views vLLM bytes)" if cache.shares_vllm_storage else ""
            bytes_str = ("0 MB shared" if cache.shares_vllm_storage
                         else f"{cache.bytes_total()/1e6:.1f}MB")
            print(f"[glq_vllm.kv_compression] sidecar alloc bin "
                  f"#{len(_e8_sidecar)}: bpw={bpw} shape="
                  f"{(num_blocks, block_size, num_kv_heads, head_size)} "
                  f"bytes={bytes_str}{shared}",
                  flush=True)
        except Exception:
            pass

    # Key is already 3D here (guarded above when deriving head_size).
    try:
        write_kv(quant, key, value, cache, slot_mapping)
        _sidecar_write_counter += 1
    except Exception as e:
        # Sidecar is best-effort during Stage 2b; the main monkey-patch
        # is the correctness path. Don't take the engine down for a
        # shape edge case — log once and disable for this layer.
        bin_key = (ptr, head_size)
        if bin_key in _e8_sidecar:
            del _e8_sidecar[bin_key]
        try:
            print(f"[glq_vllm.kv_compression] sidecar write failed for "
                  f"layer ptr={ptr:#x} head_size={head_size}: "
                  f"{type(e).__name__}: {e}",
                  flush=True)
        except Exception:
            pass


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
        _sidecar_write(key, value, key_cache, slot_mapping, value_cache)
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
        _sidecar_write(key, value, key_cache, slot_mapping, value_cache)
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
        _sidecar_write(key, value, key_cache, slot_mapping, value_cache)
        if _compressed_allocation_active:
            # vLLM's cache is now compressed bytes our code never
            # reads; the original fp16 write would crash. Sidecar has
            # the real data — return without touching vLLM's buffer.
            return
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


# ---------------------------------------------------------------- #
# Phase 5.3 Stage 2b sidecar control
# ---------------------------------------------------------------- #

def enable_sidecar() -> None:
    """Turn on the E8PagedKVCache sidecar (allocated lazily per layer
    on the first ``_sidecar_write`` call). The main monkey-patch
    (Phase 5.1) continues to populate vLLM's fp16 cache so attention
    still works through vLLM's existing path; the sidecar runs alongside.

    Call after ``enable()`` (or set ``GLQ_KV_E8_SIDECAR=1`` env var)."""
    global _sidecar_enabled
    _sidecar_enabled = True


def enable_sidecar_read() -> None:
    """Stage 2c-1: route attention K/V reads through the sidecar.

    Monkey-patches ``triton_attn.unified_attention`` so that when
    vLLM's TritonAttentionImpl.forward calls it with the layer's
    paged cache, we swap in the decompressed fp16 K/V from the E8
    sidecar instead. The original ``unified_attention`` kernel
    handles attention; the only thing different is where its K, V
    inputs came from.

    Bit-exact in steady state with the Stage 2b path because both
    feed the kernel ``dequantize(quantize(K))`` — just stored
    differently. No memory savings yet (sidecar still parallel to
    vLLM's cache); Stage 2c-2 shrinks vLLM's allocation by
    declaring our compressed page size in ``AttentionSpec``.

    Requires ``enable_sidecar()`` first."""
    global _sidecar_read_enabled, _original_unified
    if not _sidecar_enabled:
        raise RuntimeError(
            "enable_sidecar_read requires enable_sidecar to be active")
    try:
        import vllm.v1.attention.backends.triton_attn as ta
    except Exception as e:
        print(f"[glq_vllm.kv_compression] cannot enable sidecar read: "
              f"triton_attn unavailable ({e})", flush=True)
        return
    if _original_unified is None:
        _original_unified = ta.unified_attention
    ta.unified_attention = _patched_unified_attention
    _sidecar_read_enabled = True


def disable_sidecar() -> None:
    global _sidecar_enabled, _sidecar_read_enabled
    _sidecar_enabled = False
    _sidecar_read_enabled = False
    _e8_sidecar.clear()
    if _original_unified is not None:
        try:
            import vllm.v1.attention.backends.triton_attn as ta
            ta.unified_attention = _original_unified
        except Exception:
            pass


def _patched_unified_attention(*, q, k, v, **kwargs):
    """Stage 2c-1 attention-read hook.

    Replaces vLLM's K/V paged-cache tensors with decompressed fp16
    versions sourced from the E8 sidecar before delegating to
    ``unified_attention``. Falls back to vLLM's cache on any failure
    (sidecar not allocated for this layer, quantizer not found, etc.)
    so a Stage 2c-1 bug never breaks generation — the main monkey-patch
    keeps vLLM's fp16 cache valid as a safety net.
    """
    global _sidecar_read_counter
    if not _sidecar_read_enabled:
        return _original_unified(q=q, k=k, v=v, **kwargs)
    # Real head_size comes from q (the model produces q at full
    # head_size regardless of how vLLM stores cache). k.shape[-1] is
    # unreliable under Stage 2c-2b because vLLM's cache uses the
    # compressed last-dim.
    real_head_size = q.shape[-1]
    bin_key = (k.data_ptr(), real_head_size)
    cache = _e8_sidecar.get(bin_key)
    if cache is None:
        return _original_unified(q=q, k=k, v=v, **kwargs)
    quant, _ = _quantizer_for_bpw(cache.bpw)
    if quant is None:
        return _original_unified(q=q, k=k, v=v, **kwargs)
    block_table = kwargs.get("block_table")
    try:
        from glq_vllm.e8_paged_cache import (
            gather_kv_to_paged_fp16,
            gather_kv_to_paged_fp16_fused,
            remap_block_table,
        )
        # Stage 3b: opt-in fused Triton kernel collapses the 5-take gather
        # + dequant into one launch per side (eliminates ~10k Python-side
        # aten::index calls per decode at ctx=4k). Falls through to the
        # slow path automatically on CPU or if Triton is missing; the
        # outer try/except below catches any fused-path bugs and falls
        # back to slow.
        _gather = (gather_kv_to_paged_fp16_fused
                   if os.environ.get("GLQ_KV_E8_FUSED_GATHER", "0") == "1"
                   else gather_kv_to_paged_fp16)
        if block_table is not None and block_table.numel() > 0:
            # Scoped: decompress only the blocks this batch references.
            # Bounds the transient fp16 workspace by # unique referenced
            # blocks instead of cache.num_blocks (typically ~100x smaller
            # at non-saturated load).
            unique_blocks = block_table.flatten().unique().long()
            k_e8, v_e8 = _gather(
                quant, cache, block_indices=unique_blocks)
            new_block_table = remap_block_table(
                block_table, unique_blocks, cache.num_blocks)
            kwargs = {**kwargs, "block_table": new_block_table}
        else:
            k_e8, v_e8 = _gather(quant, cache)
        if k_e8.dtype != k.dtype:
            k_e8 = k_e8.to(k.dtype)
            v_e8 = v_e8.to(v.dtype)
    except Exception as e:
        try:
            print(f"[glq_vllm.kv_compression] sidecar-read fallback "
                  f"(ptr={k.data_ptr():#x} head_size={k.shape[-1]}): "
                  f"{type(e).__name__}: {e}", flush=True)
        except Exception:
            pass
        return _original_unified(q=q, k=k, v=v, **kwargs)
    _sidecar_read_counter += 1
    return _original_unified(q=q, k=k_e8, v=v_e8, **kwargs)


def sidecar_read_count() -> int:
    return _sidecar_read_counter


# ---------------------------------------------------------------- #
# Phase 5.3 Stage 2c-2b — compressed allocation in vLLM
# ---------------------------------------------------------------- #

def enable_compressed_allocation() -> None:
    """Stage 2c-2b: declare compressed page size to vLLM's allocator.

    Hooks:
    * ``Attention.get_kv_cache_spec`` returns ``E8FullAttentionSpec`` /
      ``E8SlidingWindowSpec`` with ``real_page_size_bytes`` set to our
      compressed page size, so vLLM allocates a ~3-4x smaller paged
      buffer.
    * ``TritonAttentionBackend.get_kv_cache_shape`` returns the
      compact ``(num_blocks, 2, elems_per_block_per_kv)`` shape so
      ``_reshape_kv_cache``'s view math succeeds at the smaller byte
      budget.
    * ``_patched_triton`` skips the original fp16 write — the cache
      buffer is now compressed bytes our code never touches; only
      the sidecar carries real K/V.

    Requires ``enable_sidecar()`` and ``enable_sidecar_read()``.
    """
    global _compressed_allocation_active
    global _original_get_kv_cache_spec, _original_triton_get_kv_cache_shape

    if not _sidecar_enabled or not _sidecar_read_enabled:
        raise RuntimeError(
            "enable_compressed_allocation requires sidecar + read hooks "
            "(call enable_sidecar / enable_sidecar_read first)")

    from glq_vllm.e8_kv_spec import (
        E8FullAttentionSpec,
        E8SlidingWindowSpec,
        compressed_kv_cache_shape,
    )
    if E8FullAttentionSpec is None or E8SlidingWindowSpec is None:
        raise RuntimeError(
            "E8 spec subclasses unavailable (vllm import failed)")

    # -- 1) patch Attention.get_kv_cache_spec --
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec, SlidingWindowSpec,
        )
    except ImportError as e:
        raise RuntimeError(
            f"compressed allocation requires vllm: {e}") from e

    if _original_get_kv_cache_spec is None:
        _original_get_kv_cache_spec = Attention.get_kv_cache_spec

    def _e8_get_kv_cache_spec(self, vllm_config):
        spec = _original_get_kv_cache_spec(self, vllm_config)
        bpw = _active_config.get("bpw")
        if bpw is None or bpw not in (2, 3, 4, 5, 6, 7):
            return spec
        if isinstance(spec, FullAttentionSpec):
            return E8FullAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=spec.head_size_v,
                dtype=spec.dtype,
                sliding_window=spec.sliding_window,
                attention_chunk_size=spec.attention_chunk_size,
                bpw=bpw,
            )
        if isinstance(spec, SlidingWindowSpec):
            return E8SlidingWindowSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                dtype=spec.dtype,
                sliding_window=spec.sliding_window,
                bpw=bpw,
            )
        return spec

    Attention.get_kv_cache_spec = _e8_get_kv_cache_spec

    # -- 1b) register E8 spec subclasses with the manager dispatch --
    # vLLM's get_manager_for_kv_cache_spec uses type(spec) as a dict
    # key, so subclassing isn't enough — we must add the entries.
    from vllm.v1.core.single_type_kv_cache_manager import (
        spec_manager_map,
        FullAttentionManager,
        SlidingWindowManager,
    )
    spec_manager_map[E8FullAttentionSpec] = FullAttentionManager
    spec_manager_map[E8SlidingWindowSpec] = SlidingWindowManager

    # -- 2) patch TritonAttentionBackend.get_kv_cache_shape --
    try:
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionBackend,
        )
    except ImportError as e:
        raise RuntimeError(
            f"compressed allocation requires triton_attn backend: {e}") from e

    if _original_triton_get_kv_cache_shape is None:
        _original_triton_get_kv_cache_shape = (
            TritonAttentionBackend.get_kv_cache_shape)

    @staticmethod
    def _e8_triton_get_kv_cache_shape(num_blocks, block_size,
                                       num_kv_heads, head_size,
                                       cache_dtype_str="auto"):
        bpw = _active_config.get("bpw")
        if bpw is None or bpw not in (2, 3, 4, 5, 6, 7):
            return _original_triton_get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size,
                cache_dtype_str)
        # dtype_size = 2 (bf16/fp16) for now — TODO: read from spec
        return compressed_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size,
            bpw=bpw, dtype_size=2)

    TritonAttentionBackend.get_kv_cache_shape = _e8_triton_get_kv_cache_shape

    _compressed_allocation_active = True
    print("[glq_vllm.kv_compression] compressed allocation hooks "
          "installed (Stage 5.3-2c-2b): Attention.get_kv_cache_spec + "
          "TritonAttentionBackend.get_kv_cache_shape", flush=True)


def disable_compressed_allocation() -> None:
    global _compressed_allocation_active
    _compressed_allocation_active = False
    if _original_get_kv_cache_spec is not None:
        try:
            from vllm.model_executor.layers.attention.attention import (
                Attention)
            Attention.get_kv_cache_spec = _original_get_kv_cache_spec
        except Exception:
            pass
    if _original_triton_get_kv_cache_shape is not None:
        try:
            from vllm.v1.attention.backends.triton_attn import (
                TritonAttentionBackend)
            TritonAttentionBackend.get_kv_cache_shape = (
                _original_triton_get_kv_cache_shape)
        except Exception:
            pass


def sidecar_stats() -> list[dict[str, Any]]:
    """Snapshot of every allocated bin's E8PagedKVCache for inspection."""
    return [
        {
            "ptr": ptr,
            "key_head_size": key_head_size,
            "bpw": c.bpw,
            "n_primary": c.n_primary,
            "n_secondary": c.n_secondary,
            "num_blocks": c.num_blocks,
            "block_size": c.block_size,
            "num_kv_heads": c.num_kv_heads,
            "head_size": c.head_size,
            "bytes_total": c.bytes_total(),
            "bytes_per_token_per_head": c.bytes_per_token_per_head(),
        }
        for (ptr, key_head_size), c in _e8_sidecar.items()
    ]


def sidecar_write_count() -> int:
    return _sidecar_write_counter


def gather_sidecar_layer(ptr: int, head_size: int,
                         slot_mapping: torch.Tensor):
    """Decompress a slice of a sidecar bin back to fp16 K, V.

    The bin is identified by (ptr, head_size) since one vLLM
    KVCacheTensor can be shared by multiple logical layers with
    different head_size views.
    """
    from glq_vllm.e8_paged_cache import gather_kv
    cache = _e8_sidecar[(ptr, head_size)]
    quant, _ = _quantizer_for_bpw(cache.bpw)
    return gather_kv(quant, cache, slot_mapping)


def _quantizer_for_bpw(bpw: int):
    """Look up a quantizer by bpw alone (uniform or mixed mode)."""
    if _bpw_list:
        return _bpw_quantizers.get(bpw), bpw
    return _quantizer, bpw
