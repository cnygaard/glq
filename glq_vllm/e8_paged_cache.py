"""Stage-1 prototype of a vLLM-shaped paged KV cache backed by E8 indices.

This is the *layout-and-correctness* prototype for Phase 5.3 — it
allocates a paged cache at the **compressed size** (one cache slot per
token stores indices + scale, never fp16 K/V), so the bytes-on-card
genuinely shrink. The compute paths are pure PyTorch; the eventual
CUDA kernels (Stage 3) will replace ``write_kv`` and ``gather_kv``
with one fused launch each, but the byte layout fixed here is what
those kernels will target.

Layout
------
Mirrors vLLM's flash_attn paged cache:

    cache_shape = [num_blocks, block_size, num_kv_heads, head_size]

For our compressed form we replace the last (head_size) axis with one
or more per-group axes of size ``n_groups = head_size // 8``:

    idx1   : int16  [num_blocks, block_size, num_kv_heads, n_groups]
    idx2   : int16  [.] (only when bpw >= 4 — second primary stage)
    idx3   : int16  [.] (only when bpw >= 6 — third primary stage)
    idx_s1 : uint8  [.] (only when bpw is in {3, 5, 7} — secondary 8-bit)
    scale  : fp16   [.] (always present)

K and V each get their own bag of these tensors (they're stored
independently in vLLM and they will be here too). Bit-cost per group:

    bpw   primary stages   secondary stages   bits/group
    2     1                0                  16 +  0 + 16 = 32
    3     1                1                  16 +  8 + 16 = 40
    4     2                0                  32 +  0 + 16 = 48
    5     2                1                  32 +  8 + 16 = 56
    6     3                0                  48 +  0 + 16 = 64
    7     3                1                  48 +  8 + 16 = 72

Bit-exact equivalence with HF path
----------------------------------
``E8KVQuantizer.quantize`` produces the dict {idx1, idx2?, idx3?,
idx_s1?, scale, shape, dtype}. ``E8KVQuantizer.dequantize`` consumes
that same dict. ``write_kv`` here calls ``quantize`` and stores the
fields into the paged buffers; ``gather_kv`` reads them back and
hands the assembled dict to ``dequantize``. So the math is the
existing math — only the storage layout is new.

Memory check (per-token-per-head, head_size=128, n_groups=16):
   fp16 baseline: 128 * 2          = 256 B
   2 bpw        : (16 + 16) * 2    =  64 B  (0.25x)
   3 bpw        : 16*2 + 16*1 + 16*2 = 80 B (0.31x)
   4 bpw        : 16*2 + 16*2 + 16*2 = 96 B (0.38x)

These match the cache_bytes measurements from
``benchmarks/kv_compress_memory.py`` end-to-end on Gemma-4-E4B-it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


_BPW_RECIPE = {
    2: (1, 0),
    3: (1, 1),
    4: (2, 0),
    5: (2, 1),
    6: (3, 0),
    7: (3, 1),
}


@dataclass
class E8PagedKVCache:
    """vLLM-shaped paged cache backed by E8 indices + scale.

    Always exposes ``k_idx1, v_idx1, k_scale, v_scale``. Higher-bpw
    recipes attach additional ``k_idx{2,3}/k_idx_s1`` (and the V
    twins) — None for tiers that don't need them.
    """

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    n_groups: int
    bpw: int
    n_primary: int
    n_secondary: int
    device: torch.device
    dtype: torch.dtype          # original K/V dtype (fp16/bf16)

    k_idx1: torch.Tensor
    v_idx1: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    k_idx2: Optional[torch.Tensor] = None
    v_idx2: Optional[torch.Tensor] = None
    k_idx3: Optional[torch.Tensor] = None
    v_idx3: Optional[torch.Tensor] = None
    k_idx_s1: Optional[torch.Tensor] = None
    v_idx_s1: Optional[torch.Tensor] = None

    @classmethod
    def alloc(cls, *, num_blocks: int, block_size: int,
              num_kv_heads: int, head_size: int, bpw: int,
              device: str | torch.device = "cuda",
              dtype: torch.dtype = torch.float16) -> "E8PagedKVCache":
        if head_size % 8 != 0:
            raise ValueError(
                f"head_size ({head_size}) must be a multiple of 8")
        if bpw not in _BPW_RECIPE:
            raise ValueError(
                f"bpw={bpw} not supported; allowed {sorted(_BPW_RECIPE)}")
        n_primary, n_secondary = _BPW_RECIPE[bpw]
        n_groups = head_size // 8
        device = torch.device(device)
        layout = (num_blocks, block_size, num_kv_heads, n_groups)

        def _z(t):
            return torch.zeros(layout, dtype=t, device=device)

        out = cls(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=num_kv_heads, head_size=head_size,
            n_groups=n_groups, bpw=bpw, n_primary=n_primary,
            n_secondary=n_secondary, device=device, dtype=dtype,
            k_idx1=_z(torch.int16), v_idx1=_z(torch.int16),
            k_scale=_z(dtype), v_scale=_z(dtype),
        )
        if n_primary >= 2:
            out.k_idx2 = _z(torch.int16)
            out.v_idx2 = _z(torch.int16)
        if n_primary >= 3:
            out.k_idx3 = _z(torch.int16)
            out.v_idx3 = _z(torch.int16)
        if n_secondary >= 1:
            out.k_idx_s1 = _z(torch.uint8)
            out.v_idx_s1 = _z(torch.uint8)
        return out

    def bytes_total(self) -> int:
        """Total bytes on card. Sum of all tensor sizes."""
        n = 0
        for t in (self.k_idx1, self.v_idx1, self.k_scale, self.v_scale,
                  self.k_idx2, self.v_idx2, self.k_idx3, self.v_idx3,
                  self.k_idx_s1, self.v_idx_s1):
            if t is not None:
                n += t.numel() * t.element_size()
        return n

    def bytes_per_token_per_head(self) -> int:
        """Storage per (token, head). For K *and* V combined: double this."""
        # idx1 (always)
        b = 2 * self.n_groups                        # int16
        if self.n_primary >= 2: b += 2 * self.n_groups
        if self.n_primary >= 3: b += 2 * self.n_groups
        if self.n_secondary >= 1: b += 1 * self.n_groups   # uint8
        b += self.dtype.itemsize * self.n_groups          # scale fp16/bf16
        return b


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #

def _scatter_into_cache(cache: E8PagedKVCache, slot_mapping: torch.Tensor,
                        *, side: str, qt: dict) -> None:
    """Scatter the contents of one ``E8KVQuantizer.quantize`` dict into
    the cache buffers for the given side ('k' or 'v'). slot_mapping is
    the absolute slot index (block_idx * block_size + within_block)."""
    block = slot_mapping // cache.block_size
    slot = slot_mapping % cache.block_size
    nT = slot_mapping.shape[0]
    H, G = cache.num_kv_heads, cache.n_groups

    def store(name: str, src: torch.Tensor):
        buf = getattr(cache, f"{side}_{name}")
        if buf is None:
            raise RuntimeError(
                f"recipe has no buffer for {side}_{name} at bpw={cache.bpw}")
        # src has shape [nT * H, G]; reshape to [nT, H, G] and scatter.
        src = src.reshape(nT, H, G)
        buf[block, slot] = src.to(buf.dtype)

    store("idx1", qt["idx1"])
    if cache.n_primary >= 2: store("idx2", qt["idx2"])
    if cache.n_primary >= 3: store("idx3", qt["idx3"])
    if cache.n_secondary >= 1: store("idx_s1", qt["idx_s1"])

    # scale comes back as [nT, H, G] already.
    scale_buf = getattr(cache, f"{side}_scale")
    scale_buf[block, slot] = qt["scale"].to(scale_buf.dtype)


def write_kv(quantizer, key: torch.Tensor, value: torch.Tensor,
             cache: E8PagedKVCache, slot_mapping: torch.Tensor) -> None:
    """Quantize K, V and write the encoded form into ``cache`` at slots.

    Args:
        quantizer: ``E8KVQuantizer`` with the matching ``n_stages`` /
            ``secondary_stages`` for ``cache.bpw``.
        key, value: ``[num_tokens, num_kv_heads, head_size]`` fp16/bf16.
        cache: paged cache allocated with ``E8PagedKVCache.alloc``.
        slot_mapping: ``[num_tokens]`` int64, absolute slot indices.
    """
    assert key.shape == value.shape, (
        f"key/value shape mismatch: {key.shape} vs {value.shape}")
    assert key.shape[-1] == cache.head_size, (
        f"head_size mismatch: input {key.shape[-1]} vs cache {cache.head_size}")
    assert key.shape[-2] == cache.num_kv_heads
    if quantizer.bpw != cache.bpw:
        raise ValueError(
            f"quantizer bpw={quantizer.bpw} does not match cache bpw={cache.bpw}")

    qt_k = quantizer.quantize(key)
    qt_v = quantizer.quantize(value)
    _scatter_into_cache(cache, slot_mapping, side="k", qt=qt_k)
    _scatter_into_cache(cache, slot_mapping, side="v", qt=qt_v)


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #

def _gather_qt(cache: E8PagedKVCache, slot_mapping: torch.Tensor,
               *, side: str) -> dict:
    """Gather encoded values for the given slots and assemble the dict
    that ``E8KVQuantizer.dequantize`` expects."""
    block = slot_mapping // cache.block_size
    slot = slot_mapping % cache.block_size
    nT = slot_mapping.shape[0]
    H, G = cache.num_kv_heads, cache.n_groups

    def load(name: str) -> torch.Tensor:
        buf = getattr(cache, f"{side}_{name}")
        # buf shape: [num_blocks, block_size, H, G]
        x = buf[block, slot]  # [nT, H, G]
        return x.reshape(nT * H, G)

    out = {
        "idx1": load("idx1"),
        "shape": (nT, H, cache.head_size),
        "dtype": cache.dtype,
    }
    if cache.n_primary >= 2: out["idx2"] = load("idx2")
    if cache.n_primary >= 3: out["idx3"] = load("idx3")
    if cache.n_secondary >= 1: out["idx_s1"] = load("idx_s1")

    scale_buf = getattr(cache, f"{side}_scale")
    out["scale"] = scale_buf[block, slot]  # [nT, H, G]
    return out


def gather_kv(quantizer, cache: E8PagedKVCache,
              slot_mapping: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Read encoded K, V from the cache at ``slot_mapping`` and return
    the dequantized fp16/bf16 tensors of shape
    ``[num_tokens, num_kv_heads, head_size]``.

    For attention this is the "decompress to fp16 workspace" step that
    feeds into flash_attn.
    """
    qt_k = _gather_qt(cache, slot_mapping, side="k")
    qt_v = _gather_qt(cache, slot_mapping, side="v")
    key = quantizer.dequantize(qt_k)
    value = quantizer.dequantize(qt_v)
    return key, value


def gather_kv_to_paged_fp16(quantizer, cache: E8PagedKVCache,
                             block_indices: torch.Tensor | None = None,
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompress (a subset of) the paged cache to ``flash_attn``-shaped
    fp16/bf16 paged buffers.

    Args:
        quantizer: matching ``E8KVQuantizer``.
        cache: the compressed paged cache.
        block_indices: ``[num_selected_blocks]`` int64. If ``None``,
            decompresses every block.

    Returns:
        ``(k_paged, v_paged)`` of shape
        ``[num_selected_blocks, block_size, num_kv_heads, head_size]``
        — the layout that ``flash_attn_with_kvcache`` consumes
        directly via its ``block_table`` argument.
    """
    if block_indices is None:
        block_indices = torch.arange(
            cache.num_blocks, device=cache.device, dtype=torch.long)
    NB = block_indices.shape[0]
    BS = cache.block_size
    H = cache.num_kv_heads
    G = cache.n_groups

    def _gather_side(side: str) -> torch.Tensor:
        # Build the qt dict for the selected blocks, treating each
        # block as a (BS, H) batch of group-vectors.
        def take(name: str):
            buf = getattr(cache, f"{side}_{name}")
            return buf[block_indices]  # [NB, BS, H, G]

        qt: dict = {
            "idx1": take("idx1").reshape(NB * BS * H, G),
            "shape": (NB, BS, H, cache.head_size),
            "dtype": cache.dtype,
        }
        if cache.n_primary >= 2: qt["idx2"] = take("idx2").reshape(NB * BS * H, G)
        if cache.n_primary >= 3: qt["idx3"] = take("idx3").reshape(NB * BS * H, G)
        if cache.n_secondary >= 1:
            qt["idx_s1"] = take("idx_s1").reshape(NB * BS * H, G)
        scale_buf = getattr(cache, f"{side}_scale")
        qt["scale"] = scale_buf[block_indices]            # [NB, BS, H, G]

        return quantizer.dequantize(qt)        # [NB, BS, H, head_size]

    return _gather_side("k"), _gather_side("v")
