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
        """Total *new* bytes on card. Returns 0 for shared-storage caches
        (built via from_paged_storage) — those views point at memory
        vLLM already accounts for."""
        if getattr(self, "_shared_storage", False):
            return 0
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

    @classmethod
    def from_paged_storage(cls, k_buf: torch.Tensor, v_buf: torch.Tensor, *,
                           head_size: int, bpw: int,
                           dtype: torch.dtype) -> "E8PagedKVCache":
        """Construct an E8PagedKVCache whose component tensors are
        strided views into vLLM's already-allocated paged buffer.

        This is the Stage 2c-2c primitive that eliminates parallel
        allocation: vLLM allocated the right byte count at the
        compressed page size (Stage 2c-2b), and this method reinterprets
        those bytes as our idx1/idx_s1/idx2/idx3/scale layout *in place*
        — no new memory.

        Per (token, head) the byte layout is:
            [idx1 (n_groups * 2 B int16)]
            [idx2 (n_groups * 2 B int16)]   if bpw in {4, 5, 6, 7}
            [idx3 (n_groups * 2 B int16)]   if bpw in {6, 7}
            [idx_s1 (n_groups * 1 B uint8)] if bpw in {3, 5, 7}
            [scale (n_groups * 2 B dtype)]

        Args:
            k_buf, v_buf: vLLM's K, V views — typically the result of
                ``kv_cache.unbind(1)`` on a layer's paged tensor.
                Expected shape ``[num_blocks, block_size, num_kv_heads,
                compressed_elems_per_tok_per_head]`` with the compressed
                last dim summing to ``bth = n_groups * bytes_per_group``.
            head_size: the *real* head size (not the compressed
                last-dim count). Drives ``n_groups``.
            bpw: 2..7.
            dtype: dtype of the scale component (matches k_buf.dtype).
        """
        if bpw not in _BPW_RECIPE:
            raise ValueError(f"bpw {bpw} not in {sorted(_BPW_RECIPE)}")
        if head_size % 8 != 0:
            raise ValueError(f"head_size {head_size} not divisible by 8")
        n_primary, n_secondary = _BPW_RECIPE[bpw]
        n_groups = head_size // 8
        # bytes per (token, head) — must match the spec's compressed page
        # math (see e8_kv_spec.compressed_page_size_bytes).
        from glq_vllm.e8_kv_spec import E8_BYTES_PER_GROUP
        bytes_per_group = E8_BYTES_PER_GROUP[bpw]
        bth = n_groups * bytes_per_group

        if k_buf.shape != v_buf.shape:
            raise ValueError(
                f"k_buf/v_buf shape mismatch {k_buf.shape} vs {v_buf.shape}")
        num_blocks, block_size, num_kv_heads, compressed_last = k_buf.shape
        elem_size = k_buf.dtype.itemsize
        if compressed_last * elem_size != bth:
            raise ValueError(
                f"buffer last dim {compressed_last} * elem_size {elem_size} "
                f"!= bth {bth}; buffer/spec mismatch at bpw={bpw}, "
                f"head_size={head_size}")

        # Walk the layout, computing each component's byte offset.
        # Order matches the write/read code below — keep it stable.
        layout: list[tuple[str, int, int, torch.dtype]] = []  # (name, off, nbytes, dtype)
        cur = 0
        layout.append(("idx1", cur, 2 * n_groups, torch.int16))
        cur += 2 * n_groups
        if n_primary >= 2:
            layout.append(("idx2", cur, 2 * n_groups, torch.int16))
            cur += 2 * n_groups
        if n_primary >= 3:
            layout.append(("idx3", cur, 2 * n_groups, torch.int16))
            cur += 2 * n_groups
        if n_secondary >= 1:
            layout.append(("idx_s1", cur, n_groups, torch.uint8))
            cur += n_groups
        layout.append(("scale", cur, 2 * n_groups, dtype))
        cur += 2 * n_groups
        if cur != bth:
            raise RuntimeError(
                f"layout sum {cur} != bth {bth} (bpw={bpw}); recipe bug")

        # Strides in BYTES for side_buf's outer dims (B, BS, H). These
        # already encode any V-side interspersion (when side_buf is
        # ``raw[:, 0]`` or ``raw[:, 1]`` the B stride covers both K and
        # V worth of bytes), so we don't have to reason about whether
        # vLLM uses interleaved storage or not — just inherit the
        # strides from the input.
        side_byte_strides = [
            s * k_buf.element_size() for s in k_buf.stride()[:3]
        ]

        def _make_view(side_buf: torch.Tensor, byte_off: int,
                       comp_dtype: torch.dtype) -> torch.Tensor:
            """Build a strided view of one component for one (K or V) side."""
            storage = side_buf.untyped_storage()
            base_offset_bytes = side_buf.data_ptr() - storage.data_ptr()
            comp_elem_size = comp_dtype.itemsize
            total_offset_bytes = base_offset_bytes + byte_off
            if total_offset_bytes % comp_elem_size != 0:
                raise RuntimeError(
                    f"byte offset {total_offset_bytes} not aligned to "
                    f"{comp_dtype} (size {comp_elem_size})")
            # torch.Tensor.set_ takes storage_offset in tensor-dtype
            # elements, not raw bytes — even when the storage is an
            # untyped_storage().
            storage_offset_elems = total_offset_bytes // comp_elem_size
            # Convert outer-dim byte strides to component-dtype elem strides.
            outer_strides = tuple(
                s // comp_elem_size for s in side_byte_strides)
            # Inside one (B, T, H) chunk the component bytes are packed,
            # so the last-dim stride is just 1 component elem.
            strides = (*outer_strides, 1)
            n_elems_last = (
                2 * n_groups // comp_elem_size if comp_dtype != torch.uint8
                else n_groups)
            size = (num_blocks, block_size, num_kv_heads, n_elems_last)
            view = torch.empty(0, dtype=comp_dtype, device=side_buf.device)
            view.set_(storage, storage_offset=storage_offset_elems,
                      size=size, stride=strides)
            return view

        obj = cls(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=num_kv_heads, head_size=head_size,
            n_groups=n_groups, bpw=bpw, n_primary=n_primary,
            n_secondary=n_secondary, device=k_buf.device, dtype=dtype,
            k_idx1=_make_view(k_buf, layout[0][1], torch.int16),
            v_idx1=_make_view(v_buf, layout[0][1], torch.int16),
            # scale always last in layout list
            k_scale=_make_view(k_buf, layout[-1][1], dtype),
            v_scale=_make_view(v_buf, layout[-1][1], dtype),
        )
        # Optional components by name lookup.
        for name, off, _, comp_dtype in layout:
            if name in ("idx1", "scale"):
                continue
            setattr(obj, f"k_{name}", _make_view(k_buf, off, comp_dtype))
            setattr(obj, f"v_{name}", _make_view(v_buf, off, comp_dtype))
        # Mark as shared so bytes_total can report 0 for memory accounting.
        obj._shared_storage = True
        return obj

    @property
    def shares_vllm_storage(self) -> bool:
        return getattr(self, "_shared_storage", False)


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


def remap_block_table(block_table: torch.Tensor,
                      unique_blocks: torch.Tensor,
                      num_blocks: int) -> torch.Tensor:
    """Translate ``block_table`` from absolute block ids into positions
    within ``unique_blocks``.

    Used together with the compact form of ``gather_kv_to_paged_fp16``:
    the caller decompresses only the unique blocks the current attention
    call references, builds a remapped ``block_table`` pointing into the
    compact output, and hands that pair to ``flash_attn`` (or any other
    paged-attention kernel that consumes ``block_table``).
    """
    remap = torch.full(
        (num_blocks,), -1, dtype=torch.int32, device=block_table.device)
    pos = torch.arange(
        unique_blocks.shape[0], dtype=torch.int32, device=block_table.device)
    remap[unique_blocks] = pos
    return remap[block_table.long()].to(block_table.dtype)


# --------------------------------------------------------------------------- #
# Stage 3b — fused Triton dequant-gather kernel
# --------------------------------------------------------------------------- #
# The slow path below (``gather_kv_to_paged_fp16``) fires 5+ ``aten::index``
# calls per side per layer, each of which triggers its own gather kernel
# plus several ``aten::copy_`` follow-ups. The Stage 3a profile measured
# ~10k Python-side dispatches per decode and ~57% of CUDA time spent in
# this plumbing. The fused kernel below replaces the entire plumbing with
# one launch: gather indices + scale, dequant via codebook lookup, residual
# math, and per-group 8x8 inverse-Hadamard, all inline.
#
# ``gather_kv_to_paged_fp16_fused`` is signature-compatible with the slow
# path. On CPU caches (used by unit tests) it transparently falls back to
# the slow path; on CUDA it dispatches to the Triton kernel below.

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:  # pragma: no cover - triton is a required runtime dep on CUDA
    _TRITON_OK = False


def _hadamard_8_for_kernel(dtype: torch.dtype, device) -> torch.Tensor:
    """Same matrix as ``glq.kv_e8._hadamard_8`` but cast to the kernel's
    working dtype. Sized for the per-group 8x8 transform."""
    import math as _math
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    H2 = torch.kron(H1, H1)
    H3 = torch.kron(H2, H1)
    return (H3 / _math.sqrt(8.0)).to(dtype=dtype, device=device).contiguous()


if _TRITON_OK:
    @triton.jit
    def _fused_gather_dequant_kernel(
        # cache buffers (one side, K or V):
        idx1_ptr, idx2_ptr, idx3_ptr, idx_s1_ptr, scale_ptr,
        # codebook tables:
        cb_primary_ptr,    # [K_primary, 8]
        cb_secondary_ptr,  # [K_secondary, 8] or null
        # Hadamard matrix (symmetric -> equals its transpose):
        h_ptr,             # [8, 8] same dtype as out
        # gather indices:
        block_indices_ptr,  # [NB_sel] int64
        # output:
        out_ptr,           # [NB_sel, BS, H, head_size] bf16/fp16
        # sizes:
        NB_sel,
        BS: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
        # strides — idx1/idx2/idx3 share the [num_blocks, BS, H, G] layout:
        idx_stride_nb,
        idx_stride_bs: tl.constexpr,
        idx_stride_h: tl.constexpr,
        # uint8 idx_s1 may live in a separately strided buffer:
        idxu_stride_nb,
        idxu_stride_bs: tl.constexpr,
        idxu_stride_h: tl.constexpr,
        # scale buffer strides (same shape as idx tensors, dtype-strided):
        sc_stride_nb,
        sc_stride_bs: tl.constexpr,
        sc_stride_h: tl.constexpr,
        # output strides [NB_sel, BS, H, head_size]:
        out_stride_nb: tl.constexpr,
        out_stride_bs: tl.constexpr,
        out_stride_h: tl.constexpr,
        # math constants:
        RS: tl.constexpr,
        N_PRIMARY: tl.constexpr,
        N_SECONDARY: tl.constexpr,
    ):
        """One program per (block_in_sel, slot, head). Each program
        decodes all G groups for that (block, slot, head), so it emits
        head_size = G * 8 output elements.

        Grid: (NB_sel, BS, H). Total launches = NB_sel * BS * H, which
        replaces ~10 ``aten::index`` + gather kernels per (block, slot,
        head) in the slow path.
        """
        bi = tl.program_id(0)         # 0 .. NB_sel-1
        s_idx = tl.program_id(1)      # 0 .. BS-1
        h_idx = tl.program_id(2)      # 0 .. H-1

        # Look up the absolute block id this program is responsible for.
        nb_g = tl.load(block_indices_ptr + bi).to(tl.int64)

        # Per-group offset within the cache buffers (idx1 stride is the same
        # for idx2/idx3 since they share shape; idx_s1 may differ).
        g_arange = tl.arange(0, G)
        idx_base = (nb_g * idx_stride_nb
                    + s_idx * idx_stride_bs
                    + h_idx * idx_stride_h)
        # int16 indices: load all G groups at once. The cache stores
        # codebook indices in int16; values 32768..65535 are stored as
        # negative two's-complement. PyTorch's ``cb[idx.long()]`` rescues
        # them via negative-indexing wraparound, but Triton would
        # sign-extend to int32 and load out-of-bounds memory. Mask to
        # 16-bit unsigned before computing the codebook offset.
        idx1_vec = tl.load(idx1_ptr + idx_base + g_arange).to(tl.int32) & 0xFFFF

        # Primary stage 1: codebook[idx1] for each of the G groups.
        # Output: dec[g, k] where g in [0, G), k in [0, 8).
        cb_off = idx1_vec[:, None] * 8 + tl.arange(0, 8)[None, :]
        dec = tl.load(cb_primary_ptr + cb_off).to(tl.float32)

        # Primary stage 2.
        if N_PRIMARY >= 2:
            idx2_vec = tl.load(idx2_ptr + idx_base + g_arange).to(tl.int32) & 0xFFFF
            cb_off2 = idx2_vec[:, None] * 8 + tl.arange(0, 8)[None, :]
            dec = dec + tl.load(cb_primary_ptr + cb_off2).to(tl.float32) / RS

        # Primary stage 3.
        if N_PRIMARY >= 3:
            idx3_vec = tl.load(idx3_ptr + idx_base + g_arange).to(tl.int32) & 0xFFFF
            cb_off3 = idx3_vec[:, None] * 8 + tl.arange(0, 8)[None, :]
            dec = dec + tl.load(cb_primary_ptr + cb_off3).to(tl.float32) / (RS * RS)

        # Secondary stage 1 (8-bit small codebook).
        if N_SECONDARY >= 1:
            idxu_base = (nb_g * idxu_stride_nb
                         + s_idx * idxu_stride_bs
                         + h_idx * idxu_stride_h)
            idxs_vec = tl.load(idx_s1_ptr + idxu_base + g_arange).to(tl.int32)
            cb_offs = idxs_vec[:, None] * 8 + tl.arange(0, 8)[None, :]
            # divisor = rs ** (n_primary - 1 + 1) = rs ** n_primary
            if N_PRIMARY == 1:
                div_s = RS
            elif N_PRIMARY == 2:
                div_s = RS * RS
            else:
                div_s = RS * RS * RS
            dec = dec + tl.load(cb_secondary_ptr + cb_offs).to(tl.float32) / div_s

        # Per-group scale.
        sc_base = (nb_g * sc_stride_nb
                   + s_idx * sc_stride_bs
                   + h_idx * sc_stride_h)
        scale_vec = tl.load(scale_ptr + sc_base + g_arange).to(tl.float32)
        dec = dec * scale_vec[:, None]

        # 8x8 inverse Hadamard, per group. Walsh-Hadamard is symmetric so
        # H.T == H, hence dec @ H.T == dec @ H. Load once, fold over the
        # group axis with tl.sum on the inner contraction.
        h_rows = tl.arange(0, 8)[:, None]
        h_cols = tl.arange(0, 8)[None, :]
        H_mat = tl.load(h_ptr + h_rows * 8 + h_cols).to(tl.float32)
        # dec is [G, 8]; we want out[g, j] = sum_k dec[g, k] * H[k, j].
        # Triton: broadcast dec to [G, 8, 1], H to [1, 8, 8], multiply, sum.
        out_block = tl.sum(
            dec[:, :, None] * H_mat[None, :, :], axis=1
        )  # [G, 8]

        # Write to output [NB_sel, BS, H, head_size = G*8].
        out_base = (bi * out_stride_nb
                    + s_idx * out_stride_bs
                    + h_idx * out_stride_h)
        out_off = (tl.arange(0, G)[:, None] * 8
                   + tl.arange(0, 8)[None, :])
        tl.store(out_ptr + out_base + out_off,
                 out_block.to(out_ptr.dtype.element_ty))


# Module-level Hadamard cache keyed by (dtype, device) — built once.
_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}


def _get_hadamard(dtype: torch.dtype, device) -> torch.Tensor:
    """Return a cached 8x8 Hadamard matrix sized for ``dtype`` on
    ``device``. The matrix is symmetric so we use it for the inverse
    transform directly (H == H.T)."""
    key = (dtype, torch.device(device).index if torch.device(device).index is not None
           else str(device))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None and cached.device == torch.device(device):
        return cached
    H = _hadamard_8_for_kernel(dtype=dtype, device=device)
    _HADAMARD_CACHE[key] = H
    return H


def gather_kv_to_paged_fp16_fused(quantizer, cache: E8PagedKVCache,
                                   block_indices: torch.Tensor | None = None,
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused replacement for ``gather_kv_to_paged_fp16``.

    Signature-compatible with the slow path. On CPU caches (or if Triton
    is missing) falls back to the slow path. On CUDA dispatches one
    Triton kernel per side (K, V) — each launch handles every (block,
    slot, head, group) tuple for that side.

    Stage 3a profile target: the slow path issues ~10k Python-side
    ``aten::index`` calls per decode (5 takes × 2 sides × ~32 layers ×
    ~32 tokens). This kernel collapses each side's gather + dequant to
    one launch, eliminating the gather plumbing entirely.
    """
    if cache.device.type != "cuda" or not _TRITON_OK:
        return gather_kv_to_paged_fp16(quantizer, cache,
                                       block_indices=block_indices)

    if block_indices is None:
        block_indices = torch.arange(
            cache.num_blocks, device=cache.device, dtype=torch.long)
    else:
        block_indices = block_indices.to(device=cache.device, dtype=torch.long)
    NB_sel = int(block_indices.shape[0])
    BS = cache.block_size
    H = cache.num_kv_heads
    G = cache.n_groups
    head_size = cache.head_size
    dtype = cache.dtype

    # Codebooks: use the half-precision tensor (.codebook_half) cast to
    # the cache's dtype so loads match the slow path.
    cb_pri = quantizer.codebook.codebook_half.to(dtype=dtype, device=cache.device)
    if not cb_pri.is_contiguous():
        cb_pri = cb_pri.contiguous()
    if cache.n_secondary >= 1:
        cb_sec = quantizer.secondary_codebook.codebook_half.to(
            dtype=dtype, device=cache.device)
        if not cb_sec.is_contiguous():
            cb_sec = cb_sec.contiguous()
    else:
        # Triton wants a real pointer; use the primary codebook tensor
        # (its data is unused when N_SECONDARY == 0).
        cb_sec = cb_pri

    rs = float(quantizer.codebook.resid_scale)
    n_primary = cache.n_primary
    n_secondary = cache.n_secondary

    H_mat = _get_hadamard(dtype, cache.device)

    def _run_side(side: str) -> torch.Tensor:
        idx1 = getattr(cache, f"{side}_idx1")
        idx2 = getattr(cache, f"{side}_idx2", None)
        idx3 = getattr(cache, f"{side}_idx3", None)
        idx_s1 = getattr(cache, f"{side}_idx_s1", None)
        scale = getattr(cache, f"{side}_scale")

        out = torch.empty(
            (NB_sel, BS, H, head_size), dtype=dtype, device=cache.device)

        # Strides for the [num_blocks, BS, H, G] index buffers.
        idx_st_nb, idx_st_bs, idx_st_h, idx_st_g = idx1.stride()
        assert idx_st_g == 1, "idx1 must be contiguous along G"
        # idx2, idx3 share idx1's stride/layout if present.
        # idx_s1 may have its own (uint8 storage):
        if idx_s1 is not None:
            idxu_st_nb, idxu_st_bs, idxu_st_h, idxu_st_g = idx_s1.stride()
            assert idxu_st_g == 1
        else:
            idxu_st_nb = idxu_st_bs = idxu_st_h = 0
        # scale tensor strides (same shape, different dtype):
        sc_st_nb, sc_st_bs, sc_st_h, sc_st_g = scale.stride()
        assert sc_st_g == 1
        # out strides [NB_sel, BS, H, head_size]:
        out_st_nb, out_st_bs, out_st_h, out_st_g = out.stride()
        assert out_st_g == 1

        grid = (NB_sel, BS, H)
        idx2_ptr = idx2 if idx2 is not None else idx1
        idx3_ptr = idx3 if idx3 is not None else idx1
        idx_s1_ptr = idx_s1 if idx_s1 is not None else idx1
        _fused_gather_dequant_kernel[grid](
            idx1, idx2_ptr, idx3_ptr, idx_s1_ptr, scale,
            cb_pri, cb_sec, H_mat,
            block_indices,
            out,
            NB_sel,
            BS, H, G,
            idx_st_nb, idx_st_bs, idx_st_h,
            idxu_st_nb, idxu_st_bs, idxu_st_h,
            sc_st_nb, sc_st_bs, sc_st_h,
            out_st_nb, out_st_bs, out_st_h,
            RS=rs,
            N_PRIMARY=n_primary,
            N_SECONDARY=n_secondary,
        )
        return out

    return _run_side("k"), _run_side("v")


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
