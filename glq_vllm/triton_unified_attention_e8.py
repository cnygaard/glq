"""v0.5 forked vLLM Triton attention kernel with inline E8 dequant.

This is a fork of vLLM 0.20's
``vllm/v1/attention/ops/triton_unified_attention.py::kernel_unified_attention_2d``.
The eventual aim (Phase 1.5 and beyond) is to fuse E8-KV dequantisation
into the attention tile loop, eliminating the v0.3.5 workspace that
costs ~500 µs / layer / token of pure overhead.

Phase 1.0 (this file): skeleton fork with K/V still passed as
pre-decompressed tensors, identical to upstream. The forked kernel
must produce bit-identical output to upstream ``unified_attention``
on the Phase-1 fixture (head_size=128, single sequence, single KV
head, causal-only masking).

Stripped vs upstream (out of Phase-1 scope):

- ``USE_FP8`` — Phase 1 is fp16 K/V only.
- ``USE_ALIBI_SLOPES`` / ``USE_ALIBI_SQRT`` — Gemma-4 doesn't use them.
- ``USE_QQ_BIAS`` — only ever used by Bamba/Mamba hybrid heads.
- ``USE_SOFTCAP`` — Gemma-4 doesn't use it in attention.
- ``USE_SINKS`` — GPT-OSS sinks variant.
- ``USE_MM_PREFIX`` / ``MAX_MM_RANGES`` — multimodal PrefixLM.
- ``SLIDING_WINDOW`` — added back in Phase 2 for Gemma-4-E4B.

The stripped branches will be re-introduced in Phase 2+ as they're
needed for real Gemma-4 shapes. Until then the simpler kernel is
faster to debug — fewer constexpr paths means fewer ways the
dequant splice can interact with masking.

Sub-phase progression (each lands as its own diff + test):

- 1.0: this skeleton — pre-decompressed K/V, no dequant.
- 1.1: inline stage-1 dequant (``idx1`` only).
- 1.2: stage-2 residual (``idx2 / RS``).
- 1.3: per-group ``scale`` broadcast.
- 1.4: per-group Hadamard inverse.
- 1.5: full attention end-to-end on tiny shapes.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d_e8(
    output_ptr,                  # [num_tokens, num_query_heads, head_size]
    query_ptr,                   # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,               # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,             # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,            # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,                # [num_seqs]
    scale,                       # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,    # should be equal to head_size
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,   # should be equal to head_size
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,     # must be power of 2
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,  # must be power of 2
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,         # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequence
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # iterate through tiles
    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        # V : (TILE_SIZE, HEAD_SIZE)
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # compute running maximum
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For full masking there's a chance the max is -inf. Set m_j 0
        # to avoid NaN in subsequent exp.
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M,)
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ───────────────────────────────────────────────────────────────────────
# Phase 1.1 — Stage-1 inline dequant (idx1 → codebook lookup only).
#
# Difference vs Phase 1.0:
# - ``key_cache_ptr`` / ``value_cache_ptr`` → ``K_idx1_ptr`` /
#   ``V_idx1_ptr`` (int16 indices into the 65 536-entry primary
#   codebook), plus a shared ``codebook_ptr`` for the lookup.
# - The K/V load is no longer ``tl.load(cache)``: it's a two-step
#   ``load(idx) → load(codebook[idx])`` that materialises the dequanted
#   tile inside the kernel.
# - Per-group ``scale`` and the Hadamard ``H_mat`` are NOT applied yet
#   — those go in Phase 1.3 and Phase 1.4. The test asserts the kernel
#   matches the *kernel-truth* path (decode→reshape→SDPA), not the
#   full ``reference_e8_attention``.
# ───────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_unified_attention_2d_e8_v1_1(
    output_ptr,                  # [num_tokens, num_query_heads, head_size]
    query_ptr,                   # [num_tokens, num_query_heads, head_size]
    K_idx1_ptr,                  # [num_blocks, block_size, num_kv_heads, n_groups] int16
    V_idx1_ptr,                  # [num_blocks, block_size, num_kv_heads, n_groups] int16
    codebook_ptr,                # [65536, 8] fp16
    block_tables_ptr,            # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,                # [num_seqs]
    scale,                       # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    N_GROUPS: tl.constexpr,      # head_size // 8
    # K idx1 strides (4D layout: blocks, slot, kv_head, group)
    stride_k_idx1_0: tl.int64,
    stride_k_idx1_1: tl.int64,
    stride_k_idx1_2: tl.int64,
    stride_k_idx1_3: tl.constexpr,
    # V idx1 strides
    stride_v_idx1_0: tl.int64,
    stride_v_idx1_1: tl.int64,
    stride_v_idx1_2: tl.int64,
    stride_v_idx1_3: tl.constexpr,
    query_start_len_ptr,         # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # Pre-compute group and codebook-coord vectors (constant across tiles).
    g_arange = tl.arange(0, N_GROUPS)        # [N_GROUPS]
    cb_inner = tl.arange(0, 8)               # [8]

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        slot = seq_offset % BLOCK_SIZE       # [TILE_SIZE]

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ───── K dequant ───────────────────────────────────────────
        # idx1: [TILE_SIZE, N_GROUPS] int16 → int32 (masked to unsigned 16-bit)
        k_idx_off = (
            physical_block_idx[:, None] * stride_k_idx1_0
            + slot[:, None] * stride_k_idx1_1
            + kv_head_idx * stride_k_idx1_2
            + g_arange[None, :] * stride_k_idx1_3
        )
        k_idx1 = tl.load(
            K_idx1_ptr + k_idx_off,
            mask=tile_mask[:, None],
            other=0,
        ).to(tl.int32) & 0xFFFF

        # Codebook lookup → [TILE_SIZE, N_GROUPS, 8] fp16 (codebook dtype).
        # Codebook is contiguous [65536, 8], so stride is (8, 1).
        k_cb_off = k_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        k_dec = tl.load(codebook_ptr + k_cb_off)
        # Reshape group dim into head dim, then transpose to [HEAD, TILE].
        K = tl.trans(tl.reshape(k_dec, (TILE_SIZE, HEAD_SIZE)))
        K = tl.where(tile_mask[None, :], K, 0.0)

        # ───── V dequant ───────────────────────────────────────────
        v_idx_off = (
            physical_block_idx[:, None] * stride_v_idx1_0
            + slot[:, None] * stride_v_idx1_1
            + kv_head_idx * stride_v_idx1_2
            + g_arange[None, :] * stride_v_idx1_3
        )
        v_idx1 = tl.load(
            V_idx1_ptr + v_idx_off,
            mask=tile_mask[:, None],
            other=0,
        ).to(tl.int32) & 0xFFFF
        v_cb_off = v_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        v_dec = tl.load(codebook_ptr + v_cb_off)
        V = tl.reshape(v_dec, (TILE_SIZE, HEAD_SIZE))     # [TILE, HEAD]
        V = tl.where(tile_mask[:, None], V, 0.0)

        # ───── Causal mask + Q @ K → S ─────────────────────────────
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # ───── Online softmax ──────────────────────────────────────
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ───────────────────────────────────────────────────────────────────────
# Phase 1.2 — Stage-2 residual (idx1 + idx2 / RS).
#
# Difference vs Phase 1.1:
# - Adds ``idx2`` indices and divides the second-stage codebook
#   lookup by the residual scale RS. Matches the math in
#   ``E8KVQuantizer.dequantize`` *minus* per-group scale and the
#   Hadamard inverse (Phase 1.3 + Phase 1.4).
# - Convention is **divide on decode** (encoder multiplies by RS
#   before quantising the residual, decoder undoes by dividing).
#   See ``E8KVQuantizer.quantize`` lines 142-146 / ``dequantize``
#   line 198. Getting this direction wrong produces output with
#   the right magnitude but wrong sign — caught by the rtol gate.
# ───────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_unified_attention_2d_e8_v1_2(
    output_ptr,
    query_ptr,
    K_idx1_ptr,
    K_idx2_ptr,
    V_idx1_ptr,
    V_idx2_ptr,
    codebook_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    rs,                          # float32 — residual scale (codebook.resid_scale)
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    N_GROUPS: tl.constexpr,
    stride_k_idx_0: tl.int64,    # shared between idx1 and idx2 (same shape)
    stride_k_idx_1: tl.int64,
    stride_k_idx_2: tl.int64,
    stride_k_idx_3: tl.constexpr,
    stride_v_idx_0: tl.int64,
    stride_v_idx_1: tl.int64,
    stride_v_idx_2: tl.int64,
    stride_v_idx_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    g_arange = tl.arange(0, N_GROUPS)
    cb_inner = tl.arange(0, 8)

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        slot = seq_offset % BLOCK_SIZE

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ───── K dequant (idx1 + idx2 / RS) ─────────────────────────
        k_idx_off = (
            physical_block_idx[:, None] * stride_k_idx_0
            + slot[:, None] * stride_k_idx_1
            + kv_head_idx * stride_k_idx_2
            + g_arange[None, :] * stride_k_idx_3
        )
        k_idx1 = tl.load(
            K_idx1_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_idx2 = tl.load(
            K_idx2_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF

        # Codebook decodes (fp16) — accumulate in fp32 to keep
        # numerical headroom for the 1/RS divide.
        k_cb_off1 = k_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        k_cb_off2 = k_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        k_dec = tl.load(codebook_ptr + k_cb_off1).to(tl.float32)
        k_dec = k_dec + tl.load(codebook_ptr + k_cb_off2).to(tl.float32) / rs

        K = tl.trans(tl.reshape(k_dec, (TILE_SIZE, HEAD_SIZE)))
        K = tl.where(tile_mask[None, :], K, 0.0).to(Q.dtype)

        # ───── V dequant ───────────────────────────────────────────
        v_idx_off = (
            physical_block_idx[:, None] * stride_v_idx_0
            + slot[:, None] * stride_v_idx_1
            + kv_head_idx * stride_v_idx_2
            + g_arange[None, :] * stride_v_idx_3
        )
        v_idx1 = tl.load(
            V_idx1_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_idx2 = tl.load(
            V_idx2_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_cb_off1 = v_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        v_cb_off2 = v_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        v_dec = tl.load(codebook_ptr + v_cb_off1).to(tl.float32)
        v_dec = v_dec + tl.load(codebook_ptr + v_cb_off2).to(tl.float32) / rs

        V = tl.reshape(v_dec, (TILE_SIZE, HEAD_SIZE))
        V = tl.where(tile_mask[:, None], V, 0.0).to(Q.dtype)

        # ───── Softmax ───────────────────────────────────────────
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ───────────────────────────────────────────────────────────────────────
# Phase 1.3 — Per-group scale broadcast.
#
# Difference vs Phase 1.2:
# - Loads per-group ``scale`` tensors (one fp16 value per token / head
#   / group of 8) and multiplies into the dequant before reshape.
# - Hadamard inverse still NOT applied (Phase 1.4 adds that).
# ───────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_unified_attention_2d_e8_v1_3(
    output_ptr,
    query_ptr,
    K_idx1_ptr,
    K_idx2_ptr,
    K_scale_ptr,
    V_idx1_ptr,
    V_idx2_ptr,
    V_scale_ptr,
    codebook_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    rs,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    N_GROUPS: tl.constexpr,
    stride_k_idx_0: tl.int64,
    stride_k_idx_1: tl.int64,
    stride_k_idx_2: tl.int64,
    stride_k_idx_3: tl.constexpr,
    stride_v_idx_0: tl.int64,
    stride_v_idx_1: tl.int64,
    stride_v_idx_2: tl.int64,
    stride_v_idx_3: tl.constexpr,
    stride_k_scale_0: tl.int64,
    stride_k_scale_1: tl.int64,
    stride_k_scale_2: tl.int64,
    stride_k_scale_3: tl.constexpr,
    stride_v_scale_0: tl.int64,
    stride_v_scale_1: tl.int64,
    stride_v_scale_2: tl.int64,
    stride_v_scale_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    g_arange = tl.arange(0, N_GROUPS)
    cb_inner = tl.arange(0, 8)

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        slot = seq_offset % BLOCK_SIZE

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ───── K: idx1 + idx2/RS, then × scale ─────────────────────
        k_idx_off = (
            physical_block_idx[:, None] * stride_k_idx_0
            + slot[:, None] * stride_k_idx_1
            + kv_head_idx * stride_k_idx_2
            + g_arange[None, :] * stride_k_idx_3
        )
        k_idx1 = tl.load(
            K_idx1_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_idx2 = tl.load(
            K_idx2_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_cb_off1 = k_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        k_cb_off2 = k_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        k_dec = tl.load(codebook_ptr + k_cb_off1).to(tl.float32)
        k_dec = k_dec + tl.load(codebook_ptr + k_cb_off2).to(tl.float32) / rs

        # Per-group scale: [TILE_SIZE, N_GROUPS] broadcasts over [.., :, None].
        k_scale_off = (
            physical_block_idx[:, None] * stride_k_scale_0
            + slot[:, None] * stride_k_scale_1
            + kv_head_idx * stride_k_scale_2
            + g_arange[None, :] * stride_k_scale_3
        )
        k_scale = tl.load(
            K_scale_ptr + k_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        k_dec = k_dec * k_scale[:, :, None]

        K = tl.trans(tl.reshape(k_dec, (TILE_SIZE, HEAD_SIZE)))
        K = tl.where(tile_mask[None, :], K, 0.0).to(Q.dtype)

        # ───── V: idx1 + idx2/RS, then × scale ─────────────────────
        v_idx_off = (
            physical_block_idx[:, None] * stride_v_idx_0
            + slot[:, None] * stride_v_idx_1
            + kv_head_idx * stride_v_idx_2
            + g_arange[None, :] * stride_v_idx_3
        )
        v_idx1 = tl.load(
            V_idx1_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_idx2 = tl.load(
            V_idx2_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_cb_off1 = v_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        v_cb_off2 = v_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        v_dec = tl.load(codebook_ptr + v_cb_off1).to(tl.float32)
        v_dec = v_dec + tl.load(codebook_ptr + v_cb_off2).to(tl.float32) / rs

        v_scale_off = (
            physical_block_idx[:, None] * stride_v_scale_0
            + slot[:, None] * stride_v_scale_1
            + kv_head_idx * stride_v_scale_2
            + g_arange[None, :] * stride_v_scale_3
        )
        v_scale = tl.load(
            V_scale_ptr + v_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dec = v_dec * v_scale[:, :, None]

        V = tl.reshape(v_dec, (TILE_SIZE, HEAD_SIZE))
        V = tl.where(tile_mask[:, None], V, 0.0).to(Q.dtype)

        # ───── Softmax ────────────────────────────────────────────
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ───────────────────────────────────────────────────────────────────────
# Phase 1.4 — Hadamard inverse (per-group 8×8 transform).
#
# Difference vs Phase 1.3:
# - Applies the inverse Walsh-Hadamard transform on each 8-wide group
#   after scale, completing the dequant pipeline. ``H`` is symmetric
#   and orthogonal so ``H.T = H = H⁻¹``; we just multiply on the right
#   by ``H``.
# - Uses broadcast + ``tl.sum`` instead of ``tl.dot`` because Triton's
#   ``tl.dot`` requires the inner dim ≥ 16 on most HW (8-wide MMA isn't
#   available). The 8×8 transform is small enough that the explicit
#   reduction is fine.
# - With Phase 1.4 the kernel produces the same K/V as
#   ``E8KVQuantizer.dequantize`` (within fp16 precision); the test
#   gates against ``reference_e8_attention``.
# ───────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_unified_attention_2d_e8_v1_4(
    output_ptr,
    query_ptr,
    K_idx1_ptr,
    K_idx2_ptr,
    K_scale_ptr,
    V_idx1_ptr,
    V_idx2_ptr,
    V_scale_ptr,
    codebook_ptr,
    H_mat_ptr,                   # [8, 8] fp32 (preloaded once per kernel)
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    rs,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    N_GROUPS: tl.constexpr,
    stride_k_idx_0: tl.int64,
    stride_k_idx_1: tl.int64,
    stride_k_idx_2: tl.int64,
    stride_k_idx_3: tl.constexpr,
    stride_v_idx_0: tl.int64,
    stride_v_idx_1: tl.int64,
    stride_v_idx_2: tl.int64,
    stride_v_idx_3: tl.constexpr,
    stride_k_scale_0: tl.int64,
    stride_k_scale_1: tl.int64,
    stride_k_scale_2: tl.int64,
    stride_k_scale_3: tl.constexpr,
    stride_v_scale_0: tl.int64,
    stride_v_scale_1: tl.int64,
    stride_v_scale_2: tl.int64,
    stride_v_scale_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    g_arange = tl.arange(0, N_GROUPS)
    cb_inner = tl.arange(0, 8)
    # Load 8×8 Hadamard once (lives in fp32 registers across the loop).
    h_rows = tl.arange(0, 8)[:, None]
    h_cols = tl.arange(0, 8)[None, :]
    H_mat = tl.load(H_mat_ptr + h_rows * 8 + h_cols).to(tl.float32)  # [8, 8]

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        slot = seq_offset % BLOCK_SIZE

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ───── K: full dequant (idx1, idx2/RS, scale, Hadamard) ────
        k_idx_off = (
            physical_block_idx[:, None] * stride_k_idx_0
            + slot[:, None] * stride_k_idx_1
            + kv_head_idx * stride_k_idx_2
            + g_arange[None, :] * stride_k_idx_3
        )
        k_idx1 = tl.load(
            K_idx1_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_idx2 = tl.load(
            K_idx2_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_cb_off1 = k_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        k_cb_off2 = k_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        k_dec = tl.load(codebook_ptr + k_cb_off1).to(tl.float32)
        k_dec = k_dec + tl.load(codebook_ptr + k_cb_off2).to(tl.float32) / rs

        k_scale_off = (
            physical_block_idx[:, None] * stride_k_scale_0
            + slot[:, None] * stride_k_scale_1
            + kv_head_idx * stride_k_scale_2
            + g_arange[None, :] * stride_k_scale_3
        )
        k_scale = tl.load(
            K_scale_ptr + k_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        k_dec = k_dec * k_scale[:, :, None]

        # Hadamard: dec[t, g, j] = sum_k dec[t, g, k] * H[k, j]
        # Shapes: dec[:,:,:,None] = [TILE, G, 8, 1]; H[None,None,:,:] = [1,1,8,8]
        k_dec = tl.sum(
            k_dec[:, :, :, None] * H_mat[None, None, :, :], axis=2
        )  # [TILE, G, 8]

        K = tl.trans(tl.reshape(k_dec, (TILE_SIZE, HEAD_SIZE)))
        K = tl.where(tile_mask[None, :], K, 0.0).to(Q.dtype)

        # ───── V: same pipeline (no transpose) ─────────────────────
        v_idx_off = (
            physical_block_idx[:, None] * stride_v_idx_0
            + slot[:, None] * stride_v_idx_1
            + kv_head_idx * stride_v_idx_2
            + g_arange[None, :] * stride_v_idx_3
        )
        v_idx1 = tl.load(
            V_idx1_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_idx2 = tl.load(
            V_idx2_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_cb_off1 = v_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        v_cb_off2 = v_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        v_dec = tl.load(codebook_ptr + v_cb_off1).to(tl.float32)
        v_dec = v_dec + tl.load(codebook_ptr + v_cb_off2).to(tl.float32) / rs

        v_scale_off = (
            physical_block_idx[:, None] * stride_v_scale_0
            + slot[:, None] * stride_v_scale_1
            + kv_head_idx * stride_v_scale_2
            + g_arange[None, :] * stride_v_scale_3
        )
        v_scale = tl.load(
            V_scale_ptr + v_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dec = v_dec * v_scale[:, :, None]
        v_dec = tl.sum(
            v_dec[:, :, :, None] * H_mat[None, None, :, :], axis=2
        )

        V = tl.reshape(v_dec, (TILE_SIZE, HEAD_SIZE))
        V = tl.where(tile_mask[:, None], V, 0.0).to(Q.dtype)

        # ───── Softmax + accumulate ───────────────────────────────
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def unified_attention_e8_v1_4(
    q: torch.Tensor,
    k_idx1: torch.Tensor,
    k_idx2: torch.Tensor,
    k_scale: torch.Tensor,
    v_idx1: torch.Tensor,
    v_idx2: torch.Tensor,
    v_scale: torch.Tensor,
    codebook: torch.Tensor,
    H_mat: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    resid_scale: float,
    block_table: torch.Tensor,
):
    """Phase 1.4 launcher — full dequant pipeline (Phase 1.3 + Hadamard).

    K[t, h, d] = (((codebook[idx1] + codebook[idx2] / RS) * scale[g])
                  @ H.T) reshaped to ``[..., head_size]``. Since H is
                  symmetric, ``@ H.T == @ H``.
    """
    block_size = k_idx1.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_idx1.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    n_groups = k_idx1.shape[3]
    assert head_size == n_groups * 8
    assert k_scale.shape == k_idx1.shape
    assert codebook.shape == (65536, 8)
    assert H_mat.shape == (8, 8)
    assert H_mat.is_contiguous()

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    TILE_SIZE = 16

    kernel_unified_attention_2d_e8_v1_4[
        (total_num_q_blocks, num_kv_heads)
    ](
        output_ptr=out,
        query_ptr=q,
        K_idx1_ptr=k_idx1,
        K_idx2_ptr=k_idx2,
        K_scale_ptr=k_scale,
        V_idx1_ptr=v_idx1,
        V_idx2_ptr=v_idx2,
        V_scale_ptr=v_scale,
        codebook_ptr=codebook,
        H_mat_ptr=H_mat,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        rs=resid_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        N_GROUPS=n_groups,
        stride_k_idx_0=k_idx1.stride(0),
        stride_k_idx_1=k_idx1.stride(1),
        stride_k_idx_2=k_idx1.stride(2),
        stride_k_idx_3=k_idx1.stride(3),
        stride_v_idx_0=v_idx1.stride(0),
        stride_v_idx_1=v_idx1.stride(1),
        stride_v_idx_2=v_idx1.stride(2),
        stride_v_idx_3=v_idx1.stride(3),
        stride_k_scale_0=k_scale.stride(0),
        stride_k_scale_1=k_scale.stride(1),
        stride_k_scale_2=k_scale.stride(2),
        stride_k_scale_3=k_scale.stride(3),
        stride_v_scale_0=v_scale.stride(0),
        stride_v_scale_1=v_scale.stride(1),
        stride_v_scale_2=v_scale.stride(2),
        stride_v_scale_3=v_scale.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )


def unified_attention_e8_v1_3(
    q: torch.Tensor,
    k_idx1: torch.Tensor,
    k_idx2: torch.Tensor,
    k_scale: torch.Tensor,
    v_idx1: torch.Tensor,
    v_idx2: torch.Tensor,
    v_scale: torch.Tensor,
    codebook: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    resid_scale: float,
    block_table: torch.Tensor,
):
    """Phase 1.3 launcher — Phase 1.2 + per-group scale.

    K[t, h, d] = ((codebook[idx1] + codebook[idx2] / RS) * scale[group])
                  reshaped to [..., head_size]. Hadamard inverse still
                  not applied.
    """
    block_size = k_idx1.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_idx1.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    n_groups = k_idx1.shape[3]
    assert head_size == n_groups * 8
    assert k_scale.shape == k_idx1.shape, (
        f"scale shape mismatch: {k_scale.shape} vs idx {k_idx1.shape}"
    )
    assert codebook.shape == (65536, 8)

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    TILE_SIZE = 16

    kernel_unified_attention_2d_e8_v1_3[
        (total_num_q_blocks, num_kv_heads)
    ](
        output_ptr=out,
        query_ptr=q,
        K_idx1_ptr=k_idx1,
        K_idx2_ptr=k_idx2,
        K_scale_ptr=k_scale,
        V_idx1_ptr=v_idx1,
        V_idx2_ptr=v_idx2,
        V_scale_ptr=v_scale,
        codebook_ptr=codebook,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        rs=resid_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        N_GROUPS=n_groups,
        stride_k_idx_0=k_idx1.stride(0),
        stride_k_idx_1=k_idx1.stride(1),
        stride_k_idx_2=k_idx1.stride(2),
        stride_k_idx_3=k_idx1.stride(3),
        stride_v_idx_0=v_idx1.stride(0),
        stride_v_idx_1=v_idx1.stride(1),
        stride_v_idx_2=v_idx1.stride(2),
        stride_v_idx_3=v_idx1.stride(3),
        stride_k_scale_0=k_scale.stride(0),
        stride_k_scale_1=k_scale.stride(1),
        stride_k_scale_2=k_scale.stride(2),
        stride_k_scale_3=k_scale.stride(3),
        stride_v_scale_0=v_scale.stride(0),
        stride_v_scale_1=v_scale.stride(1),
        stride_v_scale_2=v_scale.stride(2),
        stride_v_scale_3=v_scale.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )


def unified_attention_e8_v1_2(
    q: torch.Tensor,
    k_idx1: torch.Tensor,
    k_idx2: torch.Tensor,
    v_idx1: torch.Tensor,
    v_idx2: torch.Tensor,
    codebook: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    resid_scale: float,
    block_table: torch.Tensor,
):
    """Phase 1.2 launcher — adds stage-2 residual to Phase 1.1.

    ``K[t, h, d] = (codebook[idx1] + codebook[idx2] / RS)`` reshaped
    to ``[..., head_size]`` (still no per-group scale, no Hadamard).
    """
    block_size = k_idx1.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_idx1.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    n_groups = k_idx1.shape[3]
    assert head_size == n_groups * 8
    assert k_idx2.shape == k_idx1.shape, (
        f"idx1/idx2 shape mismatch: {k_idx1.shape} vs {k_idx2.shape}"
    )
    assert codebook.shape == (65536, 8)
    assert codebook.is_contiguous()

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    TILE_SIZE = 16

    kernel_unified_attention_2d_e8_v1_2[
        (total_num_q_blocks, num_kv_heads)
    ](
        output_ptr=out,
        query_ptr=q,
        K_idx1_ptr=k_idx1,
        K_idx2_ptr=k_idx2,
        V_idx1_ptr=v_idx1,
        V_idx2_ptr=v_idx2,
        codebook_ptr=codebook,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        rs=resid_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        N_GROUPS=n_groups,
        stride_k_idx_0=k_idx1.stride(0),
        stride_k_idx_1=k_idx1.stride(1),
        stride_k_idx_2=k_idx1.stride(2),
        stride_k_idx_3=k_idx1.stride(3),
        stride_v_idx_0=v_idx1.stride(0),
        stride_v_idx_1=v_idx1.stride(1),
        stride_v_idx_2=v_idx1.stride(2),
        stride_v_idx_3=v_idx1.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )


def unified_attention_e8_v1_1(
    q: torch.Tensor,                # [num_tokens, num_q_heads, head_size]
    k_idx1: torch.Tensor,           # [num_blocks, block_size, num_kv_heads, n_groups] int16
    v_idx1: torch.Tensor,           # [num_blocks, block_size, num_kv_heads, n_groups] int16
    codebook: torch.Tensor,         # [65536, 8] fp16 (contiguous)
    out: torch.Tensor,              # [num_tokens, num_q_heads, head_size]
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    block_table: torch.Tensor,
):
    """Phase 1.1 launcher — inline stage-1 dequant.

    Decodes ``K[t, h, d] = codebook[k_idx1[block(t), slot(t), h, d//8],
    d%8]`` (no scale, no Hadamard) and runs the same attention math as
    Phase 1.0. This is intentionally NOT the production decode path —
    it's the simplest dequant arrangement that exercises the
    idx-load → codebook-gather → reshape → tl.dot pipeline.
    """
    block_size = k_idx1.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_idx1.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    n_groups = k_idx1.shape[3]
    assert head_size == n_groups * 8, (
        f"head_size {head_size} != n_groups {n_groups} * 8"
    )
    assert codebook.shape == (65536, 8), (
        f"codebook shape {codebook.shape} != (65536, 8)"
    )
    assert codebook.is_contiguous(), "codebook must be contiguous [65536, 8]"

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    TILE_SIZE = 16  # Phase 1: one tile = one block (fp16).

    kernel_unified_attention_2d_e8_v1_1[
        (total_num_q_blocks, num_kv_heads)
    ](
        output_ptr=out,
        query_ptr=q,
        K_idx1_ptr=k_idx1,
        V_idx1_ptr=v_idx1,
        codebook_ptr=codebook,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        N_GROUPS=n_groups,
        stride_k_idx1_0=k_idx1.stride(0),
        stride_k_idx1_1=k_idx1.stride(1),
        stride_k_idx1_2=k_idx1.stride(2),
        stride_k_idx1_3=k_idx1.stride(3),
        stride_v_idx1_0=v_idx1.stride(0),
        stride_v_idx1_1=v_idx1.stride(1),
        stride_v_idx1_2=v_idx1.stride(2),
        stride_v_idx1_3=v_idx1.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )


def unified_attention_e8(
    q: torch.Tensor,           # [num_tokens, num_q_heads, head_size]
    k: torch.Tensor,           # [num_blocks, block_size, num_kv_heads, head_size]
    v: torch.Tensor,           # [num_blocks, block_size, num_kv_heads, head_size]
    out: torch.Tensor,         # [num_tokens, num_q_heads, head_size]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1] int32
    seqused_k: torch.Tensor,   # [num_seqs] int32
    softmax_scale: float,
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq] int32
):
    """Phase 1.0 launcher — pre-decompressed K/V, causal-only.

    Mirrors the relevant subset of vLLM's ``unified_attention`` so the
    correctness test can swap the upstream call for ours and compare
    outputs bit-identically.

    The kernel itself only ever runs the 2-D path (no 3-D
    softmax-segments fast path) — Phase 1 stays simple and the 3-D
    optimisation can be re-introduced post-correctness.
    """
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile size mirrors upstream's non-Gemma3 default: 16 for fp16/bf16,
    # 32 for fp8. Phase 1 is fp16 only, so always 16.
    TILE_SIZE = 16

    kernel_unified_attention_2d_e8[
        (
            total_num_q_blocks,
            num_kv_heads,
        )
    ](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )


# ───────────────────────────────────────────────────────────────────────
# Phase 2.0 — Shape sweep (head_size + GQA + varlen).
#
# Differences vs Phase 1.4:
#   1. Hadamard application rewritten as a constexpr loop over the
#      8 output columns of H. The v1.4 broadcast pattern
#      ``tl.sum(dec[:, :, :, None] * H_mat[None, None, :, :], axis=2)``
#      materialises a ``[TILE=16, N_GROUPS=64, 8, 8]`` intermediate at
#      head_size=512 (256 KB conceptual). Looping over j-columns
#      reduces the intermediate to ``[TILE, N_GROUPS, 8]`` per
#      iteration — same FLOPs, friendly to Triton's register tiler.
#   2. No new constexpr / input — just refactored math.
#
# All other surface (varlen / GQA / multi-block) is structurally in
# the kernel via ``find_seq_idx`` + ``BLOCK_M / BLOCK_Q`` (copied
# verbatim from upstream); the Phase 1 fixture just didn't exercise
# those code paths. Phase 2.0 tests fill that gap.
# ───────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_unified_attention_2d_e8_v2_0(
    output_ptr,
    query_ptr,
    K_idx1_ptr,
    K_idx2_ptr,
    K_scale_ptr,
    V_idx1_ptr,
    V_idx2_ptr,
    V_scale_ptr,
    codebook_ptr,
    H_mat_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    rs,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    N_GROUPS: tl.constexpr,
    stride_k_idx_0: tl.int64,
    stride_k_idx_1: tl.int64,
    stride_k_idx_2: tl.int64,
    stride_k_idx_3: tl.constexpr,
    stride_v_idx_0: tl.int64,
    stride_v_idx_1: tl.int64,
    stride_v_idx_2: tl.int64,
    stride_v_idx_3: tl.constexpr,
    stride_k_scale_0: tl.int64,
    stride_k_scale_1: tl.int64,
    stride_k_scale_2: tl.int64,
    stride_k_scale_3: tl.constexpr,
    stride_v_scale_0: tl.int64,
    stride_v_scale_1: tl.int64,
    stride_v_scale_2: tl.int64,
    stride_v_scale_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    g_arange = tl.arange(0, N_GROUPS)
    cb_inner = tl.arange(0, 8)
    h_rows = tl.arange(0, 8)[:, None]
    h_cols = tl.arange(0, 8)[None, :]
    H_mat = tl.load(H_mat_ptr + h_rows * 8 + h_cols).to(tl.float32)

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        slot = seq_offset % BLOCK_SIZE

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ───── K: full dequant ─────────────────────────────────────
        k_idx_off = (
            physical_block_idx[:, None] * stride_k_idx_0
            + slot[:, None] * stride_k_idx_1
            + kv_head_idx * stride_k_idx_2
            + g_arange[None, :] * stride_k_idx_3
        )
        k_idx1 = tl.load(
            K_idx1_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_idx2 = tl.load(
            K_idx2_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_cb_off1 = k_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        k_cb_off2 = k_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        k_dec = tl.load(codebook_ptr + k_cb_off1).to(tl.float32)
        k_dec = k_dec + tl.load(codebook_ptr + k_cb_off2).to(tl.float32) / rs

        k_scale_off = (
            physical_block_idx[:, None] * stride_k_scale_0
            + slot[:, None] * stride_k_scale_1
            + kv_head_idx * stride_k_scale_2
            + g_arange[None, :] * stride_k_scale_3
        )
        k_scale = tl.load(
            K_scale_ptr + k_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        k_dec = k_dec * k_scale[:, :, None]

        # Hadamard: out[t, g, j] = sum_k dec[t, g, k] * H[k, j]
        # Inline the broadcast + reduction in a single expression so Triton's
        # tile-loop fuser sees the full pattern. v1.4 used the same pattern
        # at head_size=128 — at head_size=512 (n_groups=64), the conceptual
        # intermediate [TILE=16, G=64, 8, 8] is 64 KB fp32; Triton tiles
        # the reduction across threads so the per-thread footprint stays
        # below register pressure.
        k_dec = tl.sum(
            k_dec[:, :, :, None] * H_mat[None, None, :, :], axis=2
        )

        K = tl.trans(tl.reshape(k_dec, (TILE_SIZE, HEAD_SIZE)))
        K = tl.where(tile_mask[None, :], K, 0.0).to(Q.dtype)

        # ───── V: full dequant ─────────────────────────────────────
        v_idx_off = (
            physical_block_idx[:, None] * stride_v_idx_0
            + slot[:, None] * stride_v_idx_1
            + kv_head_idx * stride_v_idx_2
            + g_arange[None, :] * stride_v_idx_3
        )
        v_idx1 = tl.load(
            V_idx1_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_idx2 = tl.load(
            V_idx2_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_cb_off1 = v_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        v_cb_off2 = v_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        v_dec = tl.load(codebook_ptr + v_cb_off1).to(tl.float32)
        v_dec = v_dec + tl.load(codebook_ptr + v_cb_off2).to(tl.float32) / rs

        v_scale_off = (
            physical_block_idx[:, None] * stride_v_scale_0
            + slot[:, None] * stride_v_scale_1
            + kv_head_idx * stride_v_scale_2
            + g_arange[None, :] * stride_v_scale_3
        )
        v_scale = tl.load(
            V_scale_ptr + v_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dec = v_dec * v_scale[:, :, None]

        v_dec = tl.sum(
            v_dec[:, :, :, None] * H_mat[None, None, :, :], axis=2
        )

        V = tl.reshape(v_dec, (TILE_SIZE, HEAD_SIZE))
        V = tl.where(tile_mask[:, None], V, 0.0).to(Q.dtype)

        # ───── Softmax + accumulate ───────────────────────────────
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def unified_attention_e8_v2_0(
    q: torch.Tensor,
    k_idx1: torch.Tensor,
    k_idx2: torch.Tensor,
    k_scale: torch.Tensor,
    v_idx1: torch.Tensor,
    v_idx2: torch.Tensor,
    v_scale: torch.Tensor,
    codebook: torch.Tensor,
    H_mat: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    resid_scale: float,
    block_table: torch.Tensor,
):
    """Phase 2.0 launcher — generalised head_size / GQA / varlen.

    Same dequant pipeline as v1.4 but the Hadamard application is
    refactored to a constexpr j-loop, which keeps the intermediate
    bounded as head_size grows (16 → 64 groups).

    Accepts any combination of ``head_size ∈ {n_groups × 8}``,
    ``num_q_heads ∈ {k × num_kv_heads}``, and multi-sequence
    batches (driven by ``cu_seqlens_q`` / ``seqused_k`` / per-row
    ``block_table``).
    """
    block_size = k_idx1.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_idx1.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    n_groups = k_idx1.shape[3]
    assert head_size == n_groups * 8
    assert k_scale.shape == k_idx1.shape
    assert codebook.shape == (65536, 8)
    assert H_mat.shape == (8, 8)
    assert H_mat.is_contiguous()

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    TILE_SIZE = 16

    # Resource tuning. At head_size=512 the Hadamard's [TILE, G=64, 8, 8]
    # broadcast intermediate forces Triton's smem allocator past
    # Blackwell's 101 KB default carveout. ``num_stages=1`` disables the
    # k-loop double-buffer (~halves smem); ``num_warps=8`` spreads the
    # work over twice as many threads so per-thread register pressure
    # stays low. head_size ≤ 256 doesn't need this and uses the
    # Triton defaults.
    if head_size > 256:
        launch_kwargs = dict(num_stages=1, num_warps=8)
    else:
        launch_kwargs = {}

    kernel_unified_attention_2d_e8_v2_0[
        (total_num_q_blocks, num_kv_heads)
    ](
        output_ptr=out,
        query_ptr=q,
        K_idx1_ptr=k_idx1, K_idx2_ptr=k_idx2, K_scale_ptr=k_scale,
        V_idx1_ptr=v_idx1, V_idx2_ptr=v_idx2, V_scale_ptr=v_scale,
        codebook_ptr=codebook,
        H_mat_ptr=H_mat,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        rs=resid_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        N_GROUPS=n_groups,
        stride_k_idx_0=k_idx1.stride(0),
        stride_k_idx_1=k_idx1.stride(1),
        stride_k_idx_2=k_idx1.stride(2),
        stride_k_idx_3=k_idx1.stride(3),
        stride_v_idx_0=v_idx1.stride(0),
        stride_v_idx_1=v_idx1.stride(1),
        stride_v_idx_2=v_idx1.stride(2),
        stride_v_idx_3=v_idx1.stride(3),
        stride_k_scale_0=k_scale.stride(0),
        stride_k_scale_1=k_scale.stride(1),
        stride_k_scale_2=k_scale.stride(2),
        stride_k_scale_3=k_scale.stride(3),
        stride_v_scale_0=v_scale.stride(0),
        stride_v_scale_1=v_scale.stride(1),
        stride_v_scale_2=v_scale.stride(2),
        stride_v_scale_3=v_scale.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        **launch_kwargs,
    )


# ───────────────────────────────────────────────────────────────────────
# Phase 2.1 — Sliding-window mask.
#
# Adds ``SLIDING_WINDOW: tl.constexpr`` and three mirrored-from-upstream
# blocks:
#
#   1. Tile-pruning: skip tiles entirely outside the window
#      (vllm/v1/attention/ops/triton_unified_attention.py lines 209-230).
#      Pure optimisation — without it long-context perf collapses.
#
#   2. Per-element mask extension (upstream line 291-292):
#      ``seq_mask &= (query_abs_pos - seq_offset) < SLIDING_WINDOW``
#      The correctness-critical change.
#
#   3. V mask under sliding-window (upstream lines 378-382):
#      ``V = tl.where((context_len + qpos_lo - seq_offset[:, None]) <
#      SLIDING_WINDOW, V, 0.0)`` — belt-and-braces; in theory P=0 at
#      masked positions makes V's value irrelevant for P @ V, but
#      mirroring upstream is safer than relying on the equivalence.
#
# ``SLIDING_WINDOW=0`` disables all three branches — the kernel
# reduces to v2.0 exactly.
# ───────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_unified_attention_2d_e8_v2_1(
    output_ptr,
    query_ptr,
    K_idx1_ptr,
    K_idx2_ptr,
    K_scale_ptr,
    V_idx1_ptr,
    V_idx2_ptr,
    V_scale_ptr,
    codebook_ptr,
    H_mat_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    rs,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    N_GROUPS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    stride_k_idx_0: tl.int64,
    stride_k_idx_1: tl.int64,
    stride_k_idx_2: tl.int64,
    stride_k_idx_3: tl.constexpr,
    stride_v_idx_0: tl.int64,
    stride_v_idx_1: tl.int64,
    stride_v_idx_2: tl.int64,
    stride_v_idx_3: tl.constexpr,
    stride_k_scale_0: tl.int64,
    stride_k_scale_1: tl.int64,
    stride_k_scale_2: tl.int64,
    stride_k_scale_3: tl.constexpr,
    stride_v_scale_0: tl.int64,
    stride_v_scale_1: tl.int64,
    stride_v_scale_2: tl.int64,
    stride_v_scale_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ─── Sliding-window tile-pruning (upstream lines 209-230) ────────
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    g_arange = tl.arange(0, N_GROUPS)
    cb_inner = tl.arange(0, 8)
    h_rows = tl.arange(0, 8)[:, None]
    h_cols = tl.arange(0, 8)[None, :]
    H_mat = tl.load(H_mat_ptr + h_rows * 8 + h_cols).to(tl.float32)

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        slot = seq_offset % BLOCK_SIZE

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ───── K: full dequant ─────────────────────────────────────
        k_idx_off = (
            physical_block_idx[:, None] * stride_k_idx_0
            + slot[:, None] * stride_k_idx_1
            + kv_head_idx * stride_k_idx_2
            + g_arange[None, :] * stride_k_idx_3
        )
        k_idx1 = tl.load(
            K_idx1_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_idx2 = tl.load(
            K_idx2_ptr + k_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        k_cb_off1 = k_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        k_cb_off2 = k_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        k_dec = tl.load(codebook_ptr + k_cb_off1).to(tl.float32)
        k_dec = k_dec + tl.load(codebook_ptr + k_cb_off2).to(tl.float32) / rs

        k_scale_off = (
            physical_block_idx[:, None] * stride_k_scale_0
            + slot[:, None] * stride_k_scale_1
            + kv_head_idx * stride_k_scale_2
            + g_arange[None, :] * stride_k_scale_3
        )
        k_scale = tl.load(
            K_scale_ptr + k_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        k_dec = k_dec * k_scale[:, :, None]

        k_dec = tl.sum(
            k_dec[:, :, :, None] * H_mat[None, None, :, :], axis=2
        )

        K = tl.trans(tl.reshape(k_dec, (TILE_SIZE, HEAD_SIZE)))
        K = tl.where(tile_mask[None, :], K, 0.0).to(Q.dtype)

        # ───── V: full dequant ─────────────────────────────────────
        v_idx_off = (
            physical_block_idx[:, None] * stride_v_idx_0
            + slot[:, None] * stride_v_idx_1
            + kv_head_idx * stride_v_idx_2
            + g_arange[None, :] * stride_v_idx_3
        )
        v_idx1 = tl.load(
            V_idx1_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_idx2 = tl.load(
            V_idx2_ptr + v_idx_off, mask=tile_mask[:, None], other=0,
        ).to(tl.int32) & 0xFFFF
        v_cb_off1 = v_idx1[:, :, None] * 8 + cb_inner[None, None, :]
        v_cb_off2 = v_idx2[:, :, None] * 8 + cb_inner[None, None, :]
        v_dec = tl.load(codebook_ptr + v_cb_off1).to(tl.float32)
        v_dec = v_dec + tl.load(codebook_ptr + v_cb_off2).to(tl.float32) / rs

        v_scale_off = (
            physical_block_idx[:, None] * stride_v_scale_0
            + slot[:, None] * stride_v_scale_1
            + kv_head_idx * stride_v_scale_2
            + g_arange[None, :] * stride_v_scale_3
        )
        v_scale = tl.load(
            V_scale_ptr + v_scale_off, mask=tile_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dec = v_dec * v_scale[:, :, None]

        v_dec = tl.sum(
            v_dec[:, :, :, None] * H_mat[None, None, :, :], axis=2
        )

        V = tl.reshape(v_dec, (TILE_SIZE, HEAD_SIZE))
        V = tl.where(tile_mask[:, None], V, 0.0).to(Q.dtype)

        # ───── Causal + sliding-window mask + Q @ K → S ───────────
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            # Upstream line 291-292.
            seq_mask = seq_mask & ((query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW)

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        # ───── V mask under sliding-window (upstream lines 378-382) ──
        if SLIDING_WINDOW > 0:
            qpos_lo_v = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo_v - seq_offset[:, None]) < SLIDING_WINDOW,
                V, 0.0,
            )

        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def unified_attention_e8_v2_1(
    q: torch.Tensor,
    k_idx1: torch.Tensor,
    k_idx2: torch.Tensor,
    k_scale: torch.Tensor,
    v_idx1: torch.Tensor,
    v_idx2: torch.Tensor,
    v_scale: torch.Tensor,
    codebook: torch.Tensor,
    H_mat: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    resid_scale: float,
    block_table: torch.Tensor,
    sliding_window: int = 0,
):
    """Phase 2.1 launcher — Phase 2.0 + sliding-window mask.

    ``sliding_window=0`` disables the window (kernel reduces to v2.0
    exactly). Values >0 restrict each query to attend to keys in
    ``(query_abs_pos - sliding_window, query_abs_pos]``.
    """
    block_size = k_idx1.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_idx1.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    n_groups = k_idx1.shape[3]
    assert head_size == n_groups * 8
    assert k_scale.shape == k_idx1.shape
    assert codebook.shape == (65536, 8)
    assert H_mat.shape == (8, 8)
    assert H_mat.is_contiguous()

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    TILE_SIZE = 16

    if head_size > 256:
        launch_kwargs = dict(num_stages=1, num_warps=8)
    else:
        launch_kwargs = {}

    kernel_unified_attention_2d_e8_v2_1[
        (total_num_q_blocks, num_kv_heads)
    ](
        output_ptr=out,
        query_ptr=q,
        K_idx1_ptr=k_idx1, K_idx2_ptr=k_idx2, K_scale_ptr=k_scale,
        V_idx1_ptr=v_idx1, V_idx2_ptr=v_idx2, V_scale_ptr=v_scale,
        codebook_ptr=codebook,
        H_mat_ptr=H_mat,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        rs=resid_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        N_GROUPS=n_groups,
        SLIDING_WINDOW=sliding_window,
        stride_k_idx_0=k_idx1.stride(0),
        stride_k_idx_1=k_idx1.stride(1),
        stride_k_idx_2=k_idx1.stride(2),
        stride_k_idx_3=k_idx1.stride(3),
        stride_v_idx_0=v_idx1.stride(0),
        stride_v_idx_1=v_idx1.stride(1),
        stride_v_idx_2=v_idx1.stride(2),
        stride_v_idx_3=v_idx1.stride(3),
        stride_k_scale_0=k_scale.stride(0),
        stride_k_scale_1=k_scale.stride(1),
        stride_k_scale_2=k_scale.stride(2),
        stride_k_scale_3=k_scale.stride(3),
        stride_v_scale_0=v_scale.stride(0),
        stride_v_scale_1=v_scale.stride(1),
        stride_v_scale_2=v_scale.stride(2),
        stride_v_scale_3=v_scale.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        **launch_kwargs,
    )
