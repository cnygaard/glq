"""Phase 2.0 acceptance gate: shape / GQA / varlen sweep.

Kernel ``kernel_unified_attention_2d_e8_v2_0`` extends v1.4 along
three orthogonal axes (head_size, GQA, varlen). The Hadamard step is
rewritten as a constexpr j-loop to keep the per-iteration footprint
bounded as ``n_groups`` grows from 16 → 64 (head_size 128 → 512).

Test matrix covers:
- head_size ∈ {128, 256, 512}  (n_groups ∈ {16, 32, 64})
- (num_q_heads, num_kv_heads) ∈ {(1,1), (4,1), (8,4)}  (no GQA, 4× GQA, 2× GQA)
- num_seqs ∈ {1, 2}  (varlen via cu_seqlens_q / per-seq block_table)
- Tk ∈ {32, 64}

All against ``reference_e8_attention`` at rtol=5e-3, atol=5e-3.
"""
from __future__ import annotations

import math
from itertools import product

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


@pytest.fixture(scope="module")
def quantizer_4bpw():
    from glq.codebook import E8ShellCodebook
    from glq.kv_e8 import E8KVQuantizer
    cb = E8ShellCodebook(verbose=False)
    for name in ("codebook", "codebook_norms", "codebook_half",
                 "codebook_half_t", "codebook_norms_half",
                 "codebook_packed"):
        setattr(cb, name, getattr(cb, name).to("cuda"))
    cb.device = "cuda"
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


@pytest.fixture(scope="module")
def hadamard_8():
    from glq_vllm.e8_paged_cache import _get_hadamard
    return _get_hadamard(torch.float32, "cuda")


def _quantize_to_paged(K, V, quantizer, *, num_blocks, block_size):
    """Pack quantized K/V into the kernel's [num_blocks, block_size, H, G] layout.

    K/V have shape ``[Tk_total, H_kv, head_size]`` where ``Tk_total`` is
    the concatenation of all sequences' Tk. Each token's K/V goes to
    slot ``(t // block_size, t % block_size)`` — the simplest packing
    that matches what the test's ``block_table`` will reference.
    """
    Tk_total, H_kv, D = K.shape
    n_groups = D // 8

    qk = quantizer.quantize(K)
    qv = quantizer.quantize(V)

    def _pack_int16(qt_idx):
        flat = qt_idx.reshape(Tk_total, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=torch.int16, device=K.device,
        )
        for t in range(Tk_total):
            paged[t // block_size, t % block_size] = flat[t]
        return paged

    def _pack_scale(scale_tensor):
        flat = scale_tensor.reshape(Tk_total, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=flat.dtype, device=K.device,
        )
        for t in range(Tk_total):
            paged[t // block_size, t % block_size] = flat[t]
        return paged

    return (
        _pack_int16(qk["idx1"]), _pack_int16(qk["idx2"]),
        _pack_scale(qk["scale"]),
        _pack_int16(qv["idx1"]), _pack_int16(qv["idx2"]),
        _pack_scale(qv["scale"]),
        qk, qv,
    )


def _run_kernel(q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
                codebook, H_mat, resid_scale, *, cu_seqlens_q, seqused_k,
                block_table, softmax_scale):
    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v2_0,
    )
    out = torch.zeros_like(q)
    unified_attention_e8_v2_0(
        q=q,
        k_idx1=k_idx1, k_idx2=k_idx2, k_scale=k_scale,
        v_idx1=v_idx1, v_idx2=v_idx2, v_scale=v_scale,
        codebook=codebook, H_mat=H_mat,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        resid_scale=resid_scale,
        block_table=block_table,
    )
    return out


def _reference_per_seq(q_seq, qk_full, qv_full, quantizer, *,
                        seq_token_range, softmax_scale):
    """Run reference attention for one sequence's slice of a multi-seq batch.

    ``qk_full`` / ``qv_full`` are quantizer dicts for ALL sequences' K/V
    concatenated; ``seq_token_range`` = ``(start, end)`` slices out
    this sequence's K/V before calling ``reference_e8_attention``.
    """
    from tests.test_e8_attention_reference import reference_e8_attention
    start, end = seq_token_range
    # Slice idx/scale tensors. ``qk_full['idx1']`` has shape [Tk_total*H_kv, G].
    # Reshape via the original ``shape`` field of the quantize dict.
    orig_shape = qk_full["shape"]
    Tk_total, H_kv, D = orig_shape
    n_groups = D // 8

    def _slice(qt):
        sliced = {
            "idx1": qt["idx1"].reshape(Tk_total, H_kv, n_groups)[start:end]
                    .reshape(-1, n_groups),
            "idx2": qt["idx2"].reshape(Tk_total, H_kv, n_groups)[start:end]
                    .reshape(-1, n_groups),
            "scale": qt["scale"].reshape(Tk_total, H_kv, n_groups)[start:end]
                     .reshape((end - start) * H_kv, n_groups),
            "shape": (end - start, H_kv, D),
            "dtype": qt["dtype"],
        }
        # ``scale``'s shape should match the prefix of original shape + (G,).
        # The reference treats the leading dims as a single N, so the flat
        # `(N, G)` shape is fine.
        return sliced

    qk_slice = _slice(qk_full)
    qv_slice = _slice(qv_full)
    return reference_e8_attention(
        q_seq, qk_slice, qv_slice, quantizer,
        causal=True, softmax_scale=softmax_scale,
    )


def _build_random(Tk_per_seq, Tq_per_seq, num_q_heads, num_kv_heads,
                  head_size, num_seqs, seed):
    """Build q, K, V for a multi-seq batch.

    Returns:
      q: [Tq_total, H_q, D]
      K: [Tk_total, H_kv, D]
      V: [Tk_total, H_kv, D]
      seq_layouts: list of (start_q, end_q, start_k, end_k) per seq
    """
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(seed)

    Tq_total = num_seqs * Tq_per_seq
    Tk_total = num_seqs * Tk_per_seq

    q = torch.randn(Tq_total, num_q_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    K = torch.randn(Tk_total, num_kv_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    V = torch.randn(Tk_total, num_kv_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    layouts = [
        (s * Tq_per_seq, (s + 1) * Tq_per_seq,
         s * Tk_per_seq, (s + 1) * Tk_per_seq)
        for s in range(num_seqs)
    ]
    return q, K, V, layouts


# ── Test matrix ─────────────────────────────────────────────────────

_HEAD_SIZES = [128, 256, 512]
_HEAD_CONFIGS = [(1, 1), (4, 1), (8, 4)]   # (num_q, num_kv)
_NUM_SEQS = [1, 2]
_TKS = [32, 64]

_PARAMS = [
    (hs, nq, nkv, ns, tk)
    for hs, (nq, nkv), ns, tk in product(
        _HEAD_SIZES, _HEAD_CONFIGS, _NUM_SEQS, _TKS,
    )
]


@pytest.mark.parametrize(
    "head_size,num_q,num_kv,num_seqs,Tk_per_seq",
    _PARAMS,
    ids=[f"H{hs}_Q{nq}_KV{nkv}_S{ns}_Tk{tk}" for hs, nq, nkv, ns, tk in _PARAMS],
)
def test_phase_2_0_shape_gqa_varlen(
    quantizer_4bpw, hadamard_8,
    head_size, num_q, num_kv, num_seqs, Tk_per_seq,
):
    block_size = 16
    Tq_per_seq = Tk_per_seq  # Prefill-shaped (Tq == Tk per seq) to keep
                              # reference causal-mask convention happy.

    blocks_per_seq = (Tk_per_seq + block_size - 1) // block_size
    num_blocks = num_seqs * blocks_per_seq

    q, K, V, layouts = _build_random(
        Tk_per_seq=Tk_per_seq, Tq_per_seq=Tq_per_seq,
        num_q_heads=num_q, num_kv_heads=num_kv,
        head_size=head_size, num_seqs=num_seqs,
        seed=hash((head_size, num_q, num_kv, num_seqs, Tk_per_seq)) & 0xFFFF,
    )
    (k_idx1, k_idx2, k_scale,
     v_idx1, v_idx2, v_scale,
     qk_full, qv_full) = _quantize_to_paged(
        K, V, quantizer_4bpw,
        num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_4bpw.codebook.codebook_half
    rs = float(quantizer_4bpw.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    # Build varlen metadata. Each seq has Tq_per_seq queries, Tk_per_seq keys.
    cu_seqlens_q = torch.tensor(
        [s * Tq_per_seq for s in range(num_seqs + 1)],
        device=q.device, dtype=torch.int32,
    )
    seqused_k = torch.full(
        (num_seqs,), Tk_per_seq, device=q.device, dtype=torch.int32,
    )
    block_table = torch.zeros(
        num_seqs, blocks_per_seq, device=q.device, dtype=torch.int32,
    )
    for s in range(num_seqs):
        block_table[s] = torch.arange(
            s * blocks_per_seq, (s + 1) * blocks_per_seq,
            device=q.device, dtype=torch.int32,
        )

    out_fk = _run_kernel(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        block_table=block_table, softmax_scale=softmax_scale,
    )

    # Reference: compute per-seq SDPA on each slice, then concatenate.
    out_ref_parts = []
    for (sq, eq, sk, ek) in layouts:
        out_ref_parts.append(_reference_per_seq(
            q[sq:eq], qk_full, qv_full, quantizer_4bpw,
            seq_token_range=(sk, ek), softmax_scale=softmax_scale,
        ))
    out_ref = torch.cat(out_ref_parts, dim=0)

    max_abs = (out_ref.float() - out_fk.float()).abs().max().item()
    rmse = (out_ref.float() - out_fk.float()).pow(2).mean().sqrt().item()
    print(f"  H={head_size:>3} Q={num_q} KV={num_kv} seqs={num_seqs} Tk={Tk_per_seq}: "
          f"max-abs={max_abs:.2e} RMSE={rmse:.2e}")
    assert max_abs < 5e-3, (
        f"head_size={head_size}, num_q={num_q}, num_kv={num_kv}, "
        f"num_seqs={num_seqs}, Tk={Tk_per_seq}: max-abs={max_abs:.2e} "
        f">= 5e-3 (RMSE={rmse:.2e})"
    )
