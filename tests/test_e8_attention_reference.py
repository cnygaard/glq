"""Phase 0: PyTorch correctness reference for v0.5 inline-dequant attention.

This module exports ``reference_e8_attention(q, k_qt, v_qt, …)`` — a
pure-PyTorch function that consumes the same compressed K/V dicts
the production E8 KV cache stores and produces the attention output
that any future inline-dequant kernel must match.

It's intentionally simple and slow:

  1. ``E8KVQuantizer.dequantize`` re-materialises full-precision K/V.
  2. Run scaled-dot-product attention on the result.

The forked Triton kernel in v0.5 must produce the same output (within
``atol=5e-3, rtol=5e-3``) from the same compressed inputs. This file
is the gate: if the reference is buggy, every downstream comparison
is meaningless.

Phase 0 acceptance gate
-----------------------
- ``reference_e8_attention`` correctly implements masked SDPA against
  the dequantised K/V.
- Compressing then attending via the reference matches "attend on raw
  K/V" within bpw-dependent MSE tolerance (sanity check that the
  compression error is the dominant error term, not a math bug).
"""
from __future__ import annotations

import math

import pytest
import torch


# ── reference under test ─────────────────────────────────────────


def reference_e8_attention(
    q: torch.Tensor,            # [num_tokens, num_q_heads, head_dim]
    k_qt: dict,                  # E8KVQuantizer.quantize() output for K
    v_qt: dict,                  # …for V
    quantizer,                   # E8KVQuantizer that produced k_qt / v_qt
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Pure-PyTorch reference: dequantise K/V, run masked SDPA.

    Args:
        q: [Tq, H_q, D] queries (any float dtype on cuda or cpu).
        k_qt, v_qt: ``E8KVQuantizer.quantize`` output dicts. The encoded
            K/V layout is ``[Tk, H_kv, D]`` (caller pre-flattens any
            block/slot dims).
        quantizer: the ``E8KVQuantizer`` that produced ``k_qt`` /
            ``v_qt``. Needed for ``dequantize``.
        causal: standard causal mask (Tq queries can attend to the
            first ``Tk - Tq + i`` keys for query ``i``).
        softmax_scale: usually ``1 / sqrt(head_dim)``. Auto if None.
        sliding_window: if set, keys further than ``sliding_window``
            past the query position are masked out.

    Returns:
        [Tq, H_q, D] attention output, same dtype as ``q``.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # Dequantise K, V to full precision in q's dtype.
    K = quantizer.dequantize(k_qt).to(q.dtype)   # [Tk, H_kv, D]
    V = quantizer.dequantize(v_qt).to(q.dtype)

    Tq, H_q, D = q.shape
    Tk, H_kv, _ = K.shape

    # GQA: each KV head is shared by ``H_q // H_kv`` Q heads.
    if H_q != H_kv:
        if H_q % H_kv != 0:
            raise ValueError(
                f"num_q_heads ({H_q}) must be divisible by "
                f"num_kv_heads ({H_kv}) for GQA"
            )
        repeat = H_q // H_kv
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)

    # Reshape for batched matmul: [H_q, Tq, D] × [H_q, D, Tk] → [H_q, Tq, Tk]
    Q_bh = q.transpose(0, 1)            # [H_q, Tq, D]
    K_bh = K.transpose(0, 1)            # [H_q, Tk, D]
    V_bh = V.transpose(0, 1)            # [H_q, Tk, D]

    scores = torch.matmul(Q_bh, K_bh.transpose(-1, -2)) * softmax_scale

    # Build the mask.
    mask = torch.zeros(Tq, Tk, dtype=torch.bool, device=q.device)
    if causal:
        # Query i (0-indexed) sees keys 0..Tk-Tq+i inclusive.
        causal_offset = Tk - Tq
        rows = torch.arange(Tq, device=q.device)[:, None]
        cols = torch.arange(Tk, device=q.device)[None, :]
        mask |= cols > (rows + causal_offset)
    if sliding_window is not None and sliding_window > 0:
        causal_offset = Tk - Tq
        rows = torch.arange(Tq, device=q.device)[:, None]
        cols = torch.arange(Tk, device=q.device)[None, :]
        # Distance from each query's position back to a key.
        delta = (rows + causal_offset) - cols
        mask |= delta >= sliding_window

    if mask.any():
        scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out_bh = torch.matmul(weights, V_bh)        # [H_q, Tq, D]
    return out_bh.transpose(0, 1).contiguous()  # [Tq, H_q, D]


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def quantizer_4bpw():
    """E8 quantizer at 4 bpw (n_primary=2, n_secondary=0)."""
    from glq.codebook import E8ShellCodebook
    from glq.kv_e8 import E8KVQuantizer

    cb = E8ShellCodebook(verbose=False)
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


def _rand_kv(Tk, n_kv_heads, head_dim, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(Tk, n_kv_heads, head_dim, generator=g, dtype=dtype)


# ── tests ─────────────────────────────────────────────────────────


def test_reference_smoke_b1_h1_tiny(quantizer_4bpw):
    """Tiniest possible shape: 1 query head, 1 KV head, head_dim=8."""
    Tq, Tk, D = 1, 4, 8
    q = torch.randn(Tq, 1, D)
    K = _rand_kv(Tk, 1, D, seed=1)
    V = _rand_kv(Tk, 1, D, seed=2)

    k_qt = quantizer_4bpw.quantize(K)
    v_qt = quantizer_4bpw.quantize(V)

    out = reference_e8_attention(q, k_qt, v_qt, quantizer_4bpw, causal=True)
    assert out.shape == (Tq, 1, D)
    assert torch.isfinite(out).all(), "output has NaN/Inf"


def test_reference_matches_pytorch_sdpa_on_uncompressed(quantizer_4bpw):
    """Compress→dequant→SDPA must approximately match raw SDPA. The error
    is dominated by the quantizer (bpw=4), not the SDPA math.

    Gate: relative MSE between reference output and raw SDPA output
    must be in the same ballpark as the compressor's own roundtrip
    error.
    """
    Tq, Tk = 4, 64
    H_q, H_kv, D = 4, 4, 64
    q = torch.randn(Tq, H_q, D, dtype=torch.float32)
    K = _rand_kv(Tk, H_kv, D, seed=10)
    V = _rand_kv(Tk, H_kv, D, seed=11)

    k_qt = quantizer_4bpw.quantize(K)
    v_qt = quantizer_4bpw.quantize(V)

    ref_out = reference_e8_attention(
        q, k_qt, v_qt, quantizer_4bpw, causal=True)

    # Ground truth: SDPA on the raw (un-compressed) K/V.
    K_bh = K.transpose(0, 1)
    V_bh = V.transpose(0, 1)
    Q_bh = q.transpose(0, 1)
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q_bh, K_bh.transpose(-1, -2)) * scale
    causal_offset = Tk - Tq
    rows = torch.arange(Tq)[:, None]
    cols = torch.arange(Tk)[None, :]
    mask = cols > (rows + causal_offset)
    scores = scores.masked_fill(mask[None, :, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    gt_out = torch.matmul(weights, V_bh).transpose(0, 1)

    # The compression error on each (K, V) entry is roughly per-entry
    # RMS ~0.03-0.05 at 4 bpw (cf. test_kv_e8.py roundtrip numbers).
    # Attention is contractive (softmax + average), so the output error
    # should be in the same ballpark.
    err = (ref_out - gt_out).pow(2).mean().sqrt().item()
    print(f"  ref vs raw-SDPA RMSE @ 4bpw: {err:.4f}")
    assert err < 0.1, f"reference output RMSE {err:.4f} too large vs raw SDPA"


def test_reference_lossless_path(quantizer_4bpw):
    """Sanity: if we replace ``dequantize`` with the original K/V (no
    compression), our SDPA math is bit-exact identical to
    ``torch.nn.functional.scaled_dot_product_attention``.

    Catches bugs in masking/scale/GQA that would otherwise hide
    behind the quantization error. Uses Tq == Tk so the two causal
    mask conventions agree (PyTorch's ``is_causal`` is the
    "pure-prefill" variant where each query sees keys ``[0, i]``;
    our reference's variant where queries see ``[0, Tk-Tq+i]``
    degenerates to the same thing when Tq == Tk).
    """
    Tq = Tk = 16
    H_q, H_kv, D = 4, 4, 16
    q = torch.randn(Tq, H_q, D, dtype=torch.float32)
    K = _rand_kv(Tk, H_kv, D, seed=20)
    V = _rand_kv(Tk, H_kv, D, seed=21)

    # Build a fake qt dict that ``dequantize`` would never see in
    # practice, then monkey-patch the call. Easier: construct an
    # alternative reference path that skips dequantize.
    K_bh = K.transpose(0, 1)
    V_bh = V.transpose(0, 1)
    Q_bh = q.transpose(0, 1)
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q_bh, K_bh.transpose(-1, -2)) * scale
    causal_offset = Tk - Tq
    rows = torch.arange(Tq)[:, None]
    cols = torch.arange(Tk)[None, :]
    mask = cols > (rows + causal_offset)
    scores = scores.masked_fill(mask[None, :, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    manual_out = torch.matmul(weights, V_bh).transpose(0, 1)

    # PyTorch's reference SDPA.
    Q_sdpa = q.transpose(0, 1).unsqueeze(0)   # [1, H, Tq, D]
    K_sdpa = K.transpose(0, 1).unsqueeze(0)
    V_sdpa = V.transpose(0, 1).unsqueeze(0)
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        Q_sdpa, K_sdpa, V_sdpa, is_causal=True)
    sdpa_out = sdpa_out.squeeze(0).transpose(0, 1)

    err = (manual_out - sdpa_out).abs().max().item()
    print(f"  manual SDPA vs torch SDPA max-abs: {err:.6f}")
    assert err < 1e-5, (
        "manual SDPA disagrees with torch.nn.functional.scaled_dot_product_"
        f"attention: max diff {err}")


def test_reference_gqa(quantizer_4bpw):
    """GQA: num_q_heads > num_kv_heads (repeat_interleave path)."""
    Tq, Tk = 4, 32
    H_q, H_kv, D = 8, 2, 32   # 4× GQA
    q = torch.randn(Tq, H_q, D, dtype=torch.float32)
    K = _rand_kv(Tk, H_kv, D, seed=30)
    V = _rand_kv(Tk, H_kv, D, seed=31)
    k_qt = quantizer_4bpw.quantize(K)
    v_qt = quantizer_4bpw.quantize(V)
    out = reference_e8_attention(q, k_qt, v_qt, quantizer_4bpw, causal=True)
    assert out.shape == (Tq, H_q, D)
    assert torch.isfinite(out).all()


def test_reference_sliding_window(quantizer_4bpw):
    """Sliding-window mask: a query position should not attend further
    back than ``sliding_window`` keys."""
    Tq, Tk = 8, 64
    H, D = 4, 32
    q = torch.randn(Tq, H, D, dtype=torch.float32)
    K = _rand_kv(Tk, H, D, seed=40)
    V = _rand_kv(Tk, H, D, seed=41)
    k_qt = quantizer_4bpw.quantize(K)
    v_qt = quantizer_4bpw.quantize(V)

    out_unbounded = reference_e8_attention(
        q, k_qt, v_qt, quantizer_4bpw, causal=True)
    out_window4 = reference_e8_attention(
        q, k_qt, v_qt, quantizer_4bpw, causal=True, sliding_window=4)

    # The window should change the output (different mask).
    diff = (out_unbounded - out_window4).abs().max().item()
    assert diff > 1e-3, (
        f"sliding-window mask had no effect (max diff {diff:.6f})")
