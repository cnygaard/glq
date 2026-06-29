"""Parity gate for the batched MoE-expert quantization (Tier 2).

The 128 routed experts of a gemma-4 MoE layer are independent, identically
shaped, and share the RHT. Quantizing them batched across the expert axis must
be *mathematically per-expert-identical* to looping the single-expert path —
that is the whole correctness claim behind the launch-overhead win. These tests
pin:

  1. ``quantize_ldlq_codebook_nstage_batched`` == looping ``quantize_ldlq_codebook_nstage``:
     the stored artifact (the per-stage indices) is bit-for-bit identical on CPU
     fp32 (same Wscale passed to both removes the fp32-vs-double scale wobble;
     bmm==mm and cholesky_ex==cholesky in fp32). The reconstructed W_hat matches
     to fp32 ULP noise (~1e-8) — the batched recon uses ``add_(.., alpha=)`` whose
     FMA-order rounds a hair differently from the single path's ``+= dec*c``.
  2. The RHT ``*_batched`` transforms == stacking the per-expert 2D transforms.
  3. ``block_LDL_batched`` recovers a non-PD expert (identity fallback) without
     killing the rest of the batch — a single bad Hessian would crash the
     single-expert ``cholesky``.

CPU-only is sufficient: it is the bit-exact reference. (On CUDA the fp16 bmm
tiling differs from mm, so the GPU gate is SQNR-within-noise, checked e2e on the
box, not here.)
"""
import math

import torch

from glq.codebook_e8p import E8PCodebook
from glq.ldlq import (
    quantize_ldlq_codebook_nstage,
    quantize_ldlq_codebook_nstage_batched,
    block_LDL,
    block_LDL_batched,
)
from glq.rht import RHT


def _spd(E, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(E, n, n, generator=g, dtype=torch.float32)
    return A @ A.transpose(-1, -2) + n * torch.eye(n)        # well-conditioned SPD


def _recipe(cb):
    """2-stage e8p RVQ (== 4bpw): [E8P, E8P] with one residual scale."""
    return [cb, cb], [3.45]


def test_nstage_batched_indices_bit_identical_cpu():
    E, m, n = 4, 32, 64
    cb = E8PCodebook(device='cpu', verbose=False)
    codebooks, resid_scales = _recipe(cb)

    g = torch.Generator().manual_seed(7)
    W = torch.randn(E, m, n, generator=g, dtype=torch.float32) * 0.05
    H = _spd(E, n, seed=11)

    # Pass an explicit per-expert Wscale to both paths so the only remaining
    # difference is the (bit-exact in fp32) batched math.
    Wscale = (W.pow(2).mean(dim=(-2, -1)).sqrt() * cb.opt_scale)   # (E,)

    batched = quantize_ldlq_codebook_nstage_batched(
        W, H, codebooks=codebooks, resid_scales=resid_scales, Wscale=Wscale)

    for e in range(E):
        single = quantize_ldlq_codebook_nstage(
            W[e], H[e], codebooks=codebooks, resid_scales=resid_scales,
            Wscale=float(Wscale[e]))
        for s in range(len(codebooks)):
            # The stored artifact (indices) is bit-for-bit identical.
            assert torch.equal(batched['all_indices'][s][e], single['all_indices'][s]), \
                f"stage {s} expert {e}: batched indices != single"
        # W_hat matches to fp32 ULP noise (add_(alpha) FMA-order vs += dec*c).
        assert torch.allclose(batched['W_hat'][e], single['W_hat'], atol=1e-6, rtol=1e-5), \
            f"expert {e}: batched W_hat != single beyond fp32 noise"

    # Returned scalars carry the expert dim.
    assert batched['W_hat'].shape == (E, m, n)
    assert len(batched['all_indices']) == len(codebooks)
    assert batched['all_indices'][0].shape == (E, m, n // cb.codesz)
    assert batched['Wscale'].shape == (E,)
    assert batched['proxy_loss'].shape == (E,)


def test_nstage_batched_auto_wscale_close_cpu():
    # Without an explicit Wscale the batched path computes it per-expert in fp32
    # while the single path uses a python double — the resulting W_hat must still
    # match to fp32 noise (rare argmax tie-flips at most).
    E, m, n = 3, 16, 64
    cb = E8PCodebook(device='cpu', verbose=False)
    codebooks, resid_scales = _recipe(cb)

    g = torch.Generator().manual_seed(3)
    W = torch.randn(E, m, n, generator=g, dtype=torch.float32) * 0.1
    H = _spd(E, n, seed=5)

    batched = quantize_ldlq_codebook_nstage_batched(
        W, H, codebooks=codebooks, resid_scales=resid_scales)
    for e in range(E):
        single = quantize_ldlq_codebook_nstage(
            W[e], H[e], codebooks=codebooks, resid_scales=resid_scales)
        assert torch.allclose(batched['W_hat'][e], single['W_hat'], atol=1e-3, rtol=1e-3)
        assert abs(float(batched['Wscale'][e]) - single['Wscale']) < 1e-5


def test_block_ldl_batched_matches_single_cpu():
    E, n, b = 4, 64, 8
    H = _spd(E, n, seed=21)
    Lb, Db = block_LDL_batched(H, block_size=b)
    assert Lb.shape == (E, n, n)
    assert Db.shape == (E, n // b, b, b)
    for e in range(E):
        Ls, Ds = block_LDL(H[e], block_size=b)
        assert torch.allclose(Lb[e], Ls, atol=1e-5, rtol=1e-5)
        assert torch.allclose(Db[e], Ds, atol=1e-5, rtol=1e-5)


def test_block_ldl_batched_nonpd_recovery_cpu():
    # Expert 3 has a zero (non-PD) Hessian that would crash the single-expert
    # cholesky; the batch must recover it via the identity fallback and leave
    # the good experts untouched.
    E, n, b = 4, 64, 8
    H = _spd(E, n, seed=31)
    H[3] = 0.0
    Lb, Db = block_LDL_batched(H, block_size=b)
    assert torch.isfinite(Lb).all() and torch.isfinite(Db).all()
    # Good experts still match the single path exactly.
    for e in range(3):
        Ls, _ = block_LDL(H[e], block_size=b)
        assert torch.allclose(Lb[e], Ls, atol=1e-5, rtol=1e-5)
    # The recovered expert's L is the identity factor (L of an identity Hessian).
    assert torch.allclose(Lb[3], torch.eye(n), atol=1e-5)


def test_nstage_batched_survives_nonpd_expert_cpu():
    # End-to-end: a dead (never-routed -> zero/identity Hessian) expert in the
    # batch must not crash the whole layer's quant.
    E, m, n = 4, 16, 64
    cb = E8PCodebook(device='cpu', verbose=False)
    codebooks, resid_scales = _recipe(cb)

    g = torch.Generator().manual_seed(9)
    W = torch.randn(E, m, n, generator=g, dtype=torch.float32) * 0.05
    H = _spd(E, n, seed=13)
    H[2] = 0.0
    out = quantize_ldlq_codebook_nstage_batched(
        W, H, codebooks=codebooks, resid_scales=resid_scales)
    assert torch.isfinite(out['W_hat']).all()
    # Healthy experts match the single path.
    for e in (0, 1, 3):
        single = quantize_ldlq_codebook_nstage(
            W[e], H[e], codebooks=codebooks, resid_scales=resid_scales,
            Wscale=float(out['Wscale'][e]))
        assert torch.equal(out['all_indices'][0][e], single['all_indices'][0])


def test_rht_batched_transforms_match_stacked_cpu():
    E, m, n = 4, 32, 128
    rht = RHT(m, n, device='cpu', e8p=True)
    g = torch.Generator().manual_seed(1)

    W = torch.randn(E, m, n, generator=g, dtype=torch.float32)
    Wb = rht.transform_weights_batched(W)
    Ws = torch.stack([rht.transform_weights(W[e]) for e in range(E)])
    assert Wb.shape == Ws.shape
    assert torch.allclose(Wb, Ws, atol=1e-5, rtol=1e-5)

    # Inverse round-trips the batched forward back to (E, m, n).
    Winv = rht.inverse_transform_weights_batched(Wb)
    assert Winv.shape == (E, m, n)
    assert torch.allclose(Winv, W, atol=1e-4, rtol=1e-4)
    Winv_s = torch.stack([rht.inverse_transform_weights(Wb[e]) for e in range(E)])
    assert torch.allclose(Winv, Winv_s, atol=1e-5, rtol=1e-5)

    A = torch.randn(E, n, n, generator=g, dtype=torch.float32)
    H = A @ A.transpose(-1, -2)                                   # symmetric
    Hb = rht.transform_hessian_batched(H)
    Hs = torch.stack([rht.transform_hessian(H[e]) for e in range(E)])
    assert torch.allclose(Hb, Hs, atol=1e-5, rtol=1e-5)


def test_experts_batched_wrapper_matches_single_cpu():
    # Full wrapper (RHT batched transform -> batched LDLQ -> inverse -> artifacts)
    # vs the per-expert quantize_layer_e8_shell_rht, for the 1-stage (2bpw) and
    # 2-stage (4bpw, the 26B-A4B target) recipes.
    from glq.quantize_model import (quantize_experts_e8_shell_rht_batched,
                                    quantize_layer_e8_shell_rht)
    cb = E8PCodebook(device='cpu', verbose=False)
    E, m, n = 3, 32, 128                                  # m mult 16, n mult 64
    g = torch.Generator().manual_seed(4)
    W = torch.randn(E, m, n, generator=g, dtype=torch.float32) * 0.02
    H = _spd(E, n, seed=8)

    for bpw in (2, 4):
        batch = quantize_experts_e8_shell_rht_batched(W, H, cb, bpw=bpw)
        assert len(batch) == E
        keys = set(batch[0][1].keys())
        assert {'Qidxs_e8p', 'SU', 'SV', 'Wscale'} <= keys
        if bpw == 4:
            assert {'Qidxs2_e8p', 'inv_resid_scale'} <= keys
        for e in range(E):
            W_hat_b, arts_b, met_b = batch[e]
            _, arts_s, met_s = quantize_layer_e8_shell_rht(W[e], H[e], cb, bpw=bpw)
            assert torch.equal(arts_b['SU'], arts_s['SU'])      # shared RHT (seed 42)
            assert torch.equal(arts_b['SV'], arts_s['SV'])
            assert arts_b['Qidxs_e8p'].shape == arts_s['Qidxs_e8p'].shape
            assert tuple(W_hat_b.shape) == (m, n)
            # SQNR within fp/tie-flip noise of the per-expert path (the quality gate).
            assert abs(met_b['sqnr'] - met_s['sqnr']) < 0.5, \
                f"bpw {bpw} expert {e}: SQNR batched {met_b['sqnr']} vs single {met_s['sqnr']}"


if __name__ == "__main__":
    test_nstage_batched_indices_bit_identical_cpu()
    test_nstage_batched_auto_wscale_close_cpu()
    test_block_ldl_batched_matches_single_cpu()
    test_block_ldl_batched_nonpd_recovery_cpu()
    test_nstage_batched_survives_nonpd_expert_cpu()
    test_rht_batched_transforms_match_stacked_cpu()
    test_experts_batched_wrapper_matches_single_cpu()
    print("OK: batched MoE-expert quant parity")
