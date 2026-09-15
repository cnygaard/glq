"""Gate G0 — unit tests for the GLVQ PoC core (benchmarks/_glvq_core.py).

Pins down the math contracts BEFORE implementation (arXiv 2510.20984):
- mu-law companding is an exact roundtrip on [-1, 1]
- group extract/assemble is an exact reshape identity
- Babai nearest-plane recovers exact lattice points and reduces to round() for G=I
- z-codes are clamped to the b-bit range
- fit() has correct autograd semantics (grads to G and mu; z detached) and reduces loss
- spectral clamp bounds hold after fit; bpw accounting matches the paper formula

CPU-only, seeded, no glq dependency (keeps the gate fast + local).
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import _glvq_core as glvq  # noqa: E402


# ----------------------------------------------------------------------------
# mu-law companding
# ----------------------------------------------------------------------------
def test_mu_compand_roundtrip_is_identity_on_unit_interval():
    x = torch.linspace(-1.0, 1.0, 401, dtype=torch.float64)
    for mu in (10.0, 50.0, 255.0):
        y = glvq.mu_compand(x, mu)
        x_rec = glvq.mu_expand(y, mu)
        assert torch.allclose(x_rec, x, atol=1e-9), f"mu={mu} max err {(x_rec-x).abs().max()}"


def test_mu_compand_maps_unit_interval_into_unit_interval():
    x = torch.linspace(-1.0, 1.0, 51, dtype=torch.float64)
    y = glvq.mu_compand(x, 100.0)
    assert y.abs().max() <= 1.0 + 1e-12
    # companding expands small values: |F(x)| >= |x| near 0
    assert glvq.mu_compand(torch.tensor(0.01, dtype=torch.float64), 100.0) > 0.01


def test_mu_compand_broadcasts_per_group_mu():
    # x (G, Nv, d), mu (G,) -> per-group companding
    x = torch.randn(3, 5, 4, dtype=torch.float64).clamp(-0.9, 0.9)
    mu = torch.tensor([10.0, 100.0, 255.0], dtype=torch.float64)
    y = glvq.mu_compand(x, mu[:, None, None])
    x_rec = glvq.mu_expand(y, mu[:, None, None])
    assert torch.allclose(x_rec, x, atol=1e-9)


# ----------------------------------------------------------------------------
# group extract / assemble
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("m,n,d,gc", [(6, 256, 8, 128), (4, 128, 16, 128), (8, 384, 8, 128), (3, 64, 16, 64)])
def test_extract_assemble_is_exact_identity(m, n, d, gc):
    W = torch.randn(m, n, dtype=torch.float64)
    V = glvq.extract_groups(W, d, group_cols=gc)
    Gn = n // gc
    Nv = m * gc // d
    assert V.shape == (Gn, Nv, d)
    W_rec = glvq.assemble_groups(V, m, n, d, group_cols=gc)
    assert torch.equal(W_rec, W)


def test_extract_vector_is_d_contiguous_columns_of_one_row():
    # a lattice vector must be d consecutive columns within one row of a 128-col group
    m, n, d, gc = 2, 128, 8, 128
    W = torch.arange(m * n, dtype=torch.float64).reshape(m, n)
    V = glvq.extract_groups(W, d, group_cols=gc)  # (1, m*128//8, 8)
    # first vector = row 0, cols 0..7
    assert torch.equal(V[0, 0], W[0, 0:8])
    # second vector = row 0, cols 8..15
    assert torch.equal(V[0, 1], W[0, 8:16])


# ----------------------------------------------------------------------------
# Babai encode
# ----------------------------------------------------------------------------
def test_babai_nearest_plane_recovers_exact_lattice_points_orthogonal_G():
    torch.manual_seed(0)
    d, Gn, Nv, b = 8, 2, 64, 4
    Q, _ = torch.linalg.qr(torch.randn(Gn, d, d, dtype=torch.float64))
    G = Q * 0.7  # orthogonal basis, in-range codes
    zmax = 2 ** (b - 1) - 1
    z_true = torch.randint(-(2 ** (b - 1)), zmax + 1, (Gn, Nv, d)).double()
    t = glvq.decode_lattice(z_true, G)
    z = glvq.babai_nearest_plane(t, G, b)
    assert torch.equal(z, z_true)


def test_babai_simple_recovers_exact_lattice_points_orthogonal_G():
    torch.manual_seed(1)
    d, Gn, Nv, b = 8, 1, 32, 4
    Q, _ = torch.linalg.qr(torch.randn(Gn, d, d, dtype=torch.float64))
    G = Q * 1.3
    z_true = torch.randint(-4, 4, (Gn, Nv, d)).double()
    t = glvq.decode_lattice(z_true, G)
    z = glvq.babai_simple(t, G, b)
    assert torch.equal(z, z_true)


def test_babai_reduces_to_round_for_identity_basis():
    d, Gn, Nv, b = 6, 1, 40, 8  # b large so no clamping
    G = torch.eye(d, dtype=torch.float64)[None].expand(Gn, d, d).contiguous()
    t = torch.randn(Gn, Nv, d, dtype=torch.float64) * 3.0
    z = glvq.babai_nearest_plane(t, G, b)
    assert torch.equal(z, t.round())


def test_babai_clamps_to_b_bit_range():
    d, Gn, Nv, b = 4, 1, 100, 2  # range [-2, 1]
    G = torch.eye(d, dtype=torch.float64)[None]
    t = torch.randn(Gn, Nv, d, dtype=torch.float64) * 10.0  # way out of range
    z = glvq.babai_nearest_plane(t, G, b)
    assert z.min() >= -(2 ** (b - 1))
    assert z.max() <= 2 ** (b - 1) - 1


def test_decode_lattice_matches_matrix_vector_product():
    torch.manual_seed(2)
    d, Gn, Nv = 8, 3, 10
    G = torch.randn(Gn, d, d, dtype=torch.float64)
    z = torch.randint(-3, 3, (Gn, Nv, d)).double()
    V = glvq.decode_lattice(z, G)
    # V[g,n] == G[g] @ z[g,n]
    ref = torch.einsum("gij,gnj->gni", G, z)
    assert torch.allclose(V, ref, atol=1e-12)


# ----------------------------------------------------------------------------
# GLVQLayerQuantizer — construction + accounting
# ----------------------------------------------------------------------------
def _toy_layer(m=32, n=256, seed=0, structured=True):
    torch.manual_seed(seed)
    if structured:
        # correlated columns within each 128-group so a lattice can help
        base = torch.randn(m, n)
        corr = torch.randn(m, 8) @ torch.randn(8, n)
        W = (base + 1.5 * corr).float()
    else:
        W = torch.randn(m, n).float()
    X = torch.randn(512, n)
    H = (X.T @ X) / 512.0  # mean X^T X, like HessianCapture
    return W, H


def test_bpw_matches_paper_formula():
    W, H = _toy_layer(m=64, n=256)
    for d in (8, 16):
        q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=d, b=2, group_cols=128, compand=True)
        expected = 2 + (16 * d * d + 32) / (128 * 64)  # G + s + mu per 128-col group
        assert abs(q.bpw() - expected) < 1e-9


def test_bpw_drops_mu_term_without_companding():
    W, H = _toy_layer(m=64, n=256)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, compand=False)
    expected = 2 + (16 * 8 * 8 + 16) / (128 * 64)  # G + s only
    assert abs(q.bpw() - expected) < 1e-9


def test_per_layer_G_uses_single_group():
    W, H = _toy_layer(m=32, n=256)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, per_layer_G=True)
    assert q.G.shape[0] == 1


def test_init_clamp_rate_is_low_after_fill_range_init():
    W, H = _toy_layer(m=64, n=256, structured=False)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, group_cols=128)
    assert 0.0 <= q.clamp_rate() <= 0.10


def test_decode_shape_roundtrips_to_full_matrix():
    W, H = _toy_layer(m=32, n=256)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=3)
    W_hat = q.decode()
    assert W_hat.shape == W.shape
    assert torch.isfinite(W_hat).all()


# ----------------------------------------------------------------------------
# autograd semantics
# ----------------------------------------------------------------------------
def test_encode_produces_detached_integer_codes():
    W, H = _toy_layer()
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2)
    z = q.encode()
    assert not z.requires_grad
    assert torch.equal(z, z.round())  # integer-valued


def test_loss_has_gradient_to_G_and_mu_but_not_z():
    W, H = _toy_layer()
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, compand=True)
    z = q.encode()
    W_hat = q.reconstruct(z)
    loss = q.proxy_loss_tensor(W_hat)
    loss.backward()
    assert q.G.grad is not None and q.G.grad.abs().sum() > 0
    assert q._mu_raw.grad is not None and q._mu_raw.grad.abs().sum() > 0
    assert not z.requires_grad


# ----------------------------------------------------------------------------
# fit — convergence + spectral clamp
# ----------------------------------------------------------------------------
def test_fit_reduces_proxy_loss():
    W, H = _toy_layer(m=64, n=256, structured=True)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, group_cols=128)
    initial = q.proxy_loss()
    tele = q.fit(outer=8, inner=10)
    final = q.proxy_loss()
    assert final < initial, f"fit did not reduce proxy: {initial} -> {final}"
    assert tele["proxy"][-1] <= tele["proxy"][0]


def test_fit_never_worse_than_frozen_init():
    # alternating min re-rounds in the companded domain and can wander; fit must
    # keep the best iterate, so it never regresses below the Cholesky init.
    W, H = _toy_layer(m=48, n=256, structured=False, seed=3)  # iid: fitting may not help
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, group_cols=128)
    frozen = q.proxy_loss()
    q.fit(outer=15, inner=10)
    assert q.proxy_loss() <= frozen + 1e-9


def test_spectral_clamp_bounds_hold_after_fit():
    W, H = _toy_layer(m=64, n=256)
    smin, smax = 0.3, 3.0
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, group_cols=128)
    sig0 = torch.linalg.svdvals(q.G0).mean(dim=-1)  # per-group mean singular value of init
    q.fit(outer=6, inner=8, spectral=(smin, smax))
    sig = torch.linalg.svdvals(q.G)
    assert (sig >= smin * sig0[:, None] - 1e-4).all()
    assert (sig <= smax * sig0[:, None] + 1e-4).all()


def test_diag_only_stays_diagonal_through_fit_and_still_reduces_loss():
    # the "is the lattice illusory?" control: fit only diagonal G + mu (companded
    # per-axis scalar quant). Must stay diagonal and still improve over its init.
    W, H = _toy_layer(m=64, n=256, structured=True)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2, diag_only=True)
    p0 = q.proxy_loss()
    q.fit(outer=6, inner=8)
    assert q.offdiag_ratio() < 1e-6
    assert q.proxy_loss() <= p0


def test_offdiag_ratio_zero_for_diagonal_G():
    W, H = _toy_layer(m=32, n=256)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=2)
    with torch.no_grad():
        eye = torch.eye(8)[None].expand_as(q.G).contiguous()
        q.G.copy_(eye * 0.5)
    assert q.offdiag_ratio() < 1e-6


# ----------------------------------------------------------------------------
# quantize_block — LDLQ-hybrid seam (arm 6)
# ----------------------------------------------------------------------------
def test_quantize_block_shapes_and_determinism():
    W, H = _toy_layer(m=32, n=256)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=3, group_cols=128)
    col_start = 128  # second group
    X = W[:, col_start:col_start + 8].clone()
    dec1, z1 = q.quantize_block(X, col_start)
    dec2, z2 = q.quantize_block(X, col_start)
    assert dec1.shape == X.shape
    assert z1.shape == (W.shape[0], 8)
    assert torch.equal(z1, z2) and torch.allclose(dec1, dec2)  # deterministic
    assert torch.equal(z1, z1.round())


def test_storage_real_decode_is_finite_and_close():
    W, H = _toy_layer(m=32, n=256)
    q = glvq.GLVQLayerQuantizer.init_from_weights(W, H, d=8, b=3)
    q.fit(outer=4, inner=6)
    W_hat = q.decode()
    W_hat_fp16 = q.storage_real_decode()
    assert torch.isfinite(W_hat_fp16).all()
    # fp16-cast G/mu/s should be close to fp32 decode, not identical
    assert (W_hat - W_hat_fp16).abs().max() < 0.5 * W.abs().max()
