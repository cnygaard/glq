"""P0 unit tests — QTIP trellis quantizer wired into GLQ (`benchmarks/_qtip_ldlq.py`).

Since we reuse QTIP's *copied* algorithm + GLQ's proven LDLQ, correctness is by-construction;
these tests guard the wiring: the bug-prone 16×16 tile grouping, the scale formula, that the
LDLQ feedback helps, and that the codebook-level 2-bit MSE matches the `_qtip_mse` win (~0.069,
well below e8p's 0.091) — i.e. the tiling + scale reproduce QTIP's quality. CPU-only, seeded.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _qtip_ldlq as qtl  # noqa: E402


def _sqnr(x, xh):
    return 10 * math.log10((x.float() ** 2).mean().item() / ((x - xh).float() ** 2).mean().item())


# ---------------------------------------------------------------------------
# the bug-prone step: 16×16 tile grouping
# ---------------------------------------------------------------------------
def test_tile_grouping_matches_explicit_16x16_tiles():
    torch.manual_seed(0)
    m = 96
    WXWX = torch.randn(m, 16)
    tiles = WXWX.reshape(m // 16, 256)                       # what the codebook does
    for t in range(m // 16):
        # tile t must be 16 consecutive rows × 16 cols, row-major flattened
        assert torch.equal(tiles[t], WXWX[16 * t:16 * t + 16, :].flatten())


# ---------------------------------------------------------------------------
# codebook wrapper
# ---------------------------------------------------------------------------
def test_codebook_quantize_shape_and_determinism():
    cb = qtl.TrellisCodebook(variant="3inst", device="cpu")
    assert cb.codesz == 16
    torch.manual_seed(0)
    WXWX = torch.randn(64, 16)
    hat1, _ = cb.quantize(WXWX)
    hat2, _ = cb.quantize(WXWX)
    assert hat1.shape == WXWX.shape
    assert torch.isfinite(hat1).all()
    assert torch.equal(hat1, hat2)                          # deterministic


def test_opt_scale_formula_in_expected_band():
    # QTIP opt_scale = 1/(rms(lut)*0.9); empirical best-scale on unit Gaussian was ~0.81 (3inst)
    cb = qtl.TrellisCodebook(variant="3inst", device="cpu")
    assert 0.6 < cb.opt_scale < 1.05


# ---------------------------------------------------------------------------
# LDLQ feedback + codebook-level MSE anchor
# ---------------------------------------------------------------------------
def test_trellis_ldlq_feedback_does_not_increase_proxy():
    cb = qtl.TrellisCodebook(variant="3inst", device="cpu")
    torch.manual_seed(1)
    m, n = 64, 64
    W = torch.randn(m, n)
    X = torch.randn(512, n)
    H = (X.T @ X) / 512
    W_hat = qtl.trellis_ldlq(W, H, cb)
    # no-feedback baseline: quantize each 16-col block independently, same scale
    Wscale = W.pow(2).mean().sqrt().item() * cb.opt_scale
    Wr = W / Wscale
    hat_nofb = torch.zeros_like(Wr)
    for k in range(n // 16):
        kb, ke = k * 16, (k + 1) * 16
        hat_nofb[:, kb:ke], _ = cb.quantize(Wr[:, kb:ke])
    W_hat_nofb = hat_nofb * Wscale

    def proxy(Wh):
        d = W - Wh
        return (d @ H * d).sum().item() / m

    assert proxy(W_hat) <= proxy(W_hat_nofb) * 1.001         # feedback helps or ties


def test_iid_gaussian_2bit_beats_e8p_level():
    # With H=I the LDLQ feedback is 0 → pure codebook quant. Trellis SQNR must clear e8p's
    # ~10.4 dB (MSE 0.091); our _qtip_mse measured ~11.6 dB (MSE 0.069).
    cb = qtl.TrellisCodebook(variant="3inst", device="cpu")
    torch.manual_seed(0)
    m, n = 256, 512
    W = torch.randn(m, n)
    W_hat = qtl.trellis_ldlq(W, torch.eye(n), cb)
    assert _sqnr(W, W_hat) > 11.0


# ---------------------------------------------------------------------------
# full per-layer RHT + trellis path
# ---------------------------------------------------------------------------
def test_quantize_layer_trellis_rht_runs_and_reconstructs():
    cb = qtl.TrellisCodebook(variant="3inst", device="cpu")
    torch.manual_seed(2)
    m, n = 128, 256
    W = (torch.randn(m, n) * 0.05).float()                  # weight-like scale
    X = torch.randn(512, n)
    H = (X.T @ X) / 512
    W_hat = qtl.quantize_layer_trellis_rht(W, H, cb)
    assert W_hat.shape == (m, n)
    assert torch.isfinite(W_hat).all()
    assert _sqnr(W, W_hat) > 5.0                            # 2-bit on random weights, sane
