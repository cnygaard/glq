"""S0 storage round-trip tests for the shippable trellis codebook (`glq/trellis.py`).

The S0 gate is: a *stored* (packed, kernel-layout) trellis checkpoint decodes — via pure
torch — bit-exactly back to the same W_hat the encoder produced in memory. These tests pin
that contract on CPU with fixed seeds, so the eventual CUDA kernel (S1) has an exact
pure-torch oracle to match. They cover the three storage transforms that must round-trip:

  1. `pack_trellis`/`unpack_trellis` — the tail-biting bit-packing (QTIP-verbatim).
  2. `_PERMUTE`/`_INV_PERMUTE` — the 256-element MMA-fragment tile reorder.
  3. the `for_kernel` byte tile-flip — the MMA-fragment byte shuffle.

HYB is tested with an explicit (non-kmeans) tlut so the suite needs no scipy.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402


def _sqnr(x, xh):
    return 10 * math.log10((x.float() ** 2).mean().item() / ((x - xh).float() ** 2).mean().item())


def _hyb_cb(seed=0, K=2):
    """Decoder-mode HYB codebook with a deterministic tlut (no scipy/kmeans)."""
    torch.manual_seed(seed)
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    return gt.TrellisCodebook(variant="hyb", K=K, tlut=tlut, device="cpu")


# ---------------------------------------------------------------------------
# 1. the MMA-fragment 256-tile permutation is a true inverse permutation
# ---------------------------------------------------------------------------
def test_permute_is_true_inverse_permutation():
    p, ip = gt._PERMUTE, gt._INV_PERMUTE
    assert p.shape == (256,) and ip.shape == (256,)
    assert sorted(p.tolist()) == list(range(256))          # a permutation
    assert torch.equal(ip[p], torch.arange(256))           # inverse
    assert torch.equal(p[ip], torch.arange(256))


# ---------------------------------------------------------------------------
# 2. pack_trellis / unpack_trellis round-trip on REAL encoder state
# ---------------------------------------------------------------------------
def test_pack_unpack_trellis_roundtrip_hyb():
    cb = _hyb_cb()
    torch.manual_seed(1)
    tiles = torch.randn(6, 256)                             # 6 independent tiles
    _, state = cb.quantize_tiles(tiles)                    # (6, 256//V=128)
    packed = cb.pack_trellis(state)                        # int16 (6, ceil(256*K/16))
    assert packed.dtype == torch.int16
    un = cb.unpack_trellis(packed, cb.codesz * cb.codesz)  # T = 256
    assert torch.equal(un.to(torch.int32), state.to(torch.int32))


# ---------------------------------------------------------------------------
# 3. the for_kernel byte tile-flip is exactly invertible
# ---------------------------------------------------------------------------
def test_kernel_byte_flip_is_invertible():
    torch.manual_seed(2)
    m, n, K = 64, 96, 2
    num_tiles = (m // 16) * (n // 16)
    packed = torch.randint(-(2 ** 15), 2 ** 15, (num_tiles, 16 * K), dtype=torch.int16)
    flipped = gt.kernel_tile_flip(packed, m, n, K, forward=True)
    back = gt.kernel_tile_flip(flipped, m, n, K, forward=False)
    assert torch.equal(back, packed)
    assert not torch.equal(flipped, packed)                # actually reorders


# ---------------------------------------------------------------------------
# 4. THE S0 GATE: full-layer quantize -> pack(kernel layout) -> pure-torch decode
#    reproduces the in-memory normalized hatW bit-exactly.
# ---------------------------------------------------------------------------
def test_layer_storage_roundtrip_kernel_layout_bitexact():
    cb = _hyb_cb()
    torch.manual_seed(3)
    m, n = 96, 128
    W = (torch.randn(m, n) * 0.05).float()
    X = torch.randn(512, n)
    H = (X.T @ X) / 512
    hatWr_norm, Qidxs, Wscale = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True)
    hatWr_dec = gt.decode_layer(cb, packed, m, n, has_kernel=True)
    assert torch.equal(hatWr_dec, hatWr_norm)              # bit-exact round-trip


def test_layer_storage_roundtrip_natural_layout_bitexact():
    cb = _hyb_cb()
    torch.manual_seed(4)
    m, n = 64, 96
    W = (torch.randn(m, n) * 0.05).float()
    hatWr_norm, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=False)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=False)
    hatWr_dec = gt.decode_layer(cb, packed, m, n, has_kernel=False)
    assert torch.equal(hatWr_dec, hatWr_norm)


# ---------------------------------------------------------------------------
# 5. decoder codebook reconstructs identically from a stored tlut (no kmeans)
# ---------------------------------------------------------------------------
def test_hyb_codebook_reconstructs_from_stored_tlut():
    tlut = (torch.randn(2 ** 9, 2) * 0.968).to(torch.float16)
    cb1 = gt.TrellisCodebook(variant="hyb", K=2, tlut=tlut.clone(), device="cpu")
    cb2 = gt.TrellisCodebook(variant="hyb", K=2, tlut=tlut.clone(), device="cpu")
    assert torch.equal(cb1.cb.lut, cb2.cb.lut)
    torch.manual_seed(5)
    tiles = torch.randn(4, 256)
    h1, s1 = cb1.quantize_tiles(tiles)
    h2, s2 = cb2.quantize_tiles(tiles)
    assert torch.equal(s1, s2) and torch.equal(h1, h2)


# ---------------------------------------------------------------------------
# 6. quality sanity: kernel-layout quant still clears e8p's 2-bit MSE bar
# ---------------------------------------------------------------------------
def test_kernel_layout_quality_matches_natural():
    cb = _hyb_cb()
    torch.manual_seed(6)
    m, n = 256, 256
    W = torch.randn(m, n)
    hk, _, sk = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=True)
    hn, _, sn = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=False)
    # _PERMUTE is quality-neutral on iid weights: both clear e8p's ~10.4 dB (MSE 0.091)
    assert _sqnr(W / sk, hk) > 11.0
    assert _sqnr(W / sn, hn) > 11.0
    assert abs(_sqnr(W / sk, hk) - _sqnr(W / sn, hn)) < 0.5
