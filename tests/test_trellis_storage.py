"""S0 storage round-trip tests for the shippable trellis codebook (`glq/trellis.py`).

The S0 gate is: a *stored* (packed, kernel-layout) trellis checkpoint decodes — via pure
torch — bit-exactly back to the same W_hat the encoder produced in memory. These tests pin
that contract on CPU with fixed seeds, so the eventual CUDA kernel (S1) has an exact
pure-torch oracle to match. They cover the three storage transforms that must round-trip:

  1. `pack_trellis`/`unpack_trellis` — the tail-biting bit-packing (QTIP-verbatim).
  2. `_PERMUTE`/`_INV_PERMUTE` — the 256-element MMA-fragment tile reorder.
  3. the `for_kernel` byte tile-flip — the MMA-fragment byte shuffle.

Everything is parameterized over the native trellis rate **K in {2,3,4}** — the three rates
the CUDA kernel templates on (`tr_bits_from_packed` hard-checks `2 <= R <= 4`). K is *not*
a codebook property: the HYB tlut is K-independent (kmeans on 2-D Gaussians, no K), and the
rate is recoverable from the packed shape alone (cols == 16*K), which is exactly what the
kernel, the HF factory and the vLLM loader each rely on.

HYB is tested with an explicit (non-kmeans) tlut so the suite needs no scipy.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402

# The rates the shipped CUDA kernel templates on (R == K).
KS = [2, 3, 4]


def _sqnr(x, xh):
    return 10 * math.log10((x.float() ** 2).mean().item() / ((x - xh).float() ** 2).mean().item())


def _hyb_cb(seed=0, K=2):
    """Decoder-mode HYB codebook with a deterministic tlut (no scipy/kmeans)."""
    torch.manual_seed(seed)
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    return gt.TrellisCodebook(variant="hyb", K=K, tlut=tlut, device="cpu")


def _3inst_cb(K=2):
    """Lookup-free 3INST codebook (computed, V=1, no tlut, no kmeans)."""
    return gt.TrellisCodebook(variant="3inst", K=K, device="cpu")


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
@pytest.mark.parametrize("K", KS)
def test_pack_unpack_trellis_roundtrip_hyb(K):
    cb = _hyb_cb(K=K)
    torch.manual_seed(1)
    tiles = torch.randn(6, 256)                             # 6 independent tiles
    _, state = cb.quantize_tiles(tiles)                    # (6, 256//V=128)
    packed = cb.pack_trellis(state)                        # int16 (6, ceil(256*K/16))
    assert packed.dtype == torch.int16
    un = cb.unpack_trellis(packed, cb.codesz * cb.codesz)  # T = 256
    assert torch.equal(un.to(torch.int32), state.to(torch.int32))


@pytest.mark.parametrize("K", KS)
def test_packed_rate_is_self_describing(K):
    """cols == 16*K (32/48/64) is the ONLY record of the rate in a checkpoint —
    `tr_bits_from_packed` (CUDA), `hf_integration` and the vLLM loader all recover
    K from it. Pin the shape, or a rate mismatch becomes a silent load failure."""
    cb = _hyb_cb(K=K)
    torch.manual_seed(1)
    _, state = cb.quantize_tiles(torch.randn(6, 256))
    packed = cb.pack_trellis(state)
    assert packed.shape == (6, 16 * K)
    # ...and it really is K bits/weight: 6 tiles x 256 weights, 2 bytes/int16.
    assert packed.numel() * 16 == 6 * 256 * K


# ---------------------------------------------------------------------------
# 3. the for_kernel byte tile-flip is exactly invertible
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
def test_kernel_byte_flip_is_invertible(K):
    torch.manual_seed(2)
    m, n = 64, 96
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
@pytest.mark.parametrize("K", KS)
def test_layer_storage_roundtrip_kernel_layout_bitexact(K):
    cb = _hyb_cb(K=K)
    torch.manual_seed(3)
    m, n = 96, 128
    W = (torch.randn(m, n) * 0.05).float()
    X = torch.randn(512, n)
    H = (X.T @ X) / 512
    hatWr_norm, Qidxs, Wscale = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True)
    hatWr_dec = gt.decode_layer(cb, packed, m, n, has_kernel=True)
    assert torch.equal(hatWr_dec, hatWr_norm)              # bit-exact round-trip


@pytest.mark.parametrize("K", KS)
def test_layer_storage_roundtrip_natural_layout_bitexact(K):
    cb = _hyb_cb(K=K)
    torch.manual_seed(4)
    m, n = 64, 96
    W = (torch.randn(m, n) * 0.05).float()
    hatWr_norm, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=False)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=False)
    hatWr_dec = gt.decode_layer(cb, packed, m, n, has_kernel=False)
    assert torch.equal(hatWr_dec, hatWr_norm)


# ---------------------------------------------------------------------------
# 4b. 3INST (lookup-free, V=1) storage: the layout that travels with the codebook.
#     3inst ships CUDA kernels (the <R, IS_3INST> instantiations in glq_trellis.cu) →
#     has_kernel=True → the KERNEL (MMA-fragment) layout, exactly like HYB. The driver
#     (quantize_model.py) calls quantize_layer_trellis_rht WITHOUT for_kernel, so its
#     DEFAULT must follow cb.has_kernel — the Phase-0a invariant that store-layout ==
#     decode-layout for every variant (a mismatch silently scrambles every weight).
# ---------------------------------------------------------------------------
def test_3inst_codebook_is_lookup_free():
    cb = _3inst_cb(K=2)
    assert cb.has_kernel is True           # ships the lookup-free CUDA kernel → kernel layout
    assert cb.tlut is None                 # computed codebook, nothing to store
    assert cb.V == 1                       # one weight per trellis state


@pytest.mark.parametrize("K", KS)
def test_3inst_natural_layout_roundtrip_bitexact(K):
    """The 3inst codebook + pure-torch decode round-trip bit-exactly in the natural layout
    too (explicit has_kernel=False on both sides) — the layouts are equivalent as long as
    pack and decode agree; only the DEFAULT is kernel now."""
    cb = _3inst_cb(K=K)
    torch.manual_seed(20)
    m, n = 64, 96
    W = (torch.randn(m, n) * 0.05).float()
    hatWr_norm, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=False)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=False)
    hatWr_dec = gt.decode_layer(cb, packed, m, n, has_kernel=False)
    assert torch.equal(hatWr_dec, hatWr_norm)


@pytest.mark.parametrize("K", KS)
def test_quantize_layer_default_layout_follows_codebook_3inst(K):
    """THE driver invariant (Phase 0a): quantize_layer_trellis_rht's default `for_kernel`
    must equal the codebook's native layout. For 3inst (has_kernel=True since the lookup-free
    kernel shipped) the default call MUST store the SAME kernel layout as an explicit
    for_kernel=True call — the layout the <R, IS_3INST> CUDA kernels consume."""
    cb = _3inst_cb(K=K)
    assert cb.has_kernel is True
    torch.manual_seed(21)
    m, n = 64, 128
    W = (torch.randn(m, n) * 0.05).float()
    X = torch.randn(256, n)
    H = (X.T @ X) / 256
    _, art_default = gt.quantize_layer_trellis_rht(W, H, cb)                       # driver path
    _, art_correct = gt.quantize_layer_trellis_rht(W, H, cb, for_kernel=True)
    assert torch.equal(art_default["trellis_packed"], art_correct["trellis_packed"])
    assert "tlut" not in art_default                                              # 3inst stores no tlut


@pytest.mark.parametrize("K", KS)
def test_quantize_layer_default_layout_unchanged_for_hyb(K):
    """Regression: HYB (has_kernel=True) keeps storing the kernel layout by default."""
    cb = _hyb_cb(K=K)
    torch.manual_seed(22)
    m, n = 64, 128
    W = (torch.randn(m, n) * 0.05).float()
    X = torch.randn(256, n)
    H = (X.T @ X) / 256
    _, art_default = gt.quantize_layer_trellis_rht(W, H, cb)
    _, art_kernel = gt.quantize_layer_trellis_rht(W, H, cb, for_kernel=True)
    assert torch.equal(art_default["trellis_packed"], art_kernel["trellis_packed"])


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


def test_tlut_is_rate_independent():
    """One shared tlut serves every K — the HYB tlut is kmeans on 2-D Gaussians and
    never sees K. This is why a checkpoint stores ONE tlut regardless of rate (and
    why a future mixed-K trellis is architecturally cheap)."""
    luts = [_hyb_cb(K=K).tlut for K in KS]
    for lut in luts[1:]:
        assert torch.equal(lut, luts[0])


# ---------------------------------------------------------------------------
# 6. quality sanity: kernel-layout quant still clears e8p's 2-bit MSE bar,
#    and a higher native rate is strictly better (this is the point of K=3/4).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
def test_kernel_layout_quality_matches_natural(K):
    cb = _hyb_cb(K=K)
    torch.manual_seed(6)
    m, n = 256, 256
    W = torch.randn(m, n)
    hk, _, sk = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=True)
    hn, _, sn = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=False)
    # _PERMUTE is quality-neutral on iid weights: both clear e8p's ~10.4 dB (MSE 0.091)
    assert _sqnr(W / sk, hk) > 11.0
    assert _sqnr(W / sn, hn) > 11.0
    assert abs(_sqnr(W / sk, hk) - _sqnr(W / sn, hn)) < 0.5


def test_sqnr_strictly_improves_with_rate():
    """A native rate-K trellis must beat rate-(K-1) — the whole case for K=3/4 over
    stacking e8p RVQ stages. ~6 dB/bit is the rate-distortion slope."""
    torch.manual_seed(6)
    W = torch.randn(256, 256)
    sqnr = {}
    for K in KS:
        cb = _hyb_cb(K=K)
        hk, _, sk = gt.trellis_ldlq(W, torch.eye(256), cb, for_kernel=True)
        sqnr[K] = _sqnr(W / sk, hk)
    assert sqnr[4] > sqnr[3] > sqnr[2], sqnr
    # each added bit should buy several dB (RD slope ~6 dB/bit); a flat step would
    # mean the extra bits are not reaching the Viterbi.
    assert sqnr[3] - sqnr[2] > 3.0, sqnr
    assert sqnr[4] - sqnr[3] > 3.0, sqnr
