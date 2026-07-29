"""3INST (V=1, lookup-free) trellis kernel gates — CPU de-risk for the V=1 decode kernel.

Phase-1 plan `hazy-churning-shannon`. The 3inst kernel decodes each 16-bit trellis state with
`decode_3inst` (a uint32 hash + two-half fp16 sum) instead of HYB's smem tlut gather — V=1
(1 weight/state, K-bit stride) vs HYB's V=2. Two CPU bit-mirrors pin the kernel bit-for-bit
BEFORE any nvcc (the bug class is pure integer bit-math, seconds/iter on CPU vs a minutes-long
JIT rebuild on a reclaim-prone GPU box):

  * Mirror #1 — `decode_compressed(V=0)` (QTIP's pure-torch swizzle model of the CUDA bit-flow)
    == `decode_layer(has_kernel=True)`. Validates decode_layer is the right oracle and the
    K-bit-stride / tail-biting-window layout is understood.
  * Mirror #2 — a literal per-lane transliteration of `tr_load_reg_cs<R>` + `tr_decode_regw<R>` +
    the decompress tile-walk/scatter (glq_trellis.cu). Stage 2a transliterates the SHIPPING HYB
    (V=2) path exactly -> validates the __byte_perm / __shfl(laneId+1) / tile-walk / scatter numpy
    infra; stage 2b flips ONLY the decode to V=1 -> validates the new 8-state K-stride extraction
    and the widened reg_cs2 overflow (R=3 needs 2 overflow states, R=4 needs 3).

The validated V=1 kernel spec (what S1/S2 implement):
  tr_load_reg_cs<R,3inst>: per-window chunk (8R bits: R=2->u16, R=3->reg_24_i, R=4->r_i);
      reg_cs = chunk, reg_cs2 = (__shfl(chunk, laneId+1) >> (8R-16)) & 0xFFFF.
  tr_decode_regw<R,3inst>: Ext=(uint64)chunk<<16 | reg_cs2; state_j=(Ext >> (8R - R*j)) & 0xFFFF,
      j=0..7; decode_3inst(state_j); pair (s_2j, s_2j+1) -> half2[j]. Scatter/mma unchanged.

`decode_compressed` is inlined verbatim from `qtip/lib/utils/kernel_decompress.py` (QTIP, GPL-3),
`@torch.compile` stripped so it runs eagerly. Its line-46 stride is `R << V`, so V=0 gives the
K-bit stride of the V=1 code.
"""
import functools
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402

KS = [2, 3, 4]
# Residual-stage rates for stacked RVQ (5-8 bpw = K=4 primary + K=(bpw-4) residual). R=1 is
# reachable ONLY as a stage-2 rate (bpw 5) and never as a primary, so it is exercised by the
# V=1 decode mirror and the 3INST kernels but not by HYB, which has no lookup-free path.
KS_RESID = [1, 2, 3, 4]
MASK32 = (1 << 32) - 1


@functools.lru_cache(maxsize=None)
def _make(variant, K, m, n, seed=0):
    """Quantize a random layer in KERNEL layout (for_kernel=True). Cached so the (slow CPU-Viterbi)
    quant is shared across the decode_compressed gate and the mirror gate. HYB gets an explicit
    random tlut (decoder-style construction) — the default kmeans fit needs scipy, an encode-only
    dep absent on CI, and the bit-mirror only needs *a* tlut, not a fitted one."""
    torch.manual_seed(seed)
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543) if variant == "hyb" else None
    cb = gt.TrellisCodebook(variant=variant, K=K, tlut=tlut, device="cpu")
    W = (torch.randn(m, n) * 0.05).float()
    _, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True)
    assert packed.shape == (m // 16 * (n // 16), 16 * K)     # V-independent 16*K int16 layout
    return cb, packed


# ===========================================================================
# Mirror #1 — decode_layer == QTIP decode_compressed(V=0)
# ===========================================================================
def _decode_compressed(L, S, R, V, m, k, compressed, expanded_lut):
    """QTIP kernel_decompress.decode_compressed, @torch.compile stripped. Pure-torch model of the
    CUDA bit-flow: byte-unswizzle -> tail-biting 32-bit windows -> extract L-bit states at
    stride (R<<V) -> lut gather -> m16n8k16 de-swizzle."""
    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)
    assert compressed.shape == (R * m * k // 16,)
    BITS_PER_BLOCK = R * 16 * 16
    BLOCK_SIZE = 16 * 16
    compressed = (compressed.view(torch.uint8).reshape(
        m // 16 // 2, k // 16 // 2, BLOCK_SIZE // 8, 2, 2, R).permute(0, -2, 1, -3, 2, -1).flip(
            (-1,)).reshape(m // 16, k // 16, BITS_PER_BLOCK // 16, 2).flip(
                (-1,)).view(torch.uint16).reshape(m // 16, k // 16, BITS_PER_BLOCK // 16))
    assert L <= 16
    blocked = compressed.reshape(R * m * k // BITS_PER_BLOCK, BITS_PER_BLOCK // 16, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = torch.cat((blocked_roll, blocked), dim=-1).reshape(
        blocked.shape[0], -1).contiguous().view(torch.uint32)
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(*blocked32.shape, 16).view(torch.int32)
    shifts = torch.arange(0, 16, dtype=torch.int32).reshape(1, 1, -1).expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    indices = torch.bitwise_and(shifted.reshape(shifted.shape[0], -1)[:, 16 - L::R << V], (1 << L) - 1)
    mma_swizzled = expanded_lut[indices]
    return (mma_swizzled.reshape(m // 16, k // 16, 16, 16).reshape(
        m // 16, k // 16, 8, 4, 2, 2, 2).permute(0, -2, 2, 1, -3, 3, -1).reshape(m, k))


@pytest.mark.parametrize("K", KS)
def test_s0_decode_compressed_v0_matches_decode_layer(K):
    m, n = 64, 128
    cb, packed = _make("3inst", K, m, n)
    lut = gt.decode_3inst(torch.arange(2 ** 16)).float()
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True).float()
    dc = _decode_compressed(16, 9, K, 0, m, n, packed.reshape(-1).contiguous(), lut).float()
    assert torch.equal(dc, ref), f"K={K}: max|Δ|={(dc - ref).abs().max().item():.3e}"


def test_s0_3inst_lut_is_exact_fp16():
    """decode_3inst sums two fp16 halves in half precision, so every codebook value is an
    exactly-fp16 value widened to fp32 — this is why a torch.equal decompress gate is achievable."""
    lut = gt.decode_3inst(torch.arange(2 ** 16)).float()
    assert torch.equal(lut, lut.half().float())


# ===========================================================================
# Mirror #2 — literal per-lane tr_load_reg_cs<R> + tr_decode_regw<R> + tile-walk transliteration
# ===========================================================================
def _u32(x):
    return (np.asarray(x, dtype=np.int64) & MASK32).astype(np.int64)


def _bperm(x, y, s):
    """CUDA __byte_perm(x, y, s): out byte i = byte (s>>4i)&7 of {x[0..3], y[0..3]}."""
    x = _u32(x); y = _u32(y)
    src = [(x >> (8 * i)) & 0xFF for i in range(4)] + [(y >> (8 * i)) & 0xFF for i in range(4)]
    out = np.zeros_like(x)
    for i in range(4):
        out |= src[(s >> (4 * i)) & 0x7] << (8 * i)
    return _u32(out)


def _shfl_next(v):
    """__shfl_sync(FULL, v, laneId+1), tail-biting wrap 31->0: out[l] = v[(l+1) % 32]."""
    return np.roll(v, -1)


def _load_reg_cs(pu16, weight_idx, R):
    """tr_load_reg_cs<R> (HYB build) for all 32 lanes -> reg_cs, reg_cs2 dicts of (32,) uint32."""
    wi = np.asarray(weight_idx, dtype=np.int64)
    ld = np.stack([pu16[wi + t] for t in range(2 * R)], axis=1).astype(np.int64)
    r = [_u32(ld[:, 2 * i] | (ld[:, 2 * i + 1] << 16)) for i in range(R)]
    cs = {}; cs2 = {k: _u32(0) for k in "xyzw"}
    if R == 2:
        n1 = _shfl_next(r[0]); n2 = _shfl_next(r[1])
        cs["x"] = _bperm(n1, r[0], 0x5410); cs["y"] = _bperm(n1, r[0], 0x7632)
        cs["z"] = _bperm(n2, r[1], 0x5410); cs["w"] = _bperm(n2, r[1], 0x7632)
    elif R == 3:
        r1, r2, r3 = r
        reg = [_u32(r1 & 0xffffff), _u32(((r1 >> 24) | (r2 << 8)) & 0xffffff),
               _u32(((r2 >> 16) | (r3 << 16)) & 0xffffff), _u32((r3 >> 8) & 0xffffff)]
        p1 = _u32((reg[0] >> 8) | ((reg[1] << 8) & 0xffff0000))
        p3 = _u32((reg[2] >> 8) | ((reg[3] << 8) & 0xffff0000))
        n1 = _shfl_next(p1); n3 = _shfl_next(p3)
        cs["x"] = _bperm(n1, reg[0], 0x6541); cs["y"] = _bperm(n1, reg[1], 0x6543)
        cs["z"] = _bperm(n3, reg[2], 0x6541); cs["w"] = _bperm(n3, reg[3], 0x6543)
        cs2["x"] = _u32(((n1 >> 6) & 0x3ff) | (reg[0] << 10))
        cs2["y"] = _u32(((n1 >> 22) & 0x3ff) | (reg[1] << 10))
        cs2["z"] = _u32(((n3 >> 6) & 0x3ff) | (reg[2] << 10))
        cs2["w"] = _u32(((n3 >> 22) & 0x3ff) | (reg[3] << 10))
    else:
        r1, r2, r3, r4 = r
        p1 = _u32((r1 >> 16) | (r2 & 0xffff0000)); p3 = _u32((r3 >> 16) | (r4 & 0xffff0000))
        n1 = _shfl_next(p1); n3 = _shfl_next(p3)
        cs["x"] = r1; cs["y"] = r2; cs["z"] = r3; cs["w"] = r4
        cs2["x"] = _bperm(n1, r1, 0x0041); cs2["y"] = _bperm(n1, r2, 0x0043)
        cs2["z"] = _bperm(n3, r3, 0x0041); cs2["w"] = _bperm(n3, r4, 0x0043)
    return cs, cs2


def _hyb_states(reg_c, reg_c2, R):
    """tr_decode_regw<R> HYB extraction: 4 states (V=2). Returns (32,4) true 16-bit states."""
    out = []
    for j in range(4):
        if R == 2:
            idx = reg_c >> (4 * (4 - j))
        elif R == 3:
            idx = (reg_c >> (6 * (2 - j) + 4)) if j < 3 else reg_c2
        else:
            idx = (reg_c >> (8 * (2 - j))) if j < 3 else reg_c2
        out.append(_u32(idx) & 0xFFFF)
    return np.stack(out, axis=1)


def _load_chunks(pu16, weight_idx, R):
    """V=1 raw per-window chunks (8R bits, MSB-first) for windows x,y,z,w — the tail-biting stream
    BEFORE HYB's byte_perm (R=2:u16, R=3:reg_24_i, R=4:r_i)."""
    wi = np.asarray(weight_idx, dtype=np.int64)
    ld = np.stack([pu16[wi + t] for t in range(2 * R)], axis=1).astype(np.int64)
    if R == 1:
        # 4 chunks x 8 bits = ONE u32 per lane (vs R=2's uint2, R=4's uint4). Chunk order is
        # memory order, lowest address first, exactly as for R>=2.
        r1 = _u32(ld[:, 0] | (ld[:, 1] << 16))
        chunks = [_u32((r1 >> (8 * i)) & 0xFF) for i in range(4)]
    elif R == 2:
        chunks = [_u32(ld[:, i]) for i in range(4)]
    elif R == 3:
        r1 = _u32(ld[:, 0] | (ld[:, 1] << 16)); r2 = _u32(ld[:, 2] | (ld[:, 3] << 16))
        r3 = _u32(ld[:, 4] | (ld[:, 5] << 16))
        chunks = [_u32(r1 & 0xffffff), _u32(((r1 >> 24) | (r2 << 8)) & 0xffffff),
                  _u32(((r2 >> 16) | (r3 << 16)) & 0xffffff), _u32((r3 >> 8) & 0xffffff)]
    else:
        chunks = [_u32(ld[:, 2 * i] | (ld[:, 2 * i + 1] << 16)) for i in range(4)]
    return chunks, 8 * R


def _v1_states(chunk, R, width):
    """V=1: 8 states at K-bit stride from a window chunk. Ext = chunk (MSB at bit 8R-1) high, next
    lane's chunk-top (tail-biting continuation) low. state_j = (Ext >> (8R - R*j)) & 0xFFFF."""
    if width >= 16:
        cont = (_shfl_next(chunk) >> (width - 16)) & 0xFFFF
    else:
        # R=1 (width 8, the bpw-5 residual stage). "Continuation = top 16 bits of the NEXT
        # lane's chunk" is an identity only while 8R >= 16; here an 8-bit chunk cannot supply
        # 16 bits, and `>> (width - 16)` would be a NEGATIVE shift. The 16 stream bits
        # following this chunk live in the next TWO lanes, so both are pulled (mod-32 wrap
        # keeps the tail-biting cycle). Each fragment slot x/y/z/w is its own stream across
        # the 32 lanes, which is why the shuffle stays within a slot.
        cont = _u32(((_shfl_next(chunk) & 0xFF) << 8) | (np.roll(chunk, -2) & 0xFF))
    Ext = (_u32(chunk) << 16) | _u32(cont)
    return np.stack([(Ext >> (width - R * j)) & 0xFFFF for j in range(8)], axis=1)


_KEY = {(0, 0): "x", (1, 0): "y", (0, 1): "z", (1, 1): "w"}


def _mirror_decompress(cb, packed, m, n, V):
    """Faithful numpy transliteration of glq_trellis_decompress_kernel<R> (+ the V-parameterized
    decode). Returns hatW (m,n) float32."""
    R = cb.K
    pu16 = packed.contiguous().view(torch.int16).view(torch.uint16).numpy().reshape(-1).astype(np.int64)
    W = np.zeros((m, n), dtype=np.float32)
    TR_WARPS = 32
    tileCountM, tileCountK = m // 16, n // 16
    g = tileCountM // 2                                    # one m-tile-pair / block
    m_per_block = (tileCountM + 2 * g - 1) // (2 * g)
    k_per_block = tileCountK // (TR_WARPS * 4) * 2
    u16_per_tile = 16 * 16 * R // 16
    utb = u16_per_tile * 4
    weight_step = TR_WARPS * utb
    weight_row_step = tileCountK * u16_per_tile * 2
    lanes = np.arange(32); groupID = lanes >> 2; tig = lanes & 3
    lut = cb.cb.lut if V == 2 else None                   # (2, 2^16) fp32
    for blk in range(g):
        tileIdM = m_per_block * blk
        if tileIdM * 2 >= tileCountM:
            continue
        for warpId in range(TR_WARPS):
            this_warp_k = k_per_block + 2 if warpId < (tileCountK % (TR_WARPS * 4)) // 4 else k_per_block
            base = tileIdM * weight_row_step + warpId * utb * 2 + lanes * (utb // 32)
            for ki in range(this_warp_k):
                addr = base + (ki // 2) * 2 * weight_step + (ki % 2) * utb   # window used at this ki
                if V == 2:
                    cs, cs2 = _load_reg_cs(pu16, addr, R)
                else:
                    chunks, width = _load_chunks(pu16, addr, R)
                for subki in range(2):
                    k_tile = 4 * warpId + 2 * (ki % 2) + subki + (4 * TR_WARPS) * (ki // 2)
                    for submi in range(2):
                        wkey = _KEY[(submi, subki)]
                        if V == 2:
                            st = _hyb_states(cs[wkey], cs2[wkey], R).astype(np.int64)
                            vals = lut[:, torch.from_numpy(st.reshape(-1))].numpy().reshape(2, 32, 4)
                            rw = np.stack([vals[0], vals[1]], axis=-1)               # (32,4,2)
                        else:
                            st = _v1_states(chunks["xyzw".index(wkey)], R, width).astype(np.int64)
                            dec = gt.decode_3inst(torch.from_numpy(st.reshape(-1))).numpy().reshape(32, 8)
                            rw = dec.reshape(32, 4, 2)                               # half2[j]=(s2j,s2j+1)
                        m_tile = tileIdM * 2 + submi
                        r0 = m_tile * 16 + groupID; c0 = k_tile * 16 + 2 * tig
                        for l in range(32):
                            a, b = r0[l], c0[l]
                            W[a, b], W[a, b + 1] = rw[l, 0]
                            W[a + 8, b], W[a + 8, b + 1] = rw[l, 1]
                            W[a, b + 8], W[a, b + 9] = rw[l, 2]
                            W[a + 8, b + 8], W[a + 8, b + 9] = rw[l, 3]
    return W


@pytest.mark.parametrize("K", KS)
def test_mirror2a_hyb_transliteration_matches_decode_layer(K):
    """Stage 2a: the literal HYB (V=2) tr_load_reg_cs/tr_decode_regw/tile-walk transliteration is
    byte-identical to decode_layer — validates the __byte_perm / __shfl / scatter numpy infra
    (incl. the R=3/R=4 reg_cs2 overflow) that the V=1 mirror reuses."""
    m, n = 64, 128
    cb, packed = _make("hyb", K, m, n)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True).float().numpy()
    got = _mirror_decompress(cb, packed, m, n, V=2)
    assert np.array_equal(got, ref), f"HYB K={K}: {int((got != ref).sum())} mismatches"


@pytest.mark.parametrize("K", KS_RESID)
def test_mirror2b_v1_transliteration_matches_decode_layer(K):
    """Stage 2b — THE crux gate: the V=1 (3inst) per-lane extraction (8 states @ K-stride, widened
    reg_cs2 overflow: R=3 -> 2, R=4 -> 3 overflow states) is byte-identical to decode_layer. Pins
    the exact CUDA bit-math (tr_load_reg_cs/tr_decode_regw <R,3inst>) BEFORE any nvcc build.

    K=1 (the bpw-5 residual stage) is the structurally different case: an 8-bit chunk cannot
    supply the 16-bit continuation, so it spans TWO following lanes. See _v1_states."""
    m, n = 64, 128
    cb, packed = _make("3inst", K, m, n)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True).float().numpy()
    got = _mirror_decompress(cb, packed, m, n, V=1)
    assert np.array_equal(got, ref), f"3inst K={K}: {int((got != ref).sum())} mismatches"


# ===========================================================================
# GPU gates (S2-S4) — the <R, IS_3INST=true> CUDA instantiations vs the same oracle.
#   The CPU mirrors above prove the bit-math; these prove the PORT of it. Decompress shares
#   tr_load_reg_cs/tr_decode_regw with matvec+matmul, so GATE-1 pins the decode for all three.
# ===========================================================================
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sqnr(ref, got):
    import math
    return 10 * math.log10(ref.float().pow(2).mean().item()
                           / (got.float() - ref.float()).pow(2).mean().item())


def _ext():
    from glq import inference_kernel as ik
    assert ik._try_load_cuda_ext(), "glq CUDA extension failed to build"
    return ik._glq_cuda


@functools.lru_cache(maxsize=None)
def _quantized_3inst_cuda(m, n, K, seed=0):
    """Quantize a random 3inst layer on-GPU in KERNEL layout; return (cb, packed)."""
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(m, n, device=dev) * 0.05).float()
    X = torch.randn(512, n, device=dev)
    H = (X.T @ X) / 512
    cb = gt.TrellisCodebook(variant="3inst", K=K, device=dev)
    _, Qidxs, _ = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True).to(dev)
    assert packed.shape[1] == 16 * K          # the kernel re-derives R from this
    return cb, packed


# ---- GATE-1 (S2, the crux): CUDA 3inst decompress BIT-EXACT vs decode_layer --------------
@needs_cuda
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("m,n", [(64, 128), (128, 256), (256, 512)])
def test_cuda_3inst_decompress_bitexact_vs_decode_layer(m, n, K):
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=m)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)          # (m,n) fp32 oracle
    W = _ext().glq_decompress_trellis_3inst_cuda(packed, m, n)        # (m,n) fp16
    assert W.shape == (m, n) and W.dtype == torch.float16
    assert torch.equal(W.float(), ref.float()), \
        f"K={K} bad={int((W.float() != ref.float()).sum())} max|Δ|={(W.float() - ref).abs().max().item()}"


# ---- S3: B=1 GEMV — accuracy + bit-stable determinism ------------------------------------
@needs_cuda
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("m,n", [(256, 512), (512, 2048)])
def test_cuda_3inst_matvec_matches_reference_gemv(m, n, K):
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=m + 1)
    W = gt.decode_layer(cb, packed, m, n, has_kernel=True)            # (m,n) fp32
    torch.manual_seed(7)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    ref = x.float() @ W.t()                                           # (m,) fp32
    out = _ext().glq_decode_matvec_trellis_3inst_cuda(x, packed, m, n)
    assert out.shape == (m,) and out.dtype == torch.float32
    assert _sqnr(ref, out) > 40.0, f"K={K} SQNR {_sqnr(ref, out):.1f} dB"


@needs_cuda
@pytest.mark.parametrize("K", KS)
def test_cuda_3inst_matvec_is_deterministic(K):
    m, n = 256, 512
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=3)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    a = _ext().glq_decode_matvec_trellis_3inst_cuda(x, packed, m, n)
    b = _ext().glq_decode_matvec_trellis_3inst_cuda(x, packed, m, n)
    assert torch.equal(a, b)   # block-owns-m-range + smem reduce → no atomics, bit-stable


# ---- S4: batched GEMM — accuracy over ragged B, row-parity with the GEMV, determinism ----
@needs_cuda
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("B", [1, 2, 7, 8, 9, 63, 64, 65])
def test_cuda_3inst_matmul_batched_matches_reference(B, K):
    m, n = 256, 512
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=21)
    W = gt.decode_layer(cb, packed, m, n, has_kernel=True)
    torch.manual_seed(B)
    x = (torch.randn(B, n, device="cuda") * 0.5).to(torch.float16)
    ref = x.float() @ W.t()                                           # (B,m) fp32
    out = _ext().glq_decode_matmul_trellis_3inst_cuda(x, packed, m, n)
    assert out.shape == (B, m) and out.dtype == torch.float32
    assert _sqnr(ref, out) > 40.0, f"K={K} B={B} SQNR {_sqnr(ref, out):.1f} dB"


@needs_cuda
@pytest.mark.parametrize("K", KS)
def test_cuda_3inst_matmul_row_parity_with_gemv(K):
    """Row b of the batched GEMM must be BIT-EXACT vs the B=1 GEMV on x[b] — same mma sequence,
    same k-split, same A-fragment; only the token's N-column differs."""
    m, n = 256, 512
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=22)
    torch.manual_seed(5)
    x = (torch.randn(9, n, device="cuda") * 0.5).to(torch.float16)   # 9 → crosses a token tile
    batched = _ext().glq_decode_matmul_trellis_3inst_cuda(x, packed, m, n)
    for b in range(9):
        gemv = _ext().glq_decode_matvec_trellis_3inst_cuda(x[b].contiguous(), packed, m, n)
        assert torch.equal(batched[b], gemv), f"K={K} row {b} != GEMV"


@needs_cuda
@pytest.mark.parametrize("K", KS)
def test_cuda_3inst_matmul_is_deterministic(K):
    m, n = 256, 512
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=23)
    x = (torch.randn(16, n, device="cuda") * 0.5).to(torch.float16)
    a = _ext().glq_decode_matmul_trellis_3inst_cuda(x, packed, m, n)
    b = _ext().glq_decode_matmul_trellis_3inst_cuda(x, packed, m, n)
    assert torch.equal(a, b)


# ===========================================================================
# RS2a — CPU mirror of the FUSE_IN input-FHT prologue (de-risk BEFORE nvcc).
#   The planned fusion computes, inside every matvec/matmul block:
#     buf[i] = (i < in_features ? x[i] : 0) * sv[i]        (fp32)
#     ascending-distance butterfly, log2(n_pad) stages      (fp32, ping-pong)
#     out[i] = buf[i] * (1.0f / sqrtf(n_pad))               (fp32)  -> __float2half
#   and then feeds the mma x-fragments FROM that smem buffer with the UNCHANGED
#   x_idx/fill/consume indexing. Two mirrors pin both halves:
#     #1 value mirror  — the fp32 pipeline above, validated bit-exactly against the
#        REAL, already-built glq_input_rht_cuda kernel (no new nvcc needed);
#        the fp16 cast point is then an RN-rounding of identical fp32 values.
#     #2 index algebra — the x_buf fill/consume map reads, for fragment
#        (warpId, refresh r, slot s, lane l), exactly the fp16 quad
#        x[k_tile*16 + {2l, 2l+1, 8+2l, 9+2l}] with k_tile = 4*warpId + s + 128*r —
#        i.e. a linear buffer (global OR smem) serves the fragments unchanged.
# ===========================================================================
def _mirror_input_fht(x16, sv16, n_pad):
    """fp32-exact CPU transliteration of glq_input_rht_kernel (single-block path):
    pad + SV sign -> ascending butterfly -> * (1.0f/sqrtf(n_pad)). Returns fp32."""
    in_features = x16.numel()
    buf = torch.zeros(n_pad, dtype=torch.float32)
    buf[:in_features] = x16.float() * sv16[:in_features].float()
    if in_features < n_pad:                       # padded region: 0 * sv == 0
        buf[in_features:] = 0.0
    log_n = n_pad.bit_length() - 1
    for k in range(log_n):
        h = 1 << k
        partner = buf[torch.arange(n_pad) ^ h]
        lo = (torch.arange(n_pad) & h) == 0
        buf = torch.where(lo, buf + partner, partner - buf)
    rsqrt_n = np.float32(1.0) / np.float32(np.sqrt(np.float32(n_pad)))
    return buf * torch.tensor(float(rsqrt_n), dtype=torch.float32)


@needs_cuda
def test_rs2a_mirror_matches_cuda_input_rht_kernel():
    """Value mirror: the CPU transliteration equals the REAL CUDA glq_input_rht_cuda
    output bit-for-bit at n=2048 (the single-block shape RS2b fuses first), so the
    in-kernel FUSE_IN butterfly implementing the same loop is pinned before any nvcc."""
    from glq import inference_kernel as ik
    assert ik._try_load_cuda_ext()
    n_pad = 2048
    torch.manual_seed(41)
    x = (torch.randn(n_pad) * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n_pad) < 0.5, -1.0, 1.0).to(torch.float16)
    out = torch.empty(1, n_pad, dtype=torch.float32, device="cuda")
    rsqrt_n = float(np.float32(1.0) / np.float32(np.sqrt(np.float32(n_pad))))
    ik._glq_cuda.glq_input_rht_cuda(x.cuda().unsqueeze(0), sv.cuda(), out,
                                    n_pad, n_pad, rsqrt_n, n_pad,
                                    n_pad.bit_length() - 1)
    ref = _mirror_input_fht(x, sv, n_pad)
    assert torch.equal(out.cpu().view(-1), ref), \
        f"mirror != CUDA kernel: max|Δ|={(out.cpu().view(-1) - ref).abs().max().item():.3e}"
    # the fused kernel's __float2half of these fp32 values == the pipeline's .to(fp16)
    assert torch.equal(out.cpu().view(-1).half(), ref.half())


def test_rs2a_xbuf_fragment_map_reads_linear_buffer():
    """Index algebra: simulate the matvec's x_buf fill (x_idx = warpId*32 + laneId + r*1024,
    slot=laneId//8, col=laneId%4, u32=(laneId%8)//4) and consume (slot ki%2*2+subki, lanes
    0-3, both u32s) against a linear ramp — every consumed fp16 quad must be
    x[k_tile*16 + {2l,2l+1,8+2l,9+2l}], k_tile = 4*warpId + s + 128*r. This is what lets
    RS2b swap the global x pointer for the in-block smem FHT buffer with no index changes."""
    n = 2048
    x = torch.arange(n, dtype=torch.float32)                 # fp16-exact ramp values < 2048
    x_half2 = x.view(-1, 2)                                  # u32 j -> halves (x[2j], x[2j+1])
    for warpId in range(32):
        for r in range(2):                                   # n=2048 -> 128 k-tiles -> r in {0,1}
            xbuf = torch.zeros(4, 4, 2, 2)                   # [slot][col][u32][half]
            for lane in range(32):                           # fill (one refresh)
                x_idx = warpId * 32 + lane + r * 1024
                if x_idx < n // 2:
                    xbuf[lane // 8][lane % 4][(lane % 8) // 4] = x_half2[x_idx]
            for s in range(4):                               # consume: ki%2*2+subki == slot s
                k_tile = 4 * warpId + s + 128 * r
                if k_tile >= n // 16:
                    continue
                for l in range(4):                           # lanes 0-3 feed mma column 0
                    got = torch.cat([xbuf[s][l][0], xbuf[s][l][1]])
                    want = x[[k_tile * 16 + 2 * l, k_tile * 16 + 2 * l + 1,
                              k_tile * 16 + 8 + 2 * l, k_tile * 16 + 9 + 2 * l]]
                    assert torch.equal(got, want), (warpId, r, s, l)


# ---- RS2b: FUSE_IN — input RHT computed inside every matvec block -----------------------
@needs_cuda
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("in_features", [2048, 2000])   # exact fit + padded tail
def test_rs2b_fusein_matvec_bitexact_vs_unfused_pipeline(K, in_features):
    """The FUSE_IN GEMV (raw x + sv in, RHT in-block) must be BIT-EXACT vs the unfused
    3-kernel pipeline it replaces: glq_input_rht_cuda -> .to(fp16) -> matvec(wscale).
    Achievable because the in-block butterfly is op-order-identical (test_rs2a_*) and the
    fragment reads come from the same values via the same indexing."""
    from glq import inference_kernel as ik
    m, n = 64, 2048
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=51)
    torch.manual_seed(52)
    x_raw = (torch.randn(in_features, device="cuda") * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    w = 0.01371
    # unfused reference pipeline (the exact kernels the fused path replaces)
    x_rht = torch.empty(1, n, dtype=torch.float32, device="cuda")
    rsqrt_n = float(np.float32(1.0) / np.float32(np.sqrt(np.float32(n))))
    ik._glq_cuda.glq_input_rht_cuda(x_raw.unsqueeze(0), sv, x_rht, in_features, in_features,
                                    rsqrt_n, n, n.bit_length() - 1)
    ref = _ext().glq_decode_matvec_trellis_3inst_cuda(
        x_rht.view(-1).to(torch.float16), packed, m, n, w)
    got = _ext().glq_decode_matvec_trellis_3inst_fusein_cuda(
        x_raw, sv, packed, m, n, in_features, w)
    assert torch.equal(got, ref), \
        f"K={K} in={in_features} max|Δ|={(got - ref).abs().max().item():.3e}"


@needs_cuda
@pytest.mark.parametrize("K", KS)
def test_rs2b_fusein_matvec_is_deterministic(K):
    m, n = 64, 2048
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=51)
    x_raw = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    a = _ext().glq_decode_matvec_trellis_3inst_fusein_cuda(x_raw, sv, packed, m, n, n, 1.0)
    b = _ext().glq_decode_matvec_trellis_3inst_fusein_cuda(x_raw, sv, packed, m, n, n, 1.0)
    assert torch.equal(a, b)


# ---- RS3: FUSE_IN for block-diagonal (non-pow2) input shapes ----------------------------
@needs_cuda
@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("n", [768, 1088])     # [512,256] and [1024,64] decompositions
def test_rs3_fusein_blockdiag_bitexact_vs_unfused_pipeline(K, n):
    """Block-diag FUSE_IN must be BIT-EXACT vs the unfused pipeline it replaces:
    glq_input_rht_blockdiag_cuda (multiblock kernel, per-sub-block rsqrtf) -> .to(fp16)
    -> matvec(wscale). The in-block prologue mirrors the same per-sub-block butterfly +
    rsqrtf(bs) normalization, so torch.equal holds."""
    from glq import inference_kernel as ik
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_block_meta
    m = 64
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=61)
    blocks = _block_decompose(n)
    assert len(blocks) > 1
    bn = torch.tensor(blocks, dtype=torch.int64)
    bnm = _pack_block_meta(blocks).cuda()
    torch.manual_seed(62)
    x_raw = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    w = 0.01371
    x_rht = torch.empty(1, n, dtype=torch.float32, device="cuda")
    ik._glq_cuda.glq_input_rht_blockdiag_cuda(x_raw.unsqueeze(0), sv, x_rht, n, n, bn, bnm)
    ref = _ext().glq_decode_matvec_trellis_3inst_cuda(
        x_rht.view(-1).to(torch.float16), packed, m, n, w)
    got = _ext().glq_decode_matvec_trellis_3inst_fusein_cuda(
        x_raw, sv, packed, m, n, n, w,
        blocks_n_meta=bnm, num_blocks=len(blocks), max_bs=max(blocks))
    assert torch.equal(got, ref), \
        f"K={K} n={n} max|Δ|={(got - ref).abs().max().item():.3e}"


@needs_cuda
def test_rs3_fusein_blockdiag_is_deterministic():
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_block_meta
    m, n = 64, 768
    cb, packed = _quantized_3inst_cuda(m, n, 2, seed=61)
    blocks = _block_decompose(n)
    bnm = _pack_block_meta(blocks).cuda()
    x_raw = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    a = _ext().glq_decode_matvec_trellis_3inst_fusein_cuda(
        x_raw, sv, packed, m, n, n, 1.0,
        blocks_n_meta=bnm, num_blocks=len(blocks), max_bs=max(blocks))
    b = _ext().glq_decode_matvec_trellis_3inst_fusein_cuda(
        x_raw, sv, packed, m, n, n, 1.0,
        blocks_n_meta=bnm, num_blocks=len(blocks), max_bs=max(blocks))
    assert torch.equal(a, b)


# ---- RS1: ×wscale folded into the kernel store ------------------------------------------
@needs_cuda
@pytest.mark.parametrize("K", KS)
def test_cuda_3inst_store_folds_wscale_bitexact(K):
    """The in-store `reduced * wscale` must be BIT-EXACT vs scaling the unscaled output in
    torch — same two fp32 operands, same multiply — so folding the elementwise scale kernel
    into the decode store changes nothing numerically. Default wscale=1.0 keeps old calls."""
    m, n = 256, 512
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=31)
    torch.manual_seed(9)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    w = 0.01371
    a = _ext().glq_decode_matvec_trellis_3inst_cuda(x, packed, m, n, w)
    b = _ext().glq_decode_matvec_trellis_3inst_cuda(x, packed, m, n) * w
    assert torch.equal(a, b)
    xb = (torch.randn(5, n, device="cuda") * 0.5).to(torch.float16)
    am = _ext().glq_decode_matmul_trellis_3inst_cuda(xb, packed, m, n, w)
    bm = _ext().glq_decode_matmul_trellis_3inst_cuda(xb, packed, m, n) * w
    assert torch.equal(am, bm)


# ---- RS4a: warp-shuffle low-5 butterfly stages — independent CPU anchors ----------------
# The shuffle phase must be BIT-EXACT (same ascending stage order, same fp32 lo?a+b:b−a),
# so every pre-RS4a equality gate keeps holding. These anchors cover the paths whose only
# other cross-check is ANOTHER RS4a-modified kernel (consistent-wrong risk): the
# single-buffer variant (n=16384, 16 elems/thread through the shuffle), the <32 fallback,
# the multiblock kernel (anchored bit-exactly by harvesting the GPU's rsqrtf(bs): the
# butterfly of a one-hot at the block start is Hadamard row 0 == all-ones, so the kernel's
# output there IS rsqrtf(bs)), and the output kernel (host-side rsqrt_m -> full mirror).

@needs_cuda
@pytest.mark.parametrize("n_pad", [16, 2048, 16384])   # <32 fallback / double-buf / single-buf
def test_rs4a_input_rht_matches_mirror_all_variants(n_pad):
    from glq import inference_kernel as ik
    assert ik._try_load_cuda_ext()
    torch.manual_seed(43)
    x = (torch.randn(n_pad) * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n_pad) < 0.5, -1.0, 1.0).to(torch.float16)
    out = torch.empty(1, n_pad, dtype=torch.float32, device="cuda")
    rsqrt_n = float(np.float32(1.0) / np.float32(np.sqrt(np.float32(n_pad))))
    ik._glq_cuda.glq_input_rht_cuda(x.cuda().unsqueeze(0), sv.cuda(), out,
                                    n_pad, n_pad, rsqrt_n, n_pad,
                                    n_pad.bit_length() - 1)
    ref = _mirror_input_fht(x, sv, n_pad)
    assert torch.equal(out.cpu().view(-1), ref), \
        f"n={n_pad} max|Δ|={(out.cpu().view(-1) - ref).abs().max().item():.3e}"


def _mirror_fht_raw(v32):
    """Unnormalized ascending-distance fp32 butterfly (adds/subs only — CPU==GPU IEEE)."""
    n = v32.numel()
    buf = v32.clone()
    idx = torch.arange(n)
    h = 1
    while h < n:
        partner = buf[idx ^ h]
        buf = torch.where((idx & h) == 0, buf + partner, partner - buf)
        h <<= 1
    return buf


@needs_cuda
@pytest.mark.parametrize("n", [1088, 1040])    # [1024,64] both shuffled; [1024,16] <32 fallback
def test_rs4a_multiblock_input_rht_matches_mirror_bitexact(n):
    from glq import inference_kernel as ik
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_block_meta
    assert ik._try_load_cuda_ext()
    blocks = _block_decompose(n)
    assert len(blocks) > 1
    bn = torch.tensor(blocks, dtype=torch.int64)
    bnm = _pack_block_meta(blocks).cuda()

    def run(x16, sv16):
        out = torch.empty(1, n, dtype=torch.float32, device="cuda")
        ik._glq_cuda.glq_input_rht_blockdiag_cuda(
            x16.cuda().unsqueeze(0), sv16.cuda(), out, n, n, bn, bnm)
        return out.cpu().view(-1)

    # Harvest each block's EXACT rsqrtf(bs) (device intrinsic — not CPU-reproducible):
    # one-hot at the block start propagates as all-ones, so out[off] == rsqrtf(bs) * 1.0.
    probe = torch.zeros(n, dtype=torch.float16)
    offs = np.cumsum([0] + blocks[:-1]).tolist()
    for off in offs:
        probe[off] = 1.0
    rs = run(probe, torch.ones(n, dtype=torch.float16))
    torch.manual_seed(44)
    x = (torch.randn(n) * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n) < 0.5, -1.0, 1.0).to(torch.float16)
    got = run(x, sv)
    ref = torch.empty(n, dtype=torch.float32)
    for off, bs in zip(offs, blocks):
        seg = x[off:off + bs].float() * sv[off:off + bs].float()
        ref[off:off + bs] = _mirror_fht_raw(seg) * rs[off]      # exact harvested rsqrtf(bs)
    assert torch.equal(got, ref), \
        f"n={n} blocks={blocks} max|Δ|={(got - ref).abs().max().item():.3e}"


@needs_cuda
@pytest.mark.parametrize("m_pad,out_features", [(2048, 2000), (16384, 16384)])
def test_rs4a_output_rht_matches_mirror(m_pad, out_features):
    from glq import inference_kernel as ik
    assert ik._try_load_cuda_ext()
    torch.manual_seed(45)
    y = torch.randn(m_pad, dtype=torch.float32)
    su = torch.where(torch.rand(m_pad) < 0.5, -1.0, 1.0).to(torch.float16)
    out = torch.zeros(1, out_features, dtype=torch.float16, device="cuda")
    rsqrt_m = float(np.float32(1.0) / np.float32(np.sqrt(np.float32(m_pad))))
    ik._glq_cuda.glq_output_rht_cuda(y.cuda().unsqueeze(0), su.cuda(), out,
                                     out_features, m_pad, m_pad.bit_length() - 1, rsqrt_m)
    r = torch.tensor(rsqrt_m, dtype=torch.float32)
    ref = ((_mirror_fht_raw(y) * r) * su.float())[:out_features].half()
    assert torch.equal(out.cpu().view(-1), ref), \
        f"m={m_pad} max|Δ|={(out.cpu().view(-1) - ref).abs().max().item():.3e}"


# ---- S4b: shard-batched output RHT (qkv 3->1, gate/up 2->1 launches) --------------------
# One grid.y launch spans every sub-block of every shard of a fused linear's output row.
# Bit-exactness hinges on the meta.w normalization override: single-block shards carry the
# bit-cast HOST 1.0f/sqrtf(bs) (what glq_output_rht_cuda multiplies), multi-block shards
# carry 0 -> in-kernel rsqrtf(bs) (what the multiblock kernel computes) — so the batched
# kernel reproduces BOTH historical paths bit-for-bit.

def _s4b_sequential_reference(y_rht, su, shards):
    """The pre-S4b path: one glq_output_rht_blockdiag_cuda per shard, concatenated."""
    from glq import inference_kernel as ik
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_block_meta
    B = y_rht.shape[0]
    ref = torch.empty(B, sum(shards), dtype=torch.float16, device="cuda")
    off = 0
    for m in shards:
        blocks = _block_decompose(m)
        bn = torch.tensor(blocks, dtype=torch.int64)
        bnm = _pack_block_meta(blocks).cuda()
        seg = torch.empty(B, m, dtype=torch.float16, device="cuda")
        ik._glq_cuda.glq_output_rht_blockdiag_cuda(
            y_rht[:, off:off + m].contiguous(), su[off:off + m].contiguous(),
            seg, m, m, bn, bnm)
        ref[:, off:off + m] = seg
        off += m
    return ref


@needs_cuda
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("shards", [[2048, 512, 512], [11008, 11008], [2048, 512, 11008]])
def test_s4b_output_rht_shards_bitexact_vs_sequential(B, shards):
    from glq import inference_kernel as ik
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_shard_meta
    assert ik._try_load_cuda_ext()
    total = sum(shards)
    torch.manual_seed(71)
    y_rht = torch.randn(B, total, dtype=torch.float32, device="cuda")
    su = torch.where(torch.rand(total, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    ref = _s4b_sequential_reference(y_rht, su, shards)
    shard_blocks = [_block_decompose(m) for m in shards]
    meta = _pack_shard_meta(shard_blocks).cuda()
    max_bs = max(max(b) for b in shard_blocks)
    y = torch.empty(B, total, dtype=torch.float16, device="cuda")
    ik._glq_cuda.glq_output_rht_shards_cuda(y_rht, su, y, total, meta, max_bs)
    assert torch.equal(y, ref), \
        f"B={B} shards={shards} max|Δ|={(y.float() - ref.float()).abs().max().item():.3e}"


@needs_cuda
@pytest.mark.parametrize("B", [1, 3])
def test_s4b_yrht_entry_plus_shards_equals_fused(B):
    """Splitting the fused op at the y_rht seam must reproduce the one-shot fused op
    bit-for-bit: per-shard yrht entries write into a shared (B, total_m) fp32 buffer
    (B=1 matvec writes the contiguous row slice directly), then ONE shards-RHT launch."""
    from glq import inference_kernel as ik
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_shard_meta
    shards = [(64, 512), (32, 512)]                    # (m, n) per shard, same input x
    n = 512
    torch.manual_seed(72)
    x = (torch.randn(B, n, device="cuda") * 0.5).to(torch.float16)
    sv = torch.where(torch.rand(n, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    total = sum(m for m, _ in shards)
    su = torch.where(torch.rand(total, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    w = 0.01371
    bn = torch.tensor([n], dtype=torch.int64)
    empty_meta = torch.empty(0, dtype=torch.int32, device="cuda")
    ref_parts, packs = [], []
    off = 0
    for m, _ in shards:
        _, packed = _quantized_3inst_cuda(m, n, 2, seed=73 + m)
        packs.append(packed)
        bm = torch.tensor([m], dtype=torch.int64)
        ref_parts.append(_ext().glq_fused_linear_trellis_3inst_cuda(
            x, sv, su[off:off + m].contiguous(), packed,
            bn, bm, empty_meta, empty_meta, w, n, m, n, m))
        off += m
    ref = torch.cat(ref_parts, dim=-1)
    y_rht = torch.empty(B, total, dtype=torch.float32, device="cuda")
    off = 0
    for (m, _), packed in zip(shards, packs):
        _ext().glq_fused_linear_trellis_3inst_yrht_cuda(
            x, sv, packed, bn, empty_meta, w, n, n, m, y_rht, off)
        off += m
    meta = _pack_shard_meta([_block_decompose(m) for m, _ in shards]).cuda()
    y = torch.empty(B, total, dtype=torch.float16, device="cuda")
    ik._glq_cuda.glq_output_rht_shards_cuda(y_rht, su, y, total, meta,
                                            max(m for m, _ in shards))
    assert torch.equal(y, ref), \
        f"B={B} max|Δ|={(y.float() - ref.float()).abs().max().item():.3e}"


@needs_cuda
def test_s4b_output_rht_shards_is_deterministic():
    from glq import inference_kernel as ik
    from glq.hadamard import _block_decompose
    from glq.quantized_linear import _pack_shard_meta
    assert ik._try_load_cuda_ext()
    shards = [2048, 512, 512]
    total = sum(shards)
    y_rht = torch.randn(2, total, dtype=torch.float32, device="cuda")
    su = torch.where(torch.rand(total, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
    meta = _pack_shard_meta([_block_decompose(m) for m in shards]).cuda()
    a = torch.empty(2, total, dtype=torch.float16, device="cuda")
    b = torch.empty(2, total, dtype=torch.float16, device="cuda")
    ik._glq_cuda.glq_output_rht_shards_cuda(y_rht, su, a, total, meta, 2048)
    ik._glq_cuda.glq_output_rht_shards_cuda(y_rht, su, b, total, meta, 2048)
    assert torch.equal(a, b)


# ---- S5+S6: fused no-tlut host entry via the E8RHTLinear eager path -----------------------
def _trellis_3inst_layer(in_f=512, out_f=256, seed=11, K=2):
    """Quantize + load an E8RHTLinear with a 3inst layer (kernel layout via the has_kernel
    default). NO tlut in the artifacts — that is what routes _trellis_linear_apply to the
    no-tlut fused entry."""
    from glq.quantized_linear import E8RHTLinear
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f, device=dev) * 0.05).float()
    X = torch.randn(512, in_f, device=dev)
    H = (X.T @ X) / 512
    cb = gt.TrellisCodebook(variant="3inst", K=K, device=dev)
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cb)
    assert "tlut" not in art                       # lookup-free: nothing to store
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis").to(dev)
    layer.load_state_dict({k: v.to(dev) for k, v in {
        "trellis_packed": art["trellis_packed"],
        "SU": art["SU"], "SV": art["SV"],
        "Wscale": torch.tensor(art["Wscale"], dtype=torch.float32),
    }.items()}, strict=False)
    layer.set_codebook(cb)
    return layer, W_hat


@needs_cuda
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("B", [1, 4])
def test_fused_linear_3inst_matches_s0_reference(B, K):
    in_f, out_f = 512, 256
    layer, W_hat = _trellis_3inst_layer(in_f, out_f, K=K)
    x = torch.randn(B, in_f, device="cuda", dtype=torch.float16)
    ref = x.float() @ W_hat.float().t()            # the S0 dense reference
    y = layer(x)
    assert y.shape == (B, out_f)
    # Assert the MECHANISM, not just the output: the fused no-tlut CUDA op must have
    # engaged — the pure-torch dense fallback would produce correct values too.
    assert layer._trellis_op is True, "3inst fused CUDA path did not engage"
    assert _sqnr(ref, y) > 35.0, f"K={K} SQNR {_sqnr(ref, y):.1f} dB"


@needs_cuda
@pytest.mark.parametrize("K", KS)
def test_fused_linear_3inst_is_cudagraph_capturable(K):
    """The fused 3inst op must capture into a CUDA graph (no allocation/sync inside) — the
    decode win only materializes under HF/vLLM graph capture."""
    in_f, out_f = 512, 256
    layer, _ = _trellis_3inst_layer(in_f, out_f, seed=13, K=K)
    x = torch.randn(1, in_f, device="cuda", dtype=torch.float16)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y_eager = layer(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    assert layer._trellis_op is True, "3inst fused CUDA path did not engage"

    static_x = x.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y = layer(static_x)

    static_x.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(static_y.float(), y_eager.float(), atol=2e-2, rtol=2e-2), \
        f"K={K} graph replay != eager, max|Δ|={(static_y.float() - y_eager.float()).abs().max().item()}"


# ===========================================================================
# Stacked-RVQ (5-8 bpw) gates — stage 1 (K=4) writes, stage 2 (K=bpw-4) accumulates.
#
# Skip unless the LOADED extension actually carries the rvq2 entry. Without this a stale
# .so makes every test below silently exercise the dense eager fallback and pass green,
# proving nothing about the kernel.
# ===========================================================================
def _has_rvq2():
    if not torch.cuda.is_available():
        return False
    from glq import inference_kernel as ik
    return bool(ik._try_load_cuda_ext()) and hasattr(
        ik._glq_cuda, "glq_fused_linear_trellis_3inst_rvq2_cuda")


needs_rvq2 = pytest.mark.skipif(
    not _has_rvq2(), reason="glq CUDA ext lacks the stacked-RVQ trellis entry (stale build?)")

BPWS_RVQ = [5, 6, 7, 8]


def _trellis_3inst_rvq2_layer(in_f=512, out_f=256, seed=11, bpw=6):
    """Quantize + load a 2-stage (stacked-RVQ) E8RHTLinear. Mirrors _trellis_3inst_layer,
    but hands quantize_layer_trellis_rht the codebook LIST so it emits stage-2 artifacts."""
    from glq.quantized_linear import E8RHTLinear
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f, device=dev) * 0.05).float()
    X = torch.randn(512, in_f, device=dev)
    H = (X.T @ X) / 512
    cbs = [gt.TrellisCodebook(variant="3inst", K=k, device=dev)
           for k in gt.trellis_rvq_recipe(bpw)]
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cbs)
    assert "trellis_packed2" in art and "inv_resid_scale2" in art, "no stage-2 artifacts"
    assert "tlut" not in art                       # lookup-free: nothing to store
    sd = {"trellis_packed": art["trellis_packed"],
          "trellis_packed2": art["trellis_packed2"],
          "inv_resid_scale2": torch.as_tensor(art["inv_resid_scale2"], dtype=torch.float32),
          "SU": art["SU"], "SV": art["SV"],
          "Wscale": torch.as_tensor(art["Wscale"], dtype=torch.float32)}
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis").to(dev)
    layer.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
    layer.set_codebook(cbs[0], cbs[1])
    assert layer._trellis_has_stage2 is True
    return layer, W_hat, art, cbs


# ---- The new R=1 bit-math, straight against the torch oracle ------------------------------
@needs_rvq2
@pytest.mark.parametrize("K", KS_RESID)
@pytest.mark.parametrize("m,n", [(64, 128), (256, 512)])
def test_cuda_3inst_decompress_bitexact_all_residual_rates(m, n, K):
    """Bit-exact decompress for every residual rate, K=1 INCLUDED. K=1 is the only rate whose
    bit-unpack is structurally different (an 8-bit chunk cannot supply the 16-bit
    continuation, so it spans two lanes), and it is the one the CPU mirror already pins —
    this proves the CUDA port of that same math."""
    cb, packed = _quantized_3inst_cuda(m, n, K, seed=m + K)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)               # fp32 oracle
    W = _ext().glq_decompress_trellis_3inst_cuda(packed, m, n, K == 1)     # allow_r1
    assert torch.equal(W.float(), ref.float()), \
        f"K={K} bad={int((W.float() != ref.float()).sum())}"


@needs_rvq2
def test_r1_is_refused_as_a_primary_stage():
    """R=1 must be reachable ONLY as a residual. A K=1 buffer handed to a stage-1 call has
    to fail loudly — the launcher ladders end in a bare `else` that runs the R=4 kernel, so
    a silently-widened bound would read a neighbour's bits and return plausible garbage."""
    m, n = 64, 128
    _, packed = _quantized_3inst_cuda(m, n, 1, seed=99)
    with pytest.raises(RuntimeError, match="bits/weight"):
        _ext().glq_decompress_trellis_3inst_cuda(packed, m, n)   # allow_r1 defaults False


# ---- THE gate: decode vs W_hat (never decode-vs-decode) -----------------------------------
@needs_rvq2
@pytest.mark.parametrize("bpw", BPWS_RVQ)
@pytest.mark.parametrize("B", [1, 4])
def test_fused_rvq2_matches_W_hat(bpw, B):
    """Against the quantizer's own W_hat, NOT another decode: a fused-vs-eager A/B shares the
    scale wiring, so a stage dropped in BOTH legs passes it while quality collapses.

    bpw 8 is load-bearing here — at K1==K2 a stage swap is shape-invisible, so it is the only
    rate that can catch that class by value."""
    in_f, out_f = 512, 256
    layer, W_hat, _, _ = _trellis_3inst_rvq2_layer(in_f, out_f, bpw=bpw)
    torch.manual_seed(3)
    x = (torch.randn(B, in_f, device="cuda") * 0.5).to(torch.float16)
    ref = x.float() @ W_hat.float().t()
    y = layer(x)
    assert layer._trellis_op is True, "stacked-RVQ fused CUDA path did not engage"
    assert _sqnr(ref, y.float()) > 40.0, \
        f"bpw={bpw} B={B} SQNR {_sqnr(ref, y.float()):.1f} dB"


@needs_rvq2
@pytest.mark.parametrize("bpw", BPWS_RVQ)
def test_rvq2_stage2_actually_contributes(bpw):
    """Drop the residual buffer and the output MUST move. `_trellis_op is True` on BOTH legs
    is essential: without it this degenerates into fused-vs-eager and proves nothing about
    what the kernel dereferenced."""
    in_f, out_f = 512, 256
    layer, _, _, _ = _trellis_3inst_rvq2_layer(in_f, out_f, bpw=bpw)
    torch.manual_seed(5)
    x = (torch.randn(1, in_f, device="cuda") * 0.5).to(torch.float16)
    y_full = layer(x).float().clone()
    assert layer._trellis_op is True

    layer1, _, _, _ = _trellis_3inst_rvq2_layer(in_f, out_f, bpw=bpw)
    layer1.trellis_packed2 = torch.zeros(0, dtype=torch.int16, device="cuda")
    layer1._trellis_has_stage2 = False
    layer1._inv_rs2_float = 0.0
    layer1._trellis_op = None                      # re-resolve: now a 1-stage layer
    y_s1 = layer1(x).float()
    assert layer1._trellis_op is True, "stage-1-only leg fell off the fused path"
    assert (y_full - y_s1).abs().max().item() > 1e-3, \
        f"bpw={bpw}: dropping stage 2 changed nothing — the kernel ignored trellis_packed2"


@needs_rvq2
def test_rvq2_zero_scale_changes_the_output():
    """Catches 'kernel reads packed2 but ignores inv_resid_scale2'. The op refuses a zero
    scale outright (half-configured == the e8p stage-drop shape), which is itself the proof
    that the scale reaches the kernel."""
    in_f, out_f = 512, 256
    layer, _, _, _ = _trellis_3inst_rvq2_layer(in_f, out_f, bpw=6)
    x = (torch.randn(1, in_f, device="cuda") * 0.5).to(torch.float16)
    layer(x)                                        # resolve the fused path first
    layer._inv_rs2_float = 0.0
    with pytest.raises(RuntimeError, match="half-configured|requires a populated stage 2"):
        layer(x)


@needs_rvq2
@pytest.mark.parametrize("bpw", BPWS_RVQ)
def test_rvq2_is_deterministic(bpw):
    """`+=` accumulation must stay bit-stable: one writer per element per pass (disjoint
    block m-ranges), no atomics, no scratch."""
    in_f, out_f = 512, 256
    layer, _, _, _ = _trellis_3inst_rvq2_layer(in_f, out_f, bpw=bpw)
    x = (torch.randn(1, in_f, device="cuda") * 0.5).to(torch.float16)
    a, b = layer(x), layer(x)
    assert torch.equal(a, b), f"bpw={bpw} stacked decode is not bit-stable"


@needs_rvq2
@pytest.mark.parametrize("bpw", [5, 8])
def test_fused_rvq2_is_cudagraph_capturable(bpw):
    """Two launches per linear must still capture as one graph — catches a host sync or a
    conditional allocation slipping into the stage-2 path."""
    in_f, out_f = 512, 256
    layer, _, _, _ = _trellis_3inst_rvq2_layer(in_f, out_f, seed=13, bpw=bpw)
    x = torch.randn(1, in_f, device="cuda", dtype=torch.float16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y_eager = layer(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    assert layer._trellis_op is True, "stacked-RVQ fused CUDA path did not engage"
    static_x = x.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y = layer(static_x)
    static_x.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(static_y.float(), y_eager.float(), atol=2e-2, rtol=2e-2), \
        f"bpw={bpw} graph replay != eager"
