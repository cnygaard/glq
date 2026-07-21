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
MASK32 = (1 << 32) - 1


@functools.lru_cache(maxsize=None)
def _make(variant, K, m, n, seed=0):
    """Quantize a random layer in KERNEL layout (for_kernel=True). Cached so the (slow CPU-Viterbi)
    quant is shared across the decode_compressed gate and the mirror gate."""
    torch.manual_seed(seed)
    cb = gt.TrellisCodebook(variant=variant, K=K, device="cpu")
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
    if R == 2:
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
    cont = (_shfl_next(chunk) >> (width - 16)) & 0xFFFF
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


@pytest.mark.parametrize("K", KS)
def test_mirror2b_v1_transliteration_matches_decode_layer(K):
    """Stage 2b — THE crux gate: the V=1 (3inst) per-lane extraction (8 states @ K-stride, widened
    reg_cs2 overflow: R=3 -> 2, R=4 -> 3 overflow states) is byte-identical to decode_layer. Pins
    the exact CUDA bit-math (tr_load_reg_cs/tr_decode_regw <R,3inst>) BEFORE any nvcc build."""
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
