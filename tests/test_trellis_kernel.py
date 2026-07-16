"""S1/S3 CUDA trellis-kernel gates (GPU-only).

The highest-risk item in the whole port is the **encoder-pack ⇔ kernel-unpack pairing**:
`pack_layer` writes QTIP's kernel byte layout (tail-biting bit-pack + `_PERMUTE` + tile-flip)
and the CUDA kernel must reconstruct exactly those weights via warp-shuffle bit-unpack →
`idx*(idx+1)` → smem LUT → sign-flip. `glq/trellis.py:decode_layer` (pure torch, already
proven bit-exact against the encoder) is the oracle.

Because the tlut is stored fp16, the kernel's fp16 LUT values are bit-identical to the
fp32-holding-fp16 values torch produces — so the decompress gate is a true `torch.equal`,
not a tolerance. The decompress kernel shares its decode path with the matvec kernel, so
this one test pins both.

**Every gate runs at the native rate K in {2,3,4}** — the three rates the kernel templates
on (`tr_load_reg_cs<R>` reads uint2/uint3/uint4; `tr_decode_regw<R>` extracts indices at a
4/6/8-bit stride). Through S2 only K=2 was ever executed: R=3 and R=4 were compiled and
smem-configured but never launched. The pairing is the risk, and it is R-specific — the
lane load width (2R u16) has to line up with the packed width (16R) or the kernel silently
reads a neighbour's bits.
"""
import functools
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402
from glq import inference_kernel as ik  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# The rates the shipped CUDA kernel templates on (R == K).
KS = [2, 3, 4]


def _sqnr(ref, got):
    return 10 * math.log10(ref.float().pow(2).mean().item()
                           / (got.float() - ref.float()).pow(2).mean().item())


def _ext():
    assert ik._try_load_cuda_ext(), "glq CUDA extension failed to build"
    return ik._glq_cuda


@functools.lru_cache(maxsize=None)
def _quantized(m, n, seed=0, K=2):
    """Quantize a random layer in the KERNEL storage layout; return (cb, packed, tlut16).

    Cached: the Viterbi is the expensive part and every consumer here is read-only."""
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(m, n, device=dev) * 0.05).float()
    X = torch.randn(512, n, device=dev)
    H = (X.T @ X) / 512
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    cb = gt.TrellisCodebook(variant="hyb", K=K, tlut=tlut, device=dev)
    _, Qidxs, _ = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True).to(dev)
    assert packed.shape[1] == 16 * K          # the kernel re-derives R from this
    return cb, packed, tlut.to(dev)


# ---------------------------------------------------------------------------
# GATE 1 (the crux): CUDA decompress is BIT-EXACT vs the pure-torch decode_layer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("m,n", [(128, 256), (256, 512), (64, 128)])
def test_cuda_decompress_bitexact_vs_decode_layer(m, n, K):
    cb, packed, tlut16 = _quantized(m, n, seed=m, K=K)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)          # (m,n) fp32 oracle
    W = _ext().glq_decompress_trellis_cuda(packed, tlut16, m, n)     # (m,n) fp16
    assert W.shape == (m, n) and W.dtype == torch.float16
    assert torch.equal(W.float(), ref.float()), \
        f"K={K} max|Δ|={(W.float() - ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# GATE 2: the fused B=1 GEMV matches x @ W_decoded.T
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("m,n", [(256, 512), (512, 2048)])
def test_cuda_matvec_matches_reference_gemv(m, n, K):
    cb, packed, tlut16 = _quantized(m, n, seed=m + 1, K=K)
    W = gt.decode_layer(cb, packed, m, n, has_kernel=True)            # (m,n) fp32
    torch.manual_seed(7)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    ref = x.float() @ W.t()                                           # (m,) fp32
    out = _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
    assert out.shape == (m,) and out.dtype == torch.float32
    # fp16 tensor-core products w/ fp32 accum vs an fp32 reference
    assert _sqnr(ref, out) > 40.0, f"K={K} SQNR {_sqnr(ref, out):.1f} dB"


@pytest.mark.parametrize("K", KS)
def test_cuda_matvec_is_deterministic(K):
    m, n = 256, 512
    cb, packed, tlut16 = _quantized(m, n, seed=3, K=K)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    a = _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
    b = _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
    assert torch.equal(a, b)   # block-owns-m-range + smem reduce → no atomics, bit-stable


# ---------------------------------------------------------------------------
# GATE 2b: the BATCHED GEMM (B>1) — weights stay compressed (no dense W).
# m16n8k16 computes a 16x8 tile; the GEMV fills only column 0 and discards 7/8.
# The batched kernel puts one token per N-column, so B<=8 costs the same decode
# and the same tensor-core work as B=1. Ragged B is predicated, not padded.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("B", [1, 2, 7, 8, 9, 63, 64, 65])
def test_cuda_matmul_batched_matches_reference(B, K):
    m, n = 256, 512
    cb, packed, tlut16 = _quantized(m, n, seed=21, K=K)
    W = gt.decode_layer(cb, packed, m, n, has_kernel=True)        # (m,n) fp32
    torch.manual_seed(B)
    x = (torch.randn(B, n, device="cuda") * 0.5).to(torch.float16)
    ref = x.float() @ W.t()                                      # (B,m) fp32
    out = _ext().glq_decode_matmul_trellis_cuda(x, packed, tlut16, m, n)
    assert out.shape == (B, m) and out.dtype == torch.float32
    assert _sqnr(ref, out) > 40.0, f"K={K} B={B} SQNR {_sqnr(ref, out):.1f} dB"


@pytest.mark.parametrize("K", KS)
def test_cuda_matmul_row_parity_with_gemv(K):
    """Row b of the batched GEMM must be BIT-EXACT vs the B=1 GEMV on x[b].

    Each mma N-column accumulates independently, both kernels issue the same mma
    sequence over the same k-split (32 warps, fixed-order smem reduce), and the A
    (weight) fragment is identical — so the only thing that changed is which column
    the token sits in. Anything less than bit-exact means the B-fragment indexing or
    the C-fragment harvest is wrong.
    """
    m, n = 256, 512
    cb, packed, tlut16 = _quantized(m, n, seed=22, K=K)
    torch.manual_seed(5)
    x = (torch.randn(9, n, device="cuda") * 0.5).to(torch.float16)   # 9 → crosses a tile
    batched = _ext().glq_decode_matmul_trellis_cuda(x, packed, tlut16, m, n)
    for b in range(9):
        gemv = _ext().glq_decode_matvec_trellis_cuda(x[b].contiguous(), packed, tlut16, m, n)
        assert torch.equal(batched[b], gemv), f"K={K} row {b} != GEMV"


@pytest.mark.parametrize("K", KS)
def test_cuda_matmul_is_deterministic(K):
    m, n = 256, 512
    cb, packed, tlut16 = _quantized(m, n, seed=23, K=K)
    x = (torch.randn(16, n, device="cuda") * 0.5).to(torch.float16)
    a = _ext().glq_decode_matmul_trellis_cuda(x, packed, tlut16, m, n)
    b = _ext().glq_decode_matmul_trellis_cuda(x, packed, tlut16, m, n)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# GATE 3: the fused linear op == the pure-torch S0 reference (RHT in/out bracket)
# ---------------------------------------------------------------------------
def _trellis_layer(in_f=512, out_f=256, seed=11, K=2):
    from glq.quantized_linear import E8RHTLinear
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f, device=dev) * 0.05).float()
    X = torch.randn(512, in_f, device=dev)
    H = (X.T @ X) / 512
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    cb = gt.TrellisCodebook(variant="hyb", K=K, tlut=tlut, device=dev)
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cb)
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis").to(dev)
    layer.load_state_dict({k: v.to(dev) for k, v in {
        "trellis_packed": art["trellis_packed"], "tlut": art["tlut"],
        "SU": art["SU"], "SV": art["SV"],
        "Wscale": torch.tensor(art["Wscale"], dtype=torch.float32),
    }.items()}, strict=False)
    layer.set_codebook(cb)
    return layer, W_hat


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("B", [1, 4])
def test_fused_linear_trellis_matches_s0_reference(B, K):
    in_f, out_f = 512, 256
    layer, W_hat = _trellis_layer(in_f, out_f, K=K)
    x = torch.randn(B, in_f, device="cuda", dtype=torch.float16)
    ref = x.float() @ W_hat.float().t()          # the S0 dense reference
    y = layer(x)
    assert y.shape == (B, out_f)
    assert _sqnr(ref, y) > 35.0, f"K={K} SQNR {_sqnr(ref, y):.1f} dB"


# ---------------------------------------------------------------------------
# GATE 4: the fused op is CUDA-graph capturable — the reason cudaFuncSetAttribute
# and cudaGetDeviceProperties were hoisted out of the launch path. Without this the
# decode win never materializes under vLLM/HF graph capture.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
def test_fused_linear_trellis_is_cudagraph_capturable(K):
    in_f, out_f = 512, 256
    layer, _ = _trellis_layer(in_f, out_f, seed=13, K=K)
    x = torch.randn(1, in_f, device="cuda", dtype=torch.float16)

    # warm up on a side stream (required before capture; also resolves the lazy op + smem attr)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y_eager = layer(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    static_x = x.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y = layer(static_x)

    static_x.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(static_y.float(), y_eager.float(), atol=2e-2, rtol=2e-2), \
        f"K={K} graph replay != eager, max|Δ|={(static_y.float() - y_eager.float()).abs().max().item()}"


# ---------------------------------------------------------------------------
# GATE 5 (S3): decode cost is RATE-INDEPENDENT — the structural claim behind K=3/4.
#
# `tr_decode_regw<R>` loops j=0..3 for EVERY R: always 4 half2 (8 weights) from 4
# trellis states. LUT gathers, sign-flips and mma count are identical at K=2 and K=4;
# only the compressed bytes read per lane double (uint2 -> uint4). So one K=4 pass
# should cost far less than e8p's TWO 2-bpw RVQ passes at the same 4 bpw.
#
# Pinned as a ratio, not an absolute: a K=4 GEMV must not cost ~2x a K=2 GEMV. If this
# ever regresses, the "one pass beats two" case for trellis at 4 bpw collapses with it.
# ---------------------------------------------------------------------------
def test_decode_cost_is_rate_independent():
    m, n = 4096, 4096
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)

    def _time(K, iters=50):
        _, packed, tlut16 = _quantized(m, n, seed=31, K=K)
        for _ in range(10):                                       # warm
            _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters                    # ms

    t2, t4 = _time(2), _time(4)
    # 2x the bits, but the same decode work: weight DRAM traffic doubles, everything
    # else is flat. Generous bound — the claim is "sub-linear in K", not a precise ratio.
    assert t4 < 1.8 * t2, f"K=4 GEMV {t4:.4f} ms vs K=2 {t2:.4f} ms — decode is NOT rate-flat"
