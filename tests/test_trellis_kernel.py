"""S1 CUDA trellis-kernel gates (GPU-only).

The highest-risk item in the whole port is the **encoder-pack ⇔ kernel-unpack pairing**:
`pack_layer` writes QTIP's kernel byte layout (tail-biting bit-pack + `_PERMUTE` + tile-flip)
and the CUDA kernel must reconstruct exactly those weights via warp-shuffle bit-unpack →
`idx*(idx+1)` → smem LUT → sign-flip. `glq/trellis.py:decode_layer` (pure torch, already
proven bit-exact against the encoder) is the oracle.

Because the tlut is stored fp16, the kernel's fp16 LUT values are bit-identical to the
fp32-holding-fp16 values torch produces — so the decompress gate is a true `torch.equal`,
not a tolerance. The decompress kernel shares its decode path with the matvec kernel, so
this one test pins both.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402
from glq import inference_kernel as ik  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sqnr(ref, got):
    return 10 * math.log10(ref.float().pow(2).mean().item()
                           / (got.float() - ref.float()).pow(2).mean().item())


def _ext():
    assert ik._try_load_cuda_ext(), "glq CUDA extension failed to build"
    return ik._glq_cuda


def _quantized(m, n, seed=0):
    """Quantize a random layer in the KERNEL storage layout; return (cb, packed, tlut16)."""
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(m, n, device=dev) * 0.05).float()
    X = torch.randn(512, n, device=dev)
    H = (X.T @ X) / 512
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    cb = gt.TrellisCodebook(variant="hyb", K=2, tlut=tlut, device=dev)
    _, Qidxs, _ = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True).to(dev)
    return cb, packed, tlut.to(dev)


# ---------------------------------------------------------------------------
# GATE 1 (the crux): CUDA decompress is BIT-EXACT vs the pure-torch decode_layer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,n", [(128, 256), (256, 512), (64, 128)])
def test_cuda_decompress_bitexact_vs_decode_layer(m, n):
    cb, packed, tlut16 = _quantized(m, n, seed=m)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)          # (m,n) fp32 oracle
    W = _ext().glq_decompress_trellis_cuda(packed, tlut16, m, n)     # (m,n) fp16
    assert W.shape == (m, n) and W.dtype == torch.float16
    assert torch.equal(W.float(), ref.float()), \
        f"max|Δ|={(W.float() - ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# GATE 2: the fused B=1 GEMV matches x @ W_decoded.T
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,n", [(256, 512), (512, 2048)])
def test_cuda_matvec_matches_reference_gemv(m, n):
    cb, packed, tlut16 = _quantized(m, n, seed=m + 1)
    W = gt.decode_layer(cb, packed, m, n, has_kernel=True)            # (m,n) fp32
    torch.manual_seed(7)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    ref = x.float() @ W.t()                                           # (m,) fp32
    out = _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
    assert out.shape == (m,) and out.dtype == torch.float32
    # fp16 tensor-core products w/ fp32 accum vs an fp32 reference
    assert _sqnr(ref, out) > 40.0, f"SQNR {_sqnr(ref, out):.1f} dB"


def test_cuda_matvec_is_deterministic():
    m, n = 256, 512
    cb, packed, tlut16 = _quantized(m, n, seed=3)
    x = (torch.randn(n, device="cuda") * 0.5).to(torch.float16)
    a = _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
    b = _ext().glq_decode_matvec_trellis_cuda(x, packed, tlut16, m, n)
    assert torch.equal(a, b)   # block-owns-m-range + smem reduce → no atomics, bit-stable


# ---------------------------------------------------------------------------
# GATE 3: the fused linear op == the pure-torch S0 reference (RHT in/out bracket)
# ---------------------------------------------------------------------------
def _trellis_layer(in_f=512, out_f=256, seed=11):
    from glq.quantized_linear import E8RHTLinear
    dev = "cuda"
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f, device=dev) * 0.05).float()
    X = torch.randn(512, in_f, device=dev)
    H = (X.T @ X) / 512
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    cb = gt.TrellisCodebook(variant="hyb", K=2, tlut=tlut, device=dev)
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cb)
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis").to(dev)
    layer.load_state_dict({k: v.to(dev) for k, v in {
        "trellis_packed": art["trellis_packed"], "tlut": art["tlut"],
        "SU": art["SU"], "SV": art["SV"],
        "Wscale": torch.tensor(art["Wscale"], dtype=torch.float32),
    }.items()}, strict=False)
    layer.set_codebook(cb)
    return layer, W_hat


@pytest.mark.parametrize("B", [1, 4])
def test_fused_linear_trellis_matches_s0_reference(B):
    in_f, out_f = 512, 256
    layer, W_hat = _trellis_layer(in_f, out_f)
    x = torch.randn(B, in_f, device="cuda", dtype=torch.float16)
    ref = x.float() @ W_hat.float().t()          # the S0 dense reference
    y = layer(x)
    assert y.shape == (B, out_f)
    assert _sqnr(ref, y) > 35.0, f"SQNR {_sqnr(ref, y):.1f} dB"


# ---------------------------------------------------------------------------
# GATE 4: the fused op is CUDA-graph capturable — the reason cudaFuncSetAttribute
# and cudaGetDeviceProperties were hoisted out of the launch path. Without this the
# decode win never materializes under vLLM/HF graph capture.
# ---------------------------------------------------------------------------
def test_fused_linear_trellis_is_cudagraph_capturable():
    in_f, out_f = 512, 256
    layer, _ = _trellis_layer(in_f, out_f, seed=13)
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
        f"graph replay != eager, max|Δ|={(static_y.float() - y_eager.float()).abs().max().item()}"
