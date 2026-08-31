"""CPU fused-decode kernels for the 3INST trellis codebook (glq/csrc/cpu/).

Gate structure mirrors tests/test_trellis_3inst_kernel.py, with the CPU-specific twists:

* Tiers are runtime-selectable (`glq_cpu_set_isa`) so one process can regression-test
  every tier the machine supports; the `scalar` tier always exists, so the bit-math is
  CI-gated even on runners with no AVX2.
* The decode oracle is `glq.trellis.decode_3inst` / `decode_layer` — same as CUDA. The
  fp16-rounding trap is load-bearing: the fp32 sum of the two decoded halves is EXACT
  but differs from the oracle's fp16 add on 27,707/65,536 states unless rounded back to
  fp16 (RN). Gate 1 pins that recipe via the extension's own LUT.
"""
from __future__ import annotations

import functools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402

KS = [2, 3, 4]
TIERS = ["scalar", "avx2", "avx512", "avx512fp16"]


def _ext():
    from glq import inference_kernel_cpu as ikc
    assert ikc._try_load_cpu_ext(), f"glq CPU extension failed to build: {ikc.cpu_ext_status()}"
    return ikc._glq_cpu


@functools.lru_cache(maxsize=None)
def _make(K, m, n, seed=0):
    """Quantize a random 3inst layer in KERNEL layout, entirely on CPU (identity Hessian,
    CPU Viterbi). Bit-exactness is content-agnostic, so a random layer is a full gate."""
    torch.manual_seed(seed)
    cb = gt.TrellisCodebook(variant="3inst", K=K, device="cpu")
    W = (torch.randn(m, n) * 0.05).float()
    _, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=True)
    packed = gt.pack_layer(cb, Qidxs, m, n, has_kernel=True)
    assert packed.shape == (m // 16 * (n // 16), 16 * K)
    return cb, packed


@pytest.fixture(params=TIERS)
def isa(request):
    ext = _ext()
    if not ext.glq_cpu_isa_available(request.param):
        pytest.skip(f"tier {request.param} not available on this CPU/build")
    ext.glq_cpu_set_isa(request.param)
    yield request.param
    ext.glq_cpu_set_isa("auto")


# ---- Gate 1: the extension's LUT IS the oracle (pins the RN16 recipe) --------------------
def test_cpu_lut_matches_decode_3inst():
    ext = _ext()
    lut = ext.glq_trellis_3inst_lut_cpu()           # (65536,) fp16
    assert lut.dtype == torch.float16 and lut.shape == (65536,)
    ref = gt.decode_3inst(torch.arange(2 ** 16))    # fp32, every value fp16-exact
    assert torch.equal(lut.float(), ref.float())


# ---- Gate 2 (crux): decompress bit-exact vs decode_layer, all rates, every tier ----------
@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("m,n", [(64, 128), (128, 256), (256, 512)])
def test_cpu_decompress_bitexact_vs_decode_layer(m, n, K, isa):
    cb, packed = _make(K, m, n, seed=m)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)          # (m,n) fp32 oracle
    W = _ext().glq_decompress_trellis_3inst_cpu(packed, m, n)
    assert W.shape == (m, n) and W.dtype == torch.float16
    assert torch.equal(W.float(), ref.float()), \
        f"K={K} isa={isa} bad={int((W.float() != ref.float()).sum())}"


def test_cpu_decompress_r1_residual_rate(isa):
    """R=1 exists only as the stacked-RVQ residual rate; its two-lane tail-biting
    continuation is the known bit-unpack trap."""
    m, n = 64, 128
    cb, packed = _make(1, m, n, seed=7)
    ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)
    W = _ext().glq_decompress_trellis_3inst_cpu(packed, m, n, True)   # allow_r1
    assert torch.equal(W.float(), ref.float())


def test_cpu_r1_refused_as_primary():
    m, n = 64, 128
    _, packed = _make(1, m, n, seed=7)
    with pytest.raises(Exception, match="[Rr]"):
        _ext().glq_decompress_trellis_3inst_cpu(packed, m, n)         # allow_r1 defaults False


# ---- Gate 9 (loader/dispatch diagnostics) ------------------------------------------------
def test_cpu_unknown_isa_tier_raises():
    ext = _ext()
    with pytest.raises(Exception, match="isa|tier|unknown|unavailable"):
        ext.glq_cpu_set_isa("avx99")


def test_cpu_active_isa_reports_a_known_tier():
    ext = _ext()
    ext.glq_cpu_set_isa("auto")
    assert ext.glq_cpu_active_isa() in TIERS


def test_cpu_kernel_supported_predicate():
    ext = _ext()
    assert ext.glq_trellis_cpu_kernel_supported(64, 128)
    assert not ext.glq_trellis_cpu_kernel_supported(48, 128)   # m % 32
    assert not ext.glq_trellis_cpu_kernel_supported(64, 96)    # k % 64
