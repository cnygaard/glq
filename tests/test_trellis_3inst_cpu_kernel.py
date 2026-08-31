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


# ---- Gate 3: decode micro-variants agree (arith SIMD vs LUT gather) ----------------------
@pytest.mark.parametrize("variant", ["arith", "lut"])
def test_cpu_decode_variants_agree(variant, isa):
    ext = _ext()
    if isa == "scalar" and variant == "arith":
        pytest.skip("scalar tier decodes via the LUT only")
    ext.glq_cpu_set_decode_variant(variant)
    try:
        m, n, K = 128, 256, 4
        cb, packed = _make(K, m, n, seed=3)
        ref = gt.decode_layer(cb, packed, m, n, has_kernel=True)
        W = ext.glq_decompress_trellis_3inst_cpu(packed, m, n)
        assert torch.equal(W.float(), ref.float()), f"variant={variant} isa={isa}"
    finally:
        ext.glq_cpu_set_decode_variant("auto")


# ---- Gate 4: fused GEMV — accuracy vs fp64 reference, determinism, thread-independence ---
@pytest.mark.parametrize("K", KS)
def test_cpu_matvec_sqnr_vs_fp64_reference(K, isa):
    m, n = 256, 512
    cb, packed = _make(K, m, n, seed=11)
    W64 = gt.decode_layer(cb, packed, m, n, has_kernel=True).double()
    torch.manual_seed(5)
    x = torch.randn(n) * 0.5
    ref = (x.double() @ W64.t()).float()
    y = _ext().glq_decode_matvec_trellis_3inst_cpu(x.contiguous(), packed, m, n, 1.0)
    assert y.shape == (m,) and y.dtype == torch.float32
    err = (y.double() - ref.double()).pow(2).mean()
    sqnr = 10 * torch.log10(ref.double().pow(2).mean() / err).item()
    assert sqnr > 90.0, f"K={K} isa={isa} SQNR {sqnr:.1f} dB"


def test_cpu_matvec_is_deterministic_across_runs_and_threads(isa):
    m, n, K = 256, 512, 4
    cb, packed = _make(K, m, n, seed=12)
    x = (torch.randn(n) * 0.5).contiguous()
    ext = _ext()
    prev = torch.get_num_threads()
    try:
        outs = []
        for nt in (1, 2, prev):
            torch.set_num_threads(nt)
            outs.append(ext.glq_decode_matvec_trellis_3inst_cpu(x, packed, m, n, 1.0))
            outs.append(ext.glq_decode_matvec_trellis_3inst_cpu(x, packed, m, n, 1.0))
        for o in outs[1:]:
            assert torch.equal(outs[0], o)
    finally:
        torch.set_num_threads(prev)


def test_cpu_matvec_wscale_and_accum_fold(isa):
    """wscale multiplies the stored result; accum=True adds onto the caller's buffer —
    the RVQ stage-2 contract (y = y1 + wscale2*y2 exactly, __fadd-style single add)."""
    m, n, K = 64, 128, 2
    cb, packed = _make(K, m, n, seed=13)
    x = (torch.randn(n) * 0.5).contiguous()
    ext = _ext()
    y1 = ext.glq_decode_matvec_trellis_3inst_cpu(x, packed, m, n, 1.0)
    y2 = ext.glq_decode_matvec_trellis_3inst_cpu(x, packed, m, n, 0.5)
    assert torch.equal(y2, y1 * 0.5)
    acc = y1.clone()
    ext.glq_decode_matvec_trellis_3inst_cpu(x, packed, m, n, 0.5, acc, True)
    assert torch.equal(acc, y1 + y2)


# ---- Gate 5: batched GEMM — row b bit-identical to the GEMV on x[b] ----------------------
@pytest.mark.parametrize("B", [1, 2, 7, 8])
def test_cpu_matmul_row_parity_with_gemv(B, isa):
    m, n, K = 128, 256, 4
    cb, packed = _make(K, m, n, seed=21)
    torch.manual_seed(B)
    x = (torch.randn(B, n) * 0.5).contiguous()
    ext = _ext()
    out = ext.glq_decode_matmul_trellis_3inst_cpu(x, packed, m, n, 1.0)
    assert out.shape == (B, m) and out.dtype == torch.float32
    for b in range(B):
        gemv = ext.glq_decode_matvec_trellis_3inst_cpu(x[b].contiguous(), packed, m, n, 1.0)
        assert torch.equal(out[b], gemv), f"isa={isa} B={B} row {b} != GEMV"


def test_cpu_matmul_is_deterministic_across_threads(isa):
    m, n, K = 128, 256, 3
    cb, packed = _make(K, m, n, seed=22)
    x = (torch.randn(5, n) * 0.5).contiguous()
    ext = _ext()
    prev = torch.get_num_threads()
    try:
        outs = []
        for nt in (1, prev):
            torch.set_num_threads(nt)
            outs.append(ext.glq_decode_matmul_trellis_3inst_cpu(x, packed, m, n, 1.0))
        assert torch.equal(outs[0], outs[1])
    finally:
        torch.set_num_threads(prev)


# ---- Gate 6: fused linear (FHT -> decode-GEMM -> FHT -> signs) + stacked RVQ -------------
def _blocks_meta(n):
    from glq.hadamard import _block_decompose
    blocks = _block_decompose(n)
    return blocks, torch.tensor(
        [[sum(blocks[:i]), bs, bs.bit_length() - 1, 0] for i, bs in enumerate(blocks)],
        dtype=torch.int32)


def _reference_linear(x, sv, su, W_rht, wscale, blocks_n, blocks_m):
    from glq.hadamard import block_diagonal_fht
    xr = block_diagonal_fht((x.float() * sv.float()).clone(), blocks_n)
    y = xr @ (W_rht.float() * wscale).t()
    y = block_diagonal_fht(y, blocks_m)
    return y * su.float()


@pytest.mark.parametrize("B", [1, 3, 8, 17])
def test_cpu_fused_linear_matches_reference_bracket(B, isa):
    """B=1 GEMV, B<=8 fused GEMM, B>8 dense-transient path — all through one entry,
    against the pure-torch reference bracket. fp32 ordering differs from the reference
    matmul, so the gate is SQNR (the kernels' own exactness is gated elsewhere)."""
    m, n, K = 128, 256, 4
    cb, packed = _make(K, m, n, seed=31)
    W_rht = gt.decode_layer(cb, packed, m, n, has_kernel=True)
    torch.manual_seed(B)
    x = (torch.randn(B, n) * 0.5).contiguous()
    sv = torch.where(torch.rand(n) < 0.5, -1.0, 1.0).half()
    su = torch.where(torch.rand(m) < 0.5, -1.0, 1.0).half()
    blocks_n, meta_n = _blocks_meta(n)
    blocks_m, meta_m = _blocks_meta(m)
    ref = _reference_linear(x, sv, su, W_rht, 0.37, blocks_n, blocks_m)
    y = _ext().glq_fused_linear_trellis_3inst_cpu(
        x, sv, su, packed, meta_n, meta_m, 0.37, n, m, n, m)
    assert y.shape == (B, m) and y.dtype == torch.float32
    err = (y.double() - ref.double()).pow(2).mean()
    sqnr = 10 * torch.log10(ref.double().pow(2).mean() / err).item()
    assert sqnr > 80.0, f"isa={isa} B={B} SQNR {sqnr:.1f} dB"


def test_cpu_fused_linear_rvq2_two_stages(isa):
    """5-8 bpw stacked RVQ: y = y1 + wscale*inv_rs2*y2 by matmul linearity; the fused
    rvq2 entry must equal composing the two per-stage kernel calls exactly, and track
    the dense two-stage reference."""
    m, n = 128, 256
    cb1, packed1 = _make(4, m, n, seed=41)
    cb2, packed2 = _make(2, m, n, seed=42)
    torch.manual_seed(9)
    x = (torch.randn(2, n) * 0.5).contiguous()
    sv = torch.where(torch.rand(n) < 0.5, -1.0, 1.0).half()
    su = torch.where(torch.rand(m) < 0.5, -1.0, 1.0).half()
    blocks_n, meta_n = _blocks_meta(n)
    blocks_m, meta_m = _blocks_meta(m)
    wscale, irs2 = 0.5, 0.25
    ext = _ext()

    y = ext.glq_fused_linear_trellis_3inst_rvq2_cpu(
        x, sv, su, packed1, packed2, meta_n, meta_m, wscale, irs2, n, m, n, m)

    W = gt.decode_layer(cb1, packed1, m, n, has_kernel=True) \
        + irs2 * gt.decode_layer(cb2, packed2, m, n, has_kernel=True)
    ref = _reference_linear(x, sv, su, W, wscale, blocks_n, blocks_m)
    err = (y.double() - ref.double()).pow(2).mean()
    sqnr = 10 * torch.log10(ref.double().pow(2).mean() / err).item()
    assert sqnr > 80.0, f"isa={isa} SQNR {sqnr:.1f} dB"

    a = ext.glq_decode_matmul_trellis_3inst_cpu(x, packed1, m, n, 1.0)
    b = ext.glq_decode_matmul_trellis_3inst_cpu(x, packed2, m, n, 1.0)
    assert not torch.equal(a, b)   # the stages genuinely differ (guards a silent drop)


# ---- Gate 7: in-op block-diagonal FHT bit-exact vs the torch reference -------------------
@pytest.mark.parametrize("blocks", [[2048], [1024, 512, 256], [64, 64]])
def test_cpu_fht_blockdiag_bitexact_vs_pytorch_fht(blocks):
    from glq.hadamard import _pytorch_fht
    n = sum(blocks)
    torch.manual_seed(17)
    x = torch.randn(3, n).float()
    ref = x.clone()
    off = 0
    for bs in blocks:
        ref[..., off:off + bs] = _pytorch_fht(ref[..., off:off + bs].contiguous())
        off += bs
    meta = torch.tensor(
        [[sum(blocks[:i]), bs, bs.bit_length() - 1, 0] for i, bs in enumerate(blocks)],
        dtype=torch.int32)
    x0 = x.clone()
    out = _ext().glq_blockdiag_fht_cpu(x.contiguous(), meta)
    assert torch.equal(out, ref)
    assert torch.equal(x, x0)   # unlike block_diagonal_fht, the op must NOT mutate its input


# ---- Gate 8: E8RHTLinear on CPU — the fused path ENGAGES and the dense cache does not ----
def _cpu_layer(in_f=512, out_f=256, seed=11, K=2):
    from glq.quantized_linear import E8RHTLinear
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f) * 0.05).float()
    X = torch.randn(256, in_f)
    H = (X.T @ X) / 256
    cb = gt.TrellisCodebook(variant="3inst", K=K, device="cpu")
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cb)
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis")
    layer.load_state_dict({
        "trellis_packed": art["trellis_packed"],
        "SU": art["SU"], "SV": art["SV"],
        "Wscale": torch.tensor(art["Wscale"], dtype=torch.float32),
    }, strict=False)
    layer.set_codebook(cb)
    return layer, W_hat


def _sqnr(ref, got):
    import math
    return 10 * math.log10(ref.float().pow(2).mean().item()
                           / (got.float() - ref.float()).pow(2).mean().item())


@pytest.mark.parametrize("B", [1, 4])
def test_cpu_fused_linear_engages_and_matches(B):
    """The mechanism gate: on a CPU tensor the layer must take the fused path (compressed
    weights only — the dense fp32 cache must NOT materialize) and track x @ W_hat.T."""
    _ext()
    layer, W_hat = _cpu_layer()
    torch.manual_seed(3)
    x = (torch.randn(B, 512) * 0.5).half()
    y = layer(x)
    assert layer._trellis_op_cpu is True
    assert layer._trellis_W_rht is None, "dense cache materialized — fused path not engaged"
    ref = x.float() @ W_hat.t().float()
    assert _sqnr(ref, y) > 35.0, f"B={B} SQNR {_sqnr(ref, y):.1f} dB"


def test_cpu_fused_kill_switch_restores_dense_path(monkeypatch):
    _ext()
    import glq.quantized_linear as ql
    layer, _ = _cpu_layer(seed=12)
    x = (torch.randn(2, 512) * 0.5).half()
    y_fused = layer(x)
    assert layer._trellis_op_cpu is True

    monkeypatch.setattr(ql, "_GLQ_FUSED_TRELLIS_CPU_ENABLED", False)
    layer2, _ = _cpu_layer(seed=12)
    y_dense = layer2(x)
    assert layer2._trellis_op_cpu is False
    assert layer2._trellis_W_rht is not None      # the dense fallback ran
    assert _sqnr(y_dense, y_fused) > 80.0         # same math, different accumulation order


def test_cpu_path_does_not_latch_the_cuda_resolver():
    """A CPU forward must leave the CUDA-path cache untouched: the old single-slot bool
    would latch False on the first CPU token and never re-examine after .to('cuda')."""
    _ext()
    layer, _ = _cpu_layer(seed=13)
    layer((torch.randn(1, 512) * 0.5).half())
    assert layer._trellis_op is None
    assert layer._trellis_op_cpu is True


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
