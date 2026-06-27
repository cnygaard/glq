"""Regression tests for e8p N-stage (5-8 bpw) RVQ decode.

The headline gate is **decode-vs-W_hat SQNR**: the fused-op forward must reproduce
the quantize-side dequantized weight (x @ W_hat.T) to within fp16 precision for
every bpw 4-8. A decode-vs-decode A/B is NOT sufficient — it shares the scale
wiring, so a dropped residual stage (e.g. the set_codebook stale-scale bug that
zeroed inv_resid_scale2/3 → C++ `if(scale==0)continue` silently dropped stage 3/4)
passes the A/B while collapsing quality. Here we compare against W_hat directly.
"""
import os

import pytest
import torch

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for e8p kernels")


def _sqnr(ref, got):
    d = ref.float() - got.float()
    return 10 * torch.log10(ref.float().pow(2).sum()
                            / d.pow(2).sum().clamp(min=1e-20)).item()


@requires_gpu
@pytest.mark.parametrize("bpw", [4, 5, 6, 7, 8])
def test_e8p_nstage_decode_matches_what(bpw):
    """The fused-op forward must reproduce x @ W_hat.T (the quantize recon) for
    all bpw 4-8. A dropped top stage caps the SQNR at ~21 dB; a correct N-stage
    decode is fp16-limited (~60+ dB)."""
    try:
        import glq.inference_kernel as ik
        ik._try_load_cuda_ext()
        if ik._glq_cuda is None:
            pytest.skip("glq CUDA ext not built")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"glq CUDA ext unavailable: {e}")
    import glq.hf_integration  # noqa: F401
    from glq.codebook_e8p import E8PCodebook, E81BCodebook
    from glq.quantize_model import quantize_layer_e8_shell_rht
    from glq.quantized_linear import E8RHTLinear

    dev = "cuda"
    torch.manual_seed(0)
    M, N = 256, 512
    e8p = E8PCodebook(device=dev, verbose=False)
    e81b = E81BCodebook(device=dev, verbose=False)

    W = torch.randn(M, N) * 0.1
    H = torch.eye(N)
    W_hat, arts, _ = quantize_layer_e8_shell_rht(W.clone(), H.clone(), e8p, bpw=bpw)
    W_hat = W_hat.to(dev).float()

    lin = E8RHTLinear(N, M, bias=False, codebook_type="e8p")
    lin.load_state_dict({k: v for k, v in arts.items()}, strict=False)
    lin.set_codebook(e8p, e81b)
    lin = lin.to(dev)
    lin.train(False)

    x = torch.randn(8, N, device=dev, dtype=torch.float16)
    with torch.no_grad():
        y = lin(x).float()
    y_ref = x.float() @ W_hat.t()
    sqnr = _sqnr(y_ref, y)
    n_stages = len([k for k in arts if k.startswith("Qidxs")])
    assert sqnr > 40.0, (
        f"bpw={bpw} ({n_stages} stages): decode-vs-W_hat SQNR={sqnr:.1f}dB "
        f"(<40 => a residual stage is being dropped at decode)")


@requires_gpu
def test_e8p_stage3_actually_contributes():
    """Dropping stage-3 from a 6bpw e8p layer MUST change the forward output.
    The stale-scale bug left them identical (|delta|==0)."""
    try:
        import glq.inference_kernel as ik
        ik._try_load_cuda_ext()
        if ik._glq_cuda is None:
            pytest.skip("glq CUDA ext not built")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"glq CUDA ext unavailable: {e}")
    import glq.hf_integration  # noqa: F401
    from glq.codebook_e8p import E8PCodebook, E81BCodebook
    from glq.quantize_model import quantize_layer_e8_shell_rht
    from glq.quantized_linear import E8RHTLinear

    dev = "cuda"
    torch.manual_seed(0)
    M, N = 256, 512
    e8p = E8PCodebook(device=dev, verbose=False)
    e81b = E81BCodebook(device=dev, verbose=False)
    W = torch.randn(M, N) * 0.1
    _, arts, _ = quantize_layer_e8_shell_rht(W.clone(), torch.eye(N), e8p, bpw=6)

    def build(drop3):
        a = {k: v for k, v in arts.items()}
        if drop3:
            a.pop("Qidxs3_e8p", None)
            a.pop("inv_resid_scale2", None)
        lin = E8RHTLinear(N, M, bias=False, codebook_type="e8p")
        lin.load_state_dict(a, strict=False)
        lin.set_codebook(e8p, e81b)
        return lin.to(dev).train(False)

    x = torch.randn(8, N, device=dev, dtype=torch.float16)
    with torch.no_grad():
        y3 = build(False)(x).float()
        y2 = build(True)(x).float()
    delta = (y3 - y2).norm().item()
    assert delta > 1e-3, (
        f"stage-3 contributes nothing (|delta|={delta:.2e}) — it is being dropped")
