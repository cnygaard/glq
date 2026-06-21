"""E8P-RVQ integration tests: codebook roundtrip + full quantize→pack→load→forward.

The forward test quantizes a random weight via ``quantize_layer_e8_shell_rht`` with the
E8P codebook, loads the int64 packed artifacts into an ``E8RHTLinear(codebook_type='e8p')``,
and checks that both the B=1 (TC-GEMV / WMMA matvec) and B>1 (decompress + dense matmul)
forward paths reproduce ``x @ W_hat.T`` — i.e. the kernels + RHT round-trip is correct.
Requires CUDA (the E8P decode kernels are CUDA-only)."""
import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from glq.codebook_e8p import E8PCodebook, E81BCodebook                    # noqa: E402
from glq.quantize_model import quantize_layer_e8_shell_rht                # noqa: E402
from glq.quantized_linear import E8RHTLinear                              # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="E8P kernels are CUDA-only")


def _build_e8p_linear(W, bpw, dev):
    """Quantize W with the E8P codebook and load the artifacts into an E8RHTLinear."""
    out_f, in_f = W.shape
    H = torch.eye(in_f, device=dev)                       # identity Hessian → near-RTN, trivial LDLQ
    cb = E8PCodebook(device=dev, verbose=False)
    W_hat, arts, met = quantize_layer_e8_shell_rht(W, H, cb, bpw=bpw)
    lin = E8RHTLinear(in_f, out_f, bias=False, codebook_type='e8p').to(dev)
    lin.load_state_dict({k: v.to(dev) for k, v in arts.items()}, strict=False)
    cb2 = cb if bpw == 4 else (E81BCodebook(device=dev, verbose=False) if bpw == 3 else None)
    lin.set_codebook(cb, codebook2=cb2)
    return lin, W_hat.float()


@requires_cuda
def test_e8p_codebook_roundtrip():
    dev = 'cuda'
    torch.manual_seed(0)
    cb = E8PCodebook(device=dev, verbose=False)
    x = torch.randn(20000, 8, device=dev) / cb.opt_scale
    dec, idx = cb.quantize_fast(x.half())
    sqnr = 10 * torch.log10(x.var() / (x - dec.float()).var()).item()
    assert idx.dtype == torch.int64 and 0 <= idx.min() and idx.max() < 65536
    assert sqnr > 8.0, f"E8P 2-bit SQNR too low: {sqnr:.1f} dB"


@requires_cuda
def test_e81b_codebook_grid():
    dev = 'cuda'
    cb = E81BCodebook(device=dev, verbose=False)
    assert cb.e81b_grid.shape == (256, 8) and cb.e81b_grid.dtype == torch.float16
    _, idx = cb.quantize(torch.randn(1000, 8, device=dev))
    assert idx.max() < 256


@requires_cuda
@pytest.mark.parametrize("bpw", [2, 3, 4])
def test_e8p_forward_matches_dequant(bpw):
    """forward(x) must reproduce x @ W_hat.T for both the B=1 and B>1 kernel paths."""
    dev = 'cuda'
    torch.manual_seed(bpw)
    out_f, in_f = 512, 512
    W = torch.randn(out_f, in_f, device=dev) * 0.02
    lin, W_hat = _build_e8p_linear(W, bpw, dev)
    x = torch.randn(4, in_f, device=dev, dtype=torch.float16)
    y_ref = x.float() @ W_hat.T

    y_multi = lin(x).float()                              # B>1 → decompress + dense matmul
    rel_multi = (y_multi - y_ref).norm() / y_ref.norm().clamp(min=1e-6)
    assert rel_multi < 0.05, f"bpw={bpw} B>1 rel err {rel_multi:.4f}"

    y_one = lin(x[:1]).float()                            # B=1 → decode_matvec_e8p / lookupmatmul_e81b_k8
    rel_one = (y_one - y_ref[:1]).norm() / y_ref[:1].norm().clamp(min=1e-6)
    assert rel_one < 0.05, f"bpw={bpw} B=1 rel err {rel_one:.4f}"


@requires_cuda
def test_e8p_stage2_buffer_shapes():
    """4bpw registers Qidxs2_e8p (4-D); 3bpw registers Qidxs2_e81b (2-D); 2bpw neither."""
    dev = 'cuda'
    torch.manual_seed(0)
    W = torch.randn(512, 512, device=dev) * 0.02
    lin4, _ = _build_e8p_linear(W, 4, dev)
    assert lin4.Qidxs2_e8p.numel() > 0 and lin4.Qidxs2_e8p.dim() == 4
    assert lin4.Qidxs2_e81b.numel() == 0
    lin3, _ = _build_e8p_linear(W, 3, dev)
    assert lin3.Qidxs2_e81b.numel() > 0 and lin3.Qidxs2_e81b.dim() == 2
    assert lin3.Qidxs2_e8p.numel() == 0
    lin2, _ = _build_e8p_linear(W, 2, dev)
    assert lin2.Qidxs2_e8p.numel() == 0 and lin2.Qidxs2_e81b.numel() == 0
    assert lin2.inv_resid_scale.item() == 0.0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required"); sys.exit(1)
    test_e8p_codebook_roundtrip()
    print("codebook roundtrip OK")
    test_e81b_codebook_grid()
    print("e81b grid OK")
    for _bpw in (2, 3, 4):
        test_e8p_forward_matches_dequant(_bpw)
        print(f"forward bpw={_bpw} OK")
    test_e8p_stage2_buffer_shapes()
    print("stage-2 buffer shapes OK")
    print("\nALL E8P INTEGRATION TESTS PASSED")
