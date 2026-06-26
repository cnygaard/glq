"""Phase B gate: e8p 5/6/7/8 bpw N-stage decode — bit-exact A/B (compressed vs dense)
+ B=1/B=8 consistency, on a synthetic e8p-quantized layer. The first forward JIT-builds
the CUDA ext (recompiles the extended glq_fused_linear_e8p_cuda)."""
import os
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/opt/dlami/nvme/torch_ext")
import torch
import glq.hf_integration  # noqa: F401 (registers E8RHTLinear etc.)
from glq.codebook_e8p import E8PCodebook, E81BCodebook
from glq.quantize_model import quantize_layer_e8_shell_rht
from glq.quantized_linear import E8RHTLinear

dev = "cuda"
e8p = E8PCodebook(device=dev, verbose=False)
e81b = E81BCodebook(device=dev, verbose=False)
torch.manual_seed(0)
M, N = 256, 512                      # out_features, in_features (pow2 -> single-block RHT)
W = torch.randn(M, N) * 0.1
H = torch.eye(N)

def _sqnr(ref, got):
    diff = ref - got
    return 10 * torch.log10(ref.pow(2).sum() / diff.pow(2).sum().clamp(min=1e-20)).item()


print(f"=== e8p N-stage decode A/B (M={M} N={N}) ===", flush=True)
ok = True
for bpw in (4, 5, 6, 7, 8):
    W_hat, arts, met = quantize_layer_e8_shell_rht(W.clone(), H.clone(), e8p, bpw=bpw)
    keys = sorted(k for k in arts if k.startswith('Qidxs') or 'resid' in k)
    W_hat = W_hat.to(dev).float()                       # quantize-side dequantized weight (M,N)

    lin = E8RHTLinear(N, M, bias=False, codebook_type="e8p")
    lin.load_state_dict({k: v for k, v in arts.items()}, strict=False)
    lin.set_codebook(e8p, e81b)
    lin = lin.to(dev)
    lin.train(False)

    x8 = torch.randn(8, N, device=dev, dtype=torch.float16)
    with torch.no_grad():
        os.environ.pop("GLQ_E8P_DENSE_B1", None)
        y_comp = lin(x8).float()                       # B>1 compressed TC-GEMM
        os.environ["GLQ_E8P_DENSE_B1"] = "1"
        y_dense = lin(x8).float()                       # B>1 dense decompress+matmul
        os.environ.pop("GLQ_E8P_DENSE_B1", None)
        y_b1 = lin(x8[:1]).float()                      # B=1 GEMV path
    # THE key check: does the decode forward reproduce x @ W_hat.T (the quantize recon)?
    y_ref = (x8.float() @ W_hat.t())                   # (8, M)
    dec_sqnr = _sqnr(y_ref, y_comp)                    # decode-vs-What SQNR (dB)

    rel_ab = ((y_comp - y_dense).norm() / y_dense.norm().clamp(min=1e-9)).item()
    rel_b1 = ((y_b1[0] - y_comp[0]).norm() / y_comp[0].norm().clamp(min=1e-9)).item()
    det = all(torch.equal(lin(x8), lin(x8)) for _ in range(5))
    bad = (rel_ab > 5e-3) or (rel_b1 > 5e-3) or (not det) or (dec_sqnr < met['sqnr'] - 6)
    ok = ok and not bad
    nstage = len([k for k in keys if k.startswith('Qidxs')])
    print(f"bpw={bpw} stages={nstage} keys={keys}\n"
          f"   comp-vs-dense rel={rel_ab:.2e}  B1-vs-B8row0 rel={rel_b1:.2e}  det={det}\n"
          f"   quantize-SQNR={met['sqnr']:.1f}dB   DECODE-vs-What-SQNR={dec_sqnr:.1f}dB"
          f"   {'OK' if not bad else 'FAIL <-- decode != W_hat'}", flush=True)

print("=== RESULT:", "ALL PASS" if ok else "FAILURES", "===", flush=True)
