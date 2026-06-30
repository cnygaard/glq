"""Phase A: synthetic grouped-MoE parity for the N-stage e8p fused op.

Validates the new grouped kernels (Phase 1 E8P stage-2/3 + Phase 2 E81B graft)
reproduce the eager per-expert single-linear N-stage path. Quantizes synthetic
experts at the real gemma-4-26B-A4B expert dims (w13 out=1408/in=2816, w2
out=2816/in=704 — both mult-of-32 so E81B's WMMA is valid) via the REAL batched
MoE quant, loads them into a REAL GLQFusedMoEMethod layer, and runs apply() twice:
fused op (GLQ_MOE_FORCE_FALLBACK=0) vs eager loop (=1). bpw 6 exercises the E8P
stage-2 accumulate; bpw 5 exercises the E81B grouped graft.
"""
import os

import torch

import glq.hf_integration  # noqa: F401
import glq.inference_kernel as ik
from glq.codebook_e8p import E8PCodebook, E81BCodebook
from glq.quantize_model import quantize_experts_e8_shell_rht_batched
from glq_vllm.fused_moe_method import GLQFusedMoEMethod

ik._try_load_cuda_ext()
DEV = "cuda"


class _QC:
    codebook = "e8p"

    def __init__(self, bpw):
        self.bpw = bpw


def _sqnr(ref, got):
    d = ref.float() - got.float()
    return 10 * torch.log10(ref.float().pow(2).sum() / d.pow(2).sum().clamp(min=1e-20)).item()


def _fill(layer, pfx, arts):
    """Stack per-expert arts into the layer's (E, ...) buffers (SV is shared)."""
    for k, v in arts[0].items():
        buf = getattr(layer, f"{pfx}_{k}", None)
        if buf is None:
            continue
        if k == "SV":
            buf.data = v.to(buf.dtype).to(DEV)   # replace data (also moves to CUDA)
            continue
        stacked = torch.stack([a[k].to(buf.dtype) for a in arts]).to(DEV)
        if stacked.shape == buf.shape:
            buf.data = stacked
        else:
            print(f"  [warn] {pfx}_{k} shape {stacked.shape} != buf {buf.shape}", flush=True)


def build(E, hidden, inter, w13_out, bpw, e8p):
    torch.manual_seed(0)
    W13 = torch.randn(E, w13_out, hidden) * 0.1
    W2 = torch.randn(E, hidden, inter) * 0.1
    H13 = torch.eye(hidden).unsqueeze(0).repeat(E, 1, 1)
    H2 = torch.eye(inter).unsqueeze(0).repeat(E, 1, 1)
    arts13 = [r[1] for r in quantize_experts_e8_shell_rht_batched(W13, H13, e8p, bpw=bpw)]
    arts2 = [r[1] for r in quantize_experts_e8_shell_rht_batched(W2, H2, e8p, bpw=bpw)]

    m = GLQFusedMoEMethod.__new__(GLQFusedMoEMethod)
    m.quant_config = _QC(bpw)
    m.codebook_type = "e8p"
    layer = torch.nn.Module()
    m._create_weights_e8p(layer, E, hidden, inter, True, w13_out)
    layer.activation = "silu"
    _fill(layer, "w13", arts13)
    _fill(layer, "w2", arts2)
    m._process_e8p(layer)
    return m, layer


def main():
    e8p = E8PCodebook(device=DEV, verbose=False)
    E8PCodebook  # noqa
    E81BCodebook(device=DEV, verbose=False)  # lazily build the e81b grid path
    E, hidden, inter, w13_out, top_k = 8, 2816, 704, 1408, 2
    print("=== grouped N-stage e8p MoE parity (fused op vs eager loop) ===", flush=True)
    ok_all = True
    for bpw in (6, 5):
        m, layer = build(E, hidden, inter, w13_out, bpw, e8p)
        stage = "E8P stage-2 (3 E8P stages)" if bpw == 6 else "E81B stage (E8P+E8P+E81B)"
        fused_ok = getattr(layer, "glq_e8p_fused_ok", None)
        torch.manual_seed(1)
        T = 32
        x = torch.randn(T, hidden, device=DEV, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.randint(0, E, (T, top_k), device=DEV, dtype=torch.int64)
        topk_w = torch.softmax(torch.randn(T, top_k, device=DEV), dim=-1)
        with torch.no_grad():
            os.environ["GLQ_MOE_FORCE_FALLBACK"] = "0"
            yf = m.apply(layer, x, topk_w, topk_ids).float()
            os.environ["GLQ_MOE_FORCE_FALLBACK"] = "1"
            yl = m.apply(layer, x, topk_w, topk_ids).float()
        rel = (yf - yl).abs().max().item() / (yl.abs().max().item() + 1e-9)
        sqnr = _sqnr(yl, yf)
        passed = fused_ok and rel < 2e-2
        ok_all = ok_all and passed
        print(f"bpw={bpw} [{stage}] fused_ok={fused_ok}: max-rel(fused vs loop)={rel:.2e} "
              f"SQNR={sqnr:6.1f} dB  {'PASS' if passed else '*** FAIL ***'}", flush=True)
    os.environ["GLQ_MOE_FORCE_FALLBACK"] = "0"
    print("VERDICT:", "PASS — grouped N-stage matches the eager loop" if ok_all else "FAIL", flush=True)


if __name__ == "__main__":
    main()
