"""Decode-vs-W_hat SQNR for the BLOCK-DIAG e8p path — the suspect for the
gemma-4-26B-A4B-it e8p-4bpw MoE quality gap (MMLU-Pro 83.3% vs the shell 4bpw's
93.3%).

The shipped gate `tests/test_e8p_nstage.py::test_e8p_nstage_decode_matches_what`
runs only **pow2** dims (256x512), so it never exercises the block-diagonal RHT
that every non-pow2 gemma-4 expert linear uses. This runs the SAME gate
(serving forward must reproduce `x @ W_hat.T`) at the **real 26B-A4B expert
shapes** (non-pow2 -> block-diag), bpw=4 (the e8p MoE recipe).

Why this settles the "is there a block-diag e8p decode bug" question: the MMLU
run used the fused grouped-e8p MoE op, which is **bit-exact** to the per-expert
loop, which literally calls `E8RHTLinear._e8p_linear_apply` (the path exercised
here). So if this forward reproduces W_hat at the expert dims, the MoE decode
does too — and the MMLU gap is the e8p codebook's quant quality, not a bug.

Read: SQNR(x@W_hat.T, lin(x)). ~60+ dB => decode faithful (no bug); <40 dB => a
block-diag decode/stage bug (a dropped residual stage caps at ~21 dB).
"""
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import glq.hf_integration  # noqa: F401,E402  (registers factory)
import glq.inference_kernel as ik  # noqa: E402
from glq.codebook_e8p import E8PCodebook, E81BCodebook  # noqa: E402
from glq.quantize_model import quantize_layer_e8_shell_rht  # noqa: E402
from glq.quantized_linear import E8RHTLinear  # noqa: E402


def _sqnr(ref, got):
    d = ref.float() - got.float()
    return 10 * torch.log10(ref.float().pow(2).sum()
                            / d.pow(2).sum().clamp(min=1e-20)).item()


def run(out_f, in_f, bpw, e8p, e81b, dev, label):
    torch.manual_seed(0)
    W = torch.randn(out_f, in_f) * 0.1
    H = torch.eye(in_f)
    W_hat, arts, _ = quantize_layer_e8_shell_rht(W.clone(), H.clone(), e8p, bpw=bpw)
    W_hat = W_hat.to(dev).float()

    lin = E8RHTLinear(in_f, out_f, bias=False, codebook_type="e8p")
    lin.load_state_dict({k: v for k, v in arts.items()}, strict=False)
    lin.set_codebook(e8p, e81b)
    lin = lin.to(dev).train(False)

    pow2 = (in_f & (in_f - 1)) == 0 and (out_f & (out_f - 1)) == 0
    kind = "pow2     " if pow2 else "BLOCK-DIAG"
    worst = 1e9
    for B in (1, 8):
        x = torch.randn(B, in_f, device=dev, dtype=torch.float16)
        with torch.no_grad():
            y = lin(x).float()
        y_ref = x.float() @ W_hat.t()
        sqnr = _sqnr(y_ref, y)
        worst = min(worst, sqnr)
        print(f"  [{label:13s}] out={out_f:<5d} in={in_f:<5d} {kind} bpw={bpw} B={B}: "
              f"decode-vs-What SQNR={sqnr:6.1f} dB  {'PASS' if sqnr > 40 else '*** FAIL <40 ***'}",
              flush=True)
    qf = _sqnr(W.to(dev).float(), W_hat)
    print(f"  [{label:13s}] quant fidelity SQNR(W, W_hat)={qf:5.1f} dB (expect ~15-20 @ 4bpw)\n",
          flush=True)
    return worst


def main():
    ik._try_load_cuda_ext()
    if ik._glq_cuda is None:
        raise SystemExit("glq CUDA ext not built")
    dev = "cuda"
    e8p = E8PCodebook(device=dev, verbose=False)
    e81b = E81BCodebook(device=dev, verbose=False)
    print("=== block-diag e8p decode-vs-W_hat SQNR (gemma-4-26B-A4B expert dims) ===", flush=True)
    worsts = []
    # pow2 control (matches the shipped pow2 test — confirms the harness itself)
    worsts.append(run(512, 256, 4, e8p, e81b, dev, "pow2-control"))
    # real 26B-A4B expert linears (non-pow2 -> block-diag), 4bpw e8p recipe
    worsts.append(run(1408, 2816, 4, e8p, e81b, dev, "w13 gate||up"))   # out=1408 in=2816
    worsts.append(run(2816, 704, 4, e8p, e81b, dev, "w2 down"))         # out=2816 in=704
    ok = min(worsts) > 40
    print(f"VERDICT: {'PASS — block-diag e8p decode is faithful (no bug; gap is quant quality)' if ok else 'FAIL — block-diag e8p decode bug (worst SQNR < 40 dB)'}",
          flush=True)


if __name__ == "__main__":
    main()
