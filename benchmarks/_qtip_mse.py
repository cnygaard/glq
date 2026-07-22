"""Codebook-level MSE/SQNR bake-off: GLQ lattices (E8-shell, E8P) vs QTIP trellis.

Extends benchmarks/_codebook_sqnr_study.py (which found lattice swaps are <=0.4 dB but
NEVER tested trellis coding) with the QTIP trellis codebook. Tests the deep-research
headline: 2-bit iid-Gaussian MSE E8P 0.089 -> QTIP 0.069 (RD bound 2^-2R = 0.0625).

Every codebook is scored on the SAME source with a fitted scalar scale (fair rate-
distortion comparison), on (a) iid unit Gaussian and (b) real RHT'd SmolLM3 weight
blocks. Bitrate held fixed per column: QTIP K bits/weight; E8P/shell 2 bpw single stage.
"""
import argparse
import math
import sys

import torch

sys.path.insert(0, "benchmarks")
sys.path.insert(0, ".")
import _qtip_trellis as qt  # noqa: E402
from glq.codebook import E8ShellCodebook  # noqa: E402
from glq.codebook_e8p import E8PCodebook  # noqa: E402


def sqnr(x, xh):
    return 10 * math.log10((x.float() ** 2).mean().item()
                           / ((x - xh).float() ** 2).mean().item())


def best_scale(quant_fn, X, coarse=(0.3, 1.3, 21), refine=9):
    """Min-MSE over a scalar scale a: hatX = a * quant_fn(X/a). Returns (mse, a)."""
    lo, hi, n = coarse
    grid = torch.linspace(lo, hi, n).tolist()
    best = (float("inf"), 1.0)
    for _ in range(2):
        for a in grid:
            hatX = a * quant_fn(X / a)
            mse = ((X - hatX) ** 2).mean().item()
            if mse < best[0]:
                best = (mse, a)
        step = (grid[1] - grid[0])
        grid = torch.linspace(best[1] - step, best[1] + step, refine).tolist()
    return best


def qtip_cb(K, variant, dev):
    if variant == "3inst":                       # lookup-free computed, V=1
        cb = qt.bitshift_codebook(L=16, K=K, V=1, tlut_bits=0, decode_mode="3inst")
    elif variant == "hyb":                        # QTIP's best: kmeans LUT, V=2, Q=9
        cb = qt.bitshift_codebook(L=16, K=K, V=2, tlut_bits=9, decode_mode="quantlut_sym")
    else:
        raise ValueError(variant)
    cb = cb.to(dev)
    cb.fakeinf = cb.fakeinf.to(dev)              # plain attr, not moved by .to()
    return cb


def make_qtip_fn(cb, rows):
    def fn(Xflat):
        X2d = Xflat.reshape(rows, -1).to(torch.float16).to(cb.state.device)
        hat, _ = cb.quantize(X2d)
        return hat.reshape(Xflat.shape).to(Xflat.dtype).to(Xflat.device)
    return fn


def make_block_fn(cb):
    def fn(Xflat):
        Xb = Xflat.reshape(-1, 8).to(cb.device)
        dec, _ = cb.quantize(Xb)
        return dec.reshape(Xflat.shape).to(Xflat.dtype).to(Xflat.device)
    return fn


def run_source(name, X, rows, dev):
    print(f"\n=== {name}  (N={X.numel()}, unit-var-normalized) ===", flush=True)
    X = (X / X.std()).to(dev)                     # unit variance source
    shell = E8ShellCodebook(device=dev, verbose=False)
    e8p = E8PCodebook(device=dev, verbose=False)
    rows_dev = {}
    # 2-bit head-to-head: shell, e8p, qtip-3inst, qtip-hyb
    arms = [
        ("e8-shell 2b", make_block_fn(shell)),
        ("e8p     2b", make_block_fn(e8p)),
        ("qtip3inst 2b", make_qtip_fn(qtip_cb(2, "3inst", dev), rows)),
        ("qtip-hyb  2b", make_qtip_fn(qtip_cb(2, "hyb", dev), rows)),
        # trellis scaling at higher bits (no lattice baseline here; RVQ recipe differs)
        ("qtip3inst 3b", make_qtip_fn(qtip_cb(3, "3inst", dev), rows)),
        ("qtip-hyb  3b", make_qtip_fn(qtip_cb(3, "hyb", dev), rows)),
        ("qtip3inst 4b", make_qtip_fn(qtip_cb(4, "3inst", dev), rows)),
        ("qtip-hyb  4b", make_qtip_fn(qtip_cb(4, "hyb", dev), rows)),
    ]
    base = None
    for label, fn in arms:
        mse, a = best_scale(fn, X)
        s = sqnr(X, a * fn(X / a))
        if label.startswith("e8p"):
            base = mse
        db = f"  dB_vs_e8p={10*math.log10(base/mse):+.2f}" if (base and "2b" in label) else ""
        print(f"  {label:14s} MSE={mse:.5f}  SQNR={s:6.2f} dB  scale={a:.3f}{db}", flush=True)
    print(f"  [ref] 2-bit RD bound sigma^2*2^(-2R) = {2**-4:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers-file", default="/opt/dlami/nvme/glvq/layers.pt")
    ap.add_argument("--real-layer", default="model.layers.18.mlp.down_proj")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    # (a) iid unit Gaussian
    X = torch.randn(args.rows, args.seqlen)
    run_source("iid Gaussian", X, args.rows, dev)
    torch.cuda.empty_cache()                          # release the trellis state tensors

    # (b) real RHT'd weight block: incoherence-process a real layer, then flatten
    try:
        from glq.rht import RHT
        layers = torch.load(args.layers_file, weights_only=True)
        W = layers[args.real_layer]["W"].to(dev)
        rht = RHT(W.shape[0], W.shape[1], device=dev, block_diagonal=True, apply_left=True, e8p=False)
        Wt = rht.transform_weights(W)                 # ~iid after incoherence processing
        # chunk to bounded (rows, seqlen) — trellis MSE is asymptotic in seqlen, and the
        # full 11008-long sequence blows up the Viterbi state tensor (T*B*2^(L-KV)).
        n = args.rows * args.seqlen
        Xr = Wt.flatten()[:n].reshape(args.rows, args.seqlen)
        run_source(f"RHT'd {args.real_layer} [{args.rows}x{args.seqlen}]", Xr, args.rows, dev)
    except Exception as e:
        print(f"\n[real-weight source skipped: {type(e).__name__}: {e}]", flush=True)


if __name__ == "__main__":
    main()
