"""P1 — layer proxy-loss gate: QTIP trellis-2b vs GLQ e8p-2b on the 6 captured SmolLM3 layers.

Shared metric proxy = tr(ΔW·H·ΔWᵀ)/m (undamped H) + SQNR, both via RHT+LDLQ. Gate P1:
trellis beats e8p on ≥4/6 layers AND median dB_vs_e8p ≥ +0.5 (the +1.2 dB codebook win
partly equalizes through LDLQ; if it doesn't surface here, the wiring is wrong).
"""
import argparse
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _qtip_ldlq as qtl  # noqa: E402
from glq.codebook_e8p import E8PCodebook  # noqa: E402
from glq.quantize_model import quantize_layer_e8_shell_rht  # noqa: E402


def proxy(W, Wh, H):
    d = (W - Wh).float()
    return ((d @ H) * d).sum().item() / W.shape[0]


def sqnr(W, Wh):
    return 10 * math.log10((W.float() ** 2).mean().item() / ((W - Wh).float() ** 2).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-file", default="/opt/dlami/nvme/glvq/layers.pt")
    ap.add_argument("--variant", default="3inst")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    layers = torch.load(args.layers_file, weights_only=True)
    cb_t = qtl.TrellisCodebook(variant=args.variant, device=dev)
    print(f"trellis variant={args.variant} opt_scale={cb_t.opt_scale:.4f}; {len(layers)} layers", flush=True)

    dbs, wins = [], 0
    for name, d in layers.items():
        W, H = d["W"].to(dev), d["H"].to(dev)
        # e8p-2b (H.clone(): the fn damps H in place)
        Wh_e8p = quantize_layer_e8_shell_rht(W, H.clone(), E8PCodebook(device=dev, verbose=False),
                                             bpw=2, block_diagonal=True)[0].to(dev)
        Wh_tr = qtl.quantize_layer_trellis_rht(W, H, cb_t)
        pe, pt = proxy(W, Wh_e8p, H), proxy(W, Wh_tr, H)
        db = 10 * math.log10(pe / pt)
        dbs.append(db)
        wins += pt < pe
        print(f"  {name:40s} e8p proxy={pe:.5g} sqnr={sqnr(W, Wh_e8p):6.2f} | "
              f"trellis proxy={pt:.5g} sqnr={sqnr(W, Wh_tr):6.2f}  dB={db:+.2f}", flush=True)

    med = statistics.median(dbs)
    ok = wins >= 4 and med >= 0.5
    print(f"\nGate P1: trellis beats e8p on {wins}/{len(layers)} layers; median dB={med:+.2f}  "
          f"[{'PASS' if ok else 'fail'}]", flush=True)


if __name__ == "__main__":
    main()
