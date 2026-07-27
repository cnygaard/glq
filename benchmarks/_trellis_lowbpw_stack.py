"""Does stacking help at 2-4 bpw too, or is the native->stacked crossover really at 4/5?

`trellis_rvq_recipe` ships single-stage native for 2-4 and stacked RVQ for 5-8. The 5-8 half
was measured (stacked wins by 0.3/0.9/2.4 dB at 6/7/8); the 2-4 half was inherited, not
compared. Theory says native should win below the crossover — low rate is exactly where the
trellis coding gain is strongest, and splitting the budget into two weaker codes throws it
away, the mirror image of the high-rate collapse. This checks that.

Stacked arms get a full expanding-bracket + golden-section search on the residual scale, not
the shipped constant: `_RESID_SCALE_CAL` was fitted with a K=4 stage 1, and these splits have
a K=1/2/3 stage 1, so the constant may not transfer. Giving the stacked arm its best possible
scale keeps the comparison honest — if it still loses, the split is genuinely worse.

  python _trellis_lowbpw_stack.py --layers-file /opt/dlami/nvme/glvq/layers.pt
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glq.trellis as gt  # noqa: E402
from glq.rht import RHT  # noqa: E402
from _trellis_rs_validate import search_multiplier, sqnr_db  # noqa: E402

# Every way to split a low budget into two K>=1 trellis stages.
SPLITS = {2: [(1, 1)], 3: [(2, 1)], 4: [(2, 2), (3, 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-file", default="/opt/dlami/nvme/glvq/layers.pt")
    ap.add_argument("--bpws", default="2,3,4")
    ap.add_argument("--max-layers", type=int, default=2)
    ap.add_argument("--max-evals", type=int, default=11)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    bpws = [int(x) for x in args.bpws.split(",") if x.strip()]

    layers = torch.load(args.layers_file, weights_only=True)
    names = list(layers)[:args.max_layers]
    print(f"device={dev} bpws={bpws} layers={len(names)}", flush=True)

    wins = {b: [0, 0] for b in bpws}          # bpw -> [native wins, total]
    for name in names:
        W = layers[name]["W"].to(dev).float()
        H = layers[name]["H"].to(dev).float()
        m, n = W.shape
        Hd = H + 0.01 * torch.mean(torch.diag(H)) * torch.eye(n, device=dev)
        rht = RHT(m, n, device=dev, block_diagonal=True, apply_left=True, e8p=False)
        Wt, Ht = rht.transform_weights(W), rht.transform_hessian(Hd)
        short = ".".join(name.split(".")[-3:])

        for bpw in bpws:
            nat_cbs = [gt.TrellisCodebook(variant="3inst", K=bpw, device=dev)]
            hat, _, ws, _ = gt.trellis_ldlq_nstage(Wt, Ht, nat_cbs)
            s_nat = sqnr_db(W, rht.inverse_transform_weights(hat * ws))
            print(f"\n{short}  bpw {bpw}", flush=True)
            print(f"    native-K{bpw}       SQNR={s_nat:6.2f} dB", flush=True)

            best_stack = -1e9
            for (k1, k2) in SPLITS[bpw]:
                cbs = [gt.TrellisCodebook(variant="3inst", K=k1, device=dev),
                       gt.TrellisCodebook(variant="3inst", K=k2, device=dev)]
                h0, _, w0, cum0 = gt.trellis_ldlq_nstage(Wt, Ht, cbs)
                rs_auto = 1.0 / cum0[1]

                def f(mult, _cbs=cbs, _rs=rs_auto):
                    h, _, w, _ = gt.trellis_ldlq_nstage(Wt, Ht, _cbs,
                                                        resid_scales=[_rs * mult])
                    return sqnr_db(W, rht.inverse_transform_weights(h * w))

                bm, bs, nev, hit = search_multiplier(f, args.max_evals)
                best_stack = max(best_stack, bs)
                flag = " [censored]" if hit else ""
                print(f"    stack-{k1}+{k2}       SQNR={bs:6.2f} dB "
                      f"(best rs x{bm:.3f}, {nev} evals){flag}", flush=True)
            d = s_nat - best_stack
            wins[bpw][1] += 1
            wins[bpw][0] += (d > 0)
            print(f"    -> native {'WINS' if d > 0 else 'LOSES'} by {abs(d):.2f} dB",
                  flush=True)

    print("\n=== crossover check ===", flush=True)
    for bpw in bpws:
        w, t = wins[bpw]
        print(f"  bpw {bpw}: native beats best stacked on {w}/{t} layers", flush=True)
    print("\nThe shipped recipe (native 2-4, stacked 5-8) is justified iff native sweeps "
          "these AND stacked swept 6-8 — i.e. the crossover sits between 4 and 5.", flush=True)


if __name__ == "__main__":
    main()
