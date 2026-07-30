"""Calibrate the RVQ residual-scale estimator: is one constant per residual-rate enough?

`trellis_ldlq_nstage` fits the residual amplification once, from the first column block, as
rms(target)/rms(residual) — zero extra Viterbi passes, unlike a search (6x) which is
unaffordable on a multi-hour quantize. But that estimate is derived from the STAGE-1
residual, so it is identical for every bpw on a given layer and cannot know the residual
stage's rate. Measured: it is right at K2=2 and ~25-40% too large at K2=4.

The proposal is `rs = rms_ratio * C[K2]` — per-layer scale from the free estimate, times a
per-rate constant calibrated here. This script finds the true optimum per (layer, bpw) and
reports whether the optimal MULTIPLIER is a function of K2 alone (one constant works) or
also varies by layer (it does not).

Search is an **expanding bracket + golden section on log(multiplier)**, not a fixed grid: a
fixed grid silently censors optima that fall outside it, and an earlier ±35% grid did
exactly that (best-at-the-edge on q_proj bpw 8), which would bias the calibrated constant.

  python _trellis_rs_validate.py --layers-file .../layers.pt --bpws 5,6,7,8 --max-layers 3
"""
import argparse
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glq.trellis as gt  # noqa: E402
from glq.rht import RHT  # noqa: E402

GOLD = (math.sqrt(5.0) - 1.0) / 2.0          # 0.618
MULT_LO, MULT_HI = 0.05, 20.0                # expansion guard rails


def sqnr_db(W, Wh):
    num = W.float().pow(2).mean().item()
    den = (W.float() - Wh.float()).pow(2).mean().item()
    return 10 * math.log10(num / den) if den > 0 else 99.0


def search_multiplier(f, max_evals=14):
    """Maximize f(mult) over mult>0. Expanding bracket from 1.0, then golden section.

    Returns (best_mult, best_val, n_evals, hit_guard). ``hit_guard`` True means the optimum
    ran into MULT_LO/MULT_HI — the answer is then censored and must not be averaged in.
    """
    cache = {}

    def ev(mu):
        mu = round(mu, 4)
        if mu not in cache:
            cache[mu] = f(mu)
        return cache[mu]

    step = 1.6
    f1 = ev(1.0)
    lo_v, hi_v = ev(1.0 / step), ev(step)
    if lo_v <= f1 >= hi_v:                                   # already bracketed at 1.0
        a, b, c = 1.0 / step, 1.0, step
    else:
        # walk downhill-free: expand in the improving direction until it stops improving
        down = lo_v > hi_v
        r = (1.0 / step) if down else step
        prev_m, prev_v = 1.0, f1
        cur_m, cur_v = r, (lo_v if down else hi_v)
        while MULT_LO < cur_m < MULT_HI and len(cache) < max_evals:
            nxt_m = cur_m * ((1.0 / step) if down else step)
            nxt_v = ev(nxt_m)
            if nxt_v <= cur_v:
                break
            prev_m, prev_v = cur_m, cur_v
            cur_m, cur_v = nxt_m, nxt_v
        nxt_m = cur_m * ((1.0 / step) if down else step)
        a, b, c = (nxt_m, cur_m, prev_m) if down else (prev_m, cur_m, nxt_m)
        a, c = min(a, c), max(a, c)

    hit = not (MULT_LO < b < MULT_HI)
    # golden section on log-scale inside [a, c]
    la, lc = math.log(a), math.log(c)
    while len(cache) < max_evals and (lc - la) > 0.06:
        m1, m2 = lc - GOLD * (lc - la), la + GOLD * (lc - la)
        if ev(math.exp(m1)) > ev(math.exp(m2)):
            lc = m2
        else:
            la = m1
    best_m = max(cache, key=cache.get)
    return best_m, cache[best_m], len(cache), hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-file", default="/opt/dlami/nvme/glvq/layers.pt")
    ap.add_argument("--bpws", default="5,6,7,8")
    ap.add_argument("--max-layers", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    bpws = [int(x) for x in args.bpws.split(",") if x.strip()]

    layers = torch.load(args.layers_file, weights_only=True)
    names = list(layers)[:args.max_layers]
    print(f"device={dev} bpws={bpws} layers={len(names)} max_evals={args.max_evals}",
          flush=True)

    by_bpw = {}
    for name in names:
        W = layers[name]["W"].to(dev).float()
        H = layers[name]["H"].to(dev).float()
        m, n = W.shape
        Hd = H + 0.01 * torch.mean(torch.diag(H)) * torch.eye(n, device=dev)
        rht = RHT(m, n, device=dev, block_diagonal=True, apply_left=True, e8p=False)
        Wt, Ht = rht.transform_weights(W), rht.transform_hessian(Hd)

        for bpw in bpws:
            cbs = [gt.TrellisCodebook(variant="3inst", K=k, device=dev)
                   for k in gt.trellis_rvq_recipe(bpw)]
            hat, _, ws, cum = gt.trellis_ldlq_nstage(Wt, Ht, cbs)
            rs_auto = 1.0 / cum[1]
            s_auto = sqnr_db(W, rht.inverse_transform_weights(hat * ws))

            def f(mult, _cbs=cbs, _Wt=Wt, _Ht=Ht, _rs=rs_auto, _rht=rht, _W=W):
                h, _, w, _ = gt.trellis_ldlq_nstage(_Wt, _Ht, _cbs,
                                                    resid_scales=[_rs * mult])
                return sqnr_db(_W, _rht.inverse_transform_weights(h * w))

            best_m, best_s, nev, hit = search_multiplier(f, args.max_evals)
            k2 = gt.trellis_rvq_recipe(bpw)[-1]
            flag = "  [GUARD-RAIL HIT — censored]" if hit else ""
            print(f"{name.split('.')[-2] + '.' + name.split('.')[-1]:14s} bpw {bpw} "
                  f"(K2={k2}): auto rs={rs_auto:8.3f} SQNR={s_auto:6.2f} | "
                  f"best x{best_m:.3f} SQNR={best_s:6.2f} (+{best_s - s_auto:.2f} dB, "
                  f"{nev} evals){flag}", flush=True)
            if not hit:
                by_bpw.setdefault(bpw, []).append((best_m, best_s - s_auto))

    print("\n=== per-residual-rate calibration ===", flush=True)
    print("If the best multiplier is a function of K2 alone, one constant suffices and the "
          "free rms-ratio already carries the layer dependence. Spread across layers is the "
          "test.", flush=True)
    for bpw in sorted(by_bpw):
        k2 = gt.trellis_rvq_recipe(bpw)[-1]
        mus = [m for m, _ in by_bpw[bpw]]
        gains = [g for _, g in by_bpw[bpw]]
        med = statistics.median(mus)
        spread = (max(mus) / min(mus)) if min(mus) > 0 else float("inf")
        print(f"  bpw {bpw} (K2={k2}): mults={[round(m, 3) for m in mus]} "
              f"median={med:.3f} spread={spread:.2f}x  "
              f"mean gain over auto={statistics.mean(gains):+.2f} dB", flush=True)
    print("\nSuggested C[K2] = median multiplier above; a spread >1.5x means one constant "
          "is NOT enough and the scale needs a per-layer search.", flush=True)


if __name__ == "__main__":
    main()
