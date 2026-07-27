"""Phase 0 de-risk: how should the 3INST trellis reach 5/6/7/8 bpw?

Two candidate architectures, measured against each other and against two references:

  native-K   one trellis at K=bpw. Encoder is already generic in K and Viterbi cost is
             FLAT in K (every step touches 2**L cells regardless), so this is free to
             *encode* — but the decode window shrinks to ceil(L/K) symbols (2 at K=8),
             so the trellis coding gain should collapse toward memoryless VQ. Also caps
             the CUDA decode: glq_trellis.cu's chunk width 8*R must fit a uint32 -> R<=4.
  stacked    RVQ: two trellis stages, each K<=4, residual-quantized. Keeps every stage
             on the existing kernels. Mirrors the shipped e8p N-stage design.
  e8p        the shipped alternative at the same bpw — the bar trellis must clear.
  scalar     RHT + LDLQ + optimal-clip uniform scalar quantizer. native-K is *supposed*
             to degenerate toward this; if it loses, that bpw is not a trellis rate.

All arms share one LDLQ sweep (``ldlq_sweep``) and differ only in the per-tile quantize
function, so no arm gets a different feedback/scale/layout treatment. The RVQ residual
stage runs INSIDE the column-block loop — matching e8p (glq/ldlq.py:486-514), where LDLQ
feedback compensates only the FINAL residual. Staging outside the loop loses that and
understates stacking.

Tier A (--tier A): iid Gaussian, H=I. Cheap; gives the rate-distortion slope. Known to be
  a weak discriminator — the earlier codebook study found shell==e8p within 0.2 dB on iid
  while the real gap only appears on structured weights.
Tier B (--tier B): real captured (W, H). Reports the Hessian-weighted proxy
  tr(dW.H.dW^T)/m as well as SQNR. This is the tier that decides.

  python _trellis_rate_study.py --tier A --device cuda
  python _trellis_rate_study.py --tier B --layers-file /opt/dlami/nvme/glvq/layers.pt
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glq.trellis as gt  # noqa: E402
from glq.ldlq import block_LDL  # noqa: E402
from glq.rht import RHT  # noqa: E402

TD = gt.TD  # 16

# Candidate stacked splits per bpw. More than one where a second decomposition is
# plausible, so the recipe sketched in quantize_model.py:831 isn't assumed optimal.
STACKED_SPLITS = {
    5: [(3, 2), (4, 1)],
    6: [(4, 2), (3, 3)],
    7: [(4, 3)],
    8: [(4, 4)],
}


def sqnr_db(W, Wh):
    num = W.float().pow(2).mean().item()
    den = (W.float() - Wh.float()).pow(2).mean().item()
    return 10 * math.log10(num / den) if den > 0 else 99.0


def proxy_loss(W, Wh, H):
    """tr(dW . H . dW^T)/m — the Hessian-weighted metric that tracks PPL."""
    d = (W.float() - Wh.float())
    return ((d @ H.float()) * d).sum().item() / W.shape[0]


# --------------------------------------------------------------------------- LDLQ sweep

def ldlq_sweep(Wr, H, quantize_fn, for_kernel=True):
    """LDLQ reverse sweep over 16-col blocks with a pluggable per-tile quantizer.

    Mirrors glq.trellis.trellis_ldlq but takes ``quantize_fn((B,256)) -> hatX (B,256)``
    so every arm shares identical feedback, damping and tile layout. ``Wr`` is already
    normalized (rms matched to the codebook range); returns the normalized hatWr.
    """
    dev = Wr.device
    m, n = Wr.shape
    b = TD
    perm = gt._PERMUTE.to(dev)
    inv_perm = gt._INV_PERMUTE.to(dev)

    damp = 0.01 * torch.diag(H).mean()
    L, _ = block_LDL(H + damp * torch.eye(n, device=dev), block_size=b)

    hatWr = torch.zeros_like(Wr)
    R = Wr.clone()
    for k in reversed(range(n // b)):
        kb, ke = k * b, (k + 1) * b
        feedback = R[:, ke:] @ L[ke:, kb:ke] if ke < n else 0.0
        WXWX = Wr[:, kb:ke] + feedback
        tiles = WXWX.reshape(m // b, b * b)
        if for_kernel:
            tiles = tiles[:, perm]
        hatX = quantize_fn(tiles)
        if for_kernel:
            hatX = hatX[:, inv_perm]
        hatWr[:, kb:ke] = hatX.reshape(m, b).to(hatWr.dtype)
        R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]
    return hatWr


# --------------------------------------------------------------------------- quantizers

def make_native_fn(cb):
    def fn(tiles):
        return cb.quantize_tiles(tiles)[0]
    return fn


def make_stacked_fn(cb1, cb2, rs):
    """Two trellis stages. resid is amplified by ``rs`` into the codebook's range and
    shrunk by 1/rs on reconstruction — the e8p cum_inv_rs scheme with one residual."""
    def fn(tiles):
        hat1 = cb1.quantize_tiles(tiles)[0]
        resid = (tiles - hat1) * rs
        hat2 = cb2.quantize_tiles(resid)[0]
        return hat1 + hat2 / rs
    return fn


def make_scalar_fn(bits, clip):
    """Symmetric uniform scalar quantizer at ``bits``, clipped at +/-clip (in units of
    the tile rms). The control for 'is the trellis structure buying anything at all'."""
    levels = 2 ** bits
    def fn(tiles):
        s = tiles.float()
        step = 2.0 * clip / (levels - 1)
        q = torch.clamp(torch.round(s / step), -(levels // 2), levels // 2 - 1)
        return q * step
    return fn


def fit_scalar_clip(Wr, bits):
    """Pick the uniform-quantizer clip that minimizes MSE — gives the control its best shot."""
    best, best_mse = None, float("inf")
    rms = Wr.float().pow(2).mean().sqrt().item()
    for c in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        fn = make_scalar_fn(bits, c * rms)
        mse = (Wr - fn(Wr)).pow(2).mean().item()
        if mse < best_mse:
            best, best_mse = c * rms, mse
    return best


def fit_resid_scale(cb1, cb2, Wr, H, for_kernel=True):
    """Fit the stage-2 residual amplification.

    Start from the measured amplitude ratio rms(Wr)/rms(resid) after a stage-1-only pass
    (the analytically right target: it lands the residual in the same calibrated range),
    then refine with a small multiplicative search on the true 2-stage MSE. e8p hardcodes
    QuIP# constants here, but a trellis residual is a different distribution — an
    unfitted scale would understate the stacked arm.
    """
    hat1 = ldlq_sweep(Wr, H, make_native_fn(cb1), for_kernel)
    resid_rms = (Wr - hat1).float().pow(2).mean().sqrt().item()
    base = Wr.float().pow(2).mean().sqrt().item() / max(resid_rms, 1e-12)
    best, best_mse = base, float("inf")
    for mult in (0.6, 0.8, 1.0, 1.25, 1.6):
        rs = base * mult
        hat = ldlq_sweep(Wr, H, make_stacked_fn(cb1, cb2, rs), for_kernel)
        mse = (Wr - hat).pow(2).mean().item()
        if mse < best_mse:
            best, best_mse = rs, mse
    return best


# --------------------------------------------------------------------------- arm runner

def _cb(K, dev):
    return gt.TrellisCodebook(variant="3inst", K=K, device=dev)


def run_arms(W, H, bpws, dev, do_e8p=True, rs_cache=None):
    """All arms on one (W, H). Returns {bpw: {arm_name: (sqnr, proxy)}}."""
    out = {}
    Wf, Hf = W.float().to(dev), H.float().to(dev)
    m, n = Wf.shape
    # RHT once, shared by every trellis/scalar arm (e8p does its own internally).
    damp = 0.01 * torch.mean(torch.diag(Hf))
    Hd = Hf + damp * torch.eye(n, device=dev)
    rht = RHT(m, n, device=dev, block_diagonal=True, apply_left=True, e8p=False)
    Wt = rht.transform_weights(Wf)
    Ht = rht.transform_hessian(Hd)

    for bpw in bpws:
        res = {}
        # ---- native K = bpw
        cb = _cb(bpw, dev)
        Wscale = Wt.pow(2).mean().sqrt().item() * cb.opt_scale
        Wr = Wt / Wscale
        hat = ldlq_sweep(Wr, Ht, make_native_fn(cb))
        Wh = rht.inverse_transform_weights(hat * Wscale)
        res[f"native-K{bpw}"] = (sqnr_db(Wf, Wh), proxy_loss(Wf, Wh, Hf))

        # ---- stacked splits
        for (k1, k2) in STACKED_SPLITS[bpw]:
            cb1, cb2 = _cb(k1, dev), _cb(k2, dev)
            Wscale_s = Wt.pow(2).mean().sqrt().item() * cb1.opt_scale
            Wr_s = Wt / Wscale_s
            key = (k1, k2)
            if rs_cache is not None and key in rs_cache:
                rs = rs_cache[key]
            else:
                rs = fit_resid_scale(cb1, cb2, Wr_s, Ht)
                if rs_cache is not None:
                    rs_cache[key] = rs
            hat = ldlq_sweep(Wr_s, Ht, make_stacked_fn(cb1, cb2, rs))
            Wh = rht.inverse_transform_weights(hat * Wscale_s)
            res[f"stack-{k1}+{k2}"] = (sqnr_db(Wf, Wh), proxy_loss(Wf, Wh, Hf))

        # ---- scalar control (same RHT + LDLQ, uniform quantizer)
        clip = fit_scalar_clip(Wt, bpw)
        hat = ldlq_sweep(Wt, Ht, make_scalar_fn(bpw, clip))
        Wh = rht.inverse_transform_weights(hat)
        res[f"scalar-{bpw}b"] = (sqnr_db(Wf, Wh), proxy_loss(Wf, Wh, Hf))

        # ---- e8p reference at matched bpw
        if do_e8p:
            try:
                from glq.codebook_e8p import E8PCodebook
                from glq.quantize_model import quantize_layer_e8_shell_rht
                Wh_e = quantize_layer_e8_shell_rht(
                    Wf, Hf.clone(), E8PCodebook(device=dev, verbose=False),
                    bpw=bpw, block_diagonal=True)[0].to(dev)
                res[f"e8p-{bpw}"] = (sqnr_db(Wf, Wh_e), proxy_loss(Wf, Wh_e, Hf))
            except Exception as exc:            # loud, not silent — a missing reference matters
                print(f"    !! e8p-{bpw} failed: {type(exc).__name__}: {exc}", flush=True)
        out[bpw] = res
    return out


def print_table(tag, results, show_proxy):
    print(f"\n=== {tag} ===", flush=True)
    for bpw in sorted(results):
        arms = results[bpw]
        print(f"  bpw {bpw}:", flush=True)
        ranked = sorted(arms.items(), key=lambda kv: -kv[1][0])
        for name, (s, p) in ranked:
            extra = f"  proxy={p:.5g}" if show_proxy else ""
            print(f"    {name:16s} SQNR={s:6.2f} dB{extra}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bpws", default="5,6,7,8")
    ap.add_argument("--shape", default="256x256", help="Tier A only, MxN")
    ap.add_argument("--layers-file", default="/opt/dlami/nvme/glvq/layers.pt")
    ap.add_argument("--no-e8p", action="store_true")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--resid-scales", default=None,
                    help="JSON from a previous run ({'3+2': 7.46, ...}). Reuses those scales "
                         "instead of re-fitting — the fit costs 6 extra LDLQ sweeps per split, "
                         "which dominates on real (large) layers. RHT Gaussian-izes the "
                         "weights, so a scale fitted on iid transfers.")
    ap.add_argument("--layers", default=None,
                    help="comma-substrings; only run captured layers whose name matches one")
    args = ap.parse_args()

    dev = args.device
    bpws = [int(x) for x in args.bpws.split(",") if x.strip()]
    print(f"device={dev} torch={torch.__version__} bpws={bpws} tier={args.tier}", flush=True)
    # The 3inst lut is K-invariant (decode_3inst over the 2**16 states), so opt_scale is
    # identical at every K — worth stating, it means Wscale is not an arm confound.
    print(f"opt_scale K=2/4/8: {_cb(2, dev).opt_scale:.4f} / {_cb(4, dev).opt_scale:.4f} / "
          f"{_cb(8, dev).opt_scale:.4f}", flush=True)

    preset_rs = {}
    if args.resid_scales:
        with open(args.resid_scales) as f:
            raw = json.load(f)
        raw = raw.get("resid_scales", raw)
        for k, v in raw.items():
            a, b = k.split("+")
            preset_rs[(int(a), int(b))] = float(v)
        print(f"reusing fitted residual scales: {raw}", flush=True)

    all_json = {}
    if args.tier == "A":
        m, n = (int(v) for v in args.shape.split("x"))
        torch.manual_seed(0)
        W = torch.randn(m, n, device=dev)
        H = torch.eye(n, device=dev)
        rs_cache = {}
        res = run_arms(W, H, bpws, dev, do_e8p=not args.no_e8p, rs_cache=rs_cache)
        print_table(f"Tier A — iid Gaussian {m}x{n}, H=I", res, show_proxy=False)
        print("\nfitted residual scales:", {f"{k[0]}+{k[1]}": round(v, 3)
                                            for k, v in rs_cache.items()}, flush=True)
        all_json["tierA"] = {str(b): {k: v for k, v in r.items()} for b, r in res.items()}
        all_json["resid_scales"] = {f"{k[0]}+{k[1]}": v for k, v in rs_cache.items()}
    else:
        layers = torch.load(args.layers_file, weights_only=True)
        if args.layers:
            want = [s for s in args.layers.split(",") if s.strip()]
            layers = {k: v for k, v in layers.items() if any(s in k for s in want)}
        print(f"{len(layers)} captured layers from {args.layers_file}", flush=True)
        per_layer, rs_cache = {}, dict(preset_rs)
        for name, d in layers.items():
            W, H = d["W"].to(dev), d["H"].to(dev)
            print(f"  layer {name} {tuple(W.shape)}", flush=True)
            res = run_arms(W, H, bpws, dev, do_e8p=not args.no_e8p, rs_cache=rs_cache)
            per_layer[name] = res
            print_table(f"Tier B — {name}", res, show_proxy=True)
        all_json["tierB"] = {ln: {str(b): r for b, r in rr.items()}
                             for ln, rr in per_layer.items()}
        all_json["resid_scales"] = {f"{k[0]}+{k[1]}": v for k, v in rs_cache.items()}

        # ---- the pre-registered decision rule, applied
        print("\n=== DECISION (pre-registered: native-K adopted only if it beats the best "
              "stacked split on the Hessian proxy on >=4/6 layers) ===", flush=True)
        for bpw in bpws:
            wins, margins, scalar_loss = 0, [], 0
            for name, rr in per_layer.items():
                arms = rr[bpw]
                nat = arms.get(f"native-K{bpw}")
                st = [v for k, v in arms.items() if k.startswith("stack-")]
                sc = arms.get(f"scalar-{bpw}b")
                if not nat or not st:
                    continue
                best_st = min(st, key=lambda t: t[1])          # lowest proxy = best
                if nat[1] < best_st[1]:
                    wins += 1
                margins.append(10 * math.log10(best_st[1] / nat[1]))
                if sc and min(nat[1], best_st[1]) > sc[1]:
                    scalar_loss += 1
            med = statistics.median(margins) if margins else float("nan")
            # Pre-registered bar: >=4 of 6 layers. Expressed as ceil(2n/3) so it stays
            # well-defined (and equals 4 at n=6) if --layers narrows the set.
            need = math.ceil(2 * len(per_layer) / 3)
            verdict = "native-K" if wins >= need else "stacked"
            flag = f"  [!! loses to scalar on {scalar_loss} layers]" if scalar_loss else ""
            print(f"  bpw {bpw}: native wins {wins}/{len(per_layer)} layers, "
                  f"median margin {med:+.2f} dB -> adopt {verdict}{flag}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_json, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
