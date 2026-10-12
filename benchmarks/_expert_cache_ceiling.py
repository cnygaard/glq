"""Turn a measured offload slope + a measured hit rate into a PREDICTED tok/s.

This is the gate the expert-cache plan needed and did not have. S0 produced a hit-rate curve
(`benchmarks/expert_cache_s0.md`), but a hit rate is not a speedup: it only pays to the
extent the decode step is actually spent moving expert bytes over PCIe. This script makes that
conversion explicit and falsifiable, so the decision to build an arena rests on arithmetic
somebody can check rather than on the hit rate looking large.

The model, which is deliberately the simplest one that the data constrains:

    traffic(h)   = (1 - h) * expert_bytes_per_token          # bytes still crossing PCIe
    step(h)      = step_floor + traffic(h) / effective_bw
    step_floor   = measured_step - traffic(h_now) / effective_bw

`effective_bw` is *fitted from two measured arms*, not taken from a PCIe spec sheet. That
matters: fine-grained UVA reads of a routed expert land well under link peak, and using peak
would understate the cache's value by the same factor it overstates the link.

Run it:

    python benchmarks/_expert_cache_ceiling.py          # the measured L4 case
    python benchmarks/_expert_cache_ceiling.py --help   # to re-point it at another box
"""
from __future__ import annotations

import argparse

GIB = 1024.0 ** 3
MIB = 1024.0 ** 2

#: Flash-Next 3bpw, read off the safetensors header
#: (`tests/models/qwen3_8_flash_next_glq_3bpw.json`): 48 layers x 512 experts, and per
#: (layer, expert) 3 x trellis_packed I16[6400, 48] + SU/SV/Wscale = 1.776 MiB.
BLOB_MIB = 1.776
N_LAYERS, N_EXPERTS, TOP_K = 48, 512, 10
#: Total pageable expert bytes, and what one decode token actually touches.
EXPERT_GIB = N_LAYERS * N_EXPERTS * BLOB_MIB / 1024.0
TOKEN_MIB = N_LAYERS * TOP_K * BLOB_MIB


def resident_fraction(offload_gib: float) -> float:
    """Share of expert bytes left on the card at a given `--cpu-offload-gb`.

    vLLM offloads whole parameters with no knowledge of routing, so a resident expert is
    resident by luck — which is exactly why its hit rate is just this fraction, and why a
    routing-aware arena of the same size should do better.
    """
    return max(0.0, min(1.0, (EXPERT_GIB - offload_gib) / EXPERT_GIB))


def fit_bandwidth(arms: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of ms/token against *bytes of PCIe traffic* per token.

    Returns `(effective_GB_per_s, step_floor_ms)`. Fitting against traffic rather than against
    GiB-offloaded is the point: GiB-offloaded is only a proxy, and the proxy breaks as soon as
    a cache changes the hit rate without changing the offload size.
    """
    pts = [((1.0 - resident_fraction(off)) * TOKEN_MIB, 1000.0 / tokps) for off, tokps in arms]
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if n < 2 or abs(den) < 1e-12:
        raise SystemExit("need >= 2 arms at DIFFERENT offload sizes to fit a slope")
    slope = (n * sxy - sx * sy) / den          # ms per MiB
    floor = (sy - slope * sx) / n              # ms with zero expert traffic
    bw = (MIB / (slope / 1000.0)) / 1e9        # MiB/ms -> GB/s
    return bw, floor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=None, metavar="OFFLOAD_GIB:TOKPS",
                    help="a measured arm; repeat. Default: the L4 arms measured 2026-10-11")
    ap.add_argument("--arena-gib", type=float, default=10.0,
                    help="VRAM an arena would get (default 10)")
    ap.add_argument("--hit-rate", type=float, default=0.8233,
                    help="measured decode hit rate at that arena size (default 0.8233, "
                         "per-layer + LRU at 115 slots/layer)")
    ap.add_argument("--baseline", type=float, default=None, metavar="OFFLOAD_GIB",
                    help="which arm the speedup is quoted AGAINST (default: the first "
                         "--arm). This is not cosmetic: quoting 1.60x against the 41.48 GiB "
                         "arm and 1.58x against the 40.30 GiB one are both true, and only "
                         "the shipped configuration's number may go in a table.")
    ap.add_argument("--resident", action="append", default=None,
                    metavar="OFFLOAD_GIB:RESIDENT_GIB",
                    help="vLLM's own 'Model loading took X GiB' per arm, to check the byte "
                         "accounting against the engine rather than trusting it")
    args = ap.parse_args(argv)

    arms = ([tuple(float(v) for v in a.split(":")) for a in args.arm] if args.arm
            else [(40.3, 6.521), (36.18, 6.834)])

    print(f"expert bytes      : {EXPERT_GIB:.2f} GiB ({N_LAYERS} x {N_EXPERTS} x "
          f"{BLOB_MIB} MiB)")
    print(f"touched per token : {TOKEN_MIB:.1f} MiB ({N_LAYERS} layers x top_k {TOP_K})")
    print()
    print("measured arms:")
    print(f"  {'offload':>8} {'tok/s':>8} {'ms/token':>9} {'experts resident':>17} "
          f"{'PCIe MiB/token':>15}")
    for off, tokps in arms:
        rf = resident_fraction(off)
        print(f"  {off:>8.2f} {tokps:>8.3f} {1000.0 / tokps:>9.2f} "
              f"{rf * 100:>16.1f}% {(1 - rf) * TOKEN_MIB:>15.1f}")

    # ---- does the byte model agree with the ENGINE? ----------------------
    # The absolute resident figure depends on how much non-expert weight sits on the card,
    # which this script does not model. The DELTA does not — un-offloading a GiB must put a
    # GiB of experts back on the card, 1:1 — and the delta is the only thing the fit uses.
    # So check the delta and say plainly that the offset is unchecked.
    res = [tuple(float(v) for v in r.split(":")) for r in (args.resident or [])]
    if len(res) >= 2:
        print()
        print("byte-accounting check against vLLM's own 'Model loading took':")
        res.sort()
        (o0, r0), (o1, r1) = res[0], res[-1]
        d_off, d_res = o0 - o1, r1 - r0
        print(f"  offload {o1:.2f} -> {o0:.2f} GiB  ({d_off:+.2f}), "
              f"resident {r0:.2f} -> {r1:.2f} GiB ({d_res:+.2f})")
        if abs(d_off - d_res) > 0.15:
            print(f"  ** MISMATCH of {abs(d_off - d_res):.2f} GiB. Un-offloading a GiB must "
                  f"return a GiB of experts to the card; if it does not, --offload-params is "
                  f"not moving what this script thinks it is and the fit is invalid.")
            return 1
        print(f"  1:1 within {abs(d_off - d_res):.2f} GiB -> the slope's byte axis is sound. "
              f"(The absolute\n  offset is NOT checked by this: it depends on non-expert "
              f"weights, which are not modelled.)")

    bw, floor = fit_bandwidth(arms)
    print()
    offs = sorted(a[0] for a in arms)
    print(f"fitted effective PCIe bandwidth : {bw:6.2f} GB/s")
    print(f"EXTRAPOLATED intercept at zero traffic: {floor:6.2f} ms/token  "
          f"-> {1000.0 / floor:5.2f} tok/s")
    print(f"  ** NOT a floor and NOT a ceiling. ** Fitted over offload "
          f"{offs[0]:.2f}-{offs[-1]:.2f} GiB")
    print(f"  ({offs[-1] - offs[0]:.1f} GiB wide); zero traffic is ~{offs[-1]:.0f} GiB "
          f"outside that window.")
    print("  Measured consequence: this intercept read 84.24 ms and called 1.82x a hard")
    print("  ceiling; the arena then measured 58.1 ms and 2.674x. Locally verified linearity")
    print("  (3 intervals within 4.0%) was treated as global, and it is not. The fitted")
    print("  bandwidth is also the UNCOALESCED UVA one, while an arena miss is a bulk copy,")
    print("  so both errors push the same way. Use the rows below to rank options, never as")
    print("  a bound on what is achievable.")
    if len(arms) == 2:
        print("  (two arms = a slope with no residual; a third point is what tests whether")
        print("   it is a line rather than a tangent, and the extrapolation below needs it)")
    if floor <= 0:
        print("  WARNING: a non-positive floor means the fit is not physical — the arms are")
        print("  too close together, or something other than offload changed between them.")
        return 1

    # The baseline arm is the one the cache has to beat.
    if args.baseline is not None:
        match = [a for a in arms if abs(a[0] - args.baseline) < 0.01]
        if not match:
            raise SystemExit(f"--baseline {args.baseline} matches no --arm; have "
                             f"{[a[0] for a in arms]}")
        base_off, base_tokps = match[0]
    else:
        base_off, base_tokps = arms[0]
    base_ms = 1000.0 / base_tokps
    h_now = resident_fraction(base_off)

    print()
    print(f"baseline: offload {base_off:.2f} GiB, {base_tokps:.3f} tok/s, "
          f"hit rate {h_now:.4f} (resident by luck, not by routing)")
    print()
    print(f"  {'arena':>7} {'hit rate':>9} {'PCIe MiB/tok':>13} {'ms/token':>9} "
          f"{'tok/s':>7} {'vs baseline':>12}")
    print("  " + "-" * 64)
    rows = [(args.arena_gib, args.hit_rate, "measured"),
            (args.arena_gib, 1.0, "PERFECT cache (extrapolated, NOT a bound)")]
    for gib, h, note in rows:
        ms = floor + (1.0 - h) * TOKEN_MIB * (MIB / (bw * 1e9)) * 1000.0
        print(f"  {gib:>6.1f}G {h:>9.4f} {(1 - h) * TOKEN_MIB:>13.1f} {ms:>9.2f} "
              f"{1000.0 / ms:>7.3f} {base_ms / ms:>11.2f}x   {note}")

    print()
    print("Caveats that bound this prediction:")
    print("  * An arena FILL is a bulk contiguous copy; the UVA reads it replaces are")
    print("    fine-grained and uncoalesced. The fitted bandwidth is the UVA one, so a fill")
    print("    is likely CHEAPER per byte than modelled -> this prediction is conservative.")
    print("  * The arena costs VRAM that KV currently holds. Single-stream decode barely")
    print("    uses KV (6.5% of 71,338 tokens measured), but a concurrent serve would feel")
    print("    it, so this is a B=1 argument only.")
    print("  * The hit rate came from a 367-step trace whose compulsory misses cap it at")
    print("    0.9201; a longer run would do slightly better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
