"""Is the fused ACS step's cost difference a BUG, or is it more accurate than its reference?

`_trellis_step_ulp.py` established: `prev` is bit-identical on every combo, `compiled ==
eager` bit-identically (so inductor did not drift), and only the Triton kernel differs, by
<= 2 ULP scattered over 0.02-7% of elements.

Because `prev` matches, the argmin selects the same candidate, so `best.values` is a
bit-identical SELECTION rather than arithmetic; and `new_cost = state_err + best` is a single
fp32 add. The difference therefore has to live in `state_err`, and
`glq/trellis_step_kernel.py:68-77` shows why:

    err = 0.0
    for vi in range(V):
        d = lut[vi, s] - x[vi]
        err += d * d                 # Triton contracts to fma(d, d, err): ONE rounding
    store(err + best)

torch's reference is `(recons_state - thing).square().sum(dim=0)`, which rounds every `d*d`
to fp32 and THEN rounds the additions. The FMA keeps the full product before adding, so it
performs FEWER roundings -- which should make it CLOSER to the exact value, not farther.

That is the falsifiable claim this script tests. Exact `state_err` is computed in float64 and
correctly rounded to float32; then, on the elements where fused and compiled disagree, we ask
which one the correctly-rounded value actually agrees with.

    fused wins overwhelmingly  -> not a bug; the gate pins the LESS accurate of the two
    compiled wins             -> the kernel is losing precision; a real defect
    split roughly evenly      -> both are 1-ulp noise around the true value

    python benchmarks/_trellis_step_fma.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch  # noqa: E402

from _trellis_step_ulp import COMBOS, three_ways, ulp_diff  # noqa: E402


def exact_new_cost(cb, cost, thing):
    """`state_err + best` with state_err accumulated in float64, then correctly rounded.

    float64 has 52 mantissa bits against float32's 23, so for a V-term sum of squares of
    float32-representable values it is exact for our purposes: the fp64 result rounded once
    to fp32 IS the correctly-rounded answer, which is the yardstick both fp32 paths are
    approximating.
    """
    # The same gather/min the reference uses. `best.values` is a selection out of `cost`, so
    # it carries no rounding of its own and needs no fp64 treatment.
    cand = torch.gather(
        cost.unsqueeze(-2).expand(-1, cb.state_cand.shape[1], -1), -1,
        cb.state_cand.expand(len(cost), -1, 2 ** (cb.K * cb.V)))
    best = torch.min(cand, dim=-1).values

    # recons_state is already (V, 1, NSTATES) -- see trellis_step_kernel.py's lut_ptr comment
    # -- so it broadcasts against (V, B, 1) directly. Adding an unsqueeze here produced a 3-D
    # err and an IndexError two lines later; keep the shape identical to `_init_cost`'s.
    err64 = ((cb.recons_state.double() - thing.double().unsqueeze(-1)) ** 2).sum(dim=0)
    exact = err64 + best.unsqueeze(-1).expand(
        -1, -1, 2 ** (cb.K * cb.V)).reshape(err64.shape).double()
    return exact.to(torch.float32)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    import test_trellis_fused_step as T

    try:
        import triton
        tv = triton.__version__
    except Exception:                                      # noqa: BLE001
        tv = "n/a"
    print(f"torch={torch.__version__} triton={tv} "
          f"gpu={torch.cuda.get_device_name(0)}")
    print(f"{'combo':<28} {'V':>2} {'differing':>10} {'fused==exact':>13} "
          f"{'compiled==exact':>16} {'verdict':>10}")

    for variant, K, B, masked in COMBOS:
        cb = T._cb(K, variant)
        torch.manual_seed(10_000 * K + B)
        X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
        cost = T._init_cost(cb, X, masked, seed=K * 31 + B)
        thing = X[cb.V:2 * cb.V]
        (_, c_c), (_, c_e), (_, c_f) = three_ways(cb, cost, thing)

        ref = exact_new_cost(cb, cost.clone(), thing)
        # Only where the two fp32 paths actually disagree, and only where everything is
        # finite -- the masked combos carry fakeinf, where "closer" is meaningless.
        d = (c_f != c_c) & torch.isfinite(c_f) & torch.isfinite(c_c) & torch.isfinite(ref)
        n = int(d.sum())
        if n == 0:
            print(f"{variant} K={K} B={B} m={masked:<5}  {cb.V:>2} {'0':>10}  (identical)")
            continue
        fused_exact = int((c_f[d] == ref[d]).sum())
        comp_exact = int((c_c[d] == ref[d]).sum())
        # Neither matching means both are a rounding step away from truth.
        uf = ulp_diff(c_f[d], ref[d]).float().mean()
        uc = ulp_diff(c_c[d], ref[d]).float().mean()
        verdict = ("FUSED closer" if fused_exact > comp_exact
                   else "COMPILED closer" if comp_exact > fused_exact else "tie")
        print(f"{variant} K={K} B={B} m={masked:<5}  {cb.V:>2} {n:>10} "
              f"{100.0 * fused_exact / n:>12.2f}% {100.0 * comp_exact / n:>15.2f}% "
              f"{verdict:>10}   mean_ulp_from_exact fused={float(uf):.3f} "
              f"compiled={float(uc):.3f}")

        # Bias check: a systematic one-directional error is a different animal from symmetric
        # rounding noise, so report the sign split rather than only magnitudes.
        sf = torch.sign(c_f[d].double() - ref[d].double())
        print(f"{'':28} {'':>2} fused-vs-exact sign: "
              f"+{int((sf > 0).sum())} / -{int((sf < 0).sum())} / 0:{int((sf == 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
