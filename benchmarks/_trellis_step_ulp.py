"""Step 0: WHICH of the three ACS paths moved, and by how much?

86 of `tests/test_trellis_fused_step.py`'s combinations fail on the box (torch 2.13.0+cu130,
sm_120) with `prev` bit-exact and only `cost` differing. That gate deliberately pins the
hand-written Triton kernel to inductor's exact fp-contraction choice, so the failure is
consistent with codegen drift rather than an algebra bug -- the eager-vs-eager ALGEBRA gate
(`test_trellis_update_equiv`) passes. But "consistent with" is not a measurement.

Three paths, so the pairwise results say which one moved:

    compiled  cb.update under @torch.compile  -- what the gate compares against
    eager     the same body, dynamo disabled  -- the algebra, no contraction choices
    fused     glq.trellis_step_kernel.fused_update -- the hand-written Triton kernel

    fused == eager, compiled differs  -> INDUCTOR moved; the kernel is still right
    fused differs from both           -> the TRITON KERNEL moved
    all three differ                  -> both, or something structural

and the magnitude decides whether a ULP-bounded gate is honest:

    <= 1-2 ULP, scattered   -> fp contraction, the expected class
    larger, or structured   -> a real bug; do NOT relax the gate

Setup is imported from the test module itself rather than re-implemented, so this cannot
diagnose a configuration the failing test does not actually run.

    python benchmarks/_trellis_step_ulp.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch  # noqa: E402

# (variant, K, B, masked) -- drawn from the observed failures, plus a masked and a large-B
# case so a difference that only appears with the overlap head-mask is not missed.
COMBOS = [
    ("3inst", 1, 12, False),
    ("hyb", 2, 12, False),
    ("3inst", 4, 60, True),
    ("3inst", 2, 256, False),
    ("hyb", 4, 36, True),
]

_INT_FOR = {torch.float32: torch.int32, torch.float16: torch.int16,
            torch.bfloat16: torch.int16, torch.float64: torch.int64}


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in representable floats between a and b, elementwise.

    IEEE-754 bits are monotonic within a sign, so reinterpreting as a signed integer and
    folding the negative half onto a continuous ordering makes |key_a - key_b| exactly the
    number of representable values between them. Done in int64 because the fold overflows
    the source width at -0.0.
    """
    itype = _INT_FOR[a.dtype]
    imin = torch.iinfo(itype).min

    def key(t):
        r = t.contiguous().view(itype).to(torch.int64)
        return torch.where(r >= 0, r, imin - r)

    return (key(a) - key(b)).abs()


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"{name}: SHAPE/DTYPE MISMATCH {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    if torch.equal(a, b):
        return f"{name}: bit-identical"
    finite = torch.isfinite(a) & torch.isfinite(b)
    nan_a, nan_b = int(a.isnan().sum()), int(b.isnan().sum())
    diff = a != b
    n_diff, n_tot = int(diff.sum()), a.numel()
    u = ulp_diff(a, b)[finite]
    absd = (a - b).abs()[finite]
    # A contraction difference is scattered; a kernel bug usually hits whole rows.
    rows_touched = int(diff.any(dim=-1).sum()) if diff.dim() > 1 else -1
    rows_total = diff.shape[0] if diff.dim() > 1 else -1
    return (f"{name}: DIFFERS  {n_diff}/{n_tot} elems ({100.0 * n_diff / n_tot:.2f}%)  "
            f"rows {rows_touched}/{rows_total}  "
            f"max_ulp={int(u.max()) if u.numel() else 0}  "
            f"mean_ulp={float(u.float().mean()) if u.numel() else 0:.3f}  "
            f"max_abs={float(absd.max()) if absd.numel() else 0:.3e}  "
            f"nan a/b={nan_a}/{nan_b}")


def three_ways(cb, cost, thing):
    """(prev, cost) from compiled, eager and fused, each on its own clone of the input."""
    ngroup = 2 ** (cb.L - cb.K * cb.V)
    B = cost.shape[0]

    def fresh_prev():
        return torch.empty(B, ngroup, dtype=torch.int32, device="cuda")

    torch._dynamo.reset()
    p_c = fresh_prev()
    c_c = cb.update(cost.clone(), thing, p_c)

    was = torch._dynamo.config.disable
    torch._dynamo.config.disable = True          # the pattern test_trellis_cudagraph.py uses
    try:
        p_e = fresh_prev()
        c_e = cb.update(cost.clone(), thing, p_e)
    finally:
        torch._dynamo.config.disable = was

    from glq.trellis_step_kernel import fused_update
    p_f = fresh_prev()
    c_f = fused_update(cb, cost.clone(), thing, p_f)
    return (p_c, c_c), (p_e, c_e), (p_f, c_f)


def kernel_count(cb, B=60):
    """What test_update_is_two_kernels counts, so its `== 2` can be judged too."""
    torch._dynamo.reset()
    torch.manual_seed(12)
    X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
    cost = (cb.recons_state - X[:cb.V].unsqueeze(-1)).square().sum(dim=0)
    out = torch.empty(B, 2 ** (cb.L - cb.K * cb.V), dtype=torch.int32, device="cuda")
    for _ in range(3):
        cb.update(cost.clone(), X[cb.V:2 * cb.V], out)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        cb.update(cost.clone(), X[cb.V:2 * cb.V], out)
        torch.cuda.synchronize()
    return [e.key for e in prof.key_averages()
            if e.self_device_time_total > 0
            and "Memcpy" not in e.key and "Memset" not in e.key]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    import test_trellis_fused_step as T                      # the failing test's own setup

    try:
        import triton
        tv = triton.__version__
    except Exception:                                        # noqa: BLE001
        tv = "n/a"
    print(f"torch={torch.__version__} triton={tv} gpu={torch.cuda.get_device_name(0)} "
          f"cap={'.'.join(str(x) for x in torch.cuda.get_device_capability(0))}")

    for variant, K, B, masked in COMBOS:
        print(f"\n=== {variant} K={K} B={B} masked={masked} ===")
        cb = T._cb(K, variant)
        torch.manual_seed(10_000 * K + B)                    # identical to the test
        X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
        cost = T._init_cost(cb, X, masked, seed=K * 31 + B)
        thing = X[cb.V:2 * cb.V]
        (p_c, c_c), (p_e, c_e), (p_f, c_f) = three_ways(cb, cost, thing)
        print(f"  cost dtype={c_c.dtype} shape={tuple(c_c.shape)}")
        # prev decides the emitted path -- if any of these differ, the checkpoint moves.
        print("  PREV  " + compare("fused vs compiled", p_f, p_c))
        print("  PREV  " + compare("fused vs eager   ", p_f, p_e))
        print("  PREV  " + compare("compiled vs eager", p_c, p_e))
        print("  COST  " + compare("fused vs compiled", c_f, c_c))
        print("  COST  " + compare("fused vs eager   ", c_f, c_e))
        print("  COST  " + compare("compiled vs eager", c_c, c_e))

    ks = kernel_count(T._cb(4, "3inst"))
    print(f"\n=== inductor kernel count for update: {len(ks)} (test asserts == 2) ===")
    for k in ks:
        print(f"    {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
