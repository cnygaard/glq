"""Why does trellis CUDA-graph capture fail for ('pair', 256, 80)?

Quantizing Qwen3.8-Flash-Next produced

    trellis pair CUDA-graph capture failed for ('pair', 256, 80) ...
    AcceleratorError: CUDA error: operation failed due to a previous error during capture

Needs a GPU; it is a diagnostic, not a test. Walks a grid of (T, B) shapes through capture
and reports which succeed, so the boundary is visible rather than inferred from one data
point.

The leading hypothesis, from reading rather than measurement — this script exists to
FALSIFY it, not confirm it:

  _pair_body calls viterbi twice, once without overlap and once with. The overlap branch
  allocates `mask` and runs scatter_, and fused_update (trellis_step_kernel.py:98) launches
  a Triton kernel whose grid depends on B. Triton JIT-compiles per specialization and
  compiling during capture is illegal. The 3-iteration warmup is meant to cover this, so
  the question is what B=80 reaches that the warmup does not.

Also worth ruling out: the shared _vit_graph_pool interacting with an earlier
_capture_viterbi, and torch.compile recompiling `update` / `_tb_step` for the shape.

    python benchmarks/_trellis_capture_repro.py [--variant 3inst] [--K 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import trellis as T  # noqa: E402


def _try_capture(cb, T_dim, B, warm):
    """Capture one (T, B) pair shape. Returns (ok, detail)."""
    X = torch.randn(T_dim, B, device="cuda", dtype=torch.float16)
    try:
        cb._vit_graphs.clear()
        cb._vit_graph_pool = None
        T._GLQ_TRELLIS_CUDAGRAPH_ENABLED = True     # a prior failure disables it
        cb._capture_pair(X, warmup_iters=warm)
        return True, "captured"
    except Exception as e:                          # noqa: BLE001 - that is the subject
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        # Never leave a poisoned context for the next shape, or every later result is noise.
        ok = T._recover_cuda_context()
        if not ok:
            print("  !! context unrecoverable — later rows are not trustworthy", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="3inst")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--shapes", default="256:80,256:64,256:128,256:32,128:80,512:80")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU")
        return 2

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"variant={args.variant} K={args.K} warmup={args.warmup}")
    print(f"fused step enabled: {T._fused_step_on()}")
    print(f"GLQ_TRELLIS_CUDAGRAPH_MAX_B={T._GLQ_TRELLIS_CUDAGRAPH_MAX_B}")

    cb = T.TrellisCodebook(variant=args.variant, K=args.K, device="cuda").cb
    print(f"codebook: L={cb.L} V={cb.V} K={cb.K}\n")

    print(f"{'T':>6} {'B':>6}  result")
    fails = 0
    for spec in args.shapes.split(","):
        t_dim, b = (int(x) for x in spec.split(":"))
        ok, detail = _try_capture(cb, t_dim, b, args.warmup)
        print(f"{t_dim:6d} {b:6d}  {'OK' if ok else 'FAIL'}  {detail}", flush=True)
        fails += (not ok)

    # The hypothesis check: does capture succeed when the fused Triton step is off?
    # If so, the Triton launch inside the captured region is implicated.
    if fails:
        print("\n--- retry the failing shapes with the fused step disabled ---")
        T._GLQ_TRELLIS_FUSED_STEP_ENABLED = False
        for spec in args.shapes.split(","):
            t_dim, b = (int(x) for x in spec.split(":"))
            ok, detail = _try_capture(cb, t_dim, b, args.warmup)
            print(f"{t_dim:6d} {b:6d}  {'OK' if ok else 'FAIL'}  {detail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
