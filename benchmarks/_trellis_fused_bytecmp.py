"""Does the <=2 ULP ACS cost difference change the EMITTED WEIGHTS?

This replaces a question I was answering badly. `_trellis_step_ulp.py` showed the fused Triton
ACS differs from the compiled reference by <=2 ULP in `cost` while `prev` stays bit-identical,
and `_trellis_step_fma.py` tried to adjudicate which is "more correct" against an fp64
reference -- a question about the quality of an intermediate, and one my yardstick got wrong
(it computed `d = lut - x` in fp64 while both real paths compute it in fp32).

The question that actually governs the product is simpler and is the one CLAUDE.md states:
**encode-path changes must produce byte-identical checkpoints.** So quantize the same layer
twice, once through each ACS path, and compare the artifacts.

    identical -> the ULP difference is invisible in the output. The per-step bit-equality gate
                 is stricter than the contract, and can be re-anchored on this comparison.
    differs   -> the fused path and the fallback emit different checkpoints. That is a real
                 defect no ULP argument excuses, and the per-step gate was doing its job.

`RHT` seeds its own Generator (`glq/rht.py:81`, seed=42) and the tlut init is seeded too, so
the encode path is deterministic and the arms differ ONLY in the kernel.

The mechanism is asserted, not assumed: `_fused_step_on()` is checked per arm. Without that,
a run where both arms silently took the same path would report a clean "identical" -- the same
trap that cost a full A/B round on the CPU MoE work.

    python benchmarks/_trellis_fused_bytecmp.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

import glq.trellis as gt  # noqa: E402

# (variant, K, m, n) -- K is bpw for the native stages. 512x512 keeps a full quantize inside a
# few seconds while still exercising real LDLQ column blocks and many Viterbi steps.
CASES = [
    ("3inst", 2, 512, 512),
    ("3inst", 3, 512, 512),
    ("3inst", 4, 512, 512),
    ("hyb", 2, 512, 512),
    ("hyb", 4, 256, 1024),
]


def make_layer(m: int, n: int, seed: int):
    """A weight and a genuine PSD Hessian, both fixed by seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(m, n, generator=g) * 0.02).cuda()
    A = torch.randn(n, 4 * n, generator=g).cuda()
    H = (A @ A.T) / (4 * n) + torch.eye(n, device="cuda") * 1e-3
    return W, H


def quantize_with(fused: bool, W, H, variant, K):
    """One arm. Returns (artifacts, W_hat, engaged) -- engaged is the mechanism check."""
    os.environ["GLQ_TRELLIS_FUSED_STEP"] = "1" if fused else "0"
    # A prior capture failure can latch this off process-wide (glq/trellis.py:382), which
    # would make the "fused" arm silently run the fallback.
    if not gt._GLQ_TRELLIS_FUSED_STEP_ENABLED:
        raise RuntimeError("_GLQ_TRELLIS_FUSED_STEP_ENABLED is latched off; arms not distinct")
    engaged = gt._fused_step_on()
    # The WRAPPER, not `.cb`: quantize_layer_trellis_rht reads `cb0.has_kernel` to pick the
    # stored tile layout. test_trellis_fused_step.py uses `.cb` because it calls `update`
    # directly, which lives on the inner bitshift_codebook.
    cb = gt.TrellisCodebook(
        variant=variant, K=K, device="cuda",
        tlut=(torch.randn(2 ** 9, 2, generator=torch.Generator().manual_seed(0))
              * 0.9682458365518543).to(torch.float16) if variant == "hyb" else None)
    W_hat, arts = gt.quantize_layer_trellis_rht(W, H, cb)   # 2-tuple, unlike the e8_shell one
    return arts, W_hat, engaged


def compare(a: dict, b: dict, wa, wb) -> list[str]:
    out = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        if k not in a or k not in b:
            out.append(f"    {k}: present in only one arm")
            continue
        va, vb = a[k], b[k]
        if not torch.is_tensor(va):
            out.append(f"    {k}: {'same' if va == vb else f'DIFFERS {va!r} vs {vb!r}'}")
            continue
        if va.shape != vb.shape or va.dtype != vb.dtype:
            out.append(f"    {k}: SHAPE/DTYPE {tuple(va.shape)}/{va.dtype} vs "
                       f"{tuple(vb.shape)}/{vb.dtype}")
            continue
        if torch.equal(va, vb):
            out.append(f"    {k}: byte-identical  {tuple(va.shape)} {va.dtype}")
        else:
            d = (va != vb)
            n, tot = int(d.sum()), va.numel()
            extra = ""
            if va.is_floating_point():
                extra = f"  max_abs={float((va - vb).abs().max()):.3e}"
            out.append(f"    {k}: DIFFERS  {n}/{tot} ({100.0 * n / tot:.4f}%){extra}")
    if torch.equal(wa, wb):
        out.append(f"    W_hat: byte-identical  {tuple(wa.shape)}")
    else:
        d = (wa != wb)
        out.append(f"    W_hat: DIFFERS  {int(d.sum())}/{wa.numel()} "
                   f"({100.0 * int(d.sum()) / wa.numel():.4f}%)  "
                   f"max_abs={float((wa - wb).abs().max()):.3e}")
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    try:
        import triton
        tv = triton.__version__
    except Exception:                                      # noqa: BLE001
        tv = "n/a"
    print(f"torch={torch.__version__} triton={tv} gpu={torch.cuda.get_device_name(0)}")

    all_identical = True
    for variant, K, m, n in CASES:
        print(f"\n=== {variant} K={K} {m}x{n} ===")
        W, H = make_layer(m, n, seed=1000 * K + m)
        arts_f, what_f, eng_f = quantize_with(True, W, H, variant, K)
        arts_r, what_r, eng_r = quantize_with(False, W, H, variant, K)
        print(f"  MECHANISM fused_step_on: fused-arm={eng_f} fallback-arm={eng_r}")
        if not (eng_f and not eng_r):
            print("  !! ARMS NOT DISTINCT -- any 'identical' below is meaningless")
            all_identical = False
            continue
        lines = compare(arts_f, arts_r, what_f, what_r)
        print("\n".join(lines))
        if any("DIFFERS" in ln or "SHAPE/DTYPE" in ln for ln in lines):
            all_identical = False

    print("\n" + ("ALL CASES BYTE-IDENTICAL" if all_identical
                  else "AT LEAST ONE CASE DIFFERS"))
    return 0 if all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
