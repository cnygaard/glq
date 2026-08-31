"""E2E CPU smoke: load a published trellis checkpoint with HF transformers on CPU,
generate, and ASSERT the fused CPU path engaged (no dense weight cache materialized).

Usage:
  python benchmarks/_bench_cpu_e2e_smoke.py [repo_id] [--tokens 32]

Defaults to the 360M 6bpw (small download, exercises stacked RVQ K=4+K=2). The
mechanism assertion is the point: a green generate() through the DENSE fallback would
prove nothing about the fused path (assert-the-mechanism rule).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?",
                    default="xv0y5ncu/SmolLM2-360M-Instruct-GLQ-trellis-3inst-6bpw")
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()

    # MUST precede from_pretrained: this registers the "glq" quantizer with transformers;
    # without it the model loads as plain nn.Linear with newly-initialized weights and
    # generates noise (run_model.py carries the same note).
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from glq.quantized_linear import E8RHTLinear

    tok = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(args.repo, torch_dtype=torch.float32)
    model.eval()

    ids = tok("The three primary colors are", return_tensors="pt").input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False)
    dt = time.perf_counter() - t0
    text = tok.decode(out[0], skip_special_tokens=True)

    layers = [m for m in model.modules()
              if isinstance(m, E8RHTLinear) and getattr(m, "_is_trellis", False)]
    engaged = sum(1 for m in layers if m._trellis_op_cpu is True)
    dense = sum(1 for m in layers if m._trellis_W_rht is not None)
    print(f"OUT: {text!r}")
    print(f"trellis layers={len(layers)} fused-engaged={engaged} dense-cached={dense}")
    print(f"RESULT tok/s={args.tokens / dt:.2f} ({dt:.1f}s for {args.tokens} tokens)")
    assert layers, "no trellis layers found — wrong checkpoint?"
    assert engaged == len(layers), f"fused CPU path engaged on only {engaged}/{len(layers)}"
    assert dense == 0, f"{dense} layers materialized the dense cache"
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
