"""Profiling harness for nsys and ncu.

Usage:
  # System-wide timeline (nsys):
  nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
    --output=/opt/dlami/nvme/glq_nsys \
    python benchmarks/profile_nsys.py

  # Per-kernel metrics (ncu):
  ncu --set full --kernel-name "glq_matvec_splitk_kernel" --launch-count 5 \
    --output=/opt/dlami/nvme/glq_ncu_matvec \
    python benchmarks/profile_nsys.py --b1-only

  # Text summary from nsys:
  nsys stats /opt/dlami/nvme/glq_nsys.nsys-rep
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx

import glq.hf_integration
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw")
    parser.add_argument("--b1-only", action="store_true", help="Profile B=1 decode only")
    parser.add_argument("--b16-only", action="store_true", help="Profile B=16 prefill only")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.float16)
    model.forward = torch.no_grad()(model.forward)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create inputs
    input_b1 = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    input_b16 = tokenizer(
        ["The capital of France is"] * 16, return_tensors="pt",
        padding=True).to("cuda")

    do_b1 = not args.b16_only
    do_b16 = not args.b1_only

    # Warmup (outside profiler region)
    print(f"Warming up ({args.warmup} iters)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            if do_b1:
                model(**input_b1)
            if do_b16:
                model(**input_b16)
    torch.cuda.synchronize()

    # Profiled region
    print(f"Profiling ({args.iters} iters)...")
    torch.cuda.cudart().cudaProfilerStart()

    with torch.no_grad():
        if do_b1:
            for i in range(args.iters):
                nvtx.range_push(f"forward_b1_{i}")
                model(**input_b1)
                nvtx.range_pop()

        if do_b16:
            for i in range(args.iters):
                nvtx.range_push(f"forward_b16_{i}")
                model(**input_b16)
                nvtx.range_pop()

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Done. Use nsys stats or ncu --import to analyze.")


if __name__ == "__main__":
    main()
