"""Batch throughput benchmark: GLQ vs dense bf16 across batch sizes.

Measures kernel-level, layer-level, and end-to-end model throughput
to find the batch size crossover where GLQ's memory savings translate
to competitive or superior throughput.

Usage:
    # Kernel + layer + component benchmarks (no model needed)
    python benchmarks/bench_batch_throughput.py --sections A B C

    # End-to-end model prefill (requires pre-quantized model)
    python benchmarks/bench_batch_throughput.py --sections D \
        --model-dir /opt/dlami/nvme/smollm3-3b-glq-2bpw \
        --model-id HuggingFaceTB/SmolLM3-3B-Base

    # All sections
    python benchmarks/bench_batch_throughput.py --sections A B C D \
        --model-dir /opt/dlami/nvme/smollm3-3b-glq-2bpw \
        --model-id HuggingFaceTB/SmolLM3-3B-Base
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Timing utilities ----

def cuda_median_us(fn, n_warmup=20, n_iter=200):
    """Time a function using CUDA events, return median in microseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    return times[len(times) // 2]


# ---- Section A: Kernel-level sweep ----

def bench_kernel_sweep(shapes, batch_sizes, codebook, device="cuda"):
    """Raw glq_dequant_matmul vs cuBLAS matmul at varying batch sizes."""
    from glq.inference_kernel import glq_dequant_matmul

    results = []
    K = codebook.codebook.shape[0]
    cb_half = codebook.codebook_half.to(device)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        Qidxs = torch.randint(0, K, (m_pad, n_pad // 8),
                               device=device, dtype=torch.int32).to(torch.int16)
        W_dense = torch.randn(M, N, device=device, dtype=torch.float16)
        Wscale = 0.01

        for B in batch_sizes:
            x = torch.randn(B, n_pad, device=device, dtype=torch.float16)
            x_dense = torch.randn(B, N, device=device, dtype=torch.float16)

            glq_us = cuda_median_us(
                lambda: glq_dequant_matmul(x, Qidxs, cb_half, Wscale))
            dense_us = cuda_median_us(
                lambda: x_dense @ W_dense.T)

            ratio = glq_us / dense_us
            results.append({
                "shape": f"{M}x{N}", "B": B,
                "glq_us": round(glq_us, 1),
                "dense_us": round(dense_us, 1),
                "ratio": round(ratio, 2),
            })
            print(f"  {M}x{N} B={B:3d}  GLQ={glq_us:8.1f}us  "
                  f"dense={dense_us:8.1f}us  ratio={ratio:.2f}x")

    return results


# ---- Section B: Layer-level sweep ----

def bench_layer_sweep(shapes, batch_sizes, codebook, device="cuda"):
    """Full E8RHTLinear.forward() vs nn.Linear at varying batch sizes."""
    from glq.quantized_linear import E8RHTLinear

    results = []
    K = codebook.codebook.shape[0]

    for M, N in shapes:
        # GLQ layer
        glq_layer = E8RHTLinear(N, M).to(device)
        glq_layer.Qidxs = torch.randint(
            0, K, glq_layer.Qidxs.shape,
            device=device, dtype=torch.int32).to(torch.int16)
        glq_layer.set_codebook(codebook)

        # Dense layer
        dense_layer = nn.Linear(N, M, bias=False).to(device).half()

        for B in batch_sizes:
            x_glq = torch.randn(B, N, device=device, dtype=torch.float16)
            x_dense = torch.randn(B, N, device=device, dtype=torch.float16)

            with torch.no_grad():
                glq_us = cuda_median_us(lambda: glq_layer(x_glq))
                dense_us = cuda_median_us(lambda: dense_layer(x_dense))

            ratio = glq_us / dense_us
            results.append({
                "shape": f"{M}x{N}", "B": B,
                "glq_us": round(glq_us, 1),
                "dense_us": round(dense_us, 1),
                "ratio": round(ratio, 2),
            })
            print(f"  {M}x{N} B={B:3d}  GLQ={glq_us:8.1f}us  "
                  f"dense={dense_us:8.1f}us  ratio={ratio:.2f}x")

        del glq_layer, dense_layer
        torch.cuda.empty_cache()

    return results


# ---- Section C: Component breakdown ----

def bench_component_breakdown(shapes, batch_sizes, codebook, device="cuda"):
    """Per-phase timing: input_rht, dequant_matmul, output_rht."""
    from glq.hadamard import fast_hadamard_transform
    from glq.inference_kernel import glq_dequant_matmul

    results = []
    K = codebook.codebook.shape[0]
    cb_half = codebook.codebook_half.to(device)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        Qidxs = torch.randint(0, K, (m_pad, n_pad // 8),
                               device=device, dtype=torch.int32).to(torch.int16)
        SU = torch.where(torch.rand(m_pad, device=device) > 0.5,
                         torch.ones(1, device=device, dtype=torch.float16),
                         -torch.ones(1, device=device, dtype=torch.float16))
        SV = torch.where(torch.rand(n_pad, device=device) > 0.5,
                         torch.ones(1, device=device, dtype=torch.float16),
                         -torch.ones(1, device=device, dtype=torch.float16))
        sv_f = SV.float()
        su_f = SU.float()
        Wscale = 0.01

        for B in batch_sizes:
            x = torch.randn(B, N, device=device, dtype=torch.float16)

            # Phase 1: Input RHT (cast + pad + SV + FHT)
            def input_rht():
                x_f = x.float()
                x_pad = F.pad(x_f, (0, n_pad - N))
                return fast_hadamard_transform(x_pad * sv_f.unsqueeze(0))

            input_us = cuda_median_us(input_rht, n_warmup=10, n_iter=100)

            # Cache x_rht for dequant phase
            x_rht = input_rht()

            # Phase 2: Dequant + matmul
            dequant_us = cuda_median_us(
                lambda: glq_dequant_matmul(x_rht, Qidxs, cb_half, Wscale),
                n_warmup=10, n_iter=100)

            # Cache y_rht for output phase
            y_rht = glq_dequant_matmul(x_rht, Qidxs, cb_half, Wscale)

            # Phase 3: Output RHT (FHT + SU + unpad)
            def output_rht():
                y = fast_hadamard_transform(y_rht) * su_f.unsqueeze(0)
                return y[:, :M]

            output_us = cuda_median_us(output_rht, n_warmup=10, n_iter=100)

            total = input_us + dequant_us + output_us
            results.append({
                "shape": f"{M}x{N}", "B": B,
                "input_rht_us": round(input_us, 1),
                "dequant_us": round(dequant_us, 1),
                "output_rht_us": round(output_us, 1),
                "total_us": round(total, 1),
                "pct_input": round(100 * input_us / total, 1),
                "pct_dequant": round(100 * dequant_us / total, 1),
                "pct_output": round(100 * output_us / total, 1),
            })
            print(f"  {M}x{N} B={B:3d}  "
                  f"in={input_us:7.1f}us ({100*input_us/total:4.1f}%)  "
                  f"dq={dequant_us:7.1f}us ({100*dequant_us/total:4.1f}%)  "
                  f"out={output_us:7.1f}us ({100*output_us/total:4.1f}%)  "
                  f"total={total:7.1f}us")

    return results


# ---- Section D: End-to-end model prefill throughput ----

def bench_model_prefill(model_dir_glq, model_id_bf16, batch_sizes,
                        seqlen=128, n_runs=5, device="cuda"):
    """Full model prefill: GLQ vs bf16 at varying batch sizes."""
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = []

    # Load bf16 baseline
    print("  Loading bf16 baseline ...")
    AutoTokenizer.from_pretrained(model_id_bf16)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_id_bf16, torch_dtype=torch.bfloat16, device_map=device)
    model_bf16.eval()

    # Load GLQ model
    print("  Loading GLQ model ...")
    model_glq = AutoModelForCausalLM.from_pretrained(
        model_dir_glq, device_map=device)
    model_glq.eval()

    # Create input tokens
    input_ids_single = torch.randint(100, 10000, (1, seqlen), device=device)

    for B in batch_sizes:
        input_ids = input_ids_single.expand(B, -1).contiguous()

        # Measure bf16 prefill
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model_bf16(input_ids, use_cache=False)
            torch.cuda.synchronize()

            times_bf16 = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model_bf16(input_ids, use_cache=False)
                torch.cuda.synchronize()
                times_bf16.append(time.perf_counter() - t0)

        times_bf16.sort()
        bf16_s = times_bf16[len(times_bf16) // 2]
        bf16_toks = B * seqlen / bf16_s
        bf16_mem = torch.cuda.max_memory_allocated() / 1e6

        # Measure GLQ prefill
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model_glq(input_ids, use_cache=False)
            torch.cuda.synchronize()

            times_glq = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model_glq(input_ids, use_cache=False)
                torch.cuda.synchronize()
                times_glq.append(time.perf_counter() - t0)

        times_glq.sort()
        glq_s = times_glq[len(times_glq) // 2]
        glq_toks = B * seqlen / glq_s
        glq_mem = torch.cuda.max_memory_allocated() / 1e6

        ratio = glq_toks / bf16_toks
        results.append({
            "B": B,
            "glq_tok_s": round(glq_toks, 1),
            "bf16_tok_s": round(bf16_toks, 1),
            "speedup": round(ratio, 3),
            "glq_mem_mb": round(glq_mem, 0),
            "bf16_mem_mb": round(bf16_mem, 0),
        })
        print(f"  B={B:3d}  GLQ={glq_toks:8.1f} tok/s ({glq_mem:6.0f} MB)  "
              f"bf16={bf16_toks:8.1f} tok/s ({bf16_mem:6.0f} MB)  "
              f"speedup={ratio:.3f}x")

    del model_bf16, model_glq
    torch.cuda.empty_cache()
    return results


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="GLQ batch throughput benchmark")
    parser.add_argument("--sections", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C", "D"],
                        help="Benchmark sections to run")
    parser.add_argument("--model-dir",
                        help="Pre-quantized GLQ model dir (for section D)")
    parser.add_argument("--model-id",
                        help="bf16 HF model ID (for section D)")
    parser.add_argument("--bpw", type=int, default=2,
                        help="Bits per weight")
    parser.add_argument("--json", default="batch_throughput.json",
                        help="Output JSON file")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # 3B-class model dimensions
    shapes = [
        (3072, 3072),   # q_proj / o_proj
        (9216, 3072),   # gate_proj / up_proj
        (3072, 9216),   # down_proj
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64]

    all_results = {"gpu": gpu_name}

    # Build codebook for kernel/layer benchmarks
    if any(s in args.sections for s in ["A", "B", "C"]):
        from glq.codebook import E8ShellCodebook
        codebook = E8ShellCodebook.build(device="cuda", verbose=False)

    if "A" in args.sections:
        print("=" * 80)
        print("Section A: Kernel-level sweep (glq_dequant_matmul vs cuBLAS)")
        print("=" * 80)
        all_results["kernel_sweep"] = bench_kernel_sweep(
            shapes, batch_sizes, codebook)
        print()

    if "B" in args.sections:
        print("=" * 80)
        print("Section B: Layer-level sweep (E8RHTLinear vs nn.Linear)")
        print("=" * 80)
        all_results["layer_sweep"] = bench_layer_sweep(
            shapes, batch_sizes, codebook)
        print()

    if "C" in args.sections:
        print("=" * 80)
        print("Section C: Component breakdown (input_rht / dequant / output_rht)")
        print("=" * 80)
        breakdown_batches = [1, 8, 32, 64]
        all_results["component_breakdown"] = bench_component_breakdown(
            shapes, breakdown_batches, codebook)
        print()

    if "D" in args.sections:
        if not args.model_dir or not args.model_id:
            print("Section D requires --model-dir and --model-id")
            return
        print("=" * 80)
        print("Section D: End-to-end model prefill throughput")
        print("=" * 80)
        model_batches = [1, 4, 8, 16, 32, 64]
        all_results["model_prefill"] = bench_model_prefill(
            args.model_dir, args.model_id, model_batches)
        print()

    # Save results
    with open(args.json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.json}")

    # Print summary tables
    print()
    print("=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    if "kernel_sweep" in all_results:
        print("\nKernel-level: GLQ/dense ratio (< 1.0 = GLQ wins)")
        shape_set = sorted(set(r["shape"] for r in all_results["kernel_sweep"]))
        header = (f"{'Shape':<14s}"
                  + "".join(f"{'B='+str(b):>8s}" for b in batch_sizes))
        print(header)
        print("-" * len(header))
        for shape in shape_set:
            row = [r for r in all_results["kernel_sweep"]
                   if r["shape"] == shape]
            line = f"{shape:<14s}"
            for r in row:
                line += f"{r['ratio']:7.2f}x"
            print(line)

    if "layer_sweep" in all_results:
        print("\nLayer-level: GLQ/dense ratio (includes RHT overhead)")
        shape_set = sorted(set(r["shape"] for r in all_results["layer_sweep"]))
        header = (f"{'Shape':<14s}"
                  + "".join(f"{'B='+str(b):>8s}" for b in batch_sizes))
        print(header)
        print("-" * len(header))
        for shape in shape_set:
            row = [r for r in all_results["layer_sweep"]
                   if r["shape"] == shape]
            line = f"{shape:<14s}"
            for r in row:
                line += f"{r['ratio']:7.2f}x"
            print(line)

    if "model_prefill" in all_results:
        print("\nEnd-to-end prefill throughput:")
        print(f"{'B':>4s} {'GLQ tok/s':>12s} {'bf16 tok/s':>12s} "
              f"{'Speedup':>9s} {'GLQ MB':>8s} {'bf16 MB':>8s}")
        print("-" * 55)
        for r in all_results["model_prefill"]:
            print(f"{r['B']:4d} {r['glq_tok_s']:12.1f} "
                  f"{r['bf16_tok_s']:12.1f} "
                  f"{r['speedup']:8.3f}x "
                  f"{r['glq_mem_mb']:8.0f} {r['bf16_mem_mb']:8.0f}")


if __name__ == "__main__":
    main()
