"""Measure real disk size and GPU memory for SmolLM2-360M at bf16 and GLQ 4bpw.

Usage (GPU machine):
    python benchmarks/measure_real_sizes.py
"""

import gc
import os
import time

import torch


def measure_bf16():
    """Load SmolLM2-360M at bf16, measure GPU memory and disk size."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("1. BF16 BASELINE")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    mem_before = torch.cuda.memory_allocated()

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    mem_after = torch.cuda.memory_allocated()
    mem_model = mem_after - mem_before
    mem_peak = torch.cuda.max_memory_allocated()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print(f"  Parameters: {total_params:,}")
    print(f"  Parameter bytes: {total_bytes / 1e6:.1f} MB")
    print(f"  GPU memory (model): {mem_model / 1e6:.1f} MB")
    print(f"  GPU memory (peak): {mem_peak / 1e6:.1f} MB")

    # Check HF cache for disk size
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download("HuggingFaceTB/SmolLM2-360M")
    safetensors_path = os.path.join(cache_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        disk_size = os.path.getsize(safetensors_path)
        print(f"  Disk (model.safetensors): {disk_size / 1e6:.1f} MB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return mem_model, total_bytes


def measure_glq_4bpw():
    """Quantize SmolLM2-360M at 4bpw, measure disk and GPU memory."""
    import sys
    sys.path.insert(0, os.path.expanduser("~/golay-leech-quant"))

    from glq.quantize_model import quantize

    print("\n" + "=" * 60)
    print("2. GLQ 4BPW QUANTIZATION")
    print("=" * 60)

    output_dir = "/tmp/smollm2-glq-4bpw"

    t0 = time.perf_counter()
    quantize(
        model_name="HuggingFaceTB/SmolLM2-360M",
        output_dir=output_dir,
        bpw=4,
        nsamples=16,
        seqlen=2048,
        device="cuda",
    )
    quant_time = time.perf_counter() - t0
    print(f"\n  Quantization time: {quant_time:.0f}s")

    # Measure disk sizes
    print(f"\n  Disk sizes:")
    total_disk = 0
    for f in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            total_disk += size
            if size > 1000:
                print(f"    {f}: {size / 1e6:.1f} MB")
    print(f"    TOTAL: {total_disk / 1e6:.1f} MB")

    # Measure safetensors specifically
    st_path = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(st_path):
        print(f"  model.safetensors: {os.path.getsize(st_path) / 1e6:.1f} MB")

    gc.collect()
    torch.cuda.empty_cache()

    return output_dir, total_disk


def measure_glq_gpu_memory(output_dir):
    """Load GLQ model, measure GPU memory with and without fused kernel."""
    print("\n" + "=" * 60)
    print("3. GLQ 4BPW GPU MEMORY")
    print("=" * 60)

    import glq.hf_integration  # noqa: F401 — registers quantizer
    from transformers import AutoModelForCausalLM

    # --- Load with materialized weights (old path) ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    mem_before = torch.cuda.memory_allocated()

    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        device_map="cuda",
    )

    mem_loaded = torch.cuda.memory_allocated() - mem_before
    print(f"\n  After loading (indices + non-quant on GPU): {mem_loaded / 1e6:.1f} MB")

    # Run a forward pass to see peak memory (dequant happens here)
    torch.cuda.reset_peak_memory_stats()
    mem_before_fwd = torch.cuda.memory_allocated()

    x = torch.randint(0, 1000, (1, 64), device="cuda")
    with torch.no_grad():
        out = model(x)

    mem_fwd_peak = torch.cuda.max_memory_allocated() - mem_before_fwd
    mem_total_peak = torch.cuda.max_memory_allocated()
    print(f"  Forward pass peak (above model): {mem_fwd_peak / 1e6:.1f} MB")
    print(f"  Total peak GPU memory: {mem_total_peak / 1e6:.1f} MB")

    # Count actual bytes on GPU
    total_gpu_bytes = 0
    for name, buf in model.named_buffers():
        if buf is not None and buf.device.type == "cuda":
            total_gpu_bytes += buf.numel() * buf.element_size()
    for name, param in model.named_parameters():
        if param.device.type == "cuda":
            total_gpu_bytes += param.numel() * param.element_size()
    print(f"  Actual tensor bytes on GPU: {total_gpu_bytes / 1e6:.1f} MB")

    del model, out
    gc.collect()
    torch.cuda.empty_cache()

    return mem_loaded, mem_total_peak


def main():
    print("SmolLM2-360M Real Size Measurements")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    bf16_mem, bf16_bytes = measure_bf16()
    output_dir, glq_disk = measure_glq_4bpw()
    glq_loaded, glq_peak = measure_glq_gpu_memory(output_dir)

    print("\n" + "=" * 60)
    print("SUMMARY: SmolLM2-360M")
    print("=" * 60)
    print(f"  {'':30s} {'bf16':>10s} {'GLQ 4bpw':>10s} {'Ratio':>8s}")
    print(f"  {'-'*58}")
    print(f"  {'Param bytes':30s} {bf16_bytes/1e6:9.1f}M")
    print(f"  {'Disk (total)':30s} {bf16_bytes/1e6:9.1f}M {glq_disk/1e6:9.1f}M {bf16_bytes/glq_disk:7.1f}x")
    print(f"  {'GPU after load':30s} {bf16_mem/1e6:9.1f}M {glq_loaded/1e6:9.1f}M {bf16_mem/glq_loaded:7.1f}x")
    print(f"  {'GPU peak (with forward)':30s} {'':>10s} {glq_peak/1e6:9.1f}M")


if __name__ == "__main__":
    main()
