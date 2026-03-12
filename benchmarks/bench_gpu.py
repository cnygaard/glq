"""
GPU benchmark: GLQ vs AWQ vs bf16 on multiple models.

Measures perplexity (WikiText-2), tokens/sec, GPU memory, and disk size.

Usage:
    python bench_gpu.py                                    # all models, all methods
    python bench_gpu.py --models SmolLM2-360M              # single model
    python bench_gpu.py --methods bf16 glq                 # skip AWQ
    python bench_gpu.py --bpw 2 4                          # specific bit widths
    python bench_gpu.py --skip-quantize                    # reuse cached models

Requires: pip install 'glq[quantize,cuda]' autoawq
"""

import argparse
import gc
import json
import math
import os
import time

import torch
import torch.nn.functional as F

MODEL_REGISTRY = {
    "SmolLM2-360M": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M",
        "awq_group_size": 32,  # hidden_size=960, must divide evenly
    },
    "Llama-3.2-1B": {
        "hf_id": "meta-llama/Llama-3.2-1B",
        "awq_group_size": 128,
    },
    "SmolLM3-3B": {
        "hf_id": "HuggingFaceTB/SmolLM3-3B-Base",
        "awq_group_size": 128,
    },
}


# ---- Utilities ----

def log(msg=""):
    print(msg, flush=True)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_calibration_texts(n_samples=128):
    """WikiText-2 train texts for AWQ calibration."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [t for t in ds["text"] if len(t.strip()) > 100][:n_samples]


def measure_disk_size(model_dir):
    """Sum of weight files in a model directory (MB)."""
    total = 0
    for f in os.listdir(model_dir):
        if f.endswith((".safetensors", ".bin", ".pt")):
            total += os.path.getsize(os.path.join(model_dir, f))
    return total / 1e6


# ---- Quantization ----

def quantize_glq(hf_id, output_dir, bpw, n_calib=128):
    """Quantize a model with GLQ and save to output_dir."""
    from glq.quantize_model import quantize
    log(f"  Quantizing {hf_id} with GLQ {bpw}bpw ...")
    t0 = time.perf_counter()
    quantize(
        model_name=hf_id,
        output_dir=output_dir,
        bpw=bpw,
        nsamples=n_calib,
        seqlen=2048,
        device="cuda",
    )
    dt = time.perf_counter() - t0
    log(f"  Done in {dt:.0f}s")
    cleanup()
    return dt


def quantize_awq(hf_id, output_dir, group_size=128, n_calib=128):
    """Quantize a model with AWQ 4-bit and save to output_dir."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    log(f"  Quantizing {hf_id} with AWQ 4-bit (gs={group_size}) ...")
    t0 = time.perf_counter()

    model = AutoAWQForCausalLM.from_pretrained(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    calib_data = get_calibration_texts(n_calib)
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    dt = time.perf_counter() - t0
    log(f"  Done in {dt:.0f}s")
    del model
    cleanup()
    return dt


# ---- Model Loading ----

def load_bf16(hf_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    return model, tokenizer


def load_glq(model_dir):
    import glq.hf_integration  # noqa: F401 - registers GLQ quantizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_awq(model_dir):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    model = AutoAWQForCausalLM.from_quantized(model_dir, fuse_layers=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


# ---- Measurements ----

@torch.no_grad()
def measure_perplexity(model, tokenizer, device="cuda"):
    """WikiText-2 test-set perplexity."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    seqlen = 2048
    n_chunks = input_ids.shape[1] // seqlen
    total_nll = 0.0
    total_tokens = 0

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        out = model(chunk, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

        if (i + 1) % 20 == 0 or i + 1 == n_chunks:
            ppl_so_far = math.exp(total_nll / total_tokens)
            log(f"    [{i+1}/{n_chunks}] ppl={ppl_so_far:.2f}")

    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def measure_tok_per_sec(model, tokenizer, device="cuda", n_tokens=128, n_runs=3):
    """Measure autoregressive generation speed (tokens/sec)."""
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_gen = output.shape[1] - inputs.input_ids.shape[1]
        times.append(n_gen / dt)

    times.sort()
    return times[len(times) // 2]  # median


def measure_gpu_memory(model, tokenizer, device="cuda"):
    """GPU memory: loaded model + peak during forward."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    mem_loaded = torch.cuda.memory_allocated() / 1e6

    inputs = tokenizer("Hello world, this is a test.", return_tensors="pt").to(device)
    with torch.no_grad():
        model(inputs.input_ids, use_cache=False)
    torch.cuda.synchronize()

    mem_peak = torch.cuda.max_memory_allocated() / 1e6
    return mem_loaded, mem_peak


# ---- Single run ----

def run_single(model_name, method, bpw, output_base, n_calib, skip_quantize):
    """Quantize (if needed), load, measure all metrics for one configuration."""
    info = MODEL_REGISTRY[model_name]
    hf_id = info["hf_id"]

    result = {
        "model": model_name,
        "method": method,
        "bpw": bpw,
        "hf_id": hf_id,
    }

    label = method + (f" {bpw}bpw" if method != "bf16" else "")
    log(f"\n  --- {model_name} / {label} ---")

    # Phase 1: Quantize (if needed)
    if method == "glq":
        model_dir = os.path.join(output_base, f"{model_name}-glq-{bpw}bpw")
        if not skip_quantize or not os.path.exists(model_dir):
            quantize_glq(hf_id, model_dir, bpw, n_calib)
        else:
            log(f"  Reusing {model_dir}")
        result["disk_mb"] = round(measure_disk_size(model_dir), 1)

    elif method == "awq":
        model_dir = os.path.join(output_base, f"{model_name}-awq-4bpw")
        if not skip_quantize or not os.path.exists(model_dir):
            quantize_awq(hf_id, model_dir, info["awq_group_size"], n_calib)
        else:
            log(f"  Reusing {model_dir}")
        result["disk_mb"] = round(measure_disk_size(model_dir), 1)

    else:  # bf16
        model_dir = None
        result["disk_mb"] = 0

    # Phase 2: Load
    log(f"  Loading ...")
    cleanup()
    torch.cuda.reset_peak_memory_stats()

    try:
        if method == "bf16":
            model, tokenizer = load_bf16(hf_id)
        elif method == "glq":
            model, tokenizer = load_glq(model_dir)
        elif method == "awq":
            model, tokenizer = load_awq(model_dir)
    except Exception as e:
        log(f"  FAILED to load: {e}")
        result["error"] = str(e)
        return result

    # Phase 3: Measure
    log(f"  Measuring GPU memory ...")
    mem_loaded, mem_peak = measure_gpu_memory(model, tokenizer)
    result["gpu_loaded_mb"] = round(mem_loaded, 1)
    result["gpu_peak_mb"] = round(mem_peak, 1)
    log(f"    Loaded: {mem_loaded:.0f} MB, Peak: {mem_peak:.0f} MB")

    log(f"  Measuring tokens/sec ...")
    tok_s = measure_tok_per_sec(model, tokenizer)
    result["tok_per_sec"] = round(tok_s, 1)
    log(f"    {tok_s:.1f} tok/s")

    log(f"  Measuring perplexity ...")
    t0 = time.perf_counter()
    ppl = measure_perplexity(model, tokenizer)
    dt = time.perf_counter() - t0
    result["ppl"] = round(ppl, 2)
    log(f"    PPL: {ppl:.2f} ({dt:.0f}s)")

    del model
    cleanup()
    return result


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="GLQ vs AWQ vs bf16 GPU benchmark")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--methods", nargs="+", default=["bf16", "glq", "awq"],
                        choices=["bf16", "glq", "awq"])
    parser.add_argument("--bpw", nargs="+", type=int, default=[2, 3, 4],
                        choices=[2, 3, 4])
    parser.add_argument("--output-dir", default="/tmp/glq-bench")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--json", default="bench_results.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    log("=" * 70)
    log("GPU Benchmark: GLQ vs AWQ vs bf16")
    log(f"GPU: {gpu_name}")
    log(f"Models: {', '.join(args.models)}")
    log(f"Methods: {', '.join(args.methods)}")
    log(f"BPW: {args.bpw}")
    log("=" * 70)

    all_results = []

    for model_name in args.models:
        log(f"\n{'='*70}")
        log(f"  MODEL: {model_name} ({MODEL_REGISTRY[model_name]['hf_id']})")
        log("=" * 70)

        model_results = []

        # bf16 baseline
        if "bf16" in args.methods:
            r = run_single(model_name, "bf16", 16, args.output_dir, args.n_calib,
                           args.skip_quantize)
            model_results.append(r)
            all_results.append(r)

        # GLQ at each bpw
        if "glq" in args.methods:
            for bpw in args.bpw:
                r = run_single(model_name, "glq", bpw, args.output_dir, args.n_calib,
                               args.skip_quantize)
                model_results.append(r)
                all_results.append(r)

        # AWQ 4-bit
        if "awq" in args.methods:
            r = run_single(model_name, "awq", 4, args.output_dir, args.n_calib,
                           args.skip_quantize)
            model_results.append(r)
            all_results.append(r)

        # Print table for this model
        log(f"\n  {model_name} Results:")
        log(f"  {'Method':<15} {'BPW':>5} {'PPL':>8} {'tok/s':>8} "
            f"{'GPU MB':>8} {'Peak MB':>8} {'Disk MB':>8}")
        log(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in model_results:
            if "error" in r:
                log(f"  {r['method']:<15} {r['bpw']:>5} {'FAILED':>8}")
                continue
            if r["method"] == "glq":
                method_label = f"GLQ {r['bpw']}bpw"
            elif r["method"] == "awq":
                method_label = "AWQ 4bpw"
            else:
                method_label = "bf16"
            ppl = f"{r['ppl']:.2f}" if "ppl" in r else "-"
            tok = f"{r['tok_per_sec']:.1f}" if "tok_per_sec" in r else "-"
            gpu = f"{r['gpu_loaded_mb']:.0f}" if "gpu_loaded_mb" in r else "-"
            peak = f"{r['gpu_peak_mb']:.0f}" if "gpu_peak_mb" in r else "-"
            disk = f"{r['disk_mb']:.0f}" if r.get("disk_mb") else "-"
            bpw_str = f"{r['bpw']}" if r["bpw"] != 16 else "16"
            log(f"  {method_label:<15} {bpw_str:>5} {ppl:>8} {tok:>8} "
                f"{gpu:>8} {peak:>8} {disk:>8}")

    # Save JSON
    json_path = os.path.join(args.output_dir, args.json)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
