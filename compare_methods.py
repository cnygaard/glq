"""
Compare GLQ vs AWQ vs GPTQ quantization methods.
Measures WikiText-2 perplexity. Reports effective bpw from model size.

Usage (GPU):
    PYTHONPATH=GPTQModel:. python -u compare_methods.py
"""

import gc
import json
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GPTQModel"))

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda"
N_CALIB = 128
SEQLEN = 2048


def get_calibration_data(n_samples=128):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_samples]
    return texts


def get_tokenized_calibration(tokenizer, n_samples=128, seqlen=2048):
    """Get tokenized calibration data for AWQ."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100]
    # Concatenate and chunk into seqlen pieces
    full_text = "\n\n".join(texts)
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    samples = []
    for i in range(min(n_samples, len(input_ids) // seqlen)):
        samples.append(input_ids[i * seqlen:(i + 1) * seqlen])
    return samples


def measure_perplexity_hf(model, tokenizer, device="cuda"):
    """Measure perplexity using a raw HF model (for AWQ)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    seqlen = 2048
    n_chunks = input_ids.shape[1] // seqlen
    nlls = []

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        with torch.no_grad():
            out = model(chunk)
            logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        nlls.append(loss.item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def measure_perplexity_gptq(model, tokenizer, device="cuda"):
    """Measure perplexity using GPTQModel wrapper (model.model is the HF model)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    seqlen = 2048
    n_chunks = input_ids.shape[1] // seqlen
    nlls = []

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        with torch.no_grad():
            out = model.model(chunk)
            logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        nlls.append(loss.item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def count_model_params(model_id):
    """Count total linear weight parameters in the model."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    total = 0
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() == 2:
            total += p.numel()
    del model
    gc.collect()
    return total


def get_model_file_size(save_dir):
    """Get total size of model weight files in bytes."""
    total = 0
    for f in os.listdir(save_dir):
        if f.endswith((".safetensors", ".bin", ".pt")):
            total += os.path.getsize(os.path.join(save_dir, f))
    return total


def quantize_gptq(bits, group_size, calib_data):
    """Quantize with GPTQ via GPTQModel."""
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig

    save_dir = os.path.join(tempfile.gettempdir(), f"smollm2-gptq-{bits}b-gs{group_size}")
    config = QuantizeConfig(bits=bits, group_size=group_size, quant_method="gptq")

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = GPTQModel.load(MODEL_ID, quantize_config=config, device=DEVICE)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"  Quantizing ({len(calib_data)} samples) ...", flush=True)
    t0 = time.perf_counter()
    model.quantize(calib_data)
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    print(f"  Saving to {save_dir} ...", flush=True)
    model.save(save_dir)
    model_size = get_model_file_size(save_dir)
    print(f"  Model size: {model_size / 1e6:.1f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload
    print(f"  Loading quantized model ...", flush=True)
    model = GPTQModel.load(save_dir, device=DEVICE)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.model.generate(**inputs, max_new_tokens=32, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generate: \"{text}\"", flush=True)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl = measure_perplexity_gptq(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": f"GPTQ-{bits}b",
        "nominal_bits": bits,
        "group_size": group_size,
        "model_size_bytes": model_size,
        "quant_time_s": round(quant_time, 1),
        "perplexity": round(ppl, 2),
        "sample_output": text,
    }


def quantize_glq(calib_data):
    """Quantize with GLQ (E8 Shell + RHT, 2bpw) via GPTQModel."""
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig

    save_dir = os.path.join(tempfile.gettempdir(), "smollm2-glq-2b")
    config = QuantizeConfig(bits=2, group_size=-1, quant_method="glq")

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = GPTQModel.load(MODEL_ID, quantize_config=config, device=DEVICE)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"  Quantizing ({len(calib_data)} samples) ...", flush=True)
    t0 = time.perf_counter()
    model.quantize(calib_data)
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    print(f"  Saving to {save_dir} ...", flush=True)
    model.save(save_dir)
    model_size = get_model_file_size(save_dir)
    print(f"  Model size: {model_size / 1e6:.1f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload
    print(f"  Loading quantized model ...", flush=True)
    model = GPTQModel.load(save_dir, device=DEVICE)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.model.generate(**inputs, max_new_tokens=32, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generate: \"{text}\"", flush=True)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl = measure_perplexity_gptq(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "GLQ-2b",
        "nominal_bits": 2,
        "group_size": -1,
        "model_size_bytes": model_size,
        "quant_time_s": round(quant_time, 1),
        "perplexity": round(ppl, 2),
        "sample_output": text,
    }


def quantize_awq(bits, group_size, calib_data_texts):
    """Quantize with AWQ via autoawq."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    save_dir = os.path.join(tempfile.gettempdir(), f"smollm2-awq-{bits}b-gs{group_size}")

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = AutoAWQForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    quant_config = {"zero_point": True, "q_group_size": group_size, "w_bit": bits, "version": "GEMM"}

    print(f"  Quantizing ({len(calib_data_texts)} samples) ...", flush=True)
    t0 = time.perf_counter()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data_texts)
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    print(f"  Saving to {save_dir} ...", flush=True)
    model.save_quantized(save_dir)
    tokenizer.save_pretrained(save_dir)
    model_size = get_model_file_size(save_dir)
    print(f"  Model size: {model_size / 1e6:.1f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload
    print(f"  Loading quantized model ...", flush=True)
    model = AutoAWQForCausalLM.from_quantized(save_dir, fuse_layers=False)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generate: \"{text}\"", flush=True)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": f"AWQ-{bits}b",
        "nominal_bits": bits,
        "group_size": group_size,
        "model_size_bytes": model_size,
        "quant_time_s": round(quant_time, 1),
        "perplexity": round(ppl, 2),
        "sample_output": text,
    }


def main():
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Calibration: {N_CALIB} samples, seqlen={SEQLEN}")

    calib_data = get_calibration_data(N_CALIB)

    # Count parameters for effective bpw calculation
    print(f"\nCounting model parameters ...", flush=True)
    n_params = count_model_params(MODEL_ID)
    print(f"  Linear weight params: {n_params:,}", flush=True)

    # SmolLM2-360M has hidden_size=960 — not divisible by 128.
    # Use group_size=32 (960/32=30) for GPTQ/AWQ compatibility.
    configs = [
        ("glq",  2, -1),
        ("awq",  4, 32),
        ("gptq", 4, 32),
        ("gptq", 3, 32),
    ]

    results = []
    for method, bits, gs in configs:
        print(f"\n{'='*70}", flush=True)
        print(f"  {method.upper()} {bits}-bit  (group_size={gs})", flush=True)
        print(f"{'='*70}", flush=True)
        try:
            if method == "glq":
                results.append(quantize_glq(calib_data))
            elif method == "awq":
                results.append(quantize_awq(bits, gs, calib_data))
            elif method == "gptq":
                results.append(quantize_gptq(bits, gs, calib_data))
        except Exception as e:
            print(f"\n  FAILED: {method} {bits}b gs={gs}: {e}", flush=True)
            traceback.print_exc()
            results.append({
                "method": f"{method.upper()}-{bits}b",
                "nominal_bits": bits, "group_size": gs,
                "model_size_bytes": 0, "quant_time_s": 0,
                "perplexity": float("inf"), "sample_output": f"FAILED: {e}",
            })
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON: SmolLM2-360M on WikiText-2")
    print(f"  (Linear weight params: {n_params:,})")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Size(MB)':>9} {'Eff.BPW':>8} {'PPL':>10} {'Time(s)':>8}")
    print(f"{'-'*15} {'-'*9} {'-'*8} {'-'*10} {'-'*8}")

    for r in results:
        size_mb = r["model_size_bytes"] / 1e6
        # Effective bpw = (model_file_size_bits) / n_params
        eff_bpw = (r["model_size_bytes"] * 8) / n_params if r["model_size_bytes"] > 0 else 0
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] < 1e6 else "FAILED"
        print(f"{r['method']:<15} {size_mb:>9.1f} {eff_bpw:>8.2f} {ppl_str:>10} {r['quant_time_s']:>8.1f}")

    # Save results
    for r in results:
        r["n_params"] = n_params
        r["eff_bpw"] = round((r["model_size_bytes"] * 8) / n_params, 2) if r["model_size_bytes"] > 0 else 0
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to comparison_results.json")


if __name__ == "__main__":
    main()
