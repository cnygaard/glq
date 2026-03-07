"""
Compare quantization methods: GLQ vs GPTQ vs AWQ vs QuIP+GPTQ vs SpinQuant+GPTQ.
Measures WikiText-2 perplexity and GPU memory at inference.

All methods use llm-compressor except standalone GPTQ (GPTQModel).

Usage (GPU):
    PYTHONPATH=llm-compressor/src:. python -u compare_methods.py
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

MODEL_ID = os.environ.get("GLQ_MODEL", "HuggingFaceTB/SmolLM2-360M")
DEVICE = "cuda"
N_CALIB = 128
SEQLEN = 2048


def load_model(model_id, device_map="cuda", dtype="auto"):
    """Load model, handling multimodal wrappers like Mistral3."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    except (ValueError, KeyError):
        # Multimodal models (e.g. Mistral3) need their generation class
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map=device_map)


def get_text_decoder_layer_class(model):
    """Get the text decoder layer class name, filtering out vision layers."""
    no_split = getattr(model, "_no_split_modules", set())
    text_layers = [c for c in no_split if "vision" not in c.lower() and "pixtral" not in c.lower()]
    return text_layers[0] if text_layers else None


def get_ignore_list(model):
    """Get ignore list that excludes vision components for multimodal models."""
    ignore = ["lm_head"]
    if hasattr(model.config, "text_config") or hasattr(model.config, "vision_config"):
        ignore.extend(["re:vision_tower", "re:multi_modal_projector"])
    return ignore


def try_generate(model, tokenizer, device="cuda"):
    """Try generating text; return empty string on failure."""
    try:
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generate: \"{text}\"", flush=True)
        return text
    except Exception as e:
        print(f"  Generate failed (non-fatal): {e}", flush=True)
        return f"(generate failed)"


def get_calibration_data(n_samples=128):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_samples]
    return texts


def measure_gpu_memory_mb():
    """Return peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / 1e6


def reset_gpu_memory_tracking():
    """Reset peak GPU memory tracking."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def measure_perplexity_hf(model, tokenizer, device="cuda"):
    """Measure perplexity and peak GPU memory using a HF model."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    seqlen = 2048
    n_chunks = input_ids.shape[1] // seqlen
    nlls = []

    reset_gpu_memory_tracking()

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
    gpu_mem = measure_gpu_memory_mb()
    return ppl, gpu_mem


def measure_perplexity_gptq(model, tokenizer, device="cuda"):
    """Measure perplexity and peak GPU memory using GPTQModel wrapper."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    seqlen = 2048
    n_chunks = input_ids.shape[1] // seqlen
    nlls = []

    reset_gpu_memory_tracking()

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
    gpu_mem = measure_gpu_memory_mb()
    return ppl, gpu_mem


def measure_baseline():
    """Measure bf16 baseline perplexity and GPU memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {MODEL_ID} (bf16) ...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    text = try_generate(model, tokenizer, DEVICE)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl, gpu_mem = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  GPU mem: {gpu_mem:.0f} MB  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "bf16",
        "nominal_bits": 16,
        "group_size": -1,
        "model_size_bytes": model_bytes,
        "quant_time_s": 0,
        "perplexity": round(ppl, 2),
        "gpu_mem_mb": round(gpu_mem, 0),
        "sample_output": text,
    }


def quantize_glq(bits, calib_data_texts):
    """Quantize with GLQ (E8 Shell + RHT + LDLQ) via llm-compressor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.glq import GLQModifier
    from datasets import Dataset

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ds = Dataset.from_dict({"text": calib_data_texts})
    seq_target = get_text_decoder_layer_class(model)
    seq_targets = [seq_target] if seq_target else None
    recipe = [GLQModifier(bits=bits, ignore=get_ignore_list(model), sequential_targets=seq_targets)]

    print(f"  Quantizing ({len(calib_data_texts)} samples) ...", flush=True)
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=SEQLEN,
        num_calibration_samples=len(calib_data_texts),
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    # Theoretical compressed size
    n_quant_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "weight" in name and p.dim() == 2 and "lm_head" not in name
    )
    n_other_bytes = sum(p.numel() * p.element_size() for name, p in model.named_parameters()
                        if not ("weight" in name and p.dim() == 2 and "lm_head" not in name))
    model_size = int(n_quant_params * bits / 8) + n_other_bytes
    print(f"  Theoretical compressed size: {model_size / 1e6:.1f} MB", flush=True)

    text = try_generate(model, tokenizer, DEVICE)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl, gpu_mem = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  GPU mem: {gpu_mem:.0f} MB  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": f"GLQ-{bits}b",
        "nominal_bits": bits,
        "group_size": -1,
        "model_size_bytes": model_size,
        "quant_time_s": round(quant_time, 1),
        "perplexity": round(ppl, 2),
        "gpu_mem_mb": round(gpu_mem, 0),
        "sample_output": text,
    }


def quantize_awq(bits, group_size, calib_data_texts):
    """Quantize with AWQ via llm-compressor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from datasets import Dataset
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    save_dir = os.path.join(tempfile.gettempdir(), f"smollm2-awq-{bits}b-gs{group_size}")

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ds = Dataset.from_dict({"text": calib_data_texts})

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=False,
            group_size=group_size, strategy="group",
        ),
    )
    ignore = get_ignore_list(model)
    recipe = [
        AWQModifier(
            ignore=ignore,
            config_groups={"group_0": scheme},
            targets=["Linear"],
            duo_scaling="both",
        ),
    ]

    print(f"  Quantizing ({len(calib_data_texts)} samples) ...", flush=True)
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=SEQLEN,
        num_calibration_samples=len(calib_data_texts),
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    print(f"  Saving to {save_dir} ...", flush=True)
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    model_size = get_model_file_size(save_dir)
    print(f"  Model size: {model_size / 1e6:.1f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Loading quantized model ...", flush=True)
    model = load_model(save_dir, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    text = try_generate(model, tokenizer, DEVICE)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl, gpu_mem = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  GPU mem: {gpu_mem:.0f} MB  ({ppl_time:.0f}s)", flush=True)

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
        "gpu_mem_mb": round(gpu_mem, 0),
        "sample_output": text,
    }


def quantize_gptq_llmc(bits, group_size, calib_data_texts):
    """Quantize with GPTQ via llm-compressor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from datasets import Dataset
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    save_dir = os.path.join(tempfile.gettempdir(), f"smollm2-gptq-llmc-{bits}b-gs{group_size}")

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ds = Dataset.from_dict({"text": calib_data_texts})

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True,
            group_size=group_size, strategy="group",
        ),
    )
    ignore = get_ignore_list(model)
    seq_target = get_text_decoder_layer_class(model)
    recipe = [
        GPTQModifier(
            ignore=ignore,
            sequential_targets=[seq_target] if seq_target else None,
            config_groups={"group_0": scheme},
            targets=["Linear"],
        ),
    ]

    print(f"  Quantizing ({len(calib_data_texts)} samples) ...", flush=True)
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=SEQLEN,
        num_calibration_samples=len(calib_data_texts),
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    print(f"  Saving to {save_dir} ...", flush=True)
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    model_size = get_model_file_size(save_dir)
    print(f"  Model size: {model_size / 1e6:.1f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Loading quantized model ...", flush=True)
    model = load_model(save_dir, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    text = try_generate(model, tokenizer, DEVICE)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl, gpu_mem = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  GPU mem: {gpu_mem:.0f} MB  ({ppl_time:.0f}s)", flush=True)

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
        "gpu_mem_mb": round(gpu_mem, 0),
        "sample_output": text,
    }


def quantize_quip_gptq(bits, group_size, calib_data_texts):
    """Quantize with QuIP transform + GPTQ via llm-compressor.

    NOTE: QuIP transforms (runtime hooks) are NOT persisted on save/reload,
    so we evaluate in-memory rather than saving/reloading the model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform import QuIPModifier
    from datasets import Dataset
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ds = Dataset.from_dict({"text": calib_data_texts})

    ignore = get_ignore_list(model)
    seq_target = get_text_decoder_layer_class(model)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True,
            group_size=group_size, strategy="group",
        ),
    )
    recipe = [
        QuIPModifier(
            rotations=["v", "u"],
            transform_type="random-hadamard",
            transform_block_size=32,
            ignore=ignore,
        ),
        GPTQModifier(
            ignore=ignore,
            sequential_targets=[seq_target] if seq_target else None,
            config_groups={"group_0": scheme},
            targets=["Linear"],
        ),
    ]

    print(f"  Quantizing ({len(calib_data_texts)} samples) ...", flush=True)
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=SEQLEN,
        num_calibration_samples=len(calib_data_texts),
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    # Theoretical compressed size (no save/reload — transforms not persisted)
    n_quant_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "weight" in name and p.dim() == 2 and "lm_head" not in name
    )
    n_other_bytes = sum(p.numel() * p.element_size() for name, p in model.named_parameters()
                        if not ("weight" in name and p.dim() == 2 and "lm_head" not in name))
    # group_size scales/zeros overhead: ~0.5 bits/param for group_size=32
    overhead_bits = 0.5 if group_size > 0 else 0
    model_size = int(n_quant_params * (bits + overhead_bits) / 8) + n_other_bytes
    print(f"  Theoretical compressed size: {model_size / 1e6:.1f} MB", flush=True)

    text = try_generate(model, tokenizer, DEVICE)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl, gpu_mem = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  GPU mem: {gpu_mem:.0f} MB  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": f"QuIP+GPTQ-{bits}b",
        "nominal_bits": bits,
        "group_size": group_size,
        "model_size_bytes": model_size,
        "quant_time_s": round(quant_time, 1),
        "perplexity": round(ppl, 2),
        "gpu_mem_mb": round(gpu_mem, 0),
        "sample_output": text,
    }


def quantize_spinquant_gptq(bits, group_size, calib_data_texts):
    """Quantize with SpinQuant transform + GPTQ via llm-compressor.

    NOTE: SpinQuant transforms (runtime hooks) are NOT persisted on save/reload,
    so we evaluate in-memory rather than saving/reloading the model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform import SpinQuantModifier
    from datasets import Dataset
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    print(f"  Loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ds = Dataset.from_dict({"text": calib_data_texts})

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True,
            group_size=group_size, strategy="group",
        ),
    )
    # SmolLM2-360M hidden_size=960, not power of 2 → need random-hadamard
    recipe = [
        SpinQuantModifier(
            rotations=["R1", "R2"],
            transform_type="random-hadamard",
            transform_block_size=32,
        ),
        GPTQModifier(
            ignore=["lm_head", "re:vision_tower", "re:multi_modal_projector"],
            config_groups={"group_0": scheme},
            targets=["Linear"],
        ),
    ]

    print(f"  Quantizing ({len(calib_data_texts)} samples) ...", flush=True)
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=SEQLEN,
        num_calibration_samples=len(calib_data_texts),
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s", flush=True)

    # Theoretical compressed size (no save/reload — transforms not persisted)
    n_quant_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "weight" in name and p.dim() == 2 and "lm_head" not in name
    )
    n_other_bytes = sum(p.numel() * p.element_size() for name, p in model.named_parameters()
                        if not ("weight" in name and p.dim() == 2 and "lm_head" not in name))
    overhead_bits = 0.5 if group_size > 0 else 0
    model_size = int(n_quant_params * (bits + overhead_bits) / 8) + n_other_bytes
    print(f"  Theoretical compressed size: {model_size / 1e6:.1f} MB", flush=True)

    text = try_generate(model, tokenizer, DEVICE)

    print(f"  Measuring perplexity ...", flush=True)
    t0 = time.perf_counter()
    ppl, gpu_mem = measure_perplexity_hf(model, tokenizer, DEVICE)
    ppl_time = time.perf_counter() - t0
    print(f"  Perplexity: {ppl:.2f}  GPU mem: {gpu_mem:.0f} MB  ({ppl_time:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": f"Spin+GPTQ-{bits}b",
        "nominal_bits": bits,
        "group_size": group_size,
        "model_size_bytes": model_size,
        "quant_time_s": round(quant_time, 1),
        "perplexity": round(ppl, 2),
        "gpu_mem_mb": round(gpu_mem, 0),
        "sample_output": text,
    }


def count_model_params(model_id):
    """Count total linear weight parameters in the model."""
    from transformers import AutoModelForCausalLM
    model = load_model(model_id, device_map="cpu")
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


def run_single_config(method, bits, gs, result_file):
    """Run a single config and write result JSON. Called via subprocess."""
    calib_data = get_calibration_data(N_CALIB)

    try:
        if method == "baseline":
            result = measure_baseline()
        elif method == "glq":
            result = quantize_glq(bits, calib_data)
        elif method == "awq":
            result = quantize_awq(bits, gs, calib_data)
        elif method == "gptq":
            result = quantize_gptq_llmc(bits, gs, calib_data)
        elif method == "quip_gptq":
            result = quantize_quip_gptq(bits, gs, calib_data)
        elif method == "spinquant_gptq":
            result = quantize_spinquant_gptq(bits, gs, calib_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        traceback.print_exc()
        result = {
            "method": f"{method.upper()}-{bits}b",
            "nominal_bits": bits, "group_size": gs,
            "model_size_bytes": 0, "quant_time_s": 0,
            "perplexity": float("inf"), "gpu_mem_mb": 0,
            "sample_output": f"FAILED: {e}",
        }

    with open(result_file, "w") as f:
        json.dump(result, f, default=str)


def main():
    import subprocess

    # Subprocess worker mode: --run method bits gs result_file
    if len(sys.argv) >= 6 and sys.argv[1] == "--run":
        method = sys.argv[2]
        bits = int(sys.argv[3])
        gs = int(sys.argv[4])
        result_file = sys.argv[5]
        run_single_config(method, bits, gs, result_file)
        return

    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Calibration: {N_CALIB} samples, seqlen={SEQLEN}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"\nCounting model parameters ...", flush=True)
    n_params = count_model_params(MODEL_ID)
    print(f"  Linear weight params: {n_params:,}", flush=True)

    # SmolLM2-360M hidden_size=960 — not divisible by 128.
    # Use group_size=32 (960/32=30) for GPTQ/AWQ.
    all_configs = [
        ("baseline",       16, -1),
        ("glq",             2, -1),
        ("glq",             3, -1),
        ("glq",             4, -1),
        ("gptq",            4, 32),
        ("gptq",            3, 32),
        ("awq",             4, 32),
        ("quip_gptq",       4, 32),
        ("quip_gptq",       3, 32),
        ("spinquant_gptq",  4, 32),
        ("spinquant_gptq",  3, 32),
    ]

    # CLI filter: pass method names to run only those (e.g. "quip_gptq spinquant_gptq")
    filter_methods = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    if filter_methods:
        configs = [(m, b, g) for m, b, g in all_configs if m in filter_methods]
        print(f"Filtering to methods: {filter_methods}")
    else:
        configs = all_configs

    # Run each config in a fresh subprocess for accurate GPU memory measurement.
    results = []
    for method, bits, gs in configs:
        print(f"\n{'='*70}", flush=True)
        print(f"  {method.upper()} {bits}-bit  (group_size={gs})", flush=True)
        print(f"{'='*70}", flush=True)

        result_file = os.path.join(tempfile.gettempdir(), f"glq_bench_{method}_{bits}b_gs{gs}.json")
        script_path = os.path.abspath(__file__)
        proc = subprocess.run(
            [sys.executable, "-u", script_path,
             "--run", method, str(bits), str(gs), result_file],
            timeout=3600,
        )

        if os.path.exists(result_file):
            with open(result_file) as f:
                r = json.load(f)
            results.append(r)
            os.unlink(result_file)
        else:
            print(f"  FAILED: no result file (exit code {proc.returncode})", flush=True)
            results.append({
                "method": f"{method.upper()}-{bits}b",
                "nominal_bits": bits, "group_size": gs,
                "model_size_bytes": 0, "quant_time_s": 0,
                "perplexity": float("inf"), "gpu_mem_mb": 0,
                "sample_output": "FAILED: subprocess error",
            })

    # Print comparison table
    print(f"\n\n{'='*80}")
    model_short = MODEL_ID.split("/")[-1]
    print(f"  COMPARISON: {model_short} on WikiText-2  (GPU: {torch.cuda.get_device_name()})")
    print(f"  (Linear weight params: {n_params:,})")
    print(f"{'='*80}")
    hdr = f"{'Method':<18} {'Size(MB)':>9} {'Eff.BPW':>8} {'PPL':>10} {'GPU MB':>8} {'Time(s)':>8}"
    print(hdr)
    print(f"{'-'*18} {'-'*9} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for r in results:
        size_mb = r["model_size_bytes"] / 1e6
        eff_bpw = (r["model_size_bytes"] * 8) / n_params if r["model_size_bytes"] > 0 else 0
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] < 1e6 else "FAILED"
        gpu_str = f"{r.get('gpu_mem_mb', 0):.0f}"
        print(f"{r['method']:<18} {size_mb:>9.1f} {eff_bpw:>8.2f} {ppl_str:>10} {gpu_str:>8} {r['quant_time_s']:>8.1f}")

    # Save results
    for r in results:
        r["n_params"] = n_params
        r["eff_bpw"] = round((r["model_size_bytes"] * 8) / n_params, 2) if r["model_size_bytes"] > 0 else 0
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to comparison_results.json")


if __name__ == "__main__":
    main()
