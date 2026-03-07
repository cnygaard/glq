"""
Quantize a HuggingFace model with GLQ (E8 Shell + RHT + LDLQ).

Usage:
    python -m glq.quantize_model \
        --model mistralai/Ministral-3-3B-Reasoning-2512 \
        --output ./ministral-glq-2bpw \
        --bpw 2 --nsamples 16

Python API:
    from glq.quantize_model import quantize
    quantize("model_name", "output_dir", bpw=2)
"""

import argparse
import gc
import json
import os
import time

import torch
import torch.nn as nn

from .codebook import E8ShellCodebook
from .rht import RHT
from .ldlq import quantize_ldlq_codebook, quantize_ldlq_codebook_2stage


# ---- Hessian capture ----

class HessianCapture:
    """Hook-based Hessian (X^T X) accumulator for a linear layer."""

    def __init__(self, layer: nn.Linear):
        self.n_samples = 0
        self.H = None
        self._hook = layer.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, inp):
        x = inp[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.float()
        n = x.shape[0]
        if self.H is None:
            self.H = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32,
                                 device=x.device)
        self.H.addmm_(x.T, x)
        self.n_samples += n

    def finalize(self):
        self._hook.remove()
        if self.H is not None and self.n_samples > 0:
            self.H /= self.n_samples
        return self.H


# ---- padding helpers ----

def pad_to_multiple(W, block_size=8):
    n = W.shape[1]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        W = torch.cat([W, torch.zeros(W.shape[0], pad, dtype=W.dtype, device=W.device)], dim=1)
    return W, n, pad


def pad_hessian(H, block_size=8):
    n = H.shape[0]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        H_pad = torch.zeros(n + pad, n + pad, dtype=H.dtype, device=H.device)
        H_pad[:n, :n] = H
        for i in range(n, n + pad):
            H_pad[i, i] = 1e-6
        return H_pad
    return H


# ---- model structure helpers ----

def get_decoder_layers(text_model):
    if hasattr(text_model, 'layers'):
        return text_model.layers
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'layers'):
        return text_model.model.layers
    raise ValueError("Cannot find transformer layers")


def get_embed(text_model):
    if hasattr(text_model, 'embed_tokens'):
        return text_model.embed_tokens
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'embed_tokens'):
        return text_model.model.embed_tokens
    raise ValueError("Cannot find embedding layer")


def get_rotary_emb(text_model):
    if hasattr(text_model, 'rotary_emb'):
        return text_model.rotary_emb
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'rotary_emb'):
        return text_model.model.rotary_emb
    return None


# ---- per-layer quantization ----

def quantize_layer_e8_shell_rht(W, H, codebook, bpw=2, tune_iters=0):
    """
    Quantize a single linear layer with E8 Shell + RHT + LDLQ.

    Returns:
        W_hat: dequantized weight matrix (for error propagation to next layer)
        artifacts: dict with Qidxs, SU, SV, Wscale (for saving)
        metrics: dict with sqnr, bpw
    """
    dev = codebook.device
    m, n = W.shape
    W_f = W.float().to(dev)
    H_f = H.float().to(dev)

    rht = RHT(m, n, device=dev)
    W_tilde = rht.transform_weights(W_f)
    H_tilde = rht.transform_hessian(H_f)

    n_tilde = W_tilde.shape[1]
    W_pad, _, _ = pad_to_multiple(W_tilde, 8)
    H_pad = pad_hessian(H_tilde, 8)

    if bpw == 2:
        result = quantize_ldlq_codebook(W_pad, H_pad, codebook, tune_iters=tune_iters)
    else:
        result = quantize_ldlq_codebook_2stage(
            W_pad, H_pad, codebook, codebook,
            resid_scale=codebook.resid_scale, tune_iters=tune_iters)

    # Dequantize for error propagation
    W_hat_tilde = result['W_hat'][:, :n_tilde]
    W_hat = rht.inverse_transform_weights(W_hat_tilde)

    # SQNR
    diff = W_f - W_hat
    sqnr = 10 * torch.log10(W_f.pow(2).sum() / diff.pow(2).sum().clamp(min=1e-20)).item()

    # Artifacts for saving (these go into safetensors)
    artifacts = {
        'Qidxs': result['indices'].to(torch.int16),  # (m_pad, n_pad/8)
        'SU': rht.su.half(),                           # (m_pad,)
        'SV': rht.sv.half(),                           # (n_pad,)
        'Wscale': torch.tensor(result['Wscale'], dtype=torch.float32),  # scalar
    }

    metrics = {'sqnr': sqnr, 'bpw': result['bpw'], 'Wscale': result['Wscale']}
    return W_hat, artifacts, metrics


# ---- main quantization pipeline ----

def quantize(
    model_name: str,
    output_dir: str,
    bpw: int = 2,
    tune_iters: int = 0,
    nsamples: int = 16,
    seqlen: int = 2048,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """
    Quantize a HuggingFace model with GLQ and save to output_dir.

    Args:
        model_name: HF model ID or local path
        output_dir: where to save quantized model
        bpw: bits per weight (2 or 4)
        tune_iters: LDLQ refinement passes
        nsamples: calibration samples from WikiText-2
        seqlen: calibration sequence length
        device: 'cuda' or 'cpu'
        dtype: model dtype for loading
    """
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    os.makedirs(output_dir, exist_ok=True)

    print(f"GLQ Quantization: {model_name} -> {output_dir}")
    print(f"  bpw={bpw}, tune_iters={tune_iters}, nsamples={nsamples}")

    # ---- Load model ----
    print(f"\nLoading model ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    arch = cfg.architectures[0] if cfg.architectures else ""

    if "Mistral3" in arch:
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )
        text_model = model.model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )
        text_model = model

    n_params = sum(p.numel() for p in text_model.parameters()) / 1e9
    print(f"  {n_params:.2f}B params in {time.perf_counter() - t0:.1f}s")

    # ---- Calibration data ----
    print(f"\nLoading calibration data ...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    n_chunks = min(nsamples, input_ids.shape[0] // seqlen)
    calib_ids = input_ids[:n_chunks * seqlen].reshape(n_chunks, seqlen)
    print(f"  {n_chunks} sequences of length {seqlen}")

    # ---- Build codebook ----
    print(f"\nBuilding E8 shell codebook ...")
    codebook = E8ShellCodebook(device=device)

    # ---- Layer-by-layer quantization ----
    decoder_layers = get_decoder_layers(text_model)
    embed = get_embed(text_model)
    rotary_emb = get_rotary_emb(text_model)
    n_layers = len(decoder_layers)

    # Embed calibration data
    print(f"\nEmbedding calibration data ...")
    hidden_states = []
    with torch.no_grad():
        for i in range(calib_ids.shape[0]):
            hidden_states.append(embed(calib_ids[i:i+1].to(device)))
    hidden_states = torch.cat(hidden_states, dim=0)

    all_artifacts = {}  # layer_name -> {Qidxs, SU, SV, Wscale}
    all_sqnr = []
    t_start = time.perf_counter()

    for layer_idx in range(n_layers):
        layer = decoder_layers[layer_idx]
        t_layer = time.perf_counter()
        print(f"\n--- Layer {layer_idx}/{n_layers-1} ---")

        # Collect linear sublayers
        linears = {}
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                linears[name] = mod

        # Install Hessian hooks
        captures = {}
        for name, mod in linears.items():
            captures[name] = HessianCapture(mod)

        # Forward calibration (hooks capture inputs)
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1]
                seq_len = h.shape[1]
                cache_position = torch.arange(seq_len, device=h.device)
                position_ids = cache_position.unsqueeze(0)
                kwargs = dict(position_ids=position_ids, cache_position=cache_position,
                              use_cache=False)
                if rotary_emb is not None:
                    kwargs['position_embeddings'] = rotary_emb(h, position_ids=position_ids)
                layer(h, **kwargs)

        # Quantize each linear
        for name, cap in captures.items():
            H = cap.finalize()
            if H is None:
                print(f"  {name}: no activations, skipping")
                continue

            mod = linears[name]
            W = mod.weight.data

            if W.shape[1] < 8:
                print(f"  {name}: in={W.shape[1]} < 8, skipping")
                continue

            t0 = time.perf_counter()
            W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
                W, H, codebook, bpw=bpw, tune_iters=tune_iters)
            dt = time.perf_counter() - t0

            # Store artifacts with full layer path
            layer_prefix = f"model.layers.{layer_idx}.{name}"
            all_artifacts[layer_prefix] = artifacts

            # Replace weight for error propagation to next layer
            mod.weight.data = W_hat.to(dtype=mod.weight.dtype, device=mod.weight.device)
            all_sqnr.append(metrics['sqnr'])

            print(f"  {name:30s} {str(tuple(W.shape)):20s} "
                  f"SQNR={metrics['sqnr']:5.1f}dB  Ws={metrics['Wscale']:.3f}  {dt:.1f}s")

            del H
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Forward calibration through quantized layer
        new_hidden = []
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1]
                seq_len = h.shape[1]
                cache_position = torch.arange(seq_len, device=h.device)
                position_ids = cache_position.unsqueeze(0)
                kwargs = dict(position_ids=position_ids, cache_position=cache_position,
                              use_cache=False)
                if rotary_emb is not None:
                    kwargs['position_embeddings'] = rotary_emb(h, position_ids=position_ids)
                out = layer(h, **kwargs)
                new_hidden.append(out[0] if isinstance(out, tuple) else out)
        hidden_states = torch.cat(new_hidden, dim=0)

        dt_layer = time.perf_counter() - t_layer
        elapsed = time.perf_counter() - t_start
        remaining = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
        print(f"  Layer time: {dt_layer:.0f}s  |  "
              f"Elapsed: {elapsed/60:.1f}m  |  "
              f"ETA: {remaining/60:.1f}m remaining")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start
    avg_sqnr = sum(all_sqnr) / len(all_sqnr) if all_sqnr else 0
    print(f"\n{'='*60}")
    print(f"Quantized {len(all_sqnr)} sublayers in {total_time/60:.1f}m")
    print(f"Average SQNR = {avg_sqnr:.2f} dB")
    print(f"{'='*60}")

    # ---- Save ----
    print(f"\nSaving to {output_dir} ...")

    # 1. Build state dict: quantized layers get Qidxs/SU/SV/Wscale,
    #    non-quantized params (embeddings, lm_head, layernorms) keep original tensors
    from safetensors.torch import save_file

    quantized_prefixes = set(all_artifacts.keys())
    state_dict = {}

    # Add quantized layer artifacts
    for layer_prefix, arts in all_artifacts.items():
        for key, tensor in arts.items():
            state_dict[f"{layer_prefix}.{key}"] = tensor.cpu()

    # Add non-quantized parameters (everything that's NOT a quantized linear's weight)
    for name, param in model.named_parameters():
        # Check if this param belongs to a quantized linear
        # e.g. "model.layers.0.self_attn.q_proj.weight" -> prefix "model.layers.0.self_attn.q_proj"
        param_prefix = name.rsplit(".", 1)[0] if "." in name else ""
        if param_prefix in quantized_prefixes:
            continue  # skip — replaced by Qidxs/SU/SV/Wscale
        state_dict[name] = param.data.cpu()

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # 2. Save codebook
    codebook.save(os.path.join(output_dir, "e8_codebook.pt"))

    # 3. Save config.json with quantization_config
    config_dict = cfg.to_dict()
    config_dict["quantization_config"] = {
        "quant_method": "glq",
        "codebook": "e8_shell",
        "codesz": 8,
        "bpw": bpw,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Save quantize_config.json (metadata)
    quant_meta = {
        "quant_method": "glq",
        "codebook": "e8_shell",
        "codesz": 8,
        "bpw": bpw,
        "tune_iters": tune_iters,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "avg_sqnr_db": round(avg_sqnr, 2),
        "n_quantized_layers": len(all_sqnr),
        "total_time_s": round(total_time, 1),
        "source_model": model_name,
    }
    with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
        json.dump(quant_meta, f, indent=2)

    # 5. Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Done! Saved to {output_dir}")
    print(f"  model.safetensors: {os.path.getsize(os.path.join(output_dir, 'model.safetensors')) / 1e6:.1f} MB")
    print(f"  config.json: quantization_config.quant_method = 'glq'")

    return avg_sqnr


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Quantize a model with GLQ")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--bpw", type=int, default=2, choices=[2, 4],
                        help="Bits per weight (2 or 4)")
    parser.add_argument("--tune-iters", type=int, default=0,
                        help="LDLQ refinement iterations")
    parser.add_argument("--nsamples", type=int, default=16,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Calibration sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    quantize(
        model_name=args.model,
        output_dir=args.output,
        bpw=args.bpw,
        tune_iters=args.tune_iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        device=args.device,
    )


if __name__ == "__main__":
    main()
