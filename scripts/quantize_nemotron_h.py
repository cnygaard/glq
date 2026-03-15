#!/usr/bin/env python3
"""
Quantize NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 with GLQ (3bpw).

Layer-by-layer loading from safetensors to stay within 30GB RAM.
Each layer is loaded, moved to GPU, calibration-forwarded, Hessian-captured,
quantized, error-propagated, then freed.

Usage:
    python quantize_nemotron_h.py [--layers START END] [--nsamples 16] [--dry-run]
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
from safetensors import safe_open

# Add glq to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glq.codebook import E8ShellCodebook
from glq.quantize_model import HessianCapture, quantize_layer_e8_shell_rht


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
def _find_snapshot():
    """Auto-detect snapshot dir from HF cache."""
    snap_dir = os.path.join(
        HF_HOME, "hub",
        "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "snapshots")
    if os.path.isdir(snap_dir):
        entries = os.listdir(snap_dir)
        if entries:
            return os.path.join(snap_dir, entries[0])
    return None
DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_layer_state(weight_map, shard_paths, layer_idx):
    """Load all tensors for a single layer from safetensors shards."""
    prefix = "backbone.layers.{}.".format(layer_idx)
    layer_keys = [k for k in weight_map if k.startswith(prefix)]

    state = {}
    # Group keys by shard to minimize file opens
    from collections import defaultdict
    shard_to_keys = defaultdict(list)
    for key in layer_keys:
        shard_to_keys[weight_map[key]].append(key)

    for shard, keys in shard_to_keys.items():
        with safe_open(shard_paths[shard], framework="pt") as f:
            for key in keys:
                local_key = key[len(prefix):]
                state[local_key] = f.get_tensor(key)

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpw", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--layers", type=int, nargs=2, default=None,
                        help="Process only layers START..END-1")
    parser.add_argument("--output", type=str,
                        default="/opt/dlami/nvme/nemotron-glq-3bpw")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just time first two layers, don't save")
    args = parser.parse_args()

    # ---- Load config ----
    os.environ["HF_HOME"] = HF_HOME
    MODEL_DIR = _find_snapshot()
    assert MODEL_DIR and os.path.isdir(MODEL_DIR), \
        "Model not found in HF cache. Download it first."

    print("GLQ Quantization: {}".format(MODEL_ID))
    print("  bpw={}, nsamples={}, seqlen={}".format(
        args.bpw, args.nsamples, args.seqlen))
    print("  Model dir: {}".format(MODEL_DIR))
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Set attention implementation (normally done by from_pretrained)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ---- Import model classes from remote code ----
    # Use from_config on meta device to trigger the modeling module import
    # without allocating any memory.
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        _model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=DTYPE)
    NemotronHBlock = type(_model.backbone.layers[0])
    del _model

    # ---- Safetensors index ----
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    shard_paths = {s: os.path.join(MODEL_DIR, s) for s in shard_files}

    n_layers = config.num_hidden_layers
    layer_range = (
        range(args.layers[0], args.layers[1]) if args.layers
        else range(n_layers)
    )
    block_types = config.layers_block_type
    print("\n  {} total layers, processing {}".format(
        n_layers, len(layer_range)))
    for i in layer_range:
        print("    Layer {}: {}".format(i, block_types[i]))

    # ---- Calibration data ----
    print("\nLoading calibration data ...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    n_chunks = min(args.nsamples, input_ids.shape[0] // args.seqlen)
    calib_ids = input_ids[:n_chunks * args.seqlen].reshape(n_chunks, args.seqlen)
    print("  {} sequences of length {}".format(n_chunks, args.seqlen))

    # ---- Build codebook ----
    print("\nBuilding E8 shell codebook ...")
    codebook = E8ShellCodebook(device=DEVICE)

    # ---- Embed calibration data ----
    print("\nEmbedding calibration data ...")
    embed_shard = weight_map["backbone.embeddings.weight"]
    with safe_open(shard_paths[embed_shard], framework="pt") as f:
        embed_weight = f.get_tensor("backbone.embeddings.weight")
    embed = nn.Embedding(config.vocab_size, config.hidden_size,
                         _weight=embed_weight)
    embed = embed.to(DEVICE)
    hidden_states = []
    with torch.no_grad():
        for i in range(calib_ids.shape[0]):
            hidden_states.append(
                embed(calib_ids[i:i+1].to(DEVICE)).cpu())
    hidden_states = torch.cat(hidden_states, dim=0)
    del embed, embed_weight
    gc.collect()
    torch.cuda.empty_cache()
    print("  hidden_states: {} ({})".format(
        list(hidden_states.shape), hidden_states.dtype))

    # ---- Forward through layers before our range ----
    if layer_range.start > 0:
        print("\nForwarding through layers 0..{} (unquantized) ...".format(
            layer_range.start - 1))
        for layer_idx in range(layer_range.start):
            t0 = time.perf_counter()
            layer_state = load_layer_state(
                weight_map, shard_paths, layer_idx)
            block = NemotronHBlock(config, layer_idx)
            block.load_state_dict(layer_state, strict=False)
            block = block.to(DEVICE, dtype=DTYPE)
            block.eval()

            cache_pos = torch.arange(args.seqlen, device=DEVICE)
            new_hidden = []
            with torch.no_grad():
                for i in range(hidden_states.shape[0]):
                    h = hidden_states[i:i+1].to(DEVICE)
                    out = block(
                        h, cache_params=None,
                        cache_position=cache_pos)
                    new_hidden.append(out.cpu())
            hidden_states = torch.cat(new_hidden, dim=0)

            del block, layer_state
            gc.collect()
            torch.cuda.empty_cache()
            dt = time.perf_counter() - t0
            print("  Layer {} ({}): {:.1f}s".format(
                layer_idx, block_types[layer_idx], dt))

    # ---- Layer-by-layer quantization ----
    os.makedirs(args.output, exist_ok=True)
    all_artifacts = {}
    all_sqnr = []
    all_metrics = []
    t_start = time.perf_counter()

    for layer_idx in layer_range:
        t_layer = time.perf_counter()
        bt = block_types[layer_idx]
        print("\n" + "=" * 60)
        print("Layer {}/{} ({})".format(layer_idx, n_layers - 1, bt))
        print("=" * 60)

        # Load layer
        layer_state = load_layer_state(
            weight_map, shard_paths, layer_idx)
        n_tensors = len(layer_state)
        mem_mb = sum(
            t.numel() * t.element_size()
            for t in layer_state.values()) / 1e6
        print("  Loaded {} tensors ({:.0f} MB)".format(n_tensors, mem_mb))

        block = NemotronHBlock(config, layer_idx)
        block.load_state_dict(layer_state, strict=False)
        block = block.to(DEVICE, dtype=DTYPE)
        block.eval()
        del layer_state

        # Find all linear layers
        linears = {}
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                linears[name] = mod

        n_linears = len(linears)
        lin_params = sum(m.weight.numel() for m in linears.values())
        print("  {} linear layers ({:.1f}M params)".format(
            n_linears, lin_params / 1e6))

        # Install Hessian hooks
        captures = {}
        for name, mod in linears.items():
            captures[name] = HessianCapture(mod)

        # Forward calibration (hooks capture inputs)
        print("  Capturing Hessians ...")
        cache_pos = torch.arange(args.seqlen, device=DEVICE)
        t_hess = time.perf_counter()
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1].to(DEVICE)
                block(h, cache_params=None, cache_position=cache_pos)
        print("  Hessian capture: {:.1f}s".format(
            time.perf_counter() - t_hess))

        # Finalize Hessians — move to CPU to free GPU
        hessians = {}
        for name, cap in captures.items():
            H = cap.finalize()
            if H is not None:
                hessians[name] = H.cpu()
                del H
        del captures
        gc.collect()
        torch.cuda.empty_cache()

        n_with_hessian = len(hessians)
        print("  {}/{} linears have Hessians".format(
            n_with_hessian, n_linears))

        # Quantize each linear
        layer_sqnrs = []
        for li, (name, mod) in enumerate(linears.items()):
            if name not in hessians:
                print("  {}: no Hessian, skipping".format(name))
                continue

            W = mod.weight.data
            if W.shape[1] < 8:
                print("  {}: in_features={} < 8, skipping".format(
                    name, W.shape[1]))
                continue

            H_cpu = hessians[name]
            # For MoE experts with sparse routing, Hessians can be
            # near-singular. Add extra damping if Cholesky fails.
            t0 = time.perf_counter()
            H_gpu = H_cpu.to(DEVICE)
            try:
                W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
                    W, H_gpu, codebook, bpw=args.bpw)
            except torch._C._LinAlgError:
                # Add strong damping and retry
                damp = 0.1 * torch.mean(torch.diag(H_gpu)).clamp(min=1e-6)
                diag_idx = torch.arange(H_gpu.shape[0], device=DEVICE)
                H_gpu[diag_idx, diag_idx] += damp
                try:
                    W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
                        W, H_gpu, codebook, bpw=args.bpw)
                    print("    (recovered with extra damping)")
                except torch._C._LinAlgError:
                    # Last resort: use identity Hessian (Euclidean NN)
                    H_id = torch.eye(
                        W.shape[1], device=DEVICE, dtype=torch.float32)
                    W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
                        W, H_id, codebook, bpw=args.bpw)
                    print("    (fallback to identity Hessian)")
            del H_gpu
            dt = time.perf_counter() - t0

            # Store artifacts
            layer_prefix = "backbone.layers.{}.{}".format(layer_idx, name)
            all_artifacts[layer_prefix] = {
                k: v.cpu() for k, v in artifacts.items()}

            # Replace weight for error propagation
            mod.weight.data = W_hat.to(
                dtype=mod.weight.dtype, device=mod.weight.device)
            all_sqnr.append(metrics['sqnr'])
            layer_sqnrs.append(metrics['sqnr'])

            all_metrics.append({
                'layer': layer_idx, 'block_type': bt,
                'name': name, 'shape': list(W.shape),
                'sqnr': round(metrics['sqnr'], 2),
                'Wscale': round(metrics['Wscale'], 4),
                'time_s': round(dt, 1),
            })

            # Print progress (compact for experts)
            if '.experts.' in name and 'shared' not in name:
                parts = name.split('experts.')[1].split('.')
                expert_num = int(parts[0])
                proj = parts[-1]
                if expert_num % 16 == 0 or expert_num == 127:
                    print("  expert.{}.{:10s} {:20s} "
                          "SQNR={:5.1f}dB  {:.1f}s".format(
                              expert_num, proj,
                              str(tuple(W.shape)),
                              metrics['sqnr'], dt))
            else:
                print("  {:40s} {:20s} "
                      "SQNR={:5.1f}dB  Ws={:.3f}  {:.1f}s".format(
                          name, str(tuple(W.shape)),
                          metrics['sqnr'], metrics['Wscale'], dt))

            del H_cpu, W_hat, artifacts
            gc.collect()
            if li % 32 == 0:
                torch.cuda.empty_cache()

        # Free all Hessians
        del hessians
        gc.collect()
        torch.cuda.empty_cache()

        # Forward with quantized weights for error propagation
        print("  Forwarding with quantized weights ...")
        new_hidden = []
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1].to(DEVICE)
                out = block(
                    h, cache_params=None, cache_position=cache_pos)
                new_hidden.append(out.cpu())
        hidden_states = torch.cat(new_hidden, dim=0)

        del block
        gc.collect()
        torch.cuda.empty_cache()

        dt_layer = time.perf_counter() - t_layer
        avg_layer_sqnr = (
            sum(layer_sqnrs) / len(layer_sqnrs) if layer_sqnrs else 0)
        elapsed = time.perf_counter() - t_start
        layers_done = layer_idx - layer_range.start + 1
        layers_remaining = len(layer_range) - layers_done
        eta = (
            elapsed / layers_done * layers_remaining
            if layers_done > 0 else 0)

        print("\n  Layer {} summary: {} linears quantized, "
              "avg SQNR={:.1f}dB, {:.0f}s".format(
                  layer_idx, len(layer_sqnrs),
                  avg_layer_sqnr, dt_layer))
        print("  Elapsed: {:.1f}m | ETA: {:.1f}m remaining".format(
            elapsed / 60, eta / 60))

        # Save incremental metrics
        with open(os.path.join(args.output, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

        if args.dry_run and layer_idx >= layer_range.start + 1:
            print("\n--dry-run: stopping after 2 layers")
            break

    # ---- Summary ----
    total_time = time.perf_counter() - t_start
    avg_sqnr = sum(all_sqnr) / len(all_sqnr) if all_sqnr else 0
    print("\n" + "=" * 60)
    print("Quantized {} linear layers in {:.1f}m".format(
        len(all_sqnr), total_time / 60))
    print("Average SQNR = {:.2f} dB".format(avg_sqnr))
    print("=" * 60)

    if not args.dry_run:
        # Save artifacts
        print("\nSaving to {} ...".format(args.output))
        from safetensors.torch import save_file

        # Quantized layer artifacts
        state_dict = {}
        for layer_prefix, arts in all_artifacts.items():
            for key, tensor in arts.items():
                state_dict["{}.{}".format(layer_prefix, key)] = tensor.cpu()

        # Non-quantized parameters
        quantized_prefixes = set(all_artifacts.keys())
        for key in weight_map:
            param_prefix = (
                key.rsplit(".", 1)[0] if "." in key else "")
            if param_prefix in quantized_prefixes:
                continue
            shard = weight_map[key]
            with safe_open(shard_paths[shard], framework="pt") as f:
                state_dict[key] = f.get_tensor(key)

        save_file(
            state_dict,
            os.path.join(args.output, "model.safetensors"))
        size_gb = os.path.getsize(
            os.path.join(args.output, "model.safetensors")) / 1e9
        print("  Saved model.safetensors: {:.2f} GB".format(size_gb))

        # Save codebook
        codebook.save(os.path.join(args.output, "e8_codebook.pt"))

        # Save config
        config_dict = config.to_dict()
        config_dict["quantization_config"] = {
            "quant_method": "glq",
            "codebook": "e8_shell",
            "codesz": 8,
            "bpw": args.bpw,
        }
        with open(
                os.path.join(args.output, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save tokenizer
        tokenizer.save_pretrained(args.output)

        # Save quantize metadata
        quant_meta = {
            "quant_method": "glq",
            "bpw": args.bpw,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "avg_sqnr_db": round(avg_sqnr, 2),
            "n_quantized_layers": len(all_sqnr),
            "total_time_s": round(total_time, 1),
            "source_model": MODEL_ID,
        }
        with open(os.path.join(args.output,
                               "quantize_config.json"), "w") as f:
            json.dump(quant_meta, f, indent=2)

        print("Done!")


if __name__ == "__main__":
    main()
