#!/usr/bin/env python3
"""
Perplexity of Nemotron-3-Nano-30B on WikiText-2.

Layer-by-layer forward pass to stay within GPU memory.
Supports both bf16 baseline and GLQ-quantized modes.

Usage:
    python evaluate_nemotron_ppl.py --mode quantized
    python evaluate_nemotron_ppl.py --mode bf16
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# Add glq to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glq.codebook import E8ShellCodebook
from glq.quantized_linear import E8RHTLinear


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DEVICE = "cuda"
DTYPE = torch.bfloat16


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


def load_layer_state(weight_map, shard_paths, layer_idx):
    """Load all tensors for a single layer from safetensors shards."""
    prefix = "backbone.layers.{}.".format(layer_idx)
    layer_keys = [k for k in weight_map if k.startswith(prefix)]

    state = {}
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


def load_quantized_layer_state(quant_safetensors, layer_idx):
    """Load all tensors for a single layer from the quantized safetensors."""
    prefix = "backbone.layers.{}.".format(layer_idx)
    state = {}
    with safe_open(quant_safetensors, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                local_key = key[len(prefix):]
                state[local_key] = f.get_tensor(key)
    return state


def load_tensor(safetensors_path, key):
    """Load a single tensor from a safetensors file."""
    with safe_open(safetensors_path, framework="pt") as f:
        return f.get_tensor(key)


def load_tensor_from_shards(weight_map, shard_paths, key):
    """Load a single tensor from sharded safetensors."""
    shard = weight_map[key]
    with safe_open(shard_paths[shard], framework="pt") as f:
        return f.get_tensor(key)


def replace_linears_with_e8rht(block, layer_state, codebook, codebook2):
    """Replace nn.Linear modules with E8RHTLinear using quantized artifacts."""
    # Find which linears have quantized data (have .Qidxs key)
    quantized_names = set()
    for key in layer_state:
        if key.endswith(".Qidxs"):
            # e.g., "mixer.in_proj.Qidxs" -> "mixer.in_proj"
            quantized_names.add(key.rsplit(".", 1)[0])

    for name, mod in list(block.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if name not in quantized_names:
            # Not quantized -- load regular weight
            wkey = "{}.weight".format(name)
            if wkey in layer_state:
                mod.weight.data = layer_state[wkey]
            bkey = "{}.bias".format(name)
            if bkey in layer_state:
                mod.bias.data = layer_state[bkey]
            continue

        # This linear was quantized -- dequantize back to dense weight.
        # Using dequantization instead of live E8RHTLinear avoids
        # compatibility issues with Mamba code that accesses .weight
        e8 = E8RHTLinear(mod.in_features, mod.out_features,
                         bias=mod.bias is not None)

        # Load quantized buffers
        for buf_name in ["Qidxs", "SU", "SV", "Wscale",
                         "Qidxs2", "inv_resid_scale"]:
            key = "{}.{}".format(name, buf_name)
            if key in layer_state:
                getattr(e8, buf_name).data.copy_(layer_state[key])

        e8.set_codebook(codebook, codebook2=codebook2)

        # Dequantize to dense weight and put back into nn.Linear
        W_deq = e8.dequantize().to(DTYPE)
        mod.weight.data = W_deq
        del e8, W_deq

        if mod.bias is not None:
            bkey = "{}.bias".format(name)
            if bkey in layer_state:
                mod.bias.data = layer_state[bkey]

    # Load remaining non-linear params (norms, conv1d, A_log, D, dt_bias)
    remaining = {}
    for key, tensor in layer_state.items():
        # Skip quantized artifact keys (already handled above)
        is_quant_artifact = any(
            key.endswith(s) for s in
            [".Qidxs", ".SU", ".SV", ".Wscale",
             ".Qidxs2", ".inv_resid_scale"])
        # Skip linear weights we already loaded via dequantization
        is_handled_linear = any(
            key == "{}.weight".format(n) or key == "{}.bias".format(n)
            for n in quantized_names)
        # Skip regular linear weights (already loaded above)
        is_regular_weight = False
        for n, m in block.named_modules():
            if isinstance(m, nn.Linear) and n not in quantized_names:
                if key == "{}.weight".format(n) or key == "{}.bias".format(n):
                    is_regular_weight = True
                    break
        if not is_quant_artifact and not is_handled_linear and not is_regular_weight:
            remaining[key] = tensor

    if remaining:
        block.load_state_dict(remaining, strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quantized", "bf16"],
                        default="quantized")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--quant-dir", type=str,
                        default="/opt/dlami/nvme/nemotron-glq-3bpw")
    args = parser.parse_args()

    print("=" * 60)
    print("NemotronH Perplexity Assessment")
    print("  Mode: {}".format(args.mode))
    print("  Seqlen: {}".format(args.seqlen))
    if args.mode == "quantized":
        print("  Quant dir: {}".format(args.quant_dir))
    print("=" * 60)

    # ---- Load config and tokenizer ----
    os.environ["HF_HOME"] = HF_HOME
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ---- Import NemotronHBlock ----
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        _model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=DTYPE)
    NemotronHBlock = type(_model.backbone.layers[0])
    del _model

    # ---- Setup source data ----
    MODEL_DIR = _find_snapshot()
    weight_map = None
    shard_paths = None
    quant_st = None

    if args.mode == "bf16":
        assert MODEL_DIR and os.path.isdir(MODEL_DIR), \
            "BF16 model not found in HF cache."
        with open(os.path.join(
                MODEL_DIR, "model.safetensors.index.json")) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        shard_paths = {s: os.path.join(MODEL_DIR, s) for s in shard_files}
    else:
        quant_st = os.path.join(args.quant_dir, "model.safetensors")
        assert os.path.isfile(quant_st), \
            "Quantized model not found: {}".format(quant_st)
        # Also need bf16 weight_map for architecture init reference
        if MODEL_DIR and os.path.isdir(MODEL_DIR):
            idx_path = os.path.join(
                MODEL_DIR, "model.safetensors.index.json")
            if os.path.isfile(idx_path):
                with open(idx_path) as f:
                    index = json.load(f)
                weight_map = index["weight_map"]
                shard_files = sorted(set(weight_map.values()))
                shard_paths = {s: os.path.join(MODEL_DIR, s)
                               for s in shard_files}

    # ---- Build codebook (for quantized mode) ----
    codebook = None
    codebook2 = None
    if args.mode == "quantized":
        cb_path = os.path.join(args.quant_dir, "e8_codebook.pt")
        if os.path.isfile(cb_path):
            print("\nLoading codebook from {} ...".format(cb_path))
            codebook = E8ShellCodebook.load(cb_path, device="cpu")
        else:
            print("\nBuilding E8 shell codebook ...")
            codebook = E8ShellCodebook(device="cpu")
        # 3bpw needs secondary codebook
        codebook2 = codebook.make_small(256)

    # ---- Load WikiText-2 test set ----
    print("\nLoading WikiText-2 test set ...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    n_chunks = input_ids.shape[0] // args.seqlen
    test_ids = input_ids[:n_chunks * args.seqlen].reshape(
        n_chunks, args.seqlen)
    print("  {} tokens -> {} sequences of length {}".format(
        len(input_ids), n_chunks, args.seqlen))

    # ---- Embed test data ----
    print("\nEmbedding test data ...")
    if args.mode == "bf16":
        embed_weight = load_tensor_from_shards(
            weight_map, shard_paths, "backbone.embeddings.weight")
    else:
        embed_weight = load_tensor(quant_st, "backbone.embeddings.weight")

    embed = nn.Embedding(config.vocab_size, config.hidden_size,
                         _weight=embed_weight).to(DEVICE)
    hidden_states = []
    with torch.no_grad():
        for i in range(test_ids.shape[0]):
            hidden_states.append(
                embed(test_ids[i:i+1].to(DEVICE)).cpu())
    hidden_states = torch.cat(hidden_states, dim=0)
    del embed, embed_weight
    gc.collect()
    torch.cuda.empty_cache()
    print("  hidden_states: {} ({})".format(
        list(hidden_states.shape), hidden_states.dtype))

    # ---- Layer-by-layer forward ----
    n_layers = config.num_hidden_layers
    block_types = config.layers_block_type
    cache_pos = torch.arange(args.seqlen, device=DEVICE)
    t_start = time.perf_counter()

    for layer_idx in range(n_layers):
        t0 = time.perf_counter()
        bt = block_types[layer_idx]

        # Create block
        block = NemotronHBlock(config, layer_idx)

        if args.mode == "bf16":
            layer_state = load_layer_state(
                weight_map, shard_paths, layer_idx)
            block.load_state_dict(layer_state, strict=False)
            del layer_state
            block = block.to(DEVICE, dtype=DTYPE)
        else:
            layer_state = load_quantized_layer_state(quant_st, layer_idx)
            replace_linears_with_e8rht(
                block, layer_state, codebook, codebook2)
            del layer_state
            block = block.to(DEVICE, dtype=DTYPE)

        block.eval()

        # Forward each sequence
        new_hidden = []
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1].to(DEVICE)
                out = block(h, cache_params=None, cache_position=cache_pos)
                new_hidden.append(out.cpu())
        hidden_states = torch.cat(new_hidden, dim=0)

        del block
        gc.collect()
        torch.cuda.empty_cache()

        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
        print("  Layer {}/{} ({}): {:.1f}s  "
              "[{:.1f}m elapsed, {:.1f}m remaining]".format(
                  layer_idx, n_layers - 1, bt, dt,
                  elapsed / 60, eta / 60))

    # ---- Apply final norm ----
    print("\nApplying final norm ...")
    if args.mode == "bf16":
        norm_weight = load_tensor_from_shards(
            weight_map, shard_paths, "backbone.norm_f.weight")
    else:
        norm_weight = load_tensor(quant_st, "backbone.norm_f.weight")

    norm_weight = norm_weight.to(DEVICE, dtype=DTYPE)
    new_hidden = []
    with torch.no_grad():
        for i in range(hidden_states.shape[0]):
            h = hidden_states[i:i+1].to(DEVICE, dtype=DTYPE)
            # RMSNorm
            variance = h.pow(2).mean(-1, keepdim=True)
            h = h * torch.rsqrt(variance + config.rms_norm_eps)
            h = h * norm_weight
            new_hidden.append(h.cpu())
    hidden_states = torch.cat(new_hidden, dim=0)
    del norm_weight
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Compute logits and perplexity ----
    print("\nComputing perplexity ...")
    if args.mode == "bf16":
        lm_head_weight = load_tensor_from_shards(
            weight_map, shard_paths, "lm_head.weight")
    else:
        lm_head_weight = load_tensor(quant_st, "lm_head.weight")

    lm_head_weight = lm_head_weight.to(DEVICE, dtype=DTYPE)
    V = lm_head_weight.shape[0]

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(hidden_states.shape[0]):
            h = hidden_states[i:i+1].to(DEVICE, dtype=DTYPE)
            logits = F.linear(h, lm_head_weight)  # (1, seqlen, V)
            shift_logits = logits[:, :-1, :].float()
            shift_labels = test_ids[i:i+1, 1:].to(DEVICE)
            nll = F.cross_entropy(
                shift_logits.reshape(-1, V),
                shift_labels.reshape(-1),
                reduction="sum")
            total_nll += nll.item()
            total_tokens += shift_labels.numel()

            if (i + 1) % max(1, hidden_states.shape[0] // 5) == 0:
                ppl_so_far = math.exp(total_nll / total_tokens)
                print("  [{}/{}] ppl={:.2f}".format(
                    i + 1, hidden_states.shape[0], ppl_so_far))

    del lm_head_weight
    gc.collect()
    torch.cuda.empty_cache()

    ppl = math.exp(total_nll / total_tokens)
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 60)
    print("Results ({}):".format(args.mode))
    print("  Perplexity: {:.2f}".format(ppl))
    print("  Total NLL: {:.2f}".format(total_nll))
    print("  Total tokens: {}".format(total_tokens))
    print("  Total time: {:.1f}m".format(total_time / 60))
    print("=" * 60)

    # Save results to JSON alongside the evaluation
    result_path = os.path.join(
        args.quant_dir if args.mode == "quantized" else ".",
        "ppl_result_{}.json".format(args.mode))
    result = {
        "mode": args.mode,
        "perplexity": round(ppl, 4),
        "total_nll": round(total_nll, 4),
        "total_tokens": total_tokens,
        "seqlen": args.seqlen,
        "total_time_s": round(total_time, 1),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Results saved to {}".format(result_path))


if __name__ == "__main__":
    main()
