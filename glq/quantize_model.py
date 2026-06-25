"""
Quantize a HuggingFace model with GLQ (E8 Shell + RHT + LDLQ).

Usage:
    python -m glq.quantize_model \
        --model mistralai/Ministral-3-3B-Reasoning-2512 \
        --output ./ministral-glq-2bpw \
        --bpw 2 --nsamples 128

Python API:
    from glq.quantize_model import quantize
    quantize("model_name", "output_dir", bpw=2)
"""

import argparse
import gc
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn

from .codebook import E8ShellCodebook
from .rht import RHT
from .ldlq import quantize_ldlq_codebook, quantize_ldlq_codebook_2stage


# ---- parallel worker state ----

_worker_codebook = None


def _init_worker(cb_tensor, cb_opt_scale, cb_resid_scale, n_threads):
    """Initialize codebook in each worker process (called once per worker)."""
    global _worker_codebook
    torch.set_num_threads(n_threads)
    _worker_codebook = E8ShellCodebook.from_precomputed(
        cb_tensor, cb_opt_scale, cb_resid_scale, device='cpu')


def _quantize_sublayer(args):
    """Worker function: quantize one sublayer."""
    name, W, H, bpw, tune_iters = args
    W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
        W, H, _worker_codebook, bpw=bpw, tune_iters=tune_iters)
    return name, W_hat, {k: v.cpu() for k, v in artifacts.items()}, metrics


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


# ---- model profiles ----
# Each profile encapsulates architecture-specific paths for layer access,
# embedding access, state dict key prefix, and forward kwargs.

_MODEL_PROFILES = {
    'NemotronHForCausalLM': {
        'layers_attr': 'backbone.layers',
        'embed_attr': 'backbone.embeddings',
        'rotary_attr': None,       # no global rotary embedding
        'sd_prefix': 'backbone.layers',
        'trust_remote_code': True,
        'forward_kwargs': 'nemotron_h',
    },
    'Gemma4ForConditionalGeneration': {
        # Multimodal — language decoder lives at model.language_model. The
        # streaming branch in `quantize()` drives the right block class
        # (Gemma4TextDecoderLayer) and overrides `sd_prefix` directly, so
        # the layers_attr / embed_attr / sd_prefix entries here are the
        # non-streaming defaults that we don't currently exercise on the
        # 31B model (it doesn't fit). Kept correct for future small Gemma
        # 4 variants.
        'layers_attr': 'model.language_model.layers',
        'embed_attr': 'model.language_model.embed_tokens',
        'rotary_attr': 'model.language_model.rotary_emb',
        'sd_prefix': 'model.language_model.layers',
        'trust_remote_code': False,
        'forward_kwargs': 'gemma4',
    },
    'Gemma4UnifiedForConditionalGeneration': {
        # gemma-4-12B-it (model_type `gemma4_unified`): a vision+audio
        # multimodal wrapper around a DENSE Gemma-4 text decoder. Same module
        # layout as Gemma4ForConditionalGeneration (decoder at
        # model.language_model.*), but dense — hidden_size_per_layer_input == 0
        # (no PLE) and num_kv_shared_layers == 0 (no KV sharing), so those
        # gemma4 code paths auto-skip. Streaming branch overrides sd_prefix and
        # picks the Gemma4Unified* classes by arch/model_type.
        'layers_attr': 'model.language_model.layers',
        'embed_attr': 'model.language_model.embed_tokens',
        'rotary_attr': 'model.language_model.rotary_emb',
        'sd_prefix': 'model.language_model.layers',
        'trust_remote_code': False,
        'forward_kwargs': 'gemma4',
    },
    'SarvamMoEForCausalLM': {
        # sarvam-30b (model_type `sarvam_moe`, trust_remote_code): a dense-text
        # MoE. Standard `model.layers.*` layout; embeddings at
        # `model.word_embeddings`; global rotary at `model.rotary_emb`. Each
        # SarvamMoEDecoderLayer self-selects dense-vs-MoE MLP by layer_idx
        # (first_k_dense_replace). Experts are per-expert nn.Linear in a
        # ModuleList (+ a shared expert) so the nn.Linear quant loop picks them
        # up; the router (SarvamMoEGate) is an nn.Parameter so it is skipped and
        # stays dense. The attention requires precomputed position_embeddings,
        # so the streaming path builds the rotary from `rotary_attr`.
        'layers_attr': 'model.layers',
        'embed_attr': 'model.word_embeddings',
        'rotary_attr': 'model.rotary_emb',
        'sd_prefix': 'model.layers',
        'trust_remote_code': True,
        'forward_kwargs': 'default',
        # Routed MoE experts feed vLLM's FusedMoE, whose fused dequant kernel
        # takes a SINGLE input rotation (w13_SV) for the stacked gate+up (w13).
        # gate_proj/up_proj quantized separately each carry their OWN random
        # Hadamard SV, so they cannot be stacked under one SV (the fused-QKV
        # shared-SV trap). Fusing gate+up into one ``gate_up_proj`` matrix
        # BEFORE quantization yields one shared SV/SU/Wscale/inv_resid_scale —
        # matching FusedMoE's w13 layout exactly (same pattern as a fused QKV).
        # Applies to routed experts only; the dense (first_k_dense) MLP and the
        # shared expert stay split and use vLLM's per-shard merged-linear path.
        'fuse_expert_gate_up': True,
    },
}

_DEFAULT_PROFILE = {
    'layers_attr': None,   # use heuristic
    'embed_attr': None,    # use heuristic
    'rotary_attr': None,   # use heuristic
    'sd_prefix': 'model.layers',
    'trust_remote_code': False,
    'forward_kwargs': 'default',
}


def _detect_profile(config):
    """Detect model profile from config architecture."""
    arch = (config.architectures or [""])[0] if hasattr(config, 'architectures') else ""
    return _MODEL_PROFILES.get(arch, _DEFAULT_PROFILE)


def _resolve_attr(obj, dotted_path):
    """Resolve 'a.b.c' to obj.a.b.c."""
    for part in dotted_path.split('.'):
        obj = getattr(obj, part)
    return obj


# ---- model structure helpers ----

def get_decoder_layers(text_model, profile=None):
    if profile and profile.get('layers_attr'):
        return _resolve_attr(text_model, profile['layers_attr'])
    if hasattr(text_model, 'layers'):
        return text_model.layers
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'layers'):
        return text_model.model.layers
    if hasattr(text_model, 'backbone') and hasattr(text_model.backbone, 'layers'):
        return text_model.backbone.layers
    raise ValueError("Cannot find transformer layers")


def get_embed(text_model, profile=None):
    if profile and profile.get('embed_attr'):
        return _resolve_attr(text_model, profile['embed_attr'])
    if hasattr(text_model, 'embed_tokens'):
        return text_model.embed_tokens
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'embed_tokens'):
        return text_model.model.embed_tokens
    # NemotronH and similar use 'embeddings' instead of 'embed_tokens'
    if hasattr(text_model, 'embeddings'):
        return text_model.embeddings
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'embeddings'):
        return text_model.model.embeddings
    raise ValueError("Cannot find embedding layer")


def get_rotary_emb(text_model, profile=None):
    if profile and profile.get('rotary_attr'):
        return _resolve_attr(text_model, profile['rotary_attr'])
    if profile and profile.get('rotary_attr') is None and profile.get('layers_attr'):
        # Profile explicitly says no rotary (e.g. NemotronH)
        return None
    if hasattr(text_model, 'rotary_emb'):
        return text_model.rotary_emb
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'rotary_emb'):
        return text_model.model.rotary_emb
    return None


def _build_forward_kwargs(profile, h, rotary_emb, layer_idx=None, cfg=None,
                          per_layer_inputs=None, sample_idx=None,
                          shared_kv_cache=None):
    """Build layer forward kwargs based on model profile.

    For Gemma 4, the rotary embedding is per-layer-type ("sliding_attention" vs
    "full_attention"), so we look up the layer's type from ``cfg.text_config.layer_types``
    and pass it to ``rotary_emb`` to get the right ``(cos, sin)`` tuple.

    For E2B/E4B (``hidden_size_per_layer_input > 0``) the caller passes a
    pre-computed ``per_layer_inputs`` tensor of shape
    ``[n_samples, T, num_layers, ple_dim]`` plus the current ``sample_idx`` so we
    can slice ``per_layer_inputs[sample_idx:sample_idx+1, :, layer_idx, :]``.
    For the 31B variant (``hidden_size_per_layer_input == 0``) ``per_layer_inputs``
    stays ``None`` and ``per_layer_input`` is passed as ``None``.

    For E2B/E4B (``num_kv_shared_layers > 0``) reader layers expect a populated
    ``shared_kv_states[kv_shared_layer_index]``. The caller maintains
    ``shared_kv_cache: dict[producer_idx, list[(K, V) per sample]]`` populated by
    earlier producer-layer forwards; we look up the entry for ``sample_idx`` and
    pass it in. For 31B (``num_kv_shared_layers == 0``) no readers exist so an
    empty dict is fine.
    """
    seq_len = h.shape[1]
    cache_position = torch.arange(seq_len, device=h.device)

    if profile.get('forward_kwargs') == 'nemotron_h':
        return dict(cache_params=None, cache_position=cache_position)

    if profile.get('forward_kwargs') == 'gemma4':
        position_ids = cache_position.unsqueeze(0)
        layer_type = cfg.text_config.layer_types[layer_idx]
        # Per-Layer Embedding: slice the precomputed per_layer_inputs tensor.
        if per_layer_inputs is not None and sample_idx is not None:
            ple = per_layer_inputs[sample_idx:sample_idx+1, :, layer_idx, :]
        else:
            ple = None  # 31B: hidden_size_per_layer_input == 0, layer ignores it
        # KV sharing: hand each reader layer the cached K/V from its producer.
        # The dict is also writable so producer-layer forwards populate it
        # (we extract those entries after the call to refresh the cache).
        shared_kv = {}
        if shared_kv_cache is not None and sample_idx is not None:
            for producer_idx, samples in shared_kv_cache.items():
                if sample_idx < len(samples):
                    # transformers >=5.10 keys shared_kv_states by layer_type
                    # ('full_attention'/'sliding_attention'), not producer index
                    # (the reader does ``shared_kv_states[self.layer_type]``).
                    # Producer's layer_type == its reader's, so this resolves.
                    shared_kv[cfg.text_config.layer_types[producer_idx]] = samples[sample_idx]
        return dict(
            position_ids=position_ids,
            position_embeddings=rotary_emb(h, position_ids=position_ids,
                                          layer_type=layer_type),
            shared_kv_states=shared_kv,
            per_layer_input=ple,
            past_key_values=None,
        )

    position_ids = cache_position.unsqueeze(0)
    kwargs = dict(position_ids=position_ids, cache_position=cache_position,
                  use_cache=False)
    if rotary_emb is not None:
        kwargs['position_embeddings'] = rotary_emb(h, position_ids=position_ids)
    return kwargs


# ---- per-layer quantization ----

def quantize_layer_e8_shell_rht(W, H, codebook, bpw=2, tune_iters=0,
                                apply_left=True, block_diagonal=True):
    """
    Quantize a single linear layer with E8 Shell + RHT + LDLQ.

    When ``apply_left=False`` the row-direction RHT is skipped (SU=1, no
    left FHT). Used by the embedding-quant path so each row stays
    independent and per-row dequant at lookup is trivial.

    ``block_diagonal`` controls whether non-pow2 dimensions are split into
    a sum of pow2 FHT blocks (saves storage but the per-row kernel must
    match) or padded to the next power of 2 (~simpler, small storage cost
    on most shapes).

    Returns:
        W_hat: dequantized weight matrix (for error propagation to next layer)
        artifacts: dict with Qidxs, SU, SV, Wscale (for saving)
        metrics: dict with sqnr, bpw
    """
    dev = codebook.device
    m, n = W.shape
    W_f = W.float().to(dev)
    H_f = H.float().to(dev)

    # E8P's tensor-core decode packs weights as (m_pad//16, n_pad//64, 8, 4), so the RHT
    # must keep n_pad a multiple of 64 (cols) and m_pad a multiple of 16 (rows). RHT(e8p=True)
    # uses a block-diagonal decomposition floored to those multiples — minimal padding instead
    # of padding the whole dim to the next power of 2 (e.g. gemma-4-31B "3bpw" 27GB -> ~11GB).
    is_e8p = getattr(codebook, 'is_e8p', False)

    # Dampen Hessian diagonal for numerical stability
    damp = 0.01 * torch.mean(torch.diag(H_f))
    diag = torch.arange(H_f.shape[-1], device=dev)
    H_f[diag, diag] += damp

    rht = RHT(m, n, device=dev, block_diagonal=block_diagonal, apply_left=apply_left, e8p=is_e8p)
    W_tilde = rht.transform_weights(W_f)
    H_tilde = rht.transform_hessian(H_f)

    n_tilde = W_tilde.shape[1]
    W_pad, _, _ = pad_to_multiple(W_tilde, 8)
    H_pad = pad_hessian(H_tilde, 8)

    def _run_ldlq(W_p, H_p):
        from .ldlq import quantize_ldlq_codebook_nstage
        if is_e8p:
            # E8P always uses the generic N-stage LDLQ so every bpw returns the
            # same (all_indices, cum_inv_rs) contract. RVQ recipe + fixed QuIP#
            # residual scales (validated to match the prototype PPL): 2=[E8P],
            # 3=[E8P, E81B] rs 2.04, 4=[E8P, E8P] rs 3.45.
            if bpw == 2:
                return quantize_ldlq_codebook_nstage(
                    W_p, H_p, codebooks=[codebook], resid_scales=[],
                    tune_iters=tune_iters)
            elif bpw == 3:
                from .codebook_e8p import E81BCodebook
                cb_e81b = E81BCodebook(device=dev, verbose=False)
                return quantize_ldlq_codebook_nstage(
                    W_p, H_p, codebooks=[codebook, cb_e81b], resid_scales=[2.04],
                    tune_iters=tune_iters)
            elif bpw == 4:
                return quantize_ldlq_codebook_nstage(
                    W_p, H_p, codebooks=[codebook, codebook], resid_scales=[3.45],
                    tune_iters=tune_iters)
            else:
                raise ValueError(f"e8p codebook supports bpw 2/3/4 only, got {bpw}")
        # ``resid_scale`` between two stages must be tuned for the actual
        # (primary, secondary) pair. ``codebook.resid_scale`` was tuned for
        # self-self (= 4/6/8 bpw uniform recipes). For 3/5/7 bpw recipes
        # the last stage uses ``make_small(256)``, which has a different
        # opt_scale and residual-error profile, so derive its rs directly
        # against that pair. Both ``cb_small`` and ``rs_to_small`` are
        # constants for the whole run, but ~10 ms × ~200 layers is
        # trivial so we recompute here rather than thread state.
        rs_self = codebook.resid_scale
        if bpw == 2:
            return quantize_ldlq_codebook(W_p, H_p, codebook, tune_iters=tune_iters)
        elif bpw == 3:
            cb_small = codebook.make_small(256)
            rs_to_small = codebook.compute_paired_resid_scale(cb_small)
            return quantize_ldlq_codebook_2stage(
                W_p, H_p, codebook, cb_small,
                resid_scale=rs_to_small, tune_iters=tune_iters)
        elif bpw == 4:
            return quantize_ldlq_codebook_2stage(
                W_p, H_p, codebook, codebook,
                resid_scale=rs_self, tune_iters=tune_iters)
        elif bpw == 5:
            cb_small = codebook.make_small(256)
            rs_to_small = codebook.compute_paired_resid_scale(cb_small)
            return quantize_ldlq_codebook_nstage(
                W_p, H_p,
                codebooks=[codebook, codebook, cb_small],
                resid_scales=[rs_self, rs_to_small],
                tune_iters=tune_iters)
        elif bpw == 6:
            return quantize_ldlq_codebook_nstage(
                W_p, H_p,
                codebooks=[codebook, codebook, codebook],
                resid_scales=[rs_self, rs_self],
                tune_iters=tune_iters)
        elif bpw == 7:
            cb_small = codebook.make_small(256)
            rs_to_small = codebook.compute_paired_resid_scale(cb_small)
            return quantize_ldlq_codebook_nstage(
                W_p, H_p,
                codebooks=[codebook, codebook, codebook, cb_small],
                resid_scales=[rs_self, rs_self, rs_to_small],
                tune_iters=tune_iters)
        elif bpw == 8:
            return quantize_ldlq_codebook_nstage(
                W_p, H_p,
                codebooks=[codebook] * 4,
                resid_scales=[rs_self] * 3,
                tune_iters=tune_iters)
        else:
            raise ValueError(f"Unsupported bpw: {bpw}. Use 2, 3, 4, 5, 6, 7, or 8.")

    # 3-tier Cholesky error handling for sparse/degenerate Hessians
    # (e.g. MoE experts with few calibration tokens)
    try:
        result = _run_ldlq(W_pad, H_pad)
    except torch._C._LinAlgError:
        heavy_damp = 0.1 * torch.mean(torch.diag(H_pad)).clamp(min=1e-6)
        diag_idx = torch.arange(H_pad.shape[0], device=dev)
        H_pad[diag_idx, diag_idx] += heavy_damp
        try:
            result = _run_ldlq(W_pad, H_pad)
        except torch._C._LinAlgError:
            H_pad = torch.eye(H_pad.shape[0], device=dev, dtype=torch.float32)
            result = _run_ldlq(W_pad, H_pad)

    # Dequantize for error propagation
    W_hat_tilde = result['W_hat'][:, :n_tilde]
    W_hat = rht.inverse_transform_weights(W_hat_tilde)

    # SQNR
    diff = W_f - W_hat
    sqnr = 10 * torch.log10(W_f.pow(2).sum() / diff.pow(2).sum().clamp(min=1e-20)).item()

    # Artifacts for saving (these go into safetensors)
    artifacts = {
        'SU': rht.su.half(),                           # (m_pad,)
        'SV': rht.sv.half(),                           # (n_pad,)
        'Wscale': torch.tensor(result['Wscale'], dtype=torch.float32),  # scalar
    }

    if is_e8p:
        # E8P stores int64 tensor-core-packed weights (not the shell's int16 Qidxs):
        #   Qidxs_e8p   (m_pad//16, n_pad//64, 8, 4)  — stage-0 E8P, for decode_matvec_e8p
        #   Qidxs2_e8p  (same)                        — 4bpw stage-1 (E8P residual)
        #   Qidxs2_e81b (m_pad, n_pad//64)            — 3bpw stage-1 (E81B residual)
        all_idx = result['all_indices']
        cum_inv_rs = result['cum_inv_rs']
        m_pad = all_idx[0].shape[0]
        n_pad = all_idx[0].shape[1] * 8
        artifacts['Qidxs_e8p'] = codebook.maybe_pack_idxs(
            all_idx[0].to(torch.int64)).view(m_pad // 16, n_pad // 64, 8, 4).contiguous()
        if len(all_idx) > 1:
            artifacts['inv_resid_scale'] = torch.tensor(cum_inv_rs[1], dtype=torch.float32)
            if bpw == 4:
                artifacts['Qidxs2_e8p'] = codebook.maybe_pack_idxs(
                    all_idx[1].to(torch.int64)).view(m_pad // 16, n_pad // 64, 8, 4).contiguous()
            else:  # bpw == 3 → E81B residual stage
                from .codebook_e8p import E81BCodebook
                artifacts['Qidxs2_e81b'] = E81BCodebook.pack_e81b(all_idx[1].to(torch.int64))
    elif bpw == 2:
        artifacts['Qidxs'] = result['indices'].to(torch.int16)
        # Don't store Qidxs2/inv_resid_scale at 2bpw — they're all zeros.
        # E8RHTLinear._load_from_state_dict handles missing keys.
    elif bpw <= 4:
        artifacts['Qidxs'] = result['indices1'].to(torch.int16)
        artifacts['Qidxs2'] = result['indices2'].to(torch.int16)
        artifacts['inv_resid_scale'] = torch.tensor(
            1.0 / result['resid_scale'], dtype=torch.float32)
    else:
        # N-stage (bpw 5-8): store all_indices and cumulative inv_resid_scales
        all_idx = result['all_indices']
        cum_inv_rs = result['cum_inv_rs']
        artifacts['Qidxs'] = all_idx[0].to(torch.int16)
        for i in range(1, len(all_idx)):
            suffix = f'Qidxs{i + 1}'
            artifacts[suffix] = all_idx[i].to(torch.int16)
        # Store cumulative inverse resid_scales (stage 2 onward)
        for i in range(1, len(cum_inv_rs)):
            suffix = 'inv_resid_scale' if i == 1 else f'inv_resid_scale{i}'
            artifacts[suffix] = torch.tensor(cum_inv_rs[i], dtype=torch.float32)

    metrics = {'sqnr': sqnr, 'bpw': result['bpw'], 'Wscale': result['Wscale'],
               'proxy_loss': result['proxy_loss']}
    return W_hat, artifacts, metrics


# ---- main quantization pipeline ----

def _load_layer_state(weight_map, shard_paths, layer_idx, sd_prefix):
    """Load all tensors for a single layer from sharded safetensors.

    Handles FP8 quantized weights (e.g., Mistral/Devstral models):
    if a weight is float8_e4m3fn with a corresponding weight_scale_inv,
    dequantizes to bfloat16 in-place.
    """
    from collections import defaultdict
    from safetensors import safe_open

    prefix = f"{sd_prefix}.{layer_idx}."
    layer_keys = [k for k in weight_map if k.startswith(prefix)]

    state = {}
    shard_to_keys = defaultdict(list)
    for key in layer_keys:
        shard_to_keys[weight_map[key]].append(key)

    for shard, keys in shard_to_keys.items():
        with safe_open(shard_paths[shard], framework="pt") as f:
            for key in keys:
                local_key = key[len(prefix):]
                state[local_key] = f.get_tensor(key)

    # Dequantize FP8 weights: weight_bf16 = weight_fp8.to(bf16) * weight_scale_inv
    fp8_keys = [k for k in state if k.endswith('.weight')
                and state[k].dtype == torch.float8_e4m3fn]
    for wkey in fp8_keys:
        scale_key = wkey.replace('.weight', '.weight_scale_inv')
        if scale_key in state:
            w_fp8 = state[wkey]
            scale = state[scale_key]
            state[wkey] = (w_fp8.to(torch.bfloat16) * scale).to(torch.bfloat16)
            del state[scale_key]

    # Remove activation_scale keys (not needed for weight quantization)
    drop = [k for k in state if k.endswith('.activation_scale')]
    for k in drop:
        del state[k]

    return state


def _load_tensor_from_shards(weight_map, shard_paths, key):
    """Load a single tensor from sharded safetensors."""
    from safetensors import safe_open
    shard = weight_map[key]
    with safe_open(shard_paths[shard], framework="pt") as f:
        return f.get_tensor(key)


def _download_snapshot(model_id):
    """Download model and return snapshot directory path.

    If ``model_id`` is already a local directory (e.g. a checkpoint fetched
    via ``hf download --local-dir``), return it as-is instead of passing it to
    ``snapshot_download`` (which rejects non-repo-id paths).
    """
    import os
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id)


def quantize(
    model_name: str,
    output_dir: str,
    bpw=2,
    min_bpw: int = None,
    max_bpw: int = None,
    tune_iters: int = 0,
    nsamples: int = 128,
    seqlen: int = 2048,
    device: str = "cuda",
    dtype=torch.bfloat16,
    trust_remote_code: bool = False,
    streaming: bool = False,
    workers: int = 0,
    codebook_size: int = None,
    codebook_type: str = "e8_shell",
):
    """
    Quantize a HuggingFace model with GLQ and save to output_dir.

    Args:
        model_name: HF model ID or local path
        output_dir: where to save quantized model
        bpw: bits per weight. int (2,3,4) for uniform, float (e.g. 2.5)
            for mixed-precision auto-allocation, or dict for explicit
            per-layer assignment {layer_prefix: bpw}.
        tune_iters: LDLQ refinement passes
        nsamples: calibration samples from WikiText-2
        seqlen: calibration sequence length
        device: 'cuda' or 'cpu'
        dtype: model dtype for loading
        trust_remote_code: allow custom model code from HF Hub
        streaming: load weights from safetensors one layer at a time
            (required for models that exceed system RAM)
        workers: parallel workers for CPU quantization
            (0=auto, 1=sequential, ignored on GPU)
    """
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    os.makedirs(output_dir, exist_ok=True)

    # Parse bpw: int (uniform), float (mixed-precision target), or dict (explicit)
    # Mixed-precision is triggered by: fractional bpw, explicit map, or min/max range
    has_range = (min_bpw is not None or max_bpw is not None)
    if isinstance(bpw, dict):
        bpw_map = bpw
        mixed_precision = True
        avg_target = None
        print(f"GLQ Quantization: {model_name} -> {output_dir}")
        print(f"  mixed-precision (explicit map), tune_iters={tune_iters}, nsamples={nsamples}")
    elif has_range or (isinstance(bpw, float) and not float(bpw).is_integer()):
        avg_target = float(bpw)
        min_bpw = min_bpw if min_bpw is not None else 2
        max_bpw = max_bpw if max_bpw is not None else 4
        mixed_precision = True
        bpw_map = None  # will be computed after profiling pass
        print(f"GLQ Quantization: {model_name} -> {output_dir}")
        print(f"  target avg bpw={avg_target}, range [{min_bpw}, {max_bpw}], "
              f"tune_iters={tune_iters}, nsamples={nsamples}")
    else:
        bpw = int(bpw)
        mixed_precision = False
        bpw_map = None
        avg_target = None
        print(f"GLQ Quantization: {model_name} -> {output_dir}")
        print(f"  bpw={bpw}, tune_iters={tune_iters}, nsamples={nsamples}")
    if nsamples < 64:
        import warnings
        warnings.warn(
            f"nsamples={nsamples} is low — Hessian estimates may be noisy, "
            f"degrading quantization quality. Use nsamples>=128 for best results.",
            UserWarning,
        )
    if streaming:
        print(f"  streaming=True (layer-by-layer safetensors loading)")

    # ---- Load config and detect profile ----
    cfg = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)
    profile = _detect_profile(cfg)
    arch = cfg.architectures[0] if cfg.architectures else ""

    # Apply config fixups (e.g. NemotronH needs _attn_implementation="eager")
    if profile.get('forward_kwargs') == 'nemotron_h':
        cfg._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)

    # ---- Load model or prepare streaming ----
    print(f"\nLoading model ...")
    t0 = time.perf_counter()

    model = None
    text_model = None
    weight_map = None
    shard_paths = None
    BlockClass = None
    _rotary_cls = None  # generic-streaming rotary class (built off-meta in the rotary section)
    is_mistral3 = "Mistral3" in arch  # Mistral3ForConditionalGeneration (multimodal 24B)
    # Pure-text Ministral3 (123B Devstral-2) uses its own LM class but the same
    # Ministral3RotaryEmbedding as the multimodal variant. Careful: "Mistral3"
    # is NOT a substring of "Ministral3" (the latter has an extra 'n').
    is_ministral3_text = arch == "Ministral3ForCausalLM"
    # Gemma 4 is multimodal (Gemma4ForConditionalGeneration, and the newer
    # Gemma4UnifiedForConditionalGeneration used by gemma-4-12B-it). The
    # language decoder lives at `model.language_model` and saved keys are
    # prefixed `model.language_model.layers.…`. We quantize the text decoder
    # only. Both share the same module layout; the only differences the
    # streaming path cares about are the model + rotary class names.
    is_gemma4 = "Gemma4" in arch and "ForConditionalGeneration" in arch
    is_gemma4_unified = "Gemma4Unified" in arch

    if streaming:
        # Streaming mode: don't load model into RAM. Instead, instantiate
        # on meta device to discover the block class, then load weights
        # one layer at a time from safetensors.
        if is_mistral3:
            from transformers import Mistral3ForConditionalGeneration
            with torch.device("meta"):
                _model = Mistral3ForConditionalGeneration(cfg)
            _text = _model.model.language_model
            _layers = get_decoder_layers(_text, profile)
            sd_prefix = "language_model.model.layers"
        elif is_gemma4:
            import transformers as _tf
            # arch is 'Gemma4ForConditionalGeneration' or
            # 'Gemma4UnifiedForConditionalGeneration' — both are exported at the
            # transformers top level and both expose model.language_model.
            gemma_cls = getattr(_tf, arch)
            with torch.device("meta"):
                _model = gemma_cls(cfg)
            _text = _model.model.language_model
            _layers = list(_text.layers)
            sd_prefix = "model.language_model.layers"
        else:
            with torch.device("meta"):
                _model = AutoModelForCausalLM.from_config(
                    cfg, trust_remote_code=trust_remote_code, dtype=dtype)
            if profile.get('layers_attr'):
                _layers = _resolve_attr(_model, profile['layers_attr'])
            else:
                _layers = get_decoder_layers(_model, profile)
            # Capture the rotary class so the calibration forward can build a
            # real (off-meta) rotary later. Attentions that hard-unpack
            # position_embeddings (e.g. sarvam_moe) require it; the meta model's
            # rotary has meta-tensor inv_freq, so we re-instantiate by class.
            if profile.get('rotary_attr'):
                try:
                    _rotary_cls = type(_resolve_attr(_model, profile['rotary_attr']))
                except (AttributeError, KeyError):
                    _rotary_cls = None
        BlockClass = type(_layers[0])
        n_layers = len(_layers)
        del _model, _layers

        # Download model (if needed) and build weight map
        model_dir = _download_snapshot(model_name)
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            shard_files = sorted(set(weight_map.values()))
            shard_paths = {s: os.path.join(model_dir, s) for s in shard_files}
        elif os.path.exists(single_path):
            from safetensors import safe_open
            with safe_open(single_path, framework="pt") as sf:
                weight_map = {k: "model.safetensors" for k in sf.keys()}
            shard_paths = {"model.safetensors": single_path}
        else:
            raise FileNotFoundError(f"No safetensors found in {model_dir}")
        print(f"  Streaming from {model_dir} ({len(weight_map)} tensors)")
    else:
        # Standard mode: load full model to CPU
        if "Mistral3" in arch:
            from transformers import Mistral3ForConditionalGeneration
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            text_model = model.model.language_model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            text_model = model

        n_params_f = sum(p.numel() for p in text_model.parameters()) / 1e9
        print(f"  {n_params_f:.2f}B params in {time.perf_counter() - t0:.1f}s")

    # ---- Calibration data ----
    print(f"\nLoading calibration data ...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    n_chunks = min(nsamples, input_ids.shape[0] // seqlen)
    calib_ids = input_ids[:n_chunks * seqlen].reshape(n_chunks, seqlen)
    print(f"  {n_chunks} sequences of length {seqlen}")

    # ---- Build codebook ----
    print(f"\nBuilding {codebook_type} codebook ...")
    if codebook_type == "e8_relaxed":
        from .codebook_relaxed import E8RelaxedCodebook
        codebook = E8RelaxedCodebook(device=device, target_size=codebook_size)
    elif codebook_type == "e8p":
        from .codebook_e8p import E8PCodebook
        codebook = E8PCodebook(device=device)         # QuIP# padded-D̂8, TC-GEMV decode (2/3/4 bpw)
    else:
        codebook = E8ShellCodebook(device=device, target_size=codebook_size)

    # ---- Setup for layer-by-layer quantization ----
    use_gpu = str(device).startswith("cuda")
    # Mistral3 / Gemma4 streaming overrides sd_prefix in the streaming block above
    if not (streaming and (is_mistral3 or is_gemma4)):
        sd_prefix = profile['sd_prefix']
    rotary_emb = None

    if streaming:
        # Load embedding from safetensors. Auto-detect the checkpoint key layout
        # from the weight_map instead of pinning a transformers version:
        # transformers <5.10 saved the Mistral3 text decoder under
        # ``language_model.model.*``; >=5.10 (and gemma4) use
        # ``model.language_model.*``. Deriving embed_key + sd_prefix from the
        # actual keys keeps streaming robust across both layouts.
        if is_mistral3 or is_gemma4:
            _ek = next((k for k in weight_map
                        if k.endswith("embed_tokens.weight")
                        and "vision" not in k and "per_layer" not in k), None)
            embed_key = _ek or (
                "model.language_model.embed_tokens.weight" if is_gemma4
                else "language_model.model.embed_tokens.weight")
            # Target the LANGUAGE-MODEL decoder layers specifically. Multimodal
            # Gemma-4 / Mistral3 also expose ``*_tower.layers.0.*`` (vision AND
            # audio encoders), so matching a bare ``.layers.0.`` would pick a
            # tower prefix (e.g. ``model.audio_tower.layers``) and quantize the
            # wrong stack. The text decoder key always contains "language_model"
            # (``model.language_model.layers.*`` on tf>=5.10, ``language_model.
            # model.layers.*`` on tf<5.10), so key on that.
            _lk = next((k for k in weight_map
                        if ".layers.0." in k and "language_model" in k), None)
            if _lk:
                sd_prefix = _lk.split(".layers.")[0] + ".layers"
        else:
            embed_attr = profile.get('embed_attr') or 'model.embed_tokens'
            embed_key = f"{embed_attr}.weight"
        embed_weight = _load_tensor_from_shards(weight_map, shard_paths, embed_key)
        vocab_size = embed_weight.shape[0]
        hidden_size = embed_weight.shape[1]
        embed = nn.Embedding(vocab_size, hidden_size, _weight=embed_weight)
        if use_gpu:
            embed.to(device)

        # Create rotary embedding for streaming mode
        if is_mistral3 or is_ministral3_text:
            # Mistral3ForConditionalGeneration exposes its text config under
            # cfg.text_config; Ministral3ForCausalLM (text-only) uses cfg directly.
            text_cfg = cfg.text_config if is_mistral3 else cfg
            from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding
            rotary_emb = Ministral3RotaryEmbedding(config=text_cfg)
            if use_gpu:
                rotary_emb.to(device)
        elif is_gemma4:
            # Gemma4 text decoder needs a Gemma4-specific rotary embedding
            # (per-layer sliding/full type). The unified variant ships its own
            # copy under modeling_gemma4_unified.
            if is_gemma4_unified:
                from transformers.models.gemma4_unified.modeling_gemma4_unified import (
                    Gemma4UnifiedTextRotaryEmbedding as _GemmaRotary)
            else:
                from transformers.models.gemma4.modeling_gemma4 import (
                    Gemma4TextRotaryEmbedding as _GemmaRotary)
            rotary_emb = _GemmaRotary(config=cfg.text_config)
            if use_gpu:
                rotary_emb.to(device)
        elif _rotary_cls is not None:
            # Generic custom arch (e.g. sarvam_moe): build the rotary off-meta
            # from the class captured during meta discovery, so the calibration
            # forward can pass position_embeddings to attentions that require them
            # (sarvam attention hard-unpacks `cos, sin = position_embeddings`).
            rotary_emb = _rotary_cls(config=cfg)
            if use_gpu:
                rotary_emb.to(device)
    else:
        decoder_layers = get_decoder_layers(text_model, profile)
        embed = get_embed(text_model, profile)
        rotary_emb = get_rotary_emb(text_model, profile)
        n_layers = len(decoder_layers)
        if use_gpu:
            embed.to(device)
            if rotary_emb is not None:
                rotary_emb.to(device)

    # Embed calibration data
    print(f"\nEmbedding calibration data ...")
    hidden_states = []
    with torch.no_grad():
        for i in range(calib_ids.shape[0]):
            hidden_states.append(embed(calib_ids[i:i+1].to(device)))
    # Match the calibration activations to the compute dtype: streaming loads
    # the raw embedding (fp32 for fp32-stored checkpoints like sarvam_moe) while
    # the decoder layers are cast to `dtype` — without this the first layer's
    # matmul hits a float-vs-bf16 dtype mismatch. No-op for bf16/fp16 models.
    hidden_states = torch.cat(hidden_states, dim=0).to(dtype)

    # ---- Gemma-4 Per-Layer Embedding (PLE) precomputation ----
    # E2B / E4B have hidden_size_per_layer_input > 0 and feed each decoder
    # layer a per-layer slice of an embedding tensor that depends on
    # input_ids and inputs_embeds. The 31B variant has it set to 0 so this
    # block is a no-op there. We compute the full
    # [n_samples, T, num_layers, ple_dim] tensor once and slice per-layer in
    # _build_forward_kwargs.
    per_layer_inputs = None
    if is_gemma4 and getattr(cfg.text_config, "hidden_size_per_layer_input", 0) > 0:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
        text_cfg = cfg.text_config
        ple_dim = text_cfg.hidden_size_per_layer_input
        n_layers_cfg = text_cfg.num_hidden_layers
        # Load the three top-of-language_model PLE submodules from safetensors.
        ptl_w = _load_tensor_from_shards(
            weight_map, shard_paths,
            "model.language_model.embed_tokens_per_layer.weight")
        # Gemma4TextScaledWordEmbedding multiplies by sqrt(embed_dim) on lookup.
        # The weight dim is num_layers * ple_dim.
        embed_per_layer = nn.Embedding(
            ptl_w.shape[0], ptl_w.shape[1], _weight=ptl_w).to(device)
        embed_per_layer_scale = ple_dim ** 0.5

        proj_w = _load_tensor_from_shards(
            weight_map, shard_paths,
            "model.language_model.per_layer_model_projection.weight")
        per_layer_proj = nn.Linear(
            proj_w.shape[1], proj_w.shape[0], bias=False).to(device)
        per_layer_proj.weight.data = proj_w.to(device)

        norm_w = _load_tensor_from_shards(
            weight_map, shard_paths,
            "model.language_model.per_layer_projection_norm.weight")
        per_layer_norm = Gemma4RMSNorm(ple_dim, eps=text_cfg.rms_norm_eps).to(device)
        per_layer_norm.weight.data = norm_w.to(device)

        # Per modeling_gemma4: scales used to combine token-identity + context.
        proj_scale = text_cfg.hidden_size ** -0.5  # 1/sqrt(hidden_size)
        input_scale = 2.0 ** -0.5  # 1/sqrt(2) — half of (token + context)

        per_layer_inputs_chunks = []
        with torch.no_grad():
            for i in range(calib_ids.shape[0]):
                ids = calib_ids[i:i+1].to(device)
                emb_in = hidden_states[i:i+1]
                # token-identity component
                ple_token = embed_per_layer(ids) * embed_per_layer_scale
                ple_token = ple_token.reshape(
                    *ids.shape, n_layers_cfg, ple_dim)
                # context component (depends on inputs_embeds)
                ple_ctx = per_layer_proj(emb_in) * proj_scale
                ple_ctx = ple_ctx.reshape(
                    *emb_in.shape[:-1], n_layers_cfg, ple_dim)
                ple_ctx = per_layer_norm(ple_ctx)
                ple = (ple_ctx + ple_token) * input_scale
                per_layer_inputs_chunks.append(ple)
        per_layer_inputs = torch.cat(per_layer_inputs_chunks, dim=0)
        print(f"  PLE precomputed: {tuple(per_layer_inputs.shape)} "
              f"(n_samples × T × num_layers × ple_dim)")
        del embed_per_layer, per_layer_proj, per_layer_norm, ptl_w, proj_w, norm_w
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    # ---- Gemma-4 KV-sharing cache ----
    # E2B / E4B have num_kv_shared_layers > 0: layers from
    # first_kv_shared_layer_idx onward are READERS that look up K/V from an
    # earlier same-type PRODUCER layer via shared_kv_states[producer_idx].
    # We populate this cache during each producer layer's post-quantization
    # propagation forward, then replay into reader-layer forwards.
    shared_kv_cache = None
    producer_layer_indices = set()
    if is_gemma4 and getattr(cfg.text_config, "num_kv_shared_layers", 0) > 0:
        text_cfg = cfg.text_config
        first_kv_shared = text_cfg.num_hidden_layers - text_cfg.num_kv_shared_layers
        # Producers' kv_shared_layer_index is the most recent earlier same-type
        # layer. Replicate Gemma4TextAttention.__init__ logic to find them.
        for reader_idx in range(first_kv_shared, text_cfg.num_hidden_layers):
            prev_layers = text_cfg.layer_types[:reader_idx]
            if not prev_layers:
                continue
            same_type = text_cfg.layer_types[reader_idx]
            # last index of same_type in prev_layers
            try:
                producer_idx = (len(prev_layers) - 1
                                - prev_layers[::-1].index(same_type))
            except ValueError:
                continue
            producer_layer_indices.add(producer_idx)
        shared_kv_cache = {}  # {producer_idx: list[(K, V) per sample]}
        print(f"  KV-sharing: {len(producer_layer_indices)} producer layers, "
              f"{text_cfg.num_kv_shared_layers} reader layers")

    if streaming:
        del embed, embed_weight
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    all_artifacts = {}  # layer_name -> {Qidxs, SU, SV, Wscale}
    all_sqnr = []
    all_proxy_losses = {}  # layer_prefix -> proxy_loss (for sensitivity profiling)
    # Gemma-4 MoE: original stacked expert tensors (experts.gate_up_proj /
    # experts.down_proj) are replaced by per-expert GLQ artifacts, so they must
    # NOT be re-streamed verbatim into the saved checkpoint (would double size +
    # confuse the loader). Full source keys collected here are skipped at save.
    dropped_stacked_keys: set[str] = set()
    t_start = time.perf_counter()

    # ---- Set up parallel worker pool for CPU quantization ----
    pool = None
    if not use_gpu and workers != 1:
        import multiprocessing
        n_workers = workers if workers > 0 else min(os.cpu_count() or 1, 16)
        n_threads = max(1, (os.cpu_count() or 1) // n_workers)
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,
            initargs=(codebook.codebook.cpu(), codebook.opt_scale,
                      codebook.resid_scale, n_threads),
        )
        print(f"  Using {n_workers} parallel workers for CPU quantization")

    for layer_idx in range(n_layers):
        t_layer = time.perf_counter()
        print(f"\n--- Layer {layer_idx}/{n_layers-1} ---")

        if streaming:
            layer_state = _load_layer_state(
                weight_map, shard_paths, layer_idx, sd_prefix)
            layer_cfg = cfg.text_config if (is_mistral3 or is_gemma4) else cfg
            layer = BlockClass(layer_cfg, layer_idx)
            layer.load_state_dict(layer_state, strict=False)
            del layer_state
            if use_gpu:
                layer.to(device, dtype=dtype)
            layer.eval()
        else:
            layer = decoder_layers[layer_idx]
            if use_gpu:
                layer.to(device)

        # Collect linear sublayers
        linears = {}
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                linears[name] = mod

        # Gemma-4 MoE: the routed experts live in a Gemma4TextExperts module as
        # stacked 3D nn.Parameters (gate_up_proj [E,2I,H], down_proj [E,H,I]), so
        # the nn.Linear walk above misses them. Detect those modules and install a
        # forward hook to capture per-expert input Hessians during calibration
        # (gate+up share the routed input; down sees act(gate)*up). Also drop the
        # tiny router projection from quantization — it's routing-critical and
        # negligible in size, so we keep it bf16.
        gemma4_experts = []        # list of (module_name, experts_module)
        moe_expert_hessians = {}   # virtual_name -> accumulated input Gram (GPU)
        moe_hooks = []
        if is_gemma4:
            linears = {n: m for n, m in linears.items()
                       if not n.endswith('router.proj')}
            for mn, mod in layer.named_modules():
                gup = getattr(mod, 'gate_up_proj', None)
                if isinstance(gup, nn.Parameter) and gup.dim() == 3:
                    gemma4_experts.append((mn, mod))

            def _make_expert_hook(mn, emod):
                n_exp = emod.gate_up_proj.shape[0]

                def _hook(module, args, kwargs, output):
                    hs = kwargs.get('hidden_states', args[0] if args else None)
                    ti = kwargs.get('top_k_index',
                                    args[1] if len(args) > 1 else None)
                    if hs is None or ti is None:
                        return
                    hs = hs.reshape(-1, hs.shape[-1])
                    ti = ti.reshape(-1, ti.shape[-1])
                    for e in range(n_exp):
                        sel = (ti == e).any(dim=-1)
                        if not bool(sel.any()):
                            continue
                        xs = hs[sel]
                        xf = xs.float()
                        gk = f"{mn}.{e}.gate_up_proj"
                        moe_expert_hessians[gk] = (
                            moe_expert_hessians.get(gk, 0) + xf.t() @ xf)
                        with torch.no_grad():
                            go = nn.functional.linear(
                                xs.to(emod.gate_up_proj.dtype),
                                emod.gate_up_proj[e])
                            g, u = go.chunk(2, dim=-1)
                            di = (emod.act_fn(g) * u).float()
                        dk = f"{mn}.{e}.down_proj"
                        moe_expert_hessians[dk] = (
                            moe_expert_hessians.get(dk, 0) + di.t() @ di)
                return _hook

            for mn, emod in gemma4_experts:
                moe_hooks.append(emod.register_forward_hook(
                    _make_expert_hook(mn, emod), with_kwargs=True))

        # Install Hessian hooks
        captures = {}
        for name, mod in linears.items():
            captures[name] = HessianCapture(mod)

        # Forward calibration (hooks capture inputs)
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1]
                kwargs = _build_forward_kwargs(
                    profile, h, rotary_emb, layer_idx=layer_idx, cfg=cfg,
                    per_layer_inputs=per_layer_inputs, sample_idx=i,
                    shared_kv_cache=shared_kv_cache)
                layer(h, **kwargs)

        # Finalize Hessians to CPU to free GPU for quantization
        hessians = {}
        for name, cap in captures.items():
            H = cap.finalize()
            if H is not None:
                hessians[name] = H.cpu()
                del H
        del captures

        # Gemma-4 MoE: tear down the expert hooks, fold their per-expert Grams
        # into `hessians`, and expand each stacked experts module into per-expert
        # virtual nn.Linears so the standard quant loop handles them. The fused
        # gate_up matrix (already [gate; up]) is row-split at store time into
        # gate_proj/up_proj (one shared SV/Wscale) — vLLM's expert mapping reads
        # those back as w13. Writeback puts the dequant into the 3D slice so the
        # post-quant forward reflects quantization; original stacked tensors are
        # dropped from the save.
        for h in moe_hooks:
            h.remove()
        for k, H in moe_expert_hessians.items():
            hessians[k] = H.cpu()
        del moe_expert_hessians
        moe_writeback = {}   # virtual_name -> (param3d, expert_idx)
        moe_gate_up = {}     # gate_up virtual_name -> (gate_name, up_name, inter)
        for mn, emod in gemma4_experts:
            gup, dwn = emod.gate_up_proj, emod.down_proj
            n_exp, two_inter, hdim = gup.shape
            inter = two_inter // 2
            for e in range(n_exp):
                gu, dn = f"{mn}.{e}.gate_up_proj", f"{mn}.{e}.down_proj"
                vgu = nn.Linear(hdim, two_inter, bias=False,
                                device=gup.device, dtype=gup.dtype)
                vgu.weight.data = gup.data[e].clone()
                vdn = nn.Linear(dwn.shape[2], dwn.shape[1], bias=False,
                                device=dwn.device, dtype=dwn.dtype)
                vdn.weight.data = dwn.data[e].clone()
                linears[gu] = vgu
                linears[dn] = vdn
                moe_writeback[gu] = (gup, e)
                moe_writeback[dn] = (dwn, e)
                moe_gate_up[gu] = (f"{mn}.{e}.gate_proj", f"{mn}.{e}.up_proj", inter)
            dropped_stacked_keys.add(f"{sd_prefix}.{layer_idx}.{mn}.gate_up_proj")
            dropped_stacked_keys.add(f"{sd_prefix}.{layer_idx}.{mn}.down_proj")

        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        # ---- Jointly quantize routed-expert gate+up, store split ----
        # vLLM's FusedMoE stacks gate+up into w13 with ONE per-expert Wscale.
        # gate_proj/up_proj quantized SEPARATELY get different Wscale (the
        # magnitude lives entirely in the scalar Wscale; SU is ±1 signs), so
        # they can't share a single w13 scale. Quantizing the concatenated
        # [gate; up] matrix JOINTLY yields one Wscale (and one SU/SV) for the
        # whole w13. We then store the result split back under the original
        # gate_proj/up_proj keys (Qidxs/Qidxs2/SU row-split; SV/Wscale/
        # inv_resid_scale shared -> duplicated) so vLLM's native expert mapping
        # (gate->w1, up->w3) reassembles an exact w13 with no model patch.
        # Routed experts only; the dense (first_k_dense) MLP and the shared
        # expert keep the per-shard merged-linear path.
        gate_up_fusion = {}  # fused_name -> (gate_mod, up_mod, gate_out, g_name, u_name)
        if profile.get('fuse_expert_gate_up'):
            pairs = []
            for name in list(linears):
                if not name.endswith('.gate_proj'):
                    continue
                if '.experts.' not in name or 'shared' in name:
                    continue  # routed experts only
                up_name = name[:-len('.gate_proj')] + '.up_proj'
                if up_name in linears:
                    pairs.append((name, up_name))
            for gate_name, up_name in pairs:
                fused_name = gate_name[:-len('.gate_proj')] + '.gate_up_proj'
                real_gate, real_up = linears[gate_name], linears[up_name]
                gate_out = real_gate.weight.data.shape[0]
                fused = nn.Linear(
                    real_gate.in_features,
                    gate_out + real_up.weight.data.shape[0],
                    bias=False, device=real_gate.weight.device,
                    dtype=real_gate.weight.dtype)
                with torch.no_grad():
                    fused.weight.copy_(torch.cat(
                        [real_gate.weight.data, real_up.weight.data], dim=0))
                linears[fused_name] = fused
                # gate & up share the expert input -> identical Hessian; reuse
                # whichever was captured (un-routed experts -> identity fallback).
                H = hessians.pop(gate_name, None)
                H_up = hessians.pop(up_name, None)
                if H is None:
                    H = H_up
                if H is not None:
                    hessians[fused_name] = H
                del linears[gate_name], linears[up_name]
                gate_up_fusion[fused_name] = (
                    real_gate, real_up, gate_out, gate_name, up_name)
            if pairs:
                print(f"  joint-quant {len(pairs)} expert gate+up (one Wscale), "
                      f"stored split as gate_proj/up_proj")

        def _writeback(name, W_hat):
            """Store the dequantized weight back into the live module(s) so the
            post-quant calibration forward reflects quantization. A fused
            gate_up_proj splits back into its real gate/up modules."""
            if name in moe_writeback:
                param3d, e = moe_writeback[name]
                param3d.data[e] = W_hat.to(dtype=param3d.dtype,
                                           device=param3d.device)
                return
            if name in gate_up_fusion:
                real_gate, real_up, gate_out, _, _ = gate_up_fusion[name]
                real_gate.weight.data = W_hat[:gate_out].to(
                    dtype=real_gate.weight.dtype, device=real_gate.weight.device)
                real_up.weight.data = W_hat[gate_out:gate_out + real_up.weight.data.shape[0]].to(
                    dtype=real_up.weight.dtype, device=real_up.weight.device)
            else:
                mod = linears[name]
                mod.weight.data = W_hat.to(dtype=mod.weight.dtype,
                                           device=mod.weight.device)

        # Output-row-indexed artifacts get row-split between gate/up; the rest
        # (input-side SV, scalar Wscale/inv_resid_scale) are shared -> duplicated.
        _ROW_ARTS = {'Qidxs', 'Qidxs2', 'Qidxs3', 'SU'}
        _SHARED_ARTS = {'SV', 'Wscale', 'inv_resid_scale', 'inv_resid_scale2'}

        def _store_artifacts(name, arts):
            """Record quantized artifacts under their checkpoint prefix(es).
            A jointly-quantized gate_up result is split row-wise back into the
            gate_proj/up_proj prefixes vLLM's expert mapping expects."""
            if name in moe_gate_up:
                # Gemma-4 routed expert: the stacked gate_up matrix is already
                # [gate; up]; row-split into per-expert gate_proj/up_proj (dup the
                # shared SV/Wscale) so vLLM's FusedMoE w13 mapping reassembles it.
                gate_name, up_name, inter = moe_gate_up[name]
                gate_arts, up_arts = {}, {}
                for k, v in arts.items():
                    if k in _ROW_ARTS:
                        gate_arts[k] = v[:inter].clone()
                        up_arts[k] = v[inter:2 * inter].clone()
                    elif k in _SHARED_ARTS:
                        gate_arts[k] = v.clone()
                        up_arts[k] = v.clone()
                    else:
                        raise ValueError(
                            f"gemma4 expert gate_up split: unhandled artifact "
                            f"key {k!r}; add it to _ROW_ARTS or _SHARED_ARTS")
                all_artifacts[f"{sd_prefix}.{layer_idx}.{gate_name}"] = gate_arts
                all_artifacts[f"{sd_prefix}.{layer_idx}.{up_name}"] = up_arts
                return
            if name not in gate_up_fusion:
                all_artifacts[f"{sd_prefix}.{layer_idx}.{name}"] = arts
                return
            real_gate, real_up, gate_out, gate_name, up_name = gate_up_fusion[name]
            up_out = real_up.weight.data.shape[0]
            gate_arts, up_arts = {}, {}
            for k, v in arts.items():
                if k in _ROW_ARTS:
                    gate_arts[k] = v[:gate_out].clone()
                    up_arts[k] = v[gate_out:gate_out + up_out].clone()
                elif k in _SHARED_ARTS:
                    gate_arts[k] = v.clone()
                    up_arts[k] = v.clone()
                else:
                    raise ValueError(
                        f"gate_up split: unhandled artifact key {k!r}; add it to "
                        f"_ROW_ARTS or _SHARED_ARTS in _store_artifacts")
            all_artifacts[f"{sd_prefix}.{layer_idx}.{gate_name}"] = gate_arts
            all_artifacts[f"{sd_prefix}.{layer_idx}.{up_name}"] = up_arts

        # Quantize each linear sublayer
        # Filter to quantizable sublayers
        quant_names = []
        for name in linears:
            if linears[name].weight.data.shape[1] < 8:
                print(f"  {name}: in={linears[name].weight.data.shape[1]} < 8, skipping")
                continue
            if name not in hessians:
                # No activations (e.g., MoE expert never routed during calibration).
                # Use identity Hessian so LDLQ falls back to simple nearest-neighbor.
                n_cols = linears[name].weight.data.shape[1]
                hessians[name] = torch.eye(n_cols, dtype=torch.float32)
                print(f"  {name}: no activations, using identity Hessian")
            quant_names.append(name)

        if pool is not None and len(quant_names) > 1:
            # Parallel quantization (CPU only)
            tasks = [
                (name, linears[name].weight.data.clone(),
                 hessians[name], bpw, tune_iters)
                for name in quant_names
            ]
            t0 = time.perf_counter()
            results = list(pool.map(_quantize_sublayer, tasks))
            dt_batch = time.perf_counter() - t0

            for name, W_hat, artifacts, metrics in results:
                _store_artifacts(name, artifacts)
                _writeback(name, W_hat)
                all_sqnr.append(metrics['sqnr'])

            # Print summary
            n_done = len(results)
            avg = sum(r[3]['sqnr'] for r in results) / n_done if n_done else 0
            print(f"  {n_done} sublayers in {dt_batch:.1f}s (parallel) "
                  f"avg SQNR={avg:.1f}dB")
        else:
            # Split into experts (parallelizable) and non-experts (sequential).
            # Match both nested (`mlp.experts.0.`) and top-level (`experts.0.`,
            # Gemma-4) routed-expert names.
            expert_names = [n for n in quant_names
                            if ('.experts.' in n or n.startswith('experts.'))
                            and 'shared' not in n]
            non_expert_names = [n for n in quant_names if n not in expert_names]

            def _quantize_one(name):
                """Quantize a single sublayer. Returns (name, W_hat, artifacts, metrics)."""
                mod = linears[name]
                W = mod.weight.data
                layer_prefix = f"{sd_prefix}.{layer_idx}.{name}"
                if bpw_map is not None:
                    sub_bpw = bpw_map.get(layer_prefix, bpw_map.get('default', 2))
                elif mixed_precision:
                    sub_bpw = 2
                else:
                    sub_bpw = bpw
                H_gpu = hessians[name].to(device)
                W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
                    W, H_gpu, codebook, bpw=sub_bpw, tune_iters=tune_iters)
                del H_gpu
                artifacts_cpu = {k: v.cpu() for k, v in artifacts.items()}
                return name, W_hat, artifacts_cpu, metrics

            def _collect_result(name, W_hat, artifacts_cpu, metrics):
                """Store result from a quantized sublayer."""
                layer_prefix = f"{sd_prefix}.{layer_idx}.{name}"
                _store_artifacts(name, artifacts_cpu)
                _writeback(name, W_hat)
                all_sqnr.append(metrics['sqnr'])
                all_proxy_losses[layer_prefix] = metrics['proxy_loss']

            # Non-experts: sequential (few, large, may share GPU resources)
            for name in non_expert_names:
                t0 = time.perf_counter()
                name, W_hat, artifacts_cpu, metrics = _quantize_one(name)
                _collect_result(name, W_hat, artifacts_cpu, metrics)
                dt = time.perf_counter() - t0
                sub_bpw_label = ""
                if mixed_precision:
                    lp = f"{sd_prefix}.{layer_idx}.{name}"
                    sub_bpw_label = f"{bpw_map.get(lp, 2) if bpw_map else 2}b"
                print(f"  {name:30s} {str(tuple(linears[name].weight.shape)):20s} "
                      f"SQNR={metrics['sqnr']:5.1f}dB  "
                      f"Ws={metrics['Wscale']:.3f}  {sub_bpw_label}  {dt:.1f}s")
                gc.collect()
                if use_gpu:
                    torch.cuda.empty_cache()

            # Experts: parallel via ThreadPoolExecutor + CUDA streams
            if expert_names:
                from concurrent.futures import ThreadPoolExecutor
                n_parallel = min(8, len(expert_names))
                t0_experts = time.perf_counter()

                with ThreadPoolExecutor(max_workers=n_parallel) as expert_pool:
                    futures = {}
                    for name in expert_names:
                        f = expert_pool.submit(_quantize_one, name)
                        futures[f] = name

                    expert_sqnrs = []
                    for f in futures:
                        name, W_hat, artifacts_cpu, metrics = f.result()
                        _collect_result(name, W_hat, artifacts_cpu, metrics)
                        expert_sqnrs.append(metrics['sqnr'])

                dt_experts = time.perf_counter() - t0_experts
                avg_sqnr = sum(expert_sqnrs) / len(expert_sqnrs)
                print(f"  {len(expert_names)} experts in {dt_experts:.1f}s "
                      f"({n_parallel} parallel) avg SQNR={avg_sqnr:.1f}dB")

        del hessians

        # Forward calibration through quantized layer
        new_hidden = []
        # Initialize this layer's slot in the cache if it's a Gemma-4 producer.
        is_producer = (shared_kv_cache is not None
                       and layer_idx in producer_layer_indices)
        if is_producer and layer_idx not in shared_kv_cache:
            shared_kv_cache[layer_idx] = []
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1]
                kwargs = _build_forward_kwargs(
                    profile, h, rotary_emb, layer_idx=layer_idx, cfg=cfg,
                    per_layer_inputs=per_layer_inputs, sample_idx=i,
                    shared_kv_cache=shared_kv_cache)
                out = layer(h, **kwargs)
                new_hidden.append(out[0] if isinstance(out, tuple) else out)
                # Capture K/V from a producer layer's post-quantization
                # forward. transformers >=5.10 Gemma4TextAttention writes
                # ``shared_kv_states[self.layer_type] = (K, V)`` for non-reader
                # layers (store_full_length_kv), so the dict object we passed in
                # now holds that entry. Move to CPU to bound GPU memory.
                if is_producer:
                    producer_type = cfg.text_config.layer_types[layer_idx]
                    if producer_type in kwargs['shared_kv_states']:
                        K, V = kwargs['shared_kv_states'][producer_type]
                        shared_kv_cache[layer_idx].append((K.cpu(), V.cpu()))
        hidden_states = torch.cat(new_hidden, dim=0)

        if streaming:
            del layer
        elif use_gpu:
            layer.to("cpu")

        dt_layer = time.perf_counter() - t_layer
        elapsed = time.perf_counter() - t_start
        remaining = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
        print(f"  Layer time: {dt_layer:.0f}s  |  "
              f"Elapsed: {elapsed/60:.1f}m  |  "
              f"ETA: {remaining/60:.1f}m remaining")

        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    if pool is not None:
        pool.shutdown()

    # Mixed-precision auto-allocation: if avg_target was set and no bpw_map,
    # we just completed a 2bpw profiling pass. Compute allocation and report.
    if avg_target is not None and bpw_map is None:
        from .sensitivity import allocate_bpw, print_allocation_summary
        layer_sizes = {}
        for prefix, arts in all_artifacts.items():
            qidxs = arts['Qidxs']
            m, n_blocks = qidxs.shape
            layer_sizes[prefix] = m * n_blocks * 8  # n_weights
        bpw_map = allocate_bpw(all_proxy_losses, layer_sizes, avg_target,
                              min_bpw=min_bpw, max_bpw=max_bpw)
        print_allocation_summary(bpw_map, layer_sizes, all_proxy_losses)

        # Save the allocation as a JSON for use with --bpw-map
        alloc_path = os.path.join(output_dir, "bpw_allocation.json")
        with open(alloc_path, "w") as f:
            json.dump(bpw_map, f, indent=2)
        print(f"\nSaved bpw allocation to {alloc_path}")
        print(f"Re-run with --bpw-map {alloc_path} to quantize with mixed precision.")

        total_time = time.perf_counter() - t_start
        print(f"\nProfiling pass completed in {total_time/60:.1f}m")
        return 0

    # ---- Gemma-4 PLE embedding quantization ----
    # E2B / E4B keep a [vocab × num_layers·ple_dim] embedding (~4.4 GB at bf16
    # on E2B). We quantize it independently after the main loop using the
    # same E8 codebook but right-only RHT so per-row gather at inference is
    # trivial. Skipped on profiling passes (they uniformly use 2 bpw and
    # exit early before reaching here).
    if (streaming and is_gemma4
            and getattr(cfg.text_config, "hidden_size_per_layer_input", 0) > 0):
        ple_embed_prefix = "model.language_model.embed_tokens_per_layer"
        ple_embed_bpw = 4  # 2 stages, ~half the disk of 2bpw embed for noticeably better SQNR
        # Free hidden_states + per_layer_inputs since we only need the embed
        # weight from disk for this final step. KV cache is already on CPU.
        del hidden_states
        if per_layer_inputs is not None:
            del per_layer_inputs
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()
        ple_w = _load_tensor_from_shards(
            weight_map, shard_paths, f"{ple_embed_prefix}.weight")
        vocab_size_ple, embed_dim_ple = ple_w.shape
        print(f"\nQuantizing PLE embedding {tuple(ple_w.shape)} at {ple_embed_bpw}bpw...")
        # Rows are independent (apply_left=False) so we chunk by rows to bound
        # GPU memory: a chunk of 16k rows × 16k embed × fp32 = ~1 GB. Each
        # chunk's LDLQ calibrates its own Wscale + inv_resid_scale; we save
        # both as per-row tensors so E8RHTEmbedding applies the right scale
        # to each row at lookup (using a single chunk[0] scalar would alias
        # rows from later chunks and ~destroy embedding quality).
        chunk_rows = 16384
        ple_qidxs_chunks = []
        ple_qidxs2_chunks = []
        H_id = torch.eye(embed_dim_ple, dtype=torch.float32, device=device)
        ple_sv = None  # SV is RHT-seeded identically for every chunk
        ple_su = None
        ple_wscale_per_row = torch.empty(vocab_size_ple, dtype=torch.float32)
        ple_inv_rs_per_row = torch.empty(vocab_size_ple, dtype=torch.float32)
        ple_sqnr_chunks = []
        for r0 in range(0, vocab_size_ple, chunk_rows):
            r1 = min(r0 + chunk_rows, vocab_size_ple)
            chunk = ple_w[r0:r1].to(device)
            _, arts_chunk, met_chunk = quantize_layer_e8_shell_rht(
                chunk, H_id, codebook,
                bpw=ple_embed_bpw, tune_iters=0,
                apply_left=False, block_diagonal=False)
            ple_qidxs_chunks.append(arts_chunk['Qidxs'].cpu())
            if 'Qidxs2' in arts_chunk:
                ple_qidxs2_chunks.append(arts_chunk['Qidxs2'].cpu())
            ple_wscale_per_row[r0:r1] = arts_chunk['Wscale'].cpu().float()
            ple_inv_rs_per_row[r0:r1] = (
                arts_chunk['inv_resid_scale'].cpu().float()
                if 'inv_resid_scale' in arts_chunk
                else torch.zeros(()))
            if ple_sv is None:
                ple_sv = arts_chunk['SV']
                ple_su = arts_chunk['SU']
            ple_sqnr_chunks.append(met_chunk['sqnr'])
            del chunk
            if use_gpu:
                torch.cuda.empty_cache()
        ple_arts = {
            'SV': ple_sv,
            'SU': ple_su,
            'Wscale': ple_wscale_per_row,           # [vocab] not scalar
            'Qidxs': torch.cat(ple_qidxs_chunks, dim=0),
        }
        if ple_qidxs2_chunks:
            ple_arts['Qidxs2'] = torch.cat(ple_qidxs2_chunks, dim=0)
            ple_arts['inv_resid_scale'] = ple_inv_rs_per_row  # [vocab]
        avg_chunk_sqnr = sum(ple_sqnr_chunks) / len(ple_sqnr_chunks)
        print(f"  PLE embed avg SQNR={avg_chunk_sqnr:.2f} dB "
              f"({len(ple_qidxs_chunks)} chunks, per-row Wscale)")
        all_artifacts[ple_embed_prefix] = ple_arts
        all_sqnr.append(avg_chunk_sqnr)
        del ple_w, H_id, ple_qidxs_chunks, ple_qidxs2_chunks
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start
    avg_sqnr = sum(all_sqnr) / len(all_sqnr) if all_sqnr else 0
    print(f"\n{'='*60}")
    print(f"Quantized {len(all_sqnr)} sublayers in {total_time/60:.1f}m")
    print(f"Average SQNR = {avg_sqnr:.2f} dB")
    print(f"{'='*60}")

    # ---- Save ----
    if not streaming:
        if use_gpu:
            embed.to("cpu")
            if rotary_emb is not None:
                rotary_emb.to("cpu")
            torch.cuda.empty_cache()

    print(f"\nSaving to {output_dir} ...")

    from safetensors.torch import save_file
    from safetensors import safe_open

    quantized_prefixes = set(all_artifacts.keys())
    state_dict = {}

    # Add quantized layer artifacts
    for layer_prefix, arts in all_artifacts.items():
        for key, tensor in arts.items():
            state_dict[f"{layer_prefix}.{key}"] = tensor.cpu()

    if streaming:
        # Stream non-quantized parameters from source safetensors
        for key in weight_map:
            # Gemma-4 MoE: the original stacked expert tensors were replaced by
            # per-expert GLQ artifacts (different key prefix), so the generic
            # quantized_prefixes check below won't catch them — drop explicitly.
            if key in dropped_stacked_keys:
                continue
            param_prefix = key.rsplit(".", 1)[0] if "." in key else ""
            if param_prefix in quantized_prefixes:
                continue
            state_dict[key] = _load_tensor_from_shards(
                weight_map, shard_paths, key)
    else:
        # Add non-quantized parameters from loaded model
        is_wrapped = text_model is not model
        key_prefix = "model." if is_wrapped else ""
        for name, param in text_model.named_parameters():
            full_name = f"{key_prefix}{name}"
            param_prefix = full_name.rsplit(".", 1)[0] if "." in full_name else ""
            if param_prefix in quantized_prefixes:
                continue
            state_dict[full_name] = param.data.cpu()

        # For wrapped models, also save lm_head
        if is_wrapped and hasattr(model, "lm_head"):
            embed_key = f"{key_prefix}embed_tokens.weight"
            lm_w = model.lm_head.weight
            tied = (embed_key in state_dict
                    and lm_w.data_ptr() == state_dict[embed_key].data_ptr())
            if not tied:
                state_dict["lm_head.weight"] = lm_w.data.cpu()

    # Streaming multimodal Mistral3: promote to a text-only Ministral3ForCausalLM
    # layout — strip the ``language_model.`` wrapper and drop the vision tower /
    # projector so the checkpoint reloads as a pure text model (``model.layers.*``),
    # matching the non-streaming path. Without this the checkpoint saves as
    # Mistral3ForConditionalGeneration and glq's HF quantizer doesn't substitute
    # the nested ``language_model.layers`` on reload (Qidxs come back UNEXPECTED
    # -> GLQ weights silently not loaded -> garbage output).
    promote_text_only = streaming and is_mistral3
    if promote_text_only:
        _DROP = ("vision_tower.", "multi_modal_projector.",
                 "model.vision_tower.", "model.multi_modal_projector.")
        remapped = {}
        for _k, _v in state_dict.items():
            if _k.startswith(_DROP):
                continue
            _nk = _k[len("language_model."):] if _k.startswith("language_model.") else _k
            remapped[_nk] = _v
        print(f"  promoted to text-only (stripped language_model. wrapper, "
              f"dropped {len(state_dict) - len(remapped)} vision/projector tensors)")
        state_dict = remapped

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # 2. Save codebook
    codebook.save(os.path.join(output_dir, "e8_codebook.pt"))

    # 3. Save config.json with quantization_config
    if not streaming:
        is_wrapped = text_model is not model
        save_cfg = text_model.config if is_wrapped else cfg
    elif promote_text_only:
        # Promoted text-only checkpoint: save the text decoder config and let
        # the ``architectures`` get rewritten to the matching *ForCausalLM below.
        save_cfg = cfg.text_config
        is_wrapped = True
    else:
        save_cfg = cfg
        is_wrapped = False
    config_dict = save_cfg.to_dict()
    config_dict["use_cache"] = True  # Restore KV cache for inference
    if is_wrapped:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
        causal_cls = MODEL_FOR_CAUSAL_LM_MAPPING.get(type(save_cfg), None)
        if causal_cls is not None:
            config_dict["architectures"] = [causal_cls.__name__]
    # Compute effective average bpw
    if bpw_map is not None:
        total_w = sum(
            all_artifacts[p]['Qidxs'].shape[0] * all_artifacts[p]['Qidxs'].shape[1] * 8
            for p in all_artifacts)
        total_bits = sum(
            bpw_map.get(p, 2) * all_artifacts[p]['Qidxs'].shape[0] * all_artifacts[p]['Qidxs'].shape[1] * 8
            for p in all_artifacts)
        effective_bpw = round(total_bits / total_w, 2) if total_w > 0 else 2
    else:
        effective_bpw = bpw

    # Always emit `layer_bpw` (even uniform-bpw) so vLLM/sglang's GLQ
    # whitelist can distinguish quantized linears from bf16-only siblings
    # (e.g. Gemma-4 ``per_layer_model_projection``). Without it, the
    # plugin defaults to "quantize everything that's a Linear", which
    # silently zero-fills bf16 layers it has no GLQ buffers for.
    #
    # The layer_bpw map MUST cover every entry in ``all_artifacts`` —
    # any quantized layer absent from the map causes vLLM to load the
    # layer as bf16 (UnquantizedLinear / UnquantizedEmbedding) and trip a
    # KeyError on Qidxs at safetensors load. Two layer types can be in
    # ``all_artifacts`` but absent from a user-supplied ``bpw_map``:
    #   - ``model.language_model.embed_tokens_per_layer`` (Gemma-4 PLE,
    #     quantized at a fixed ``ple_embed_bpw=4`` in the streaming path,
    #     never seen by the sensitivity allocator).
    #   - any layer the allocator hadn't profiled yet (defensive).
    # Default the bpw for those to ``ple_embed_bpw=4`` for the embedding
    # and to the global ``bpw`` for everything else.
    PLE_PREFIX = "model.language_model.embed_tokens_per_layer"
    layer_bpw_out: dict[str, int] = {}
    for p in all_artifacts.keys():
        if bpw_map is not None and p in bpw_map:
            layer_bpw_out[p] = int(bpw_map[p])
        elif p == PLE_PREFIX:
            layer_bpw_out[p] = 4  # matches ple_embed_bpw above
        else:
            layer_bpw_out[p] = int(bpw)
    # Record the RHT layout so the runtime sizes the weight buffers correctly.
    # e8p quantizes block-diagonally by default; GLQ_E8P_POW2 forces the legacy
    # full power-of-two Hadamard (one block spanning the pow2-padded dim). Shell
    # and relaxed are always block-diagonal. block_diagonal=False tells the vLLM
    # loader to size the e8p buffers to pow2 instead of the block-diagonal dims.
    block_diagonal = not (codebook_type == "e8p" and bool(os.environ.get("GLQ_E8P_POW2")))
    config_dict["quantization_config"] = {
        "quant_method": "glq",
        "codebook": codebook_type,
        "block_diagonal": block_diagonal,
        "codesz": 8,
        "bpw": effective_bpw,
        "layer_bpw": layer_bpw_out,
    }
    if trust_remote_code:
        config_dict["quantization_config"]["trust_remote_code"] = True
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Save quantize_config.json (metadata)
    quant_meta = {
        "quant_method": "glq",
        "codebook": codebook_type,
        "block_diagonal": block_diagonal,
        "codesz": 8,
        "bpw": effective_bpw,
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

    # 5a. Save processor for multimodal models (e.g. Gemma-4 needs
    # processor_config.json so vLLM/sglang can load via the gemma4_mm
    # path). AutoProcessor.from_pretrained raises for pure text models;
    # swallow that — non-multimodal models legitimately have no
    # processor.
    #
    # Order matters: save the processor BEFORE the tokenizer fixup
    # block below. ``Processor.save_pretrained`` writes its bundled
    # tokenizer files, which we then overwrite with the corrected
    # tokenizer.json (preserving the source pre_tokenizer/decoder).
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        processor.save_pretrained(output_dir)
        print(f"  saved processor ({type(processor).__name__})")
    except (ValueError, OSError, ImportError, KeyError):
        # ValueError: model has no processor (typical for text-only LMs)
        # OSError:    no processor_config.json on hub
        # ImportError/KeyError: AutoProcessor can't infer the class
        pass

    # 5b. Save tokenizer, preserving the original tokenizer.json
    # pre_tokenizer and decoder which save_pretrained may corrupt (e.g.
    # Sequence(ByteLevel) can be flattened to plain ByteLevel or
    # replaced with Metaspace).
    src_tok_json = None
    try:
        from huggingface_hub import hf_hub_download
        src_path = hf_hub_download(model_name, "tokenizer.json")
        with open(src_path) as f:
            src_tok_json = json.load(f)
    except Exception:
        pass

    tokenizer.save_pretrained(output_dir)

    # Restore pre_tokenizer and decoder from the source tokenizer.json
    tok_json_path = os.path.join(output_dir, "tokenizer.json")
    if src_tok_json and os.path.exists(tok_json_path):
        with open(tok_json_path) as f:
            saved_tok = json.load(f)
        saved_tok["pre_tokenizer"] = src_tok_json["pre_tokenizer"]
        saved_tok["decoder"] = src_tok_json["decoder"]
        with open(tok_json_path, "w") as f:
            json.dump(saved_tok, f, ensure_ascii=False)

    # Strip tokenizer_class only if it references an external library class
    # (e.g., "TokenizersBackend" from the tokenizers lib, not transformers).
    # Keep known-good classes like GPT2Tokenizer, GPT2TokenizerFast, etc.
    _KEEP_TOKENIZER_CLASSES = {
        "GPT2Tokenizer", "GPT2TokenizerFast",
        "LlamaTokenizer", "LlamaTokenizerFast",
        "PreTrainedTokenizerFast",
    }
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        tok_cls = tc.get("tokenizer_class")
        if tok_cls and tok_cls not in _KEEP_TOKENIZER_CLASSES:
            tc.pop("tokenizer_class")
            with open(tc_path, "w") as f:
                json.dump(tc, f, indent=2)

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
    parser.add_argument("--bpw", type=float, default=2,
                        help="Bits per weight: 2, 3, 4 (uniform) or "
                             "fractional like 2.5 (auto mixed-precision)")
    parser.add_argument("--bpw-map", type=str, default=None,
                        help="JSON file with per-layer bpw assignment "
                             "(overrides --bpw)")
    parser.add_argument("--min-bpw", type=int, default=None, choices=[2, 3, 4, 5, 6, 7, 8],
                        help="Minimum per-layer bpw for mixed-precision "
                             "(triggers auto-allocation)")
    parser.add_argument("--max-bpw", type=int, default=None, choices=[2, 3, 4, 5, 6, 7, 8],
                        help="Maximum per-layer bpw for mixed-precision "
                             "(triggers auto-allocation)")
    parser.add_argument("--tune-iters", type=int, default=0,
                        help="LDLQ refinement iterations")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples (default 128; <64 may degrade quality)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Calibration sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom model code from HF Hub")
    parser.add_argument("--streaming", action="store_true",
                        help="Load weights layer-by-layer from safetensors "
                             "(for models exceeding system RAM)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers for CPU quantization "
                             "(0=auto, 1=sequential, ignored on GPU)")
    parser.add_argument("--codebook", type=str, default="e8_shell",
                        choices=["e8_shell", "e8_relaxed", "e8p"],
                        help="Codebook variant: e8_shell (default, optimal ball), "
                             "e8_relaxed (D~8 for KV), e8p (QuIP# padded-D̂8 + RVQ, "
                             "tensor-core decode, bpw 2/3/4).")
    parser.add_argument("--codebook-size", type=int, default=None,
                        help="E8 codebook entry count. Default 65 536 "
                             "(shells 0-5 in full + 8 655 from shell 6). "
                             "Use 4096 for v0.5 shared-memory-resident "
                             "attention kernel (64 KB fp16 fits Blackwell "
                             "per-CTA smem). Truncation is shell-sorted "
                             "(smaller values drop high-norm vectors).")
    args = parser.parse_args()

    # Determine bpw: explicit map, fractional target, or uniform int
    if args.bpw_map:
        with open(args.bpw_map) as f:
            bpw_arg = json.load(f)
    elif args.bpw != int(args.bpw):
        bpw_arg = args.bpw  # fractional → auto mixed-precision
    else:
        bpw_arg = int(args.bpw)

    quantize(
        model_name=args.model,
        output_dir=args.output,
        bpw=bpw_arg,
        min_bpw=args.min_bpw,
        max_bpw=args.max_bpw,
        tune_iters=args.tune_iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        streaming=args.streaming,
        workers=args.workers,
        codebook_size=args.codebook_size,
        codebook_type=args.codebook,
    )


if __name__ == "__main__":
    main()
