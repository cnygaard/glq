"""HuggingFace Transformers integration for GLQ quantization.

Usage:
    import glq.hf_integration  # registers the quantizer

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("path/to/glq-model")
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from .quantized_linear import E8RHTLinear, E8RHTEmbedding
from .codebook import E8ShellCodebook


@register_quantization_config("glq")
@dataclass
class GLQConfig(QuantizationConfigMixin):
    """Quantization config for GLQ (Golay-Leech Quantization)."""

    def __init__(
        self,
        codebook: str = "e8_shell",
        codesz: int = 8,
        bpw=2,
        layer_bpw: dict = None,
        kv_cache_bits: int = 16,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        self.quant_method = "glq"
        self.codebook = codebook
        self.codesz = codesz
        self.bpw = bpw
        self.layer_bpw = layer_bpw
        self.kv_cache_bits = kv_cache_bits
        self.trust_remote_code = trust_remote_code

    def to_dict(self):
        d = {
            "quant_method": self.quant_method,
            "codebook": self.codebook,
            "codesz": self.codesz,
            "bpw": self.bpw,
        }
        if self.layer_bpw:
            d["layer_bpw"] = self.layer_bpw
        if self.kv_cache_bits != 16:
            d["kv_cache_bits"] = self.kv_cache_bits
        if self.trust_remote_code:
            d["trust_remote_code"] = True
        return d


MODULES_TO_NOT_CONVERT = ["lm_head"]


def _peek_qidxs_keys(pretrained_path):
    """Open the saved safetensors (single-file or sharded) and return every
    key ending in ``.Qidxs`` plus the shape of one such tensor.

    Returns (key_list, sample_shape) or ([], None) on failure.
    """
    from safetensors import safe_open
    from transformers.utils.hub import cached_file

    keys = []
    sample_shape = None
    # Try sharded layout first (model.safetensors.index.json + N shards),
    # fall back to single model.safetensors.
    try:
        idx_path = cached_file(
            pretrained_path, "model.safetensors.index.json",
            _raise_exceptions_for_missing_entries=False)
    except Exception:
        idx_path = None
    if idx_path is not None:
        try:
            import json
            with open(idx_path) as f:
                idx = json.load(f)
            seen_files = set()
            for k, fname in idx.get("weight_map", {}).items():
                if k.endswith(".Qidxs"):
                    keys.append(k)
                    if sample_shape is None and fname not in seen_files:
                        seen_files.add(fname)
                        try:
                            shard_path = cached_file(
                                pretrained_path, fname,
                                _raise_exceptions_for_missing_entries=False)
                            if shard_path is not None:
                                with safe_open(shard_path, framework="pt") as st:
                                    if k in st.keys():
                                        sample_shape = st.get_tensor(k).shape
                        except Exception:
                            pass
            return keys, sample_shape
        except Exception:
            pass

    try:
        st_path = cached_file(
            pretrained_path, "model.safetensors",
            _raise_exceptions_for_missing_entries=False)
        if st_path is None:
            return [], None
        with safe_open(st_path, framework="pt") as st:
            for k in st.keys():
                if k.endswith(".Qidxs"):
                    keys.append(k)
                    if sample_shape is None:
                        sample_shape = st.get_tensor(k).shape
    except Exception:
        pass
    return keys, sample_shape


def _detect_block_diagonal(pretrained_path):
    """Peek at checkpoint to detect block-diagonal quantization."""
    _, sample_shape = _peek_qidxs_keys(pretrained_path)
    if sample_shape is None or sample_shape[0] <= 0:
        return False
    m = sample_shape[0]
    is_pow2 = (m & (m - 1)) == 0
    return not is_pow2


def _collect_quantized_layer_names(pretrained_path):
    """Return the set of nn.Linear module names that were GLQ-quantized.

    Reads the saved safetensors and strips ``.Qidxs`` from every key whose
    name ends that way. The resulting names match the dotted-path format
    used by ``model.named_modules()`` so the load-side replacer can scope
    its work to exactly the layers that have GLQ payload.

    Returns ``None`` if the safetensors can't be opened — caller should
    fall back to "replace everything" for backward compatibility with
    older callers that didn't rely on this scoping.
    """
    keys, _ = _peek_qidxs_keys(pretrained_path)
    if not keys:
        return None
    return {k[:-len(".Qidxs")] for k in keys}


def replace_with_glq_embedding(model, quantized_layers=None):
    """Replace nn.Embedding modules with E8RHTEmbedding when checkpoint has GLQ
    payload for them.

    Used by Gemma-4 E2B/E4B where ``embed_tokens_per_layer`` (a [vocab × ~9k]
    table) gets right-side-only RHT-quantized to cut its 4 GB bf16 footprint
    to ~1 GB on disk and in GPU. ``quantized_layers`` is the set returned by
    ``_collect_quantized_layer_names`` and is matched verbatim against
    ``model.named_modules()`` paths.
    """
    if quantized_layers is None:
        return False
    has_replaced = False
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Embedding):
            continue
        if name not in quantized_layers:
            continue
        # Gemma4TextScaledWordEmbedding (and friends) multiply by an
        # ``embed_scale`` factor inside forward. Carry that scale through to
        # E8RHTEmbedding so the replacement produces the same magnitude.
        # Some implementations register embed_scale as a buffer (which lives
        # on meta during preprocess); recover the original constant from the
        # embedding_dim instead in that case (matches how Gemma-4 sets it).
        raw_scale = getattr(module, "embed_scale", 1.0)
        if isinstance(raw_scale, torch.Tensor):
            try:
                embed_scale = float(raw_scale.detach().cpu().item())
            except Exception:
                # Meta tensor or otherwise unreadable. Gemma4 PLE scale is
                # sqrt(hidden_size_per_layer_input) which equals
                # sqrt(embedding_dim / num_hidden_layers).
                tc = getattr(getattr(model, "config", None), "text_config", None)
                if tc is not None and getattr(tc, "hidden_size_per_layer_input", 0) > 0:
                    embed_scale = float(tc.hidden_size_per_layer_input) ** 0.5
                else:
                    embed_scale = 1.0
        else:
            embed_scale = float(raw_scale)
        with torch.device("meta"):
            new_module = E8RHTEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                embed_scale=embed_scale,
            )
        new_module.requires_grad_(False)
        model.set_submodule(name, new_module)
        has_replaced = True
    return has_replaced


def replace_with_glq_linear(model, block_diagonal=False, quantized_layers=None):
    """Replace nn.Linear modules with E8RHTLinear on meta device.

    When ``quantized_layers`` is provided, only ``nn.Linear`` modules whose
    fully-qualified name appears in the set are replaced. Modules outside
    the set (e.g. a multimodal model's vision encoder when only the
    language decoder was quantized) are left as plain ``nn.Linear`` so
    their bf16 weights load normally and the auxiliary capability
    (vision/audio/etc.) stays intact. ``None`` (the default) preserves the
    original behaviour of replacing every ``nn.Linear`` not in the
    ``MODULES_TO_NOT_CONVERT`` exclusion list — used as the fallback when
    the checkpoint can't be peeked.
    """
    has_replaced = False
    for name, module in list(model.named_modules()):
        skip = any(excl in name for excl in MODULES_TO_NOT_CONVERT)
        if skip or not isinstance(module, nn.Linear):
            continue
        if quantized_layers is not None and name not in quantized_layers:
            continue

        with torch.device("meta"):
            new_module = E8RHTLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                block_diagonal=block_diagonal,
            )
        new_module.requires_grad_(False)
        model.set_submodule(name, new_module)
        has_replaced = True

    return has_replaced


def _patch_nemotron_h_decode_cache(model):
    """Make `use_cache=True` actually thread the cache for NemotronH models.

    NVIDIA's `modeling_nemotron_h.py` (shipped via trust_remote_code) has five
    latent bugs that together force generate() to silently fall back to
    no-cache decode (re-prefill every step, O(N²) total work):

    1. `prepare_inputs_for_generation` returns the cache under key
       `past_key_values`, but `NemotronHForCausalLM.forward` accepts it as
       `cache_params=` — the value lands in `**kwargs` and is ignored.
    2. `NemotronHCausalLMOutput` exposes the cache as `cache_params`, but
       generate's `_extract_past_from_model_output` only looks for
       `past_key_values` — between-step threading breaks.
    3. `cuda_kernels_forward` and `torch_forward` reference
       `cache_params.conv_kernel_size`, which is not a field on
       `HybridMambaAttentionDynamicCache`.
    4. `cache.update_conv_state` and `cache.update_ssm_state` read
       `self.conv_states.device` / `self.ssm_states.device`, but those are
       Python lists of tensors, not tensors.
    5. `torch_forward` reads `cache_params.ssm_states.device` (same
       list-not-tensor issue) when allocating an intermediate buffer.

    All five are pure modeling-side bugs (independent of GLQ). Fixing them
    speeds up the bf16 baseline by ~2x and the GLQ build by ~2.2x on
    Nemotron-Cascade-2-30B-A3B at long-prompt decode (RTX PRO 6000 Blackwell,
    transformers 4.56). The patch is idempotent and silently no-ops on
    non-NemotronH models or on a future NVIDIA release that fixes any of
    these inline.
    """
    import sys
    import types
    import logging

    log = logging.getLogger(__name__)

    # Find the remote modeling module that provided this model's classes.
    cls_mod = type(model).__module__
    if "modeling_nemotron_h" not in cls_mod:
        return  # not a NemotronH variant — nothing to do
    nh = sys.modules.get(cls_mod)
    if nh is None or not hasattr(nh, "HybridMambaAttentionDynamicCache"):
        return

    HybridCache = nh.HybridMambaAttentionDynamicCache

    # ---- Bugs 3 + 4: cache class needs conv_kernel_size + per-layer device ----
    if not getattr(HybridCache, "_glq_patched", False):

        class _DeviceProxy(list):
            """List of cache tensors that also exposes a `.device` property."""
            @property
            def device(self):
                for t in self:
                    if isinstance(t, torch.Tensor):
                        return t.device
                return torch.device("cpu")

            def zero_(self):
                for t in self:
                    if isinstance(t, torch.Tensor) and t.numel() > 0:
                        t.zero_()
                return self

        _orig_init = HybridCache.__init__

        def _patched_init(self, config, batch_size, dtype=torch.float16, device=None):
            _orig_init(self, config, batch_size, dtype=dtype, device=device)
            self.conv_kernel_size = config.conv_kernel  # bug 3
            self.conv_states = _DeviceProxy(self.conv_states)  # bug 4 + 5
            self.ssm_states = _DeviceProxy(self.ssm_states)

        def _patched_update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
            target_dev = self.conv_states[layer_idx].device
            if cache_init:
                self.conv_states[layer_idx] = new_conv_state.to(target_dev)
            else:
                self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
                self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(target_dev)
            return self.conv_states[layer_idx]

        def _patched_update_ssm_state(self, layer_idx, new_ssm_state):
            self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
            return self.ssm_states[layer_idx]

        HybridCache.__init__ = _patched_init
        HybridCache.update_conv_state = _patched_update_conv_state
        HybridCache.update_ssm_state = _patched_update_ssm_state
        HybridCache._glq_patched = True

    # ---- Bug 1: rename returned dict key past_key_values -> cache_params ----
    if not getattr(model, "_glq_prep_patched", False):
        _orig_prep = model.prepare_inputs_for_generation

        def _patched_prep(self, *args, **kwargs):
            out = _orig_prep(*args, **kwargs)
            if "past_key_values" in out and "cache_params" not in out:
                out["cache_params"] = out.pop("past_key_values")
            return out

        model.prepare_inputs_for_generation = types.MethodType(_patched_prep, model)
        model._glq_prep_patched = True

    # ---- Bug 2: alias cache_params -> past_key_values on forward output ----
    model_cls = type(model)
    if not getattr(model_cls, "_glq_fwd_patched", False):
        _orig_fwd = model_cls.forward

        def _patched_fwd(self, *args, **kwargs):
            out = _orig_fwd(self, *args, **kwargs)
            if hasattr(out, "cache_params") and getattr(out, "past_key_values", None) is None:
                try:
                    object.__setattr__(out, "past_key_values", out.cache_params)
                    if isinstance(out, dict):
                        out["past_key_values"] = out.cache_params
                except Exception:  # noqa: BLE001
                    pass
            return out

        model_cls.forward = _patched_fwd
        model_cls._glq_fwd_patched = True

    log.info(
        "glq: patched NemotronH decode-cache (5 bugs in NVIDIA modeling file). "
        "use_cache=True now threads cache across generate() steps."
    )


@register_quantizer("glq")
class GLQQuantizer(HfQuantizer):
    """HuggingFace quantizer for GLQ (E8 shell codebook + RHT)."""

    requires_calibration = False

    def __init__(self, quantization_config: GLQConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        pass  # glq is bundled — no external deps beyond torch

    def _process_model_before_weight_loading(self, model, **kwargs):
        cfg = getattr(model, "config", None)
        pretrained_path = getattr(cfg, "_name_or_path", None) if cfg is not None else None
        block_diag = _detect_block_diagonal(pretrained_path) if pretrained_path else False
        # Scope the nn.Linear -> E8RHTLinear replacement to exactly the
        # layers that have GLQ payload in the saved safetensors. Multimodal
        # models (e.g. Gemma-4) only quantize the language decoder; vision
        # / audio towers stay as plain nn.Linear with their bf16 weights.
        # Falls back to "replace everything" if the safetensors can't be
        # peeked (older callers / non-pretrained loads).
        quantized_layers = (
            _collect_quantized_layer_names(pretrained_path) if pretrained_path else None
        )
        # NemotronH on the native HF integration packs experts into stacked
        # tensors (see transformers/models/nemotron_h/modeling_nemotron_h.py
        # NemotronHExperts). Swap them for E8RHTFusedExperts before walking
        # nn.Linear so each per-expert E8RHTLinear gets installed and the
        # saved per-expert state-dict keys load straight in.
        from .fused_experts import _replace_nemotron_h_experts
        from .state_dict_stacker import install_nemotron_h_state_dict_renames
        n_fused = _replace_nemotron_h_experts(model, block_diagonal=block_diag)
        if n_fused:
            install_nemotron_h_state_dict_renames(model)
        replaced = replace_with_glq_linear(
            model, block_diagonal=block_diag, quantized_layers=quantized_layers)
        # Gemma-4 PLE embedding (and any other quantized nn.Embedding) gets
        # swapped to E8RHTEmbedding when the saved checkpoint has its GLQ
        # payload. No-op for older single-modal checkpoints.
        replace_with_glq_embedding(model, quantized_layers=quantized_layers)
        if not replaced and not n_fused:
            import logging
            logging.getLogger(__name__).warning(
                "GLQ: no nn.Linear or NemotronHExperts modules found to replace")
        # DO NOT add Qidxs2 etc. to _keys_to_ignore_on_load_missing — HF's
        # _move_missing_keys_from_meta_to_cpu uses the missing_keys list to
        # reinitialize meta-device buffers, and "ignored" keys still get
        # overwritten with zeros. Instead, let _load_from_state_dict inject
        # zero defaults for truly missing keys.
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        # Build the shared codebook and attach to all E8RHTLinear modules.
        # Try loading from: 1) glq package dir, 2) enumerate fresh
        codebook = None
        for path in [
            os.path.join(os.path.dirname(__file__), "e8_codebook.pt"),
        ]:
            if os.path.exists(path):
                codebook = E8ShellCodebook.load(path, device='cpu')
                break
        if codebook is None:
            codebook = E8ShellCodebook(device='cpu', verbose=False)

        # Secondary codebook for 3/4bpw
        # For mixed-precision models, use the max bpw to build the largest
        # codebook2 needed. Each E8RHTLinear auto-detects its actual bpw
        # from inv_resid_scale (zero = 2bpw, non-zero = 3/4bpw).
        layer_bpw = getattr(self.quantization_config, 'layer_bpw', None)
        global_bpw = getattr(self.quantization_config, 'bpw', 2)
        if layer_bpw:
            max_bpw = max(layer_bpw.values())
        else:
            max_bpw = int(global_bpw) if isinstance(global_bpw, (int, float)) else 2

        codebook2 = None
        if max_bpw >= 4:
            codebook2 = codebook
        elif max_bpw >= 3:
            codebook2 = codebook.make_small(256)

        # Detect the compute dtype the rest of the model is using so the
        # E8RHTEmbedding output matches (Gemma-4 lm_head is bf16; F.linear
        # rejects mismatched dtypes). Some test mocks have no `config`.
        cfg = getattr(model, "config", None)
        compute_dtype = (getattr(cfg, "torch_dtype", None)
                         or getattr(cfg, "dtype", None)
                         or torch.bfloat16) if cfg is not None else torch.bfloat16
        if isinstance(compute_dtype, str):
            compute_dtype = getattr(torch, compute_dtype, torch.bfloat16)

        for module in model.modules():
            if isinstance(module, (E8RHTLinear, E8RHTEmbedding)):
                module.set_codebook(codebook, codebook2=codebook2)
            if isinstance(module, E8RHTEmbedding):
                module._compute_dtype = compute_dtype

        # INT8 KV cache: attach factory so model.generate() uses it
        kv_bits = getattr(self.quantization_config, 'kv_cache_bits', 16)
        if kv_bits == 8:
            from .kv_cache import GLQQuantizedCache
            config = model.config
            model._glq_kv_cache_factory = lambda: GLQQuantizedCache(config)

        # NemotronH (Nemotron-Cascade, Nemotron-3-Nano, etc.) has 5 latent
        # bugs in NVIDIA's remote modeling file that silently force
        # use_cache=False at generate time. Patch them in-place if present.
        _patch_nemotron_h_decode_cache(model)

        return model

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self):
        return True
