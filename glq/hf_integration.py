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

from .quantized_linear import E8RHTLinear
from .codebook import E8ShellCodebook, E8PCodebook


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


def replace_with_glq_linear(model):
    """Replace nn.Linear modules with E8RHTLinear on meta device."""
    has_replaced = False
    for name, module in list(model.named_modules()):
        skip = any(excl in name for excl in MODULES_TO_NOT_CONVERT)
        if skip or not isinstance(module, nn.Linear):
            continue

        with torch.device("meta"):
            new_module = E8RHTLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
        new_module.requires_grad_(False)
        model.set_submodule(name, new_module)
        has_replaced = True

    return has_replaced


@register_quantizer("glq")
class GLQQuantizer(HfQuantizer):
    """HuggingFace quantizer for GLQ (E8 shell codebook + RHT)."""

    requires_calibration = False

    def __init__(self, quantization_config: GLQConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        pass  # glq is bundled — no external deps beyond torch

    def _process_model_before_weight_loading(self, model, **kwargs):
        replaced = replace_with_glq_linear(model)
        if not replaced:
            import logging
            logging.getLogger(__name__).warning(
                "GLQ: no nn.Linear modules found to replace")
        # Suppress MISSING warnings for Qidxs2/inv_resid_scale on 2bpw layers.
        # These are omitted from checkpoint to save disk; _load_from_state_dict
        # injects zero defaults. The pattern matches any layer's optional buffers.
        ignore = getattr(model, '_keys_to_ignore_on_load_missing', None) or []
        ignore.extend(["Qidxs2", "inv_resid_scale"])
        model._keys_to_ignore_on_load_missing = ignore
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        # Build the shared codebook and attach to all E8RHTLinear modules.
        # Detect codebook type from quantization_config
        cb_type = getattr(self.quantization_config, 'codebook', 'e8_shell')
        if cb_type == 'e8p':
            codebook_path = os.path.join(os.path.dirname(__file__), "e8_codebook.pt")
            if os.path.exists(codebook_path):
                data = torch.load(codebook_path, map_location='cpu', weights_only=True)
                if data.get('codebook_type') == 'e8p':
                    codebook = E8PCodebook.load(codebook_path, device='cpu')
                else:
                    codebook = E8PCodebook(device='cpu', verbose=False)
            else:
                codebook = E8PCodebook(device='cpu', verbose=False)
        else:
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

        for module in model.modules():
            if isinstance(module, E8RHTLinear):
                module.set_codebook(codebook, codebook2=codebook2)

        # INT8 KV cache: attach factory so model.generate() uses it
        kv_bits = getattr(self.quantization_config, 'kv_cache_bits', 16)
        if kv_bits == 8:
            from .kv_cache import GLQQuantizedCache
            config = model.config
            model._glq_kv_cache_factory = lambda: GLQQuantizedCache(config)

        return model

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self):
        return True
