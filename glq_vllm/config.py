"""GLQ quantization config for vLLM plugin registration."""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .linear_method import GLQLinearMethod


class GLQvLLMConfig(QuantizationConfig):
    """vLLM quantization config for GLQ (E8 lattice codebook + RHT)."""

    def __init__(self, bpw: int = 2, layer_bpw: dict | None = None):
        super().__init__()
        self.bpw = bpw
        self.layer_bpw = layer_bpw or {}

    def get_name(self) -> str:
        return "glq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Volta+

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GLQvLLMConfig":
        return cls(
            bpw=config.get("bpw", 2),
            layer_bpw=config.get("layer_bpw", None),
        )

    def _lookup_bpw(self, prefix: str) -> int | None:
        """Return the bpw for ``prefix`` from ``layer_bpw``, or None.

        vLLM rewrites checkpoint names twice on the way to the layer
        prefix passed here, so we try every form a checkpoint key
        could legally take:

        - Text-only Gemma-4 path collapses ``model.language_model.*`` to
          ``model.*`` (handled by adding the inverse form).
        - Multimodal Gemma-4 path (``Gemma4ForConditionalGeneration``)
          rewrites ``model.language_model.*`` to ``language_model.model.*``;
          translate it back so the checkpoint-form key still matches.
        - Stacked merge: q/k/v_proj into ``qkv_proj`` and
          gate/up_proj into ``gate_up_proj``. The whitelist lists the
          unmerged sublayers; the prefix here is the merged name.
        """
        if prefix in self.layer_bpw:
            return self.layer_bpw[prefix]
        forms = {prefix}
        if prefix.startswith("model.") and ".language_model." not in prefix:
            forms.add("model.language_model." + prefix[len("model."):])
        if prefix.startswith("language_model.model."):
            forms.add("model.language_model."
                      + prefix[len("language_model.model."):])
        merge_map = {".qkv_proj": (".q_proj", ".k_proj", ".v_proj"),
                     ".gate_up_proj": (".gate_proj", ".up_proj")}
        best = None
        for f in forms:
            for merged, parts in merge_map.items():
                if f.endswith(merged):
                    root = f[:-len(merged)]
                    for p in parts:
                        b = self.layer_bpw.get(root + p)
                        if b is not None:
                            best = max(best or 0, int(b))
            if f in self.layer_bpw:
                best = max(best or 0, int(self.layer_bpw[f]))
        return best

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bpw = self._lookup_bpw(prefix)
            if self.layer_bpw and bpw is None:
                # Layer is bf16 in the checkpoint — return default so
                # vLLM loads `.weight` instead of expecting GLQ buffers.
                return UnquantizedLinearMethod()
            return GLQLinearMethod(self, bpw=bpw if bpw is not None else self.bpw)

        # FusedMoE layers — lazy import to avoid circular deps
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
            if isinstance(layer, FusedMoE):
                from .fused_moe_method import GLQFusedMoEMethod
                return GLQFusedMoEMethod(self, moe=layer.moe_config)
        except (ImportError, AttributeError):
            pass

        return None
