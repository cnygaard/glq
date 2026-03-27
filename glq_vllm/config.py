"""GLQ quantization config for vLLM plugin registration."""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase
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

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bpw = self.layer_bpw.get(prefix, self.bpw)
            return GLQLinearMethod(self, bpw=bpw)

        # FusedMoE layers — lazy import to avoid circular deps
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
            if isinstance(layer, FusedMoE):
                from .fused_moe_method import GLQFusedMoEMethod
                return GLQFusedMoEMethod(self)
        except ImportError:
            pass

        return None
