"""GLQ quantization config for vLLM plugin registration."""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from .linear_method import GLQLinearMethod
from .embedding_method import GLQEmbeddingMethod


class GLQvLLMConfig(QuantizationConfig):
    """vLLM quantization config for GLQ (E8 lattice codebook + RHT)."""

    def __init__(self, bpw: int = 2, layer_bpw: dict | None = None,
                 codebook: str = "e8_shell", block_diagonal: bool = True):
        super().__init__()
        self.bpw = bpw
        self.layer_bpw = layer_bpw or {}
        self.codebook = codebook
        # RHT layout from the checkpoint: True (default) = block-diagonal padding,
        # False = legacy full pow2 Hadamard. Controls how the e8p weight buffers
        # are sized so the loader can copy in place. Absent in pre-0.6.7
        # checkpoints — default True matches the block-diagonal e8p default.
        self.block_diagonal = block_diagonal

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
            codebook=config.get("codebook", "e8_shell"),
            block_diagonal=config.get("block_diagonal", True),
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

    def _is_prefused(self, prefix: str) -> bool:
        """True iff ``prefix`` is a DIRECT layer_bpw entry — i.e. the checkpoint
        stores this fused linear as one jointly-quantized matrix (e.g.
        ``query_key_value``), rather than as separate per-shard weights merged
        at load (``qkv_proj``/``gate_up_proj``, whose q/k/v or gate/up
        sub-layers are the layer_bpw entries, reached via ``_lookup_bpw``'s
        merge_map). Such a matrix has a single row-direction Hadamard over all
        output rows and must be loaded/dequantized whole, not split into shards.
        """
        if not self.layer_bpw:
            return False
        forms = {prefix}
        if prefix.startswith("model.") and ".language_model." not in prefix:
            forms.add("model.language_model." + prefix[len("model."):])
        if prefix.startswith("language_model.model."):
            forms.add("model.language_model." + prefix[len("language_model.model."):])
        return any(f in self.layer_bpw for f in forms)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bpw = self._lookup_bpw(prefix)
            if self.layer_bpw and bpw is None:
                # Layer is bf16 in the checkpoint — return default so
                # vLLM loads `.weight` instead of expecting GLQ buffers.
                return UnquantizedLinearMethod()
            return GLQLinearMethod(
                self, bpw=bpw if bpw is not None else self.bpw,
                pre_fused=self._is_prefused(prefix),
                codebook_type=self.codebook,
                block_diagonal=self.block_diagonal)

        # VocabParallelEmbedding (and subclasses, e.g. ParallelLMHead).
        # Quantized embeddings — currently only Gemma-4's PLE — appear in
        # the checkpoint with the same Qidxs/SV/Wscale buffers as a
        # quantized linear, so they're enumerated in layer_bpw at quant
        # time. Embeddings absent from the map (main embed_tokens, lm_head)
        # fall through to UnquantizedEmbeddingMethod.
        if isinstance(layer, VocabParallelEmbedding):
            bpw = self._lookup_bpw(prefix)
            if bpw is None:
                return UnquantizedEmbeddingMethod()
            return GLQEmbeddingMethod(self, bpw=bpw)

        # FusedMoE layers — lazy import to avoid circular deps. vLLM 0.25 split the
        # 0.23 `FusedMoE` nn.Module into a factory *function* (returns a MoERunner)
        # plus the weight-holding `RoutedExperts` layer that get_quant_method now
        # receives; `isinstance(layer, FusedMoE)` throws TypeError there. Detect
        # RoutedExperts on 0.25+, falling back to the FusedMoE class on <=0.23.
        try:
            try:
                from vllm.model_executor.layers.fused_moe.routed_experts import (
                    RoutedExperts as _MoELayer)  # vLLM 0.25+
            except ImportError:
                from vllm.model_executor.layers.fused_moe.layer import (
                    FusedMoE as _MoELayer)       # vLLM <= 0.23
            if isinstance(layer, _MoELayer):
                from .fused_moe_method import GLQFusedMoEMethod
                return GLQFusedMoEMethod(self, moe=layer.moe_config)
        except (ImportError, AttributeError):
            pass

        return None
