"""GLQ embedding method for vLLM — VocabParallelEmbedding decompress-on-lookup.

Mirror of ``GLQLinearMethod`` but for ``VocabParallelEmbedding``. Lets vLLM
load a GLQ-quantized embedding (e.g. Gemma-4 ``embed_tokens_per_layer``)
directly from a checkpoint without requiring a dequant-PLE pre-step.

vLLM 0.20's ``VocabParallelEmbedding.forward_native`` calls
``self.quant_method.embedding(self, masked_input.long())`` after vocab-mask
handling, expecting a tensor of shape ``[*input_ids.shape, embedding_dim]``.
Per-row dequant is delegated to ``glq.quantized_linear._dequant_embedding_rows``
— the same helper ``E8RHTEmbedding.forward`` calls — so HF and vLLM share
one math path.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from glq.quantized_linear import _dequant_embedding_rows


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _make_param(tensor: torch.Tensor, weight_loader, output_dim: int | None = None,
                ) -> nn.Parameter:
    """Build an nn.Parameter with vLLM's weight_loader + sharding attrs.

    ``output_dim=0`` marks the parameter as vocab-sharded (TP slices it
    along axis 0). ``None`` = replicated (no narrowing).
    """
    p = nn.Parameter(tensor, requires_grad=False)
    attrs = {"weight_loader": weight_loader}
    if output_dim is not None:
        attrs["output_dim"] = output_dim
    set_weight_attrs(p, attrs)
    return p


class GLQEmbeddingMethod(QuantizeMethodBase):
    """GLQ-compressed VocabParallelEmbedding implementation.

    Storage matches ``glq.quantized_linear.E8RHTEmbedding``:
    per-row ``Qidxs[vocab, n_pad/8]`` int16 + scalar/per-row scales + an
    optional residual stage for 3+ bpw. Codebook is the shared 65536-entry
    E8Shell table, lazily moved to the input device on first call.
    """

    def __init__(self, quant_config, bpw: int):
        self.quant_config = quant_config
        self.bpw = int(bpw)

    # ------------------------------------------------------------------
    # vLLM contract
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register GLQ buffers.

        For VocabParallelEmbedding:
          - ``input_size_per_partition`` == embedding_dim
          - ``output_partition_sizes`` == [num_embeddings_per_partition]
          - ``output_size`` == num_embeddings_padded (vLLM may pad vocab to
            tp_size; loader narrowing handles it via output_dim=0)
        """
        embedding_dim = input_size_per_partition
        n_pad = _next_pow2(embedding_dim)
        vocab_per_rank = output_partition_sizes[0]

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is None:
            from vllm.model_executor.utils import default_weight_loader
            weight_loader = default_weight_loader

        # Vocab-sharded buffers (output_dim=0)
        layer.Qidxs = _make_param(
            torch.empty(vocab_per_rank, n_pad // 8, dtype=torch.int16),
            weight_loader, output_dim=0)
        layer.Qidxs2 = _make_param(
            torch.zeros(vocab_per_rank, n_pad // 8, dtype=torch.int16),
            weight_loader, output_dim=0)
        layer.Wscale = _make_param(
            torch.ones(vocab_per_rank, dtype=torch.float32),
            weight_loader, output_dim=0)
        layer.inv_resid_scale = _make_param(
            torch.zeros(vocab_per_rank, dtype=torch.float32),
            weight_loader, output_dim=0)

        # Replicated buffers (no output_dim → loader copies whole tensor)
        layer.SV = _make_param(
            torch.ones(n_pad, dtype=torch.float16),
            weight_loader, output_dim=None)
        # SU is unused at runtime but stored at ``n_pad`` length in older
        # checkpoints and ``[1]`` in newer ones — register at ``n_pad`` so
        # the older published Gemma-4 GLQ models round-trip cleanly. (The
        # newer ``[1]`` shape would also work via the loader's broadcast,
        # but matching ``[n_pad]`` is what the existing checkpoint expects.)
        layer.SU = _make_param(
            torch.ones(n_pad, dtype=torch.float16),
            weight_loader, output_dim=None)

        # Cache shape constants for embedding(); avoids re-deriving each call.
        layer.glq_n_pad = n_pad
        layer.glq_embedding_dim = embedding_dim
        # Compute dtype for the dequant output. vLLM passes params_dtype as
        # the activation dtype (e.g. bf16); the embedding lookup must
        # return that so downstream layers see the right tensor type.
        layer.glq_out_dtype = params_dtype

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        """Required by ``QuantizeMethodBase``. Embeddings never invoke
        ``apply()``; vLLM's ``VocabParallelEmbedding.forward_native`` calls
        ``embedding()`` instead. ``ParallelLMHead.forward`` does call
        ``apply()`` (LM head as a gemm), but our checkpoints only quantize
        the input embedding, never the LM head, so we should never hit
        this path. Raise loudly if we do."""
        raise NotImplementedError(
            "GLQEmbeddingMethod.apply called — only embedding lookup is "
            "supported. If you're trying to quantize the LM head, that's "
            "not currently implemented; use the unquantized path.")

    def embedding(self, layer: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-forward dequant + lookup.

        Returns ``[*input_ids.shape, embedding_dim]`` in ``params_dtype``.
        Does NOT apply Gemma-4's ``embed_scale_per_layer`` — that's
        applied externally by ``Gemma4Model.get_per_layer_inputs`` after
        the lookup. Same convention as the dequant-PLE pipeline.
        """
        cb1, cb2 = _get_codebook_pair(self.bpw, input_ids.device)
        # Stage detection: stage-2 active iff any inv_resid_scale is non-zero.
        # Per-row scales mean we check the whole tensor; cache this once.
        if not hasattr(layer, "_glq_n_stages_cached"):
            layer._glq_n_stages_cached = (
                2 if layer.inv_resid_scale.abs().any().item() else 1
            )
        n_stages = layer._glq_n_stages_cached

        return _dequant_embedding_rows(
            input_ids,
            layer.Qidxs, layer.SV, layer.Wscale, cb1,
            layer.Qidxs2 if n_stages >= 2 else None,
            layer.inv_resid_scale if n_stages >= 2 else None,
            cb2 if n_stages >= 2 else None,
            n_pad=layer.glq_n_pad,
            embedding_dim=layer.glq_embedding_dim,
            embed_scale=1.0,  # vLLM applies its own scale externally
            out_dtype=layer.glq_out_dtype,
        )


# ----------------------------------------------------------------------
# Codebook plumbing
# ----------------------------------------------------------------------

_codebook_cache: dict[torch.device, tuple] = {}


def _get_codebook_pair(bpw: int, device: torch.device):
    """Return (E8Shell codebook, secondary codebook or None) on ``device``.

    Stage-1 is always the 65536-entry E8 shell. Stage-2 uses a smaller
    codebook tied to the bpw target (matches ``GLQLinearMethod`` plumbing).
    """
    cached = _codebook_cache.get(device)
    if cached is not None:
        return cached
    # Reuse the linear-method singleton so we don't allocate twice
    from glq_vllm.linear_method import _get_codebook, _get_codebook2
    cb_full = _get_codebook()
    cb1 = cb_full.codebook.to(device)
    cb2_full = _get_codebook2(bpw)
    cb2 = cb2_full.codebook.to(device) if cb2_full is not None else None
    _codebook_cache[device] = (cb1, cb2)
    return cb1, cb2
