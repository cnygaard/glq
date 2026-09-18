"""Route GLQ into Qwen4Exp's per-layer-embedding table.

Every other quantized layer reaches GLQ through ``QuantizationConfig.get_quant_method``.
Qwen4Exp's n-gram table does not. vLLM builds it with an explicitly passed method::

    self.ngram_embedding = PLEVocabParallelEmbedding(
        padded_vocab_size, self.head_dim, ...,
        quant_method=_get_ple_embedding_quant_method(
            quant_config, f"{prefix}.ngram_embedding"))

and that helper hardcodes a single quantizer::

    \"\"\"Select global-scale FP8 only for quantized PLE checkpoint shards.\"\"\"
    if not isinstance(quant_config, Fp8Config):
        return None

GLQ therefore returns ``None``, ``VocabParallelEmbedding.__init__`` substitutes
``UnquantizedEmbeddingMethod``, and the table is built **dense in bf16**. On
Qwen3.8-Flash-Next that is one ``torch.empty(320_001_536, 160)`` — **95.37 GiB** — which
OOMs a 94.97 GiB card during ``load_model``, with a traceback that never names GLQ.

The table is not a rounding error: at 4 bpw it is 24.44 GiB of a 76.58 GiB checkpoint,
and leaving it bf16 puts the model over two 96 GiB cards instead of one.

There is no registration hook for this, so we wrap the function. That is a patch of a
private name in a third-party package, with the usual consequence: a vLLM release that
renames or removes it turns this into a silent no-op and the OOM comes back. Hence the
wrapper is idempotent, keeps ``_glq_original``, and is asserted on directly by
``tests/test_qwen4exp_ple_vllm_hook.py`` rather than inferred from "the model loaded" —
which is also what a fall-through to dense bf16 looks like until the allocator gives out.

The right long-term fix is upstream: that helper should consult
``quant_config.get_quant_method`` and fall back to FP8, rather than naming one quantizer.
"""

from __future__ import annotations


def install() -> None:
    """Wrap vLLM's FP8-only PLE quant-method helper so GLQ configs are honoured.

    No-op — deliberately silent — on vLLM builds without Qwen4Exp, since ``register()``
    calls this in every process regardless of which model is being served.
    """
    try:
        from vllm.models.qwen4_exp.nvidia import ple_layer
    except ImportError:
        return

    original = getattr(ple_layer, "_get_ple_embedding_quant_method", None)
    if original is None or getattr(original, "_glq_wrapped", False):
        return

    def _hook(quant_config, prefix):
        # Only claim tables this checkpoint actually quantized; everything else —
        # including FP8 checkpoints and configs we don't recognise — keeps vLLM's answer.
        from .config import GLQvLLMConfig
        if isinstance(quant_config, GLQvLLMConfig):
            method = quant_config.embedding_quant_method(prefix)
            if method is not None:
                return method
        return original(quant_config, prefix)

    _hook._glq_wrapped = True
    _hook._glq_original = original
    ple_layer._get_ple_embedding_quant_method = _hook
