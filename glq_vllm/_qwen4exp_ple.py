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

import warnings


def _glq_method(quant_config, prefix):
    """GLQ's method for this table, or None to defer to vLLM's answer.

    Only claims tables this checkpoint actually quantized; everything else -- including FP8
    checkpoints and configs we don't recognise -- keeps vLLM's answer.
    """
    from .config import GLQvLLMConfig
    if isinstance(quant_config, GLQvLLMConfig):
        return quant_config.embedding_quant_method(prefix)
    return None


def _install_legacy(ple_layer) -> bool:
    """vLLM <= 0.29.0: a module-level ``_get_ple_embedding_quant_method(quant_config, prefix)``."""
    original = getattr(ple_layer, "_get_ple_embedding_quant_method", None)
    if original is None:
        return False
    if getattr(original, "_glq_wrapped", False):
        return True

    def _hook(quant_config, prefix):
        return _glq_method(quant_config, prefix) or original(quant_config, prefix)

    _hook._glq_wrapped = True
    _hook._glq_original = original
    ple_layer._get_ple_embedding_quant_method = _hook
    return True


def _refuse_if_engram_cpu_offload() -> None:
    """Fail with a message that names the cause, before vLLM fails with one that doesn't.

    vLLM >= 0.30.0 defaults ``VLLM_PLE_CPU_OFFLOAD=1`` (``envs.py``), which makes
    ``Qwen4ExpNGramEmbedding`` pick ``Qwen4ExpPLEPinnedHostEmbedding`` -- a table held in
    pinned CPU memory and read through a UVA view that ``__init__`` builds from a single
    dense ``self.weight``. GLQ registers ``Qidxs``/``SU``/``SV``/``Wscale`` and no such
    tensor, so that path dies with::

        AttributeError: 'Qwen4ExpPLEPinnedHostEmbedding' object has no attribute 'weight'

    which names neither GLQ nor the setting responsible. Read the *same* source vLLM reads
    (``get_current_vllm_config().engram_config``) rather than the env var, so this cannot
    disagree with the decision it is predicting.
    """
    try:
        from vllm.config import get_current_vllm_config
        engram = get_current_vllm_config().engram_config
    except Exception:            # no ambient config yet (unit tests) -- nothing to check
        return
    if engram is not None and getattr(engram, "cpu_offload", False):
        raise NotImplementedError(
            "GLQ cannot serve the Qwen4Exp PLE table with Engram CPU offload enabled: "
            "vLLM's pinned-host embedding builds a UVA view from one dense `weight`, and "
            "GLQ's table is quantized into Qidxs/SU/SV/Wscale. Set VLLM_PLE_CPU_OFFLOAD=0, "
            "or pass --engram-config '{\"cpu_offload\": false}'. Note vLLM >= 0.30.0 "
            "defaults this ON, so it must be turned off explicitly.")


def _install_modern() -> bool:
    """vLLM >= 0.30.0: a staticmethod ``Qwen4ExpPLEEmbeddingMethod.from_quant_config``.

    0.30.0 moved ``Qwen4ExpNGramEmbedding`` into its own module, deleted the old helper, and
    replaced it with this. It also stopped falling through for unknown configs -- it now
    raises ``NotImplementedError("...does not support quantization config GLQvLLMConfig")``,
    so without this hook GLQ fails loudly rather than silently loading the table dense.
    """
    try:
        from vllm.models.qwen4_exp.nvidia import ngram_embedding
    except ImportError:
        return False
    cls = getattr(ngram_embedding, "Qwen4ExpPLEEmbeddingMethod", None)
    original = getattr(cls, "from_quant_config", None) if cls is not None else None
    if original is None:
        return False
    if getattr(original, "_glq_wrapped", False):
        return True

    # `embedding_dtype` is new in 0.30.0 and defaulted, so accept it positionally or by
    # keyword and pass it straight through rather than dropping it.
    def _hook(quant_config, prefix, *args, **kwargs):
        method = _glq_method(quant_config, prefix)
        if method is not None:
            # Only when GLQ actually claims the table: an FP8 or unquantized PLE is free to
            # use the pinned-host path, and refusing there would break configurations that
            # have nothing to do with us.
            _refuse_if_engram_cpu_offload()
            return method
        from .config import GLQvLLMConfig
        if isinstance(quant_config, GLQvLLMConfig):
            # 0.30.0 changed the decline path from "return None and let vLLM fall back" to
            # "raise NotImplementedError for any config that is not Fp8Config". A GLQ
            # checkpoint whose PLE table stayed bf16 is a legitimate configuration -- it is
            # what `test_a_table_absent_from_layer_bpw_is_left_alone` pins -- so answering
            # with vLLM's own unquantized method is correct here. Delegating instead would
            # crash on a checkpoint that works fine on 0.29.0.
            return ngram_embedding.Qwen4ExpPLEUnquantizedEmbeddingMethod()
        return original(quant_config, prefix, *args, **kwargs)

    _hook._glq_wrapped = True
    _hook._glq_original = original
    cls.from_quant_config = staticmethod(_hook)
    return True


def install() -> None:
    """Route GLQ into Qwen4Exp's PLE table on whichever vLLM API this build exposes.

    No-op -- deliberately silent -- on vLLM builds without Qwen4Exp, since ``register()``
    calls this in every process regardless of which model is being served.

    But if Qwen4Exp IS present and NEITHER dispatch point is found, that is warned about
    loudly. The previous version returned silently in that case, which is exactly how the
    0.29.0 -> 0.30.0 rename went unnoticed: the hook became a no-op and the only symptom was
    an OOM (0.29.0) or a NotImplementedError (0.30.0) with nothing naming GLQ.
    """
    try:
        from vllm.models.qwen4_exp.nvidia import ple_layer
    except ImportError:
        return

    if _install_legacy(ple_layer) or _install_modern():
        return

    warnings.warn(
        "GLQ: vLLM has Qwen4Exp but neither PLE quant-method dispatch point was found "
        "(<=0.29.0 'ple_layer._get_ple_embedding_quant_method', >=0.30.0 "
        "'ngram_embedding.Qwen4ExpPLEEmbeddingMethod.from_quant_config'). The PLE n-gram "
        "table will NOT use GLQ: expect a dense bf16 allocation (~95 GiB on "
        "Qwen3.8-Flash-Next) or a NotImplementedError naming GLQvLLMConfig. This vLLM "
        "version needs a new hook in glq_vllm/_qwen4exp_ple.py.",
        RuntimeWarning, stacklevel=2)
