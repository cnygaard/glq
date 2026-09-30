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


def _cpu_offload_requested() -> bool:
    """Is Engram CPU offload on? Read the same source vLLM reads, not the env var.

    ``get_current_vllm_config().engram_config`` is what ``Qwen4ExpNGramEmbedding.__init__``
    consults to choose the embedding class, so reading it here cannot disagree with the
    decision being predicted. Absent ambient config (unit tests) means "not offloading".
    """
    try:
        from vllm.config import get_current_vllm_config
        engram = get_current_vllm_config().engram_config
    except Exception:
        return False
    return engram is not None and bool(getattr(engram, "cpu_offload", False))


def _etp_world_size() -> int:
    """ETP shard count, or 1 when there is no distributed state (unit tests)."""
    try:
        from vllm.distributed import get_etp_group
        return int(get_etp_group().world_size)
    except Exception:
        return 1


def _refuse_unsupported_cpu_offload(method) -> None:
    """Refuse the offload combinations GLQ still cannot serve, naming the cause and the fix.

    A *trellis*-coded table at ETP=1 IS supported -- see ``_glq_pinned_host_cls``, which keeps
    ``trellis_packed`` (23.8 GiB on Qwen3.8-Flash-Next) in pinned host memory and gathers its
    rows over UVA. What remains unsupported is refused here rather than left to fail somewhere
    that names neither GLQ nor the setting.
    """
    if not _cpu_offload_requested():
        return

    codebook = getattr(method, "codebook", None)
    if codebook != "trellis":
        raise NotImplementedError(
            "GLQ supports Engram CPU offload only for a *trellis*-coded Qwen4Exp PLE table; "
            f"this checkpoint's PLE is {codebook!r}. The shell layout stores "
            "Qidxs/Qidxs2/inv_resid_scale at power-of-two row width and has no offload path "
            "yet. Set VLLM_PLE_CPU_OFFLOAD=0, or pass "
            "--engram-config '{\"cpu_offload\": false}'.")

    etp = _etp_world_size()
    if etp > 1:
        raise NotImplementedError(
            f"GLQ's Qwen4Exp PLE CPU offload is single-shard only, but ETP={etp}. vLLM's dense "
            "pinned lookup masks out-of-range vocab rows inside its own Triton kernel "
            "(org_vocab_start_index/org_vocab_end_index); GLQ's row decode does not, so a "
            "sharded table would decode foreign rows instead of failing. Set "
            "VLLM_PLE_CPU_OFFLOAD=0, or serve with ETP=1.")


def _glq_pinned_host_cls(base):
    """Build GLQ's pinned-host PLE layer as a subclass of vLLM's, overriding one method.

    vLLM's ``Qwen4ExpPLEPinnedHostEmbedding._lookup`` gathers **dense** rows with a Triton
    kernel over ``self._uva_weight`` and never consults ``embedding_method``, which is why a
    quantized table cannot ride the stock path. Everything *around* the lookup -- the prefetch
    side stream, the ETP reduce, ``forward`` -- operates on decoded activation-dtype rows and
    so needs no change, hence exactly one override.

    ``__init__`` is deliberately NOT overridden. It builds ``_uva_weight``,
    ``_prefetch_stream`` and ``_prefetch_buffer`` from ``self.weight``, and GLQ satisfies that
    with a one-element pinned placeholder (``GLQEmbeddingMethod._register_weight_placeholder``)
    whose dtype is the activation dtype and whose UVA view is on the GPU -- precisely what
    those three need. Reimplementing ``__init__`` would hardcode vLLM internals that a release
    can change; this way the only vLLM behaviour GLQ depends on is the ``_lookup`` seam.

    Defined lazily inside a function because the base class only exists on vLLM >= 0.30.0.
    """

    class GLQQwen4ExpPLEPinnedHostEmbedding(base):    # type: ignore[misc, valid-type]
        """A GLQ-quantized PLE table in pinned host memory, gathered over UVA."""

        _glq_offloaded = True

        def _lookup(self, input_ids, output=None):
            """Gather packed rows across PCIe and decode them on the GPU.

            ``embedding_method.embedding`` already performs gather + trellis decode + inverse
            RHT and returns ``[*input_ids.shape, embedding_dim]`` in the activation dtype --
            the exact shape and dtype ``_prefetch_buffer`` expects -- and it picks up the UVA
            view via ``layer.glq_uva_packed``. So this is the resident decode path, unchanged,
            pointed at host-resident rows.
            """
            rows = self.embedding_method.embedding(self, input_ids)
            if output is None:
                return rows
            output.copy_(rows)
            return output

    return GLQQwen4ExpPLEPinnedHostEmbedding


def _install_pinned_host() -> bool:
    """Route GLQ-quantized PLE tables to GLQ's pinned-host layer when offload is on.

    The class is chosen inside ``Qwen4ExpNGramEmbedding.__init__`` from a module global
    (``Qwen4ExpPLEPinnedHostEmbedding if engram_config.cpu_offload else ...Device...``).
    Because that name is resolved at call time, replacing the module attribute is enough --
    the same seam ``_install_modern`` uses, with the same exposure: a rename makes this a
    no-op, so ``install()`` warns when the attribute is absent and the test resolves the
    symbol instead of hardcoding it.

    The shim dispatches on the passed ``embedding_method``, so FP8 and unquantized PLE tables
    keep vLLM's own class byte-for-byte.
    """
    try:
        from vllm.models.qwen4_exp.nvidia import ngram_embedding
    except ImportError:
        return False
    current = getattr(ngram_embedding, "Qwen4ExpPLEPinnedHostEmbedding", None)
    if current is None:
        return False
    if getattr(current, "_glq_shim", False):
        return True

    original = current
    cache: dict[str, type] = {}

    def _shim(*args, **kwargs):
        method = kwargs.get("embedding_method")
        if method is None:
            # Positional form: `embedding_method` is keyword-only at vLLM's call site, but do
            # not assume a future release keeps it that way.
            method = next((a for a in args if hasattr(a, "embedding")), None)
        from .embedding_method import GLQEmbeddingMethod
        if isinstance(method, GLQEmbeddingMethod):
            cls = cache.get("cls")
            if cls is None:
                cls = cache["cls"] = _glq_pinned_host_cls(original)
            return cls(*args, **kwargs)
        return original(*args, **kwargs)

    _shim._glq_shim = True
    _shim._glq_original = original
    ngram_embedding.Qwen4ExpPLEPinnedHostEmbedding = _shim
    return True


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
            _refuse_unsupported_cpu_offload(method)
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

    # Independent of the quant-method seam: only >= 0.30.0 has a pinned-host class, and a
    # build without one simply has nothing to offload, so a False here is not an error.
    _install_pinned_host()

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
