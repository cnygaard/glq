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

import math

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from glq.quantized_linear import _dequant_embedding_rows


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _alloc_storage(layer: nn.Module, rows: int, cols: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """Ask the LAYER for row storage, so an offload-capable layer can hand back pinned memory.

    vLLM's own PLE methods do exactly this -- both ``Qwen4ExpPLEUnquantizedEmbeddingMethod``
    and ``...Fp8EmbeddingMethod`` call ``layer.allocate_embedding_weight(...)`` from inside
    ``create_weights`` (``vllm/models/qwen4_exp/nvidia/ngram_embedding.py``). The device layer
    answers with an ordinary ``torch.empty`` and the pinned-host layer answers with
    page-locked CPU memory, so one code path serves both and GLQ never decides placement.

    Why this matters rather than being cosmetic: vLLM constructs the model under an ambient
    CUDA device context, so a plain ``torch.empty`` here lands on the **GPU**. That is what we
    want for the resident path and exactly what we must avoid for the offloaded one -- the
    pinned allocator passes ``device="cpu", pin_memory=True`` explicitly and so overrides the
    ambient context. Falls back to ``torch.empty`` for any layer without the hook (Gemma-4's
    PLE, and every unit test that passes a bare ``nn.Module``).
    """
    alloc = getattr(layer, "allocate_embedding_weight", None)
    if alloc is None:
        return torch.empty(rows, cols, dtype=dtype)
    return alloc(rows, cols, dtype)


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

    def __init__(self, quant_config, bpw: int, codebook: str = "shell",
                 variant: str = "3inst"):
        self.quant_config = quant_config
        self.bpw = int(bpw)
        # Which codebook the TABLE was coded with, which is not always the run's codebook:
        # a PLE can be trellis-coded in an otherwise-shell checkpoint and vice versa. It has
        # to be known here because create_weights registers buffers before the checkpoint
        # loads, and the two layouts have different shapes and dtypes.
        self.codebook = codebook
        self.variant = variant

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

        if self.codebook == "trellis":
            # Block-diagonal RHT, so the row is NOT padded to a power of two: the buffer is
            # embedding_dim wide, not n_pad. ceil(width*K/16) int16 per row -- 60 B at
            # width 160, K=3, against shell's 64 B for the same row padded to 256.
            from glq.quantized_linear import _pow2_blocks
            # Routed through the layer (see ``_alloc_storage``) so a pinned-host PLE layer can
            # place this 23.8 GiB table in page-locked CPU memory. FULL-SIZE on purpose: the
            # loader ``copy_``s into it in place, which is what keeps a UVA view over it valid.
            layer.trellis_packed = _make_param(
                _alloc_storage(layer, vocab_per_rank,
                               math.ceil(embedding_dim * self.bpw / 16), torch.int16),
                weight_loader, output_dim=0)
            layer.Wscale = _make_param(
                torch.ones(vocab_per_rank, dtype=torch.float16),
                weight_loader, output_dim=0)
            layer.SV = _make_param(
                torch.ones(embedding_dim, dtype=torch.float16),
                weight_loader, output_dim=None)
            layer.rht_blocks = _make_param(
                torch.tensor(_pow2_blocks(embedding_dim), dtype=torch.int32),
                weight_loader, output_dim=None)
            layer.glq_embedding_dim = embedding_dim
            layer.glq_out_dtype = params_dtype
            self._register_weight_placeholder(layer, params_dtype)
            return

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

        self._register_weight_placeholder(layer, params_dtype)

    @staticmethod
    def _register_weight_placeholder(layer: nn.Module,
                                     params_dtype: torch.dtype) -> None:
        """A 0-element ``weight``, purely to survive a vLLM >= 0.30.0 LOG line.

        ``Qwen4ExpNGramEmbedding.__init__`` (``vllm/models/qwen4_exp/nvidia/
        ngram_embedding.py:719``) does this unconditionally, regardless of quant method::

            weight = self.ngram_embedding.weight
            logger.info("Initialized PLE embedding %s: ... weight_dtype=%s, "
                        "weight_device=%s, pinned=%s", ..., weight.dtype,
                        weight.device, weight.is_pinned())

        vLLM's own methods happen to ``register_parameter("weight", ...)`` inside
        ``create_weights``; GLQ registers ``trellis_packed``/``Qidxs``/``SU``/``SV``/
        ``Wscale`` and no dense table, so an **info log** takes the engine down with
        ``'Qwen4ExpPLEDeviceEmbedding' object has no attribute 'weight'``. That is a vLLM
        bug -- it breaks any quant method whose storage is not named ``weight`` -- and is
        worth reporting upstream; this keeps GLQ loadable meanwhile.

        Deliberately a **0-element** tensor, and a plain attribute rather than a parameter
        or buffer: the weight loader iterates ``named_parameters``/``named_buffers``, so
        this stays invisible to loading, and if a future vLLM ever *uses* it for real the
        empty shape fails loudly instead of silently returning wrong rows.
        ``dtype``/``device``/``is_pinned`` -- everything the log touches -- work on it.

        Called from **both** branches of ``create_weights``. The first version of this was
        only in the shell tail, and the trellis branch returns before reaching it, so it
        was a no-op on exactly the checkpoints that need it.
        """
        if hasattr(layer, "weight"):
            return
        if getattr(layer, "supports_prefetch", False):
            # The pinned-host PLE layer does more than log: its ``__init__`` runs
            # ``get_accelerator_view_from_cpu_tensor(self.weight)`` and then takes
            # ``_uva_weight.device`` for the prefetch stream and buffer. A 0-element,
            # non-pinned tensor cannot produce a valid UVA view, so give it the smallest
            # thing that can -- one page-locked element, allocated through the layer so it
            # is pinned by the same code that pins the table. GLQ's ``_lookup`` override
            # never reads ``_uva_weight``; this exists only so vLLM's own ``__init__``
            # composes. Still 1 element, so any attempt to use it as a real table fails on
            # the very next index rather than returning plausible rows.
            layer.weight = _alloc_storage(layer, 1, 1, params_dtype).reshape(-1)
            return
        # Report the device GLQ's own buffers are on, so the log line is not misleading.
        dev = next((p.device for p in layer.parameters()), None)
        layer.weight = torch.empty(0, dtype=params_dtype, device=dev)

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

    #: vLLM >= 0.30.0's ``Qwen4ExpPLEEmbeddingMethod`` carries this; its PLE loader reads it
    #: to decide whether post-load processing must run on the device. GLQ's tables are
    #: already in their runtime layout after ``process_weights_after_loading``, so False
    #: matches what vLLM's own FP8 and unquantized PLE methods declare. Harmless on 0.29.0.
    requires_device_loading: bool = False

    def dequantize(self, layer: nn.Module, embeddings: torch.Tensor,
                   output_dtype: torch.dtype) -> torch.Tensor:
        """Required by vLLM >= 0.30.0, which SPLIT the PLE lookup into two calls.

        0.29.0 had one step: ``embedding()`` returned activation-dtype rows. 0.30.0 calls
        ``embedding()`` for raw rows and then ``Qwen4ExpPLEEmbedding.dequantize`` ->
        ``embedding_method.dequantize`` to convert them (``ngram_embedding.py``, and
        ``ple_layer.py``'s ``_dequantize_embeddings``).

        GLQ's ``embedding()`` already decodes all the way to ``params_dtype``, so there is
        nothing left to convert and this is a pass-through — the same thing vLLM's own
        ``Qwen4ExpPLEUnquantizedEmbeddingMethod.dequantize`` does. The cast is kept only for
        the case where the model's activation dtype differs from the table's ``params_dtype``;
        it is a no-op when they agree.

        Deliberately NOT a subclass of ``Qwen4ExpPLEEmbeddingMethod``: that class does not
        exist on 0.29.0, and vLLM reaches this by duck typing
        (``self.embedding_method.dequantize(...)``), so inheriting would buy nothing and
        would pin GLQ to one vLLM version.
        """
        del layer
        if embeddings.dtype == output_dtype:
            return embeddings
        return embeddings.to(output_dtype)

    @staticmethod
    def _prepare_uva_packed(layer: nn.Module) -> torch.device:
        """Build the GPU-addressable view of a host-resident table; return the COMPUTE device.

        Called at load time, which is the only correct moment: the view wraps the storage
        ``trellis_packed`` already owns, and the weight loader has by then ``copy_``d the
        checkpoint into that storage in place. Building it in ``__init__`` would work only
        while nothing reallocates the parameter -- a fragile invariant to rely on.

        Returns the device the *decode* must run on, which is deliberately NOT
        ``trellis_packed.device``: with CPU offload the packed rows live on the host while the
        lut, SV, Wscale and the arithmetic all stay on the GPU. Reading the device off the
        packed tensor (as this used to) would quietly move the whole decode to the CPU.
        """
        packed = layer.trellis_packed
        layer.glq_uva_packed = None
        if packed.device.type == "cpu" and packed.is_pinned():
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
            layer.glq_uva_packed = get_accelerator_view_from_cpu_tensor(packed)
            # Announce the mechanism, not just the outcome: a table that quietly stayed
            # resident still generates correct tokens, so this line (and the footprint) is
            # what distinguishes "offloaded" from "the patch was a no-op".
            print(f"GLQPLE offload=ON pinned={packed.is_pinned()} "
                  f"host_rows={tuple(packed.shape)} bytes={packed.numel() * 2 / 2**30:.3f}GiB "
                  f"uva_device={layer.glq_uva_packed.device}", flush=True)
            return layer.glq_uva_packed.device
        print(f"GLQPLE offload=OFF resident_device={packed.device} "
              f"bytes={packed.numel() * 2 / 2**30:.3f}GiB", flush=True)
        return packed.device

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Cache the codebook + stage count on the layer at LOAD time.

        vLLM calls this once after the checkpoint is loaded. Doing the codebook
        acquisition (which touches the filesystem via ``os.path.exists`` +
        ``E8ShellCodebook.load``) and the stage-count ``.item()`` host sync here
        — rather than in ``embedding()`` — keeps the per-forward path free of
        ``os.stat``/disk-IO/host-syncs, so vLLM's ``torch.compile`` pass can
        trace it and CUDA graphs can capture it (mirrors
        ``GLQLinearMethod.process_weights_after_loading`` + ``_ensure_codebook``;
        the GLQ-quantized Gemma-4 PLE embedding was the one path still breaking
        compile with ``dynamo.exc.Unsupported: posix.stat``).
        """
        if self.codebook == "trellis":
            # Rebuild the codebook once, at load, and keep only its lut on the layer: the
            # per-forward path must stay free of construction and host syncs so vLLM can
            # trace and graph-capture it.
            from glq.trellis import TrellisCodebook
            dev = self._prepare_uva_packed(layer)
            tlut = getattr(layer, "tlut", None)
            cb = TrellisCodebook(variant=self.variant, K=self.bpw, device=dev,
                                 tlut=tlut)
            layer.glq_trellis_lut = cb.cb.lut.to(dev)
            layer.glq_trellis_LKV = (int(cb.cb.L), int(cb.cb.K), int(cb.cb.V))
            # Materialize the block sizes ONCE, here. Reading them from the buffer inside
            # the op is a device-to-host copy, and vLLM captures this lookup in a CUDA
            # graph, where that raises "Cannot copy between CPU and CUDA tensors".
            layer.glq_blocks_n = [int(b) for b in layer.rht_blocks.tolist()]
            return
        dev = layer.Qidxs.device
        cb1, cb2 = _get_codebook_pair(self.bpw, dev)
        layer.glq_cb1 = cb1
        layer.glq_cb2 = cb2
        # stage-2 active iff any inv_resid_scale is non-zero (per-row scales).
        layer._glq_n_stages_cached = (
            2 if layer.inv_resid_scale.abs().any().item() else 1)

    def embedding(self, layer: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-forward dequant + lookup.

        Returns ``[*input_ids.shape, embedding_dim]`` in ``params_dtype``.
        Does NOT apply Gemma-4's ``embed_scale_per_layer`` — that's applied
        externally by ``Gemma4Model.get_per_layer_inputs`` after the lookup.

        Routes the dequant through the **registered** ``torch.ops.glq.embedding_dequant``
        op (not the raw helper) so vLLM's torch.compile sees one opaque node: the
        helper's ``fast_hadamard_transform`` is a kernel dynamo can't trace, which
        otherwise breaks compile (the GLQ-quantized Gemma-4 PLE embedding was the
        last path doing so). The codebook + n_stages are cached at load time
        (``process_weights_after_loading``), so this path has no os.stat/disk/host
        sync — it compiles + CUDA-graph-captures cleanly.
        """
        if self.codebook == "trellis":
            if getattr(layer, "glq_trellis_lut", None) is None:
                self.process_weights_after_loading(layer)
            L, K, V = layer.glq_trellis_LKV
            # With CPU offload, hand the op the UVA view rather than the host parameter: the
            # op takes its device from this tensor and gathers with ``index_select``, so a
            # CUDA view means the rows cross PCIe while the decode stays on the GPU. Passing
            # the host tensor instead would run the entire decode on the CPU.
            packed = getattr(layer, "glq_uva_packed", None)
            if packed is None:
                packed = layer.trellis_packed
            return torch.ops.glq.embedding_dequant_trellis(
                input_ids, packed, layer.SV, layer.Wscale,
                layer.glq_trellis_lut, layer.glq_blocks_n,
                layer.glq_embedding_dim, L, K, V, 1.0, layer.glq_out_dtype)
        # Cached at load time; defensive lazy-fill for a direct (non-vLLM) call.
        cb1 = getattr(layer, "glq_cb1", None)
        if cb1 is None:
            self.process_weights_after_loading(layer)
            cb1 = layer.glq_cb1
        cb2 = layer.glq_cb2
        n_stages = layer._glq_n_stages_cached
        return torch.ops.glq.embedding_dequant(
            input_ids, layer.Qidxs, layer.SV, layer.Wscale, cb1,
            layer.Qidxs2 if n_stages >= 2 else None,
            layer.inv_resid_scale if n_stages >= 2 else None,
            cb2 if n_stages >= 2 else None,
            layer.glq_n_pad, layer.glq_embedding_dim, 1.0, layer.glq_out_dtype)


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
