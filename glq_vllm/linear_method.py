"""GLQ linear method for vLLM — fused dequant+matmul (Phase 2).

Keeps compressed GLQ indices (Qidxs, SU, SV) in GPU memory.
Dequant on-the-fly in apply() using GLQ's CUDA C and Triton kernels.

Supports:
- Standard linear layers (ColumnParallel, RowParallel, Replicated)
- Fused QKV layers (QKVParallelLinear) — per-shard GLQ buffers
"""

import math
import os
import sys
from typing import Callable

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import BasevLLMParameter

from glq.codebook import E8ShellCodebook
from glq import inference_kernel as _ik
from glq.inference_kernel import glq_dequant_matmul, _try_load_cuda_ext
from glq.hadamard import _block_decompose
from glq.quantized_linear import _pack_block_meta

# CUDA C FHT kernels are 1.3-1.6× faster than Triton. JIT-compiled on first use.
_VLLM_USE_TRITON = False


# Shared codebook singleton — moved to GPU on first use
_codebook = None
_codebook2_small = None
_codebook_device = None

# Per-device cache of empty sentinel tensors used in the block-diag fast
# path (passed to the kernel as ``cb_packed`` / ``Qidxs2`` etc. when the
# corresponding stage is inactive). Allocating ``torch.empty(0, ...)`` on
# every apply was ~8 extra dispatches per layer.
_empty_sentinels: dict[torch.device, tuple] = {}


def _get_empty_sentinels(device):
    s = _empty_sentinels.get(device)
    if s is None:
        s = (
            torch.empty(0, dtype=torch.int16, device=device),
            torch.empty(0, dtype=torch.float16, device=device),
        )
        _empty_sentinels[device] = s
    return s


_codebook_e8p = None
_codebook_e81b = None
_codebook_e8p_device = None


def _ensure_codebook(device, max_bpw: int = 2, codebook_type: str = "e8_shell"):
    """Lazy-load codebook and move to target device. Upgrades cb2 if higher bpw needed.

    For ``codebook_type == "e8p"`` returns the E8PCodebook (.grid_packed_abs) + E81BCodebook
    (.e81b_grid) singletons instead of the shell pair."""
    global _codebook, _codebook2_small, _codebook_device
    global _codebook_e8p, _codebook_e81b, _codebook_e8p_device

    if codebook_type == "e8p":
        if _codebook_e8p is not None and _codebook_e8p_device == device:
            return _codebook_e8p, _codebook_e81b
        from glq.codebook_e8p import E8PCodebook, E81BCodebook
        # vLLM sets the default dtype to fp16 during model init; the codebook's
        # construction matmuls (E8P nearest-codeword `round`) assume fp32, so pin it.
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            cb = E8PCodebook(device=device, verbose=False)
            cb2 = E81BCodebook(device=device, verbose=False)
        finally:
            torch.set_default_dtype(_prev_dtype)
        _codebook_e8p, _codebook_e81b, _codebook_e8p_device = cb, cb2, device
        return cb, cb2

    if _codebook is not None and _codebook_device == device:
        # Upgrade codebook2 if a higher bpw layer is encountered later
        if max_bpw >= 3 and _codebook2_small is None:
            # Always use the full codebook for cb2 — 4bpw layers have
            # indices up to 65535, 3bpw indices (0-255) are a valid subset
            _codebook2_small = _codebook
        return _codebook, _codebook2_small

    cb_path = os.path.join(os.path.dirname(__file__), "..", "glq", "e8_codebook.pt")
    if os.path.exists(cb_path):
        cb = E8ShellCodebook.load(cb_path, device="cpu")
    else:
        cb = E8ShellCodebook(device="cpu", verbose=False)

    # Always use full codebook for cb2 — supports both 3bpw and 4bpw
    cb2 = cb if max_bpw >= 3 else None

    cb._move_to_device(device)
    if cb2 is not None and cb2 is not cb:
        cb2._move_to_device(device)

    _codebook = cb
    _codebook2_small = cb2
    _codebook_device = device
    return cb, cb2


# --- Standalone helpers (no vLLM dependency, for testing) ---

def _get_codebook():
    cb_path = os.path.join(os.path.dirname(__file__), "..", "glq", "e8_codebook.pt")
    if os.path.exists(cb_path):
        return E8ShellCodebook.load(cb_path, device="cpu")
    return E8ShellCodebook(device="cpu", verbose=False)


def _get_codebook2(bpw: int):
    cb = _get_codebook()
    if bpw >= 4:
        return cb
    elif bpw >= 3:
        return cb.make_small(256)
    return None


def dequantize_glq_weight(
    Qidxs, SU, SV, Wscale, codebook,
    Qidxs2=None, inv_resid_scale=0.0, codebook2=None,
    out_features=None, in_features=None,
):
    """Dequantize GLQ indices to a dense weight matrix (CPU, for testing)."""
    from glq.hadamard import fast_hadamard_transform

    m_pad, n_blocks = Qidxs.shape
    n_pad = n_blocks * 8
    W_rht = codebook.decode(Qidxs.long().reshape(-1)).reshape(m_pad, n_pad).float()
    if Qidxs2 is not None and inv_resid_scale != 0.0 and codebook2 is not None:
        W_rht2 = codebook2.decode(Qidxs2.long().reshape(-1)).reshape(m_pad, n_pad).float()
        W_rht = W_rht + W_rht2 * inv_resid_scale
    W_rht = W_rht * Wscale.float()
    W = fast_hadamard_transform(W_rht.clone())
    W = W * SV.float().unsqueeze(0)
    W = fast_hadamard_transform(W.T.clone()).T
    W = W * SU.float().unsqueeze(1)
    if out_features is not None and in_features is not None:
        W = W[:out_features, :in_features]
    return W.half()


# --- GLQ buffer helpers ---

def _glq_pad(n):
    """Next power of 2."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _detect_block_diag(m_pad: int, n_pad: int):
    """Return (is_block_diag, blocks_m, blocks_n) for a given buffer shape.

    Block-diagonal checkpoints (Phase B, post-glq v0.2.9) land non-power-of-2
    ``m_pad``/``n_pad`` equal to the true ``out_features``/``in_features``.
    Legacy pow2 checkpoints land the padded size. Decomposing the exact dim
    into a sum of power-of-2 blocks lets the fused kernel run the multiblock
    FHT without padding waste.
    """
    is_bd = not (_is_pow2(m_pad) and _is_pow2(n_pad))
    blocks_m = _block_decompose(m_pad) if is_bd else [m_pad]
    blocks_n = _block_decompose(n_pad) if is_bd else [n_pad]
    return is_bd, blocks_m, blocks_n


def _glq_weight_loader(param, loaded_weight, *args, **kwargs):
    """GLQ weight loader — handles per-expert, shared, and standard params."""
    expert_id = kwargs.get('expert_id')
    shard_id = kwargs.get('shard_id')
    if expert_id is not None:
        if param.data.dim() >= 2:
            # Per-expert tensor: Qidxs (E, m_pad, nblk) or SU (E, m_pad).
            slot = param.data[expert_id]
            # Gated w13 = [gate; up] stacked: gate (shard "w1") fills the first
            # half of the output rows, up (shard "w3") the second half. The GLQ
            # checkpoint stores gate_proj/up_proj jointly-quantized then row-
            # split, so each loaded half has exactly m_pad/2 rows and shares one
            # Wscale/SV — placing them into the two halves reassembles an exact
            # w13. (Down-proj / shard "w2" is a single full-width projection.)
            if shard_id in ('w1', 'w3') and loaded_weight.shape[0] * 2 == slot.shape[0]:
                off = 0 if shard_id == 'w1' else slot.shape[0] // 2
                slot[off:off + loaded_weight.shape[0]].copy_(loaded_weight)
            else:
                # Full per-expert copy (w2/down, or non-split). Resize the slot
                # if its allocated shape doesn't match the loaded weight.
                if slot.shape != loaded_weight.shape:
                    new_shape = list(param.data.shape)
                    for d in range(loaded_weight.dim()):
                        new_shape[d + 1] = loaded_weight.shape[d]
                    param.data = torch.zeros(new_shape, dtype=param.data.dtype,
                                             device=param.data.device)
                param.data[expert_id].copy_(loaded_weight)
        elif param.data.dim() == 1 and loaded_weight.dim() == 0:
            # Per-expert scalar (Wscale, inv_resid_scale): param is (num_experts,)
            param.data[expert_id] = loaded_weight.item()
        elif param.data.dim() == 1 and loaded_weight.dim() >= 1:
            # Shared 1D param (SV): copy once on first expert
            if expert_id == 0:
                if param.data.shape != loaded_weight.shape:
                    param.data = torch.empty_like(loaded_weight)
                param.data.copy_(loaded_weight)
        elif param.data.dim() == 0:
            # Scalar param: just copy
            param.data.copy_(loaded_weight)
        return True
    # Non-expert path: auto-resize and copy
    if param.data.shape != loaded_weight.shape:
        param.data = torch.empty_like(loaded_weight)
    param.data.copy_(loaded_weight)
    return True


def _make_glq_param(tensor):
    """Create nn.Parameter with GLQ weight_loader attached."""
    p = torch.nn.Parameter(tensor, requires_grad=False)
    p.weight_loader = _glq_weight_loader
    return p


def _register_glq_buffers(layer, prefix, out_size, in_size):
    """Register one set of GLQ compressed buffers on the layer.

    Higher-stage buffers (Qidxs3 / Qidxs4 and their matching
    inv_resid_scale*) are allocated as **zero-sized sentinels** for
    layers whose checkpoint doesn't carry them. The
    ``weight_loader`` (see ``_glq_weight_loader`` above) replaces
    ``param.data`` wholesale with the loaded shape when the stage is
    present. Qidxs2 is left at full size because every bpw >= 3 layer
    has it and the wholesale-replacement path is rarely exercised for
    it, so empty(0) there would just churn allocations on every load.

    Memory: for a mix-3-8 5 bpw 31B checkpoint this drops the
    post-load model footprint from ~94 GiB (OOM on a 96 GiB card) to
    ~66 GiB by skipping the Qidxs3/Qidxs4 pre-alloc on the majority
    of layers (only 6-8 bpw layers populate them).
    """
    m_pad = _glq_pad(out_size)
    n_pad = _glq_pad(in_size)
    n_blocks = n_pad // 8
    p = prefix
    setattr(layer, f'Qidxs{p}', _make_glq_param(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16)))
    setattr(layer, f'SU{p}', _make_glq_param(
        torch.ones(m_pad, dtype=torch.float16)))
    setattr(layer, f'SV{p}', _make_glq_param(
        torch.ones(n_pad, dtype=torch.float16)))
    setattr(layer, f'Wscale{p}', _make_glq_param(
        torch.ones((), dtype=torch.float32)))
    setattr(layer, f'Qidxs2{p}', _make_glq_param(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16)))
    setattr(layer, f'inv_resid_scale{p}', _make_glq_param(
        torch.zeros((), dtype=torch.float32)))
    # Phase D: N-stage RVQ for 5-8 bpw — sentinels only. forward() reads
    # these only when ``_glq_n_stages >= 3 / >= 4``, which is gated by
    # ``inv_resid_scale2.item() != 0.0`` (set when the checkpoint loads
    # an actual stage-3 tensor).
    setattr(layer, f'Qidxs3{p}', _make_glq_param(
        torch.empty(0, dtype=torch.int16)))
    setattr(layer, f'inv_resid_scale2{p}', _make_glq_param(
        torch.zeros((), dtype=torch.float32)))
    setattr(layer, f'Qidxs4{p}', _make_glq_param(
        torch.empty(0, dtype=torch.int16)))
    setattr(layer, f'inv_resid_scale3{p}', _make_glq_param(
        torch.zeros((), dtype=torch.float32)))
    return m_pad, n_pad


class GLQShardedParameter(BasevLLMParameter):
    """Parameter that stores per-shard GLQ buffers for fused QKV layers.

    vLLM's stacked_params_mapping renames q_proj.Qidxs → qkv_proj.Qidxs
    and calls load_qkv_weight with shard_id="q"|"k"|"v". This parameter
    routes each shard to a separate internal buffer (different padded sizes
    because GLQ pads to power-of-2 and each shard has independent RHT).
    """

    def __new__(cls, shard_sizes, inner_dim, dtype, sentinel: bool = False,
                **kwargs):
        # Phase B: param.data is a zero-byte placeholder. Fused load goes
        # through _glq_shard_loader → _shard_data[i], bypassing the standard
        # vLLM shape-check + resize path that needs a real param.data.
        data = torch.empty(0, dtype=dtype)
        return super().__new__(cls, data=data, **kwargs)

    def __init__(self, shard_sizes: list[int], inner_dim: int,
                 dtype: torch.dtype, weight_loader: Callable,
                 sentinel: bool = False, **kwargs):
        data = torch.empty(0, dtype=dtype)
        super().__init__(data=data, weight_loader=weight_loader)
        self._shard_sizes = shard_sizes
        self._inner_dim = inner_dim
        self._dtype = dtype
        # Phase A: when sentinel=True, allocate zero-size per-shard buffers
        # and rely on load_qkv_weight / load_merged_column_weight to resize
        # on first store. Used for stage-3/4 indices that only exist for
        # bpw ≥ 5 / ≥ 7 layers; saves ~20 GB on Gemma-4-31B mix-3-8.
        self._shard_data = []
        if sentinel:
            for _ in shard_sizes:
                self._shard_data.append(torch.empty(0, dtype=dtype))
        else:
            for sz in shard_sizes:
                m_pad = _glq_pad(sz) if sz > 1 else 1
                if inner_dim > 0:
                    self._shard_data.append(
                        torch.zeros(m_pad, inner_dim, dtype=dtype))
                elif inner_dim == -1:
                    # 1D vector per shard (SU sign vector)
                    self._shard_data.append(
                        torch.ones(m_pad, dtype=dtype))
                else:
                    # Scalar per shard (Wscale, inv_resid_scale)
                    self._shard_data.append(torch.zeros((), dtype=dtype))

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        idx = self._shard_id_as_int(shard_id)
        if idx < len(self._shard_data):
            if self._shard_data[idx].shape != loaded_weight.shape:
                self._shard_data[idx] = torch.empty_like(loaded_weight)
            self._shard_data[idx].copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        if shard_id is not None:
            idx = self._shard_id_as_int(shard_id)
        else:
            idx = 0
        if idx < len(self._shard_data):
            if self._shard_data[idx].shape != loaded_weight.shape:
                self._shard_data[idx] = torch.empty_like(loaded_weight)
            self._shard_data[idx].copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        if len(self._shard_data) == 1:
            self._shard_data[0].copy_(loaded_weight)

    @property
    def weight_loader(self) -> Callable:
        """Override BasevLLMParameter.weight_loader to return our shard router,
        not the layer's weight_loader method."""
        return self._glq_shard_loader

    def _glq_shard_loader(self, param, loaded_weight, *args, **kwargs):
        """Route to load_qkv_weight or load_merged_column_weight."""
        shard_id = kwargs.get('shard_id') or (args[0] if args else None)
        if shard_id is not None:
            self.load_qkv_weight(loaded_weight, shard_id=shard_id, **{k: v for k, v in kwargs.items() if k != 'shard_id'})
        else:
            self.load_column_parallel_weight(loaded_weight)

    def get_shard(self, idx: int) -> torch.Tensor:
        return self._shard_data[idx]

    @property
    def num_shards(self) -> int:
        return len(self._shard_data)

    def to(self, *args, **kwargs):
        """Move per-shard buffers in place. Since Phase B made param.data
        empty(0), we can't rely on ``super().to()`` round-tripping back to
        a ``GLQShardedParameter`` (PyTorch may downcast to plain Tensor
        when the wrapped storage is empty). Operate on ``self`` directly."""
        self.data = self.data.to(*args, **kwargs)
        self._shard_data = [s.to(*args, **kwargs) for s in self._shard_data]
        return self

    def cuda(self, device=None):
        target = device if device is not None else torch.cuda.current_device()
        return self.to(device=target)


def _glq_apply_shard(x, device, cb, cb2, Qidxs, SU, SV, wscale,
                     has_stage2, inv_rs, Qidxs2, out_features, in_features,
                     m_pad, n_pad, log_n, log_m,
                     Qidxs3=None, inv_rs2=0.0, Qidxs4=None, inv_rs3=0.0,
                     block_diag_meta=None):
    """Run input RHT → dequant+matmul → output RHT for explicit tensors.

    Block-diagonal (Phase B) path: when ``block_diag_meta`` is provided, dispatches
    to the single-call ``glq_fused_linear_block_diag_cuda`` which handles
    input-RHT + dequant+matmul + output-RHT for non-power-of-2 dims in one
    host call. Pow2 path falls through to the legacy 3-call sequence.

    Stages 3-4 (Phase D, 5-8 bpw) reuse the primary 65536-entry E8 codebook —
    matches the convention in glq.quantized_linear's fallback path.
    """
    dtype = x.dtype
    B = x.shape[0]
    # custom_ops are registered at ``import glq_vllm`` time (__init__.py
    # calls _ensure_registered()). If they're missing here the install is
    # broken — fail loud, don't silently fall back to a path dynamo can't
    # trace.
    _GLQ_DEBUG = os.environ.get("GLQ_DEBUG") == "1"

    # Weight tensors are guaranteed on GPU after process_weights_after_loading
    # (it walks GLQShardedParameter._shard_data too); skipping the per-call
    # ``.to(device)`` guards saves ~6 dispatches per layer.

    primary_cb = cb.codebook_half
    cb2_half = cb2.codebook_half if has_stage2 and cb2 is not None else None

    if _GLQ_DEBUG:
        print(f"GLQ_DEBUG _glq_apply_shard: m_pad={m_pad} n_pad={n_pad} "
              f"bd_meta={'set' if block_diag_meta is not None else 'NONE'}",
              file=sys.stderr, flush=True)
    # ── Block-diagonal fast path (Phase B + D) ──────────────────────
    if (block_diag_meta is not None
            and hasattr(torch.ops.glq, "fused_linear_block_diag")
            and n_pad <= 32768 and m_pad <= 32768):
        _empty_i16, _empty_f16 = _get_empty_sentinels(device)
        bn_tensor = block_diag_meta["blocks_n_tensor"]  # CPU int64
        bm_tensor = block_diag_meta["blocks_m_tensor"]
        bn_meta = block_diag_meta["blocks_n_meta_gpu"]
        bm_meta = block_diag_meta["blocks_m_meta_gpu"]
        q2 = Qidxs2 if has_stage2 else _empty_i16
        cb2_arg = cb2_half if has_stage2 and cb2_half is not None else _empty_f16
        q3 = Qidxs3 if Qidxs3 is not None else _empty_i16
        cb3_arg = primary_cb if Qidxs3 is not None else _empty_f16
        q4 = Qidxs4 if Qidxs4 is not None else _empty_i16
        cb4_arg = primary_cb if Qidxs4 is not None else _empty_f16
        # Dispatched as a registered torch.ops.glq op so vLLM's
        # torch.compile / piecewise-CUDA-graph capture treats this call
        # as an opaque kernel boundary.
        x_half = x.half().contiguous()
        wscale_f = float(wscale)
        irs_f = float(inv_rs) if has_stage2 else 0.0
        irs2_f = float(inv_rs2) if Qidxs3 is not None else 0.0
        irs3_f = float(inv_rs3) if Qidxs4 is not None else 0.0
        y = torch.ops.glq.fused_linear_block_diag(
            x_half, SV, SU, Qidxs, primary_cb, wscale_f,
            in_features, out_features, n_pad, m_pad,
            bn_tensor, bm_tensor, bn_meta, bm_meta,
            q2, cb2_arg, irs_f,
            q3, cb3_arg, irs2_f,
            q4, cb4_arg, irs3_f,
        )
        if dtype != torch.float16:
            y = y.to(dtype)
        return y

    # Input RHT — C++ kernel for n_pad ≤ 16384, Triton wrapper for >16384.
    # Both go through ``torch.ops.glq.*`` so dynamo sees one op call per
    # branch (no untraced raw Triton-kernel calls).
    x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=device)
    rsqrt_n = 1.0 / math.sqrt(n_pad)
    x_half = x.half().contiguous()
    if n_pad <= 16384:
        torch.ops.glq.input_rht(x_half, SV, x_rht,
                                in_features, in_features, rsqrt_n, n_pad, log_n)
    else:
        torch.ops.glq.input_rht_triton(x_half, SV, x_rht,
                                        in_features, in_features, rsqrt_n,
                                        n_pad, log_n)

    if _GLQ_DEBUG:
        torch.cuda.synchronize()
        print(f"GLQ_DEBUG: input_rht OK x_rht={x_rht.shape} nan={x_rht.isnan().any().item()}", file=sys.stderr, flush=True)

    # Dequant + matmul (always through glq_dequant_matmul which handles dispatch)
    cb_packed = getattr(cb, 'codebook_packed', None)
    # Phase D: stages 3/4 reuse the primary 65536-entry codebook.
    cb3_arg = primary_cb if Qidxs3 is not None else None
    cb4_arg = primary_cb if Qidxs4 is not None else None

    y_rht = glq_dequant_matmul(
        x_rht, Qidxs, primary_cb, wscale,
        Qidxs2=Qidxs2, codebook2=cb2_half,
        inv_resid_scale=inv_rs, codebook_packed=cb_packed,
        Qidxs3=Qidxs3, codebook3=cb3_arg, inv_resid_scale2=inv_rs2,
        Qidxs4=Qidxs4, codebook4=cb4_arg, inv_resid_scale3=inv_rs3)

    if _GLQ_DEBUG:
        torch.cuda.synchronize()
        print(f"GLQ_DEBUG: matmul OK y_rht={y_rht.shape} nan={y_rht.isnan().any().item()}", file=sys.stderr, flush=True)

    # Output RHT
    rsqrt_m = 1.0 / math.sqrt(m_pad)
    if _GLQ_DEBUG:
        print(f"GLQ_DEBUG: output_rht params B={B} out={out_features} m_pad={m_pad} log_m={log_m} SU={SU.shape} SU_dev={SU.device} y_rht_dev={y_rht.device}", file=sys.stderr, flush=True)
    # Output RHT — same C++/Triton split as input_rht. Both branches go
    # through torch.ops.glq.* so dynamo traces a single op call.
    if m_pad <= 16384:
        y = torch.empty(B, out_features, dtype=torch.float16, device=device)
        torch.ops.glq.output_rht(y_rht, SU, y, out_features, m_pad, log_m, rsqrt_m)
        if _GLQ_DEBUG:
            torch.cuda.synchronize()
            print(f"GLQ_DEBUG: output_rht OK y={y.shape}", file=sys.stderr, flush=True)
        if dtype != torch.float16:
            y = y.to(dtype)
    else:
        # Allocate fp16 (matches the C++ branch), let the Triton kernel
        # store fp16, then cast to the target dtype if needed. This keeps
        # both branches numerically equivalent and avoids the Triton
        # kernel's missing bf16 store path.
        y = torch.empty(B, out_features, dtype=torch.float16, device=device)
        torch.ops.glq.output_rht_triton(y_rht, SU, y, out_features,
                                         m_pad, log_m, rsqrt_m)
        if dtype != torch.float16:
            y = y.to(dtype)
    return y


def _glq_apply_single(x, layer, prefix, cb, cb2, device):
    """Apply one set of GLQ buffers stored on the layer under `{name}{prefix}`.

    Tensors are guaranteed to live on ``device`` after
    ``process_weights_after_loading`` — no per-call ``.to(device)``.
    """
    Qidxs = getattr(layer, f'Qidxs{prefix}')
    SU = getattr(layer, f'SU{prefix}')
    SV = getattr(layer, f'SV{prefix}')
    wscale = getattr(layer, f'_glq_wscale{prefix}')
    has_stage2 = getattr(layer, f'_glq_has_stage2{prefix}')
    inv_rs = getattr(layer, f'_glq_inv_rs{prefix}')
    m_pad = getattr(layer, f'_glq_m_pad{prefix}')
    n_pad = getattr(layer, f'_glq_n_pad{prefix}')
    log_n = getattr(layer, f'_glq_log_n{prefix}')
    log_m = getattr(layer, f'_glq_log_m{prefix}')
    out_features = getattr(layer, f'_glq_out{prefix}')
    in_features = getattr(layer, f'_glq_in{prefix}')

    Qidxs2 = getattr(layer, f'Qidxs2{prefix}') if has_stage2 else None
    # Phase D N-stage: stage 3/4 tensors when _glq_n_stages >= 3 / 4.
    n_stages = getattr(layer, f'_glq_n_stages{prefix}', 2 if has_stage2 else 1)
    inv_rs2 = getattr(layer, f'_glq_inv_rs2{prefix}', 0.0)
    inv_rs3 = getattr(layer, f'_glq_inv_rs3{prefix}', 0.0)
    Qidxs3 = getattr(layer, f'Qidxs3{prefix}') if n_stages >= 3 else None
    Qidxs4 = getattr(layer, f'Qidxs4{prefix}') if n_stages >= 4 else None
    # Phase B block-diagonal metadata — set by process_weights_after_loading.
    block_diag_meta = getattr(layer, f'_glq_bd_meta{prefix}', None)
    return _glq_apply_shard(
        x, device, cb, cb2,
        Qidxs=Qidxs, SU=SU, SV=SV, wscale=wscale,
        has_stage2=has_stage2, inv_rs=inv_rs, Qidxs2=Qidxs2,
        out_features=out_features, in_features=in_features,
        m_pad=m_pad, n_pad=n_pad, log_n=log_n, log_m=log_m,
        Qidxs3=Qidxs3, inv_rs2=inv_rs2,
        Qidxs4=Qidxs4, inv_rs3=inv_rs3,
        block_diag_meta=block_diag_meta,
    )


def _glq_apply_e8p(x, layer):
    """E8P apply: one validated, torch.ops-dispatched (cudagraph-capturable) call per
    (shard), reusing E8RHTLinear._e8p_linear_apply. Single layer → one call; fused QKV/
    gate_up → one call per shard, concatenated (mirrors the shell fused loop)."""
    from glq.quantized_linear import E8RHTLinear, _pack_block_meta
    from glq.hadamard import _block_decompose
    _apply = E8RHTLinear._e8p_linear_apply
    grid = layer._glq_e8p_grid
    e81b_grid = layer._glq_e81b_grid

    def _blocks(meta):
        # Block-diagonal RHT tensors derived from the (block-diag-sized) n_pad/m_pad and
        # cached per device. A pow2 n_pad decomposes to a single block → full RHT (unchanged).
        if meta.get('_bn_dev') != x.device:
            bn = _block_decompose(meta['n_pad'])
            bm = _block_decompose(meta['m_pad'])
            meta['_bn'] = torch.tensor(bn, dtype=torch.int64)
            meta['_bm'] = torch.tensor(bm, dtype=torch.int64)
            meta['_bnm'] = _pack_block_meta(bn).to(x.device)
            meta['_bmm'] = _pack_block_meta(bm).to(x.device)
            meta['_bn_dev'] = x.device
        return meta['_bn'], meta['_bm'], meta['_bnm'], meta['_bmm']

    if getattr(layer, 'glq_is_fused', False):
        outs = []
        for i, meta in enumerate(layer._glq_e8p_meta):
            bn, bm, bnm, bmm = _blocks(meta)
            outs.append(_apply(
                x, layer.SV.get_shard(i), layer.SU.get_shard(i),
                layer.Qidxs_e8p.get_shard(i), layer.Qidxs2_e8p.get_shard(i),
                layer.Qidxs2_e81b.get_shard(i), grid, e81b_grid,
                meta['wscale'], meta['inv_rs'], meta['has_e8p2'], meta['has_e81b'],
                meta['in'], meta['out'], meta['n_pad'], meta['m_pad'],
                bn, bm, bnm, bmm,
                bias=None, out_dtype=x.dtype))
        return torch.cat(outs, dim=-1)
    meta = layer._glq_e8p_meta[0]
    bn, bm, bnm, bmm = _blocks(meta)
    return _apply(
        x, layer.SV, layer.SU, layer.Qidxs_e8p, layer.Qidxs2_e8p, layer.Qidxs2_e81b,
        grid, e81b_grid, meta['wscale'], meta['inv_rs'], meta['has_e8p2'], meta['has_e81b'],
        meta['in'], meta['out'], meta['n_pad'], meta['m_pad'],
        bn, bm, bnm, bmm, bias=None, out_dtype=x.dtype)


class GLQLinearMethod(LinearMethodBase):
    """GLQ fused dequant+matmul linear method for vLLM.

    Weights stay compressed (int16 indices + sign vectors).
    Supports standard linear layers AND fused QKV (per-shard buffers).
    """

    def __init__(self, quant_config, bpw: int = 2, pre_fused: bool = False,
                 codebook_type: str = "e8_shell"):
        self.quant_config = quant_config
        self.bpw = bpw
        self.codebook_type = codebook_type
        # ``pre_fused``: the checkpoint stores this fused linear as ONE
        # jointly-quantized matrix (e.g. sarvam/Bailing's ``query_key_value``),
        # not as separate per-shard weights merged at load. A jointly-quantized
        # matrix has a single row-direction Hadamard spanning all output rows,
        # so it MUST be dequantized as a whole — it cannot be split into q/k/v
        # shards. Load it as a single (non-sharded) GLQ matrix; its dequant
        # output [q;k;v] is exactly what QKVParallelLinear consumes.
        self.pre_fused = pre_fused

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # A pre-fused checkpoint matrix (e.g. ``query_key_value``) is loaded as a
        # single non-sharded GLQ matrix even though vLLM's layer reports
        # multiple output partitions — see GLQLinearMethod.pre_fused.
        is_fused = len(output_partition_sizes) > 1 and not self.pre_fused
        layer.glq_is_fused = is_fused
        layer.glq_in_features = input_size_per_partition
        layer.glq_bpw = self.bpw
        is_e8p = (self.codebook_type == "e8p")
        layer.glq_is_e8p = is_e8p

        weight_loader = extra_weight_attrs.get("weight_loader")

        if is_fused and is_e8p:
            # Fused QKV/gate_up for e8p: per-shard int64 TC-packed buffers. The
            # GLQShardedParameter loaders use empty_like, so sentinel=True placeholders
            # are replaced by the loaded q/k/v (or gate/up) Qidxs_e8p on load.
            layer.glq_shard_sizes = output_partition_sizes
            layer.glq_num_shards = len(output_partition_sizes)
            n_pad = _glq_pad(input_size_per_partition)
            ops = output_partition_sizes
            layer.Qidxs_e8p = GLQShardedParameter(ops, 1, torch.int64,
                                                  weight_loader=weight_loader, sentinel=True)
            layer.Qidxs2_e8p = GLQShardedParameter(ops, 1, torch.int64,
                                                   weight_loader=weight_loader, sentinel=True)
            layer.Qidxs2_e81b = GLQShardedParameter(ops, 1, torch.int64,
                                                    weight_loader=weight_loader, sentinel=True)
            layer.SU = GLQShardedParameter(ops, -1, torch.float16, weight_loader=weight_loader)
            layer.SV = GLQShardedParameter([n_pad] * len(ops), -1, torch.float16,
                                           weight_loader=weight_loader)
            layer.Wscale = GLQShardedParameter([1] * len(ops), 0, torch.float32,
                                               weight_loader=weight_loader)
            layer.inv_resid_scale = GLQShardedParameter([1] * len(ops), 0, torch.float32,
                                                        weight_loader=weight_loader)
            layer.glq_n_pad = n_pad
        elif is_fused:
            # Fused QKV: use GLQShardedParameter for per-shard storage
            layer.glq_shard_sizes = output_partition_sizes
            layer.glq_num_shards = len(output_partition_sizes)
            n_pad = _glq_pad(input_size_per_partition)
            n_blocks = n_pad // 8

            # Each GLQ buffer is a GLQShardedParameter that routes shard_id
            layer.Qidxs = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16, weight_loader=weight_loader)
            # SU is a 1D sign vector of size m_pad per shard — use -1 as sentinel
            # to allocate per-shard with correct padded size
            layer.SU = GLQShardedParameter(
                output_partition_sizes, -1, torch.float16, weight_loader=weight_loader)
            # SV is per-shard too: the quantizer uses an independent random
            # Hadamard rotation per layer, so q/k/v have different SV vectors
            # even though they share the input dim. Each shard stores a full
            # n_pad-length SV; reuse GLQShardedParameter with
            # shard_sizes=[n_pad]*num_shards to get that layout.
            layer.SV = GLQShardedParameter(
                [n_pad] * len(output_partition_sizes), -1, torch.float16,
                weight_loader=weight_loader)
            layer.Wscale = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32, weight_loader=weight_loader)
            layer.Qidxs2 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16, weight_loader=weight_loader)
            layer.inv_resid_scale = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32, weight_loader=weight_loader)
            # Stage 3/4 indices: sentinel-allocate (empty(0) per shard) — the
            # loader auto-resizes on first store, so layers with bpw < 5/7
            # never materialise these tensors. inv_resid_scale2/3 stay
            # full-size (scalar): process_weights_after_loading reads them
            # as the "stage exists?" gate.
            layer.Qidxs3 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16,
                weight_loader=weight_loader, sentinel=True)
            layer.inv_resid_scale2 = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32, weight_loader=weight_loader)
            layer.Qidxs4 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16,
                weight_loader=weight_loader, sentinel=True)
            layer.inv_resid_scale3 = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32, weight_loader=weight_loader)

            layer.glq_n_pad = n_pad

        # Dummy weight param so vLLM code that accesses layer.weight
        # (e.g. mamba_mixer2.py) doesn't crash during __init__
        if not hasattr(layer, 'weight'):
            layer.weight = _make_glq_param(torch.empty(1, dtype=params_dtype))

        if not is_fused:
            # Standard: single set of GLQ buffers (no suffix)
            out_sz = sum(output_partition_sizes)
            layer.glq_out_features = out_sz
            if is_e8p:
                m_pad, n_pad = _glq_pad(out_sz), _glq_pad(input_size_per_partition)
                mp16, nb64 = max(m_pad // 16, 1), max(n_pad // 64, 1)
                layer.Qidxs_e8p = _make_glq_param(
                    torch.zeros(mp16, nb64, 8, 4, dtype=torch.int64))
                layer.SU = _make_glq_param(torch.ones(m_pad, dtype=torch.float16))
                layer.SV = _make_glq_param(torch.ones(n_pad, dtype=torch.float16))
                layer.Wscale = _make_glq_param(torch.ones((), dtype=torch.float32))
                layer.inv_resid_scale = _make_glq_param(torch.zeros((), dtype=torch.float32))
                # Stage-2 residual: register at the FINAL shape the checkpoint
                # carries so the loader takes the in-place ``copy_`` branch (shape
                # match) and keeps the create_weights CUDA storage. The empty(0)
                # sentinel path would force ``param.data = empty_like(loaded)``,
                # repointing .data at fresh CPU storage; for a registered
                # nn.Parameter vLLM manages/restores .data, so a later move back
                # to CUDA is reverted and the decompress kernel aborts on a CPU
                # tensor at cudagraph capture. (The fused GLQShardedParameter path
                # is immune — it moves its private ``_shard_data`` list, not a
                # managed param.) bpw 4 → E8P residual; bpw 3 → E81B residual.
                if self.bpw >= 4:
                    layer.Qidxs2_e8p = _make_glq_param(
                        torch.zeros(mp16, nb64, 8, 4, dtype=torch.int64))
                else:
                    layer.Qidxs2_e8p = _make_glq_param(torch.empty(0, dtype=torch.int64))
                if self.bpw == 3:
                    layer.Qidxs2_e81b = _make_glq_param(
                        torch.zeros(m_pad, nb64, dtype=torch.int64))
                else:
                    layer.Qidxs2_e81b = _make_glq_param(torch.empty(0, dtype=torch.int64))
            else:
                m_pad, n_pad = _register_glq_buffers(
                    layer, '', out_sz, input_size_per_partition)
            layer.glq_m_pad = m_pad
            layer.glq_n_pad = n_pad

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Set up codebook and cache scalars. Dequant Mamba layers to dense."""
        device = next(layer.parameters()).device
        bpw = getattr(layer, 'glq_bpw', 2)

        if getattr(layer, 'glq_is_e8p', False):
            self._setup_e8p_weights(layer, device)
            return

        # NOTE: a previous Mamba dequant fallback (matched on
        # ``layer.weight.numel() <= 1``) accidentally caught our own dummy
        # weight stub from ``create_weights``, dequantising every non-fused
        # GLQ linear back to bf16 and silently bypassing the kernel path.
        # That fallback was removed (sglang fork reverted the equivalent
        # commit). If a future model genuinely needs dense fallback, gate
        # it on a model-specific class check rather than a generic numel
        # heuristic.

        # Ensure shared codebook is on the right device
        # Determine max_bpw across all shards. Stage 3/4 imply 5+ bpw.
        max_bpw = 2
        if getattr(layer, 'glq_is_fused', False):
            for i in range(layer.glq_num_shards):
                inv_rs = layer.inv_resid_scale.get_shard(i).item()
                inv_rs2 = (layer.inv_resid_scale2.get_shard(i).item()
                           if hasattr(layer, 'inv_resid_scale2') else 0.0)
                if inv_rs2 != 0.0:
                    max_bpw = max(max_bpw, max(bpw, 5))
                elif inv_rs != 0.0:
                    max_bpw = max(max_bpw, bpw)
        else:
            inv_rs = layer.inv_resid_scale.item()
            inv_rs2 = (layer.inv_resid_scale2.item()
                       if hasattr(layer, 'inv_resid_scale2') else 0.0)
            if inv_rs2 != 0.0:
                max_bpw = max(bpw, 5)
            elif inv_rs != 0.0:
                max_bpw = bpw

        _ensure_codebook(device, max_bpw=max_bpw)
        _try_load_cuda_ext()

        # Ensure all weight tensors are on GPU (includes Phase D stage 3/4).
        # For ``GLQShardedParameter``, the per-shard buffers live in
        # ``_shard_data`` (a Python list); moving the outer ``.data`` to
        # GPU does not propagate, so we walk the list explicitly.
        # Without this, every ``layer.Qidxs.get_shard(i)`` call returns
        # a CPU tensor and ``_glq_apply_shard`` triggers an implicit
        # ``.to(device)`` per call → ~1600 redundant ``aten::copy_``
        # dispatches per decode token on Gemma-4-E2B.
        for attr in ['Qidxs', 'SU', 'SV', 'Wscale', 'Qidxs2', 'inv_resid_scale',
                     'Qidxs3', 'inv_resid_scale2', 'Qidxs4', 'inv_resid_scale3']:
            t = getattr(layer, attr, None)
            if t is None:
                continue
            if isinstance(t, GLQShardedParameter):
                for i, sd in enumerate(t._shard_data):
                    if sd.device != device:
                        t._shard_data[i] = sd.to(device)
            elif hasattr(t, 'device') and t.device != device:
                setattr(layer, attr, torch.nn.Parameter(t.data.to(device), requires_grad=False))

        # Cache scalars for all shards
        if getattr(layer, 'glq_is_fused', False):
            layer._glq_shard_meta = []
            # Phase B: recover actual (possibly non-pow2) shard shapes from the
            # loaded per-shard Qidxs buffers. At register time we sized to pow2;
            # the weight loader overwrote with whatever the checkpoint contained.
            for i in range(layer.glq_num_shards):
                out_sz = layer.glq_shard_sizes[i]
                qidxs_i = layer.Qidxs.get_shard(i)
                if qidxs_i.dim() == 2:
                    m_pad = qidxs_i.shape[0]
                    n_pad = qidxs_i.shape[1] * 8
                else:
                    m_pad = _glq_pad(out_sz)
                    n_pad = layer.glq_n_pad
                inv_rs_val = layer.inv_resid_scale.get_shard(i).item()
                # Phase D: detect active stage count from non-zero inv_resid_scale*.
                inv_rs2_val = (layer.inv_resid_scale2.get_shard(i).item()
                               if hasattr(layer, 'inv_resid_scale2') else 0.0)
                inv_rs3_val = (layer.inv_resid_scale3.get_shard(i).item()
                               if hasattr(layer, 'inv_resid_scale3') else 0.0)
                if inv_rs3_val != 0.0:
                    n_stages = 4
                elif inv_rs2_val != 0.0:
                    n_stages = 3
                elif inv_rs_val != 0.0:
                    n_stages = 2
                else:
                    n_stages = 1
                is_bd, blocks_m, blocks_n = _detect_block_diag(m_pad, n_pad)
                bd_meta = None
                if is_bd:
                    bd_meta = {
                        "blocks_n_tensor": torch.tensor(blocks_n, dtype=torch.int64, device='cpu'),
                        "blocks_m_tensor": torch.tensor(blocks_m, dtype=torch.int64, device='cpu'),
                        "blocks_n_meta_gpu": _pack_block_meta(blocks_n).to(device, non_blocking=True),
                        "blocks_m_meta_gpu": _pack_block_meta(blocks_m).to(device, non_blocking=True),
                    }
                layer._glq_shard_meta.append({
                    'wscale': layer.Wscale.get_shard(i).item(),
                    'has_stage2': inv_rs_val != 0.0,
                    'inv_rs': inv_rs_val,
                    'n_stages': n_stages,
                    'inv_rs2': inv_rs2_val,
                    'inv_rs3': inv_rs3_val,
                    'out': out_sz,
                    'in': layer.glq_in_features,
                    'm_pad': m_pad,
                    'n_pad': n_pad,
                    'log_n': int(math.log2(n_pad)) if _is_pow2(n_pad) else 0,
                    'log_m': int(math.log2(m_pad)) if _is_pow2(m_pad) else 0,
                    'bd_meta': bd_meta,
                })
        else:
            inv_rs = layer.inv_resid_scale.item()
            layer._glq_wscale = layer.Wscale.item()
            layer._glq_has_stage2 = inv_rs != 0.0
            layer._glq_inv_rs = inv_rs
            # Phase D: detect N-stage from non-zero inv_resid_scale*.
            inv_rs2 = (layer.inv_resid_scale2.item()
                       if hasattr(layer, 'inv_resid_scale2') else 0.0)
            inv_rs3 = (layer.inv_resid_scale3.item()
                       if hasattr(layer, 'inv_resid_scale3') else 0.0)
            if inv_rs3 != 0.0:
                layer._glq_n_stages = 4
            elif inv_rs2 != 0.0:
                layer._glq_n_stages = 3
            elif inv_rs != 0.0:
                layer._glq_n_stages = 2
            else:
                layer._glq_n_stages = 1
            layer._glq_inv_rs2 = inv_rs2
            layer._glq_inv_rs3 = inv_rs3
            # Update dims from actual loaded Qidxs shape (auto-resize may have changed them)
            if hasattr(layer, 'Qidxs') and layer.Qidxs.dim() == 2:
                layer._glq_m_pad = layer.Qidxs.shape[0]
                layer._glq_n_pad = layer.Qidxs.shape[1] * 8
                layer._glq_out = layer.glq_out_features
                layer._glq_in = layer.glq_in_features
            else:
                layer._glq_m_pad = layer.glq_m_pad
                layer._glq_n_pad = layer.glq_n_pad
                layer._glq_out = layer.glq_out_features
                layer._glq_in = layer.glq_in_features
            # Phase B: detect block-diag from the loaded (possibly non-pow2)
            # buffer shape, build block decomposition + packed metadata so
            # the forward path can dispatch to glq_fused_linear_block_diag_cuda.
            is_bd, blocks_m, blocks_n = _detect_block_diag(
                layer._glq_m_pad, layer._glq_n_pad)
            if is_bd:
                layer._glq_bd_meta = {
                    "blocks_n_tensor": torch.tensor(blocks_n, dtype=torch.int64, device='cpu'),
                    "blocks_m_tensor": torch.tensor(blocks_m, dtype=torch.int64, device='cpu'),
                    "blocks_n_meta_gpu": _pack_block_meta(blocks_n).to(device, non_blocking=True),
                    "blocks_m_meta_gpu": _pack_block_meta(blocks_m).to(device, non_blocking=True),
                }
                layer._glq_log_n = (int(math.log2(layer._glq_n_pad))
                                    if _is_pow2(layer._glq_n_pad) else 0)
                layer._glq_log_m = (int(math.log2(layer._glq_m_pad))
                                    if _is_pow2(layer._glq_m_pad) else 0)
            else:
                layer._glq_bd_meta = None
                layer._glq_log_n = int(math.log2(layer._glq_n_pad))
                layer._glq_log_m = int(math.log2(layer._glq_m_pad))

        # Remove weight_loader from params — no longer needed after loading,
        # and function references prevent vLLM v1 serialization
        for name, param in layer.named_parameters():
            try:
                if hasattr(param, 'weight_loader') and not isinstance(
                    type(param).__dict__.get('weight_loader'), property
                ):
                    del param.weight_loader
            except (AttributeError, TypeError):
                pass

    def _setup_e8p_weights(self, layer, device):
        """e8p: cache codebook grids + per-(shard) scalars/flags. Pow2-padded (no block-diag)."""
        grid_cb, e81b_cb = _ensure_codebook(device, max_bpw=4, codebook_type="e8p")
        _try_load_cuda_ext()
        layer._glq_e8p_grid = grid_cb.grid_packed_abs.to(device)
        layer._glq_e81b_grid = e81b_cb.e81b_grid.to(device) if e81b_cb is not None else None

        # Move e8p buffers to device (GLQShardedParameter per-shard lists + plain params).
        for attr in ['Qidxs_e8p', 'Qidxs2_e8p', 'Qidxs2_e81b', 'SU', 'SV',
                     'Wscale', 'inv_resid_scale']:
            t = getattr(layer, attr, None)
            if t is None:
                continue
            if isinstance(t, GLQShardedParameter):
                for i, sd in enumerate(t._shard_data):
                    if sd.device != device:
                        t._shard_data[i] = sd.to(device)
            elif hasattr(t, 'device') and t.device != device:
                # Move the data IN PLACE — keep the same registered Parameter
                # object. ``setattr(layer, attr, nn.Parameter(...))`` creates a
                # NEW object; vLLM still holds the original (CPU) param it
                # registered at build time and effectively reverts ``layer.attr``
                # back to it after process_weights, so the moved data is lost and
                # the decompress kernel aborts on a CPU tensor (confirmed on the
                # non-fused stage-2 Qidxs2_e8p, which the loader resized via
                # ``empty_like(loaded)`` onto CPU). Mutating ``.data`` preserves
                # object identity — as the full-size Qidxs_e8p and the fused
                # ``_shard_data`` moves already do — so the move sticks.
                t.data = t.data.to(device)

        def _meta(Qe8p, Q2e8p, Q2e81b, wscale, inv_rs, out_sz):
            return {
                'wscale': wscale, 'inv_rs': inv_rs,
                'has_e8p2': bool(Q2e8p is not None and Q2e8p.numel() > 0),
                'has_e81b': bool(Q2e81b is not None and Q2e81b.numel() > 0),
                'm_pad': Qe8p.shape[0] * 16, 'n_pad': Qe8p.shape[1] * 64,
                'out': out_sz, 'in': layer.glq_in_features,
            }

        if getattr(layer, 'glq_is_fused', False):
            layer._glq_e8p_meta = [
                _meta(layer.Qidxs_e8p.get_shard(i), layer.Qidxs2_e8p.get_shard(i),
                      layer.Qidxs2_e81b.get_shard(i), layer.Wscale.get_shard(i).item(),
                      layer.inv_resid_scale.get_shard(i).item(), layer.glq_shard_sizes[i])
                for i in range(layer.glq_num_shards)]
        else:
            layer._glq_e8p_meta = [_meta(
                layer.Qidxs_e8p, layer.Qidxs2_e8p, layer.Qidxs2_e81b,
                layer.Wscale.item(), layer.inv_resid_scale.item(), layer.glq_out_features)]

        # Drop weight_loaders (function refs break vLLM v1 serialization).
        for _name, param in layer.named_parameters():
            try:
                if hasattr(param, 'weight_loader') and not isinstance(
                        type(param).__dict__.get('weight_loader'), property):
                    del param.weight_loader
            except (AttributeError, TypeError):
                pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Mamba layers dequantized to dense — standard matmul
        if getattr(layer, '_glq_use_dense', False):
            return F.linear(x, layer.weight.to(x.dtype), bias)
        orig_shape = x.shape
        in_features = layer.glq_in_features
        x = x.reshape(-1, in_features)
        device = x.device
        cb, cb2 = _codebook, _codebook2_small

        if getattr(layer, 'glq_is_e8p', False):
            y = _glq_apply_e8p(x, layer)
            out_features = (sum(layer.glq_shard_sizes)
                            if getattr(layer, 'glq_is_fused', False)
                            else layer.glq_out_features)
        elif getattr(layer, 'glq_is_fused', False):
            # Fused QKV: dequant each shard independently, concatenate
            shard_outputs = []
            for i in range(layer.glq_num_shards):
                meta = layer._glq_shard_meta[i]
                n_stages = meta.get('n_stages', 2 if meta['has_stage2'] else 1)
                Qidxs3 = (layer.Qidxs3.get_shard(i)
                          if n_stages >= 3 and hasattr(layer, 'Qidxs3') else None)
                Qidxs4 = (layer.Qidxs4.get_shard(i)
                          if n_stages >= 4 and hasattr(layer, 'Qidxs4') else None)
                y_shard = _glq_apply_shard(
                    x, device, cb, cb2,
                    Qidxs=layer.Qidxs.get_shard(i),
                    SU=layer.SU.get_shard(i),
                    SV=layer.SV.get_shard(i),
                    wscale=meta['wscale'],
                    has_stage2=meta['has_stage2'],
                    inv_rs=meta['inv_rs'],
                    Qidxs2=layer.Qidxs2.get_shard(i) if meta['has_stage2'] else None,
                    out_features=meta['out'],
                    in_features=meta['in'],
                    m_pad=meta['m_pad'],
                    n_pad=meta['n_pad'],
                    log_n=meta['log_n'],
                    log_m=meta['log_m'],
                    Qidxs3=Qidxs3, inv_rs2=meta.get('inv_rs2', 0.0),
                    Qidxs4=Qidxs4, inv_rs3=meta.get('inv_rs3', 0.0),
                    block_diag_meta=meta.get('bd_meta'),
                )
                shard_outputs.append(y_shard)
            y = torch.cat(shard_outputs, dim=-1)
            out_features = sum(layer.glq_shard_sizes)
        else:
            # Standard: single dequant
            y = _glq_apply_single(x, layer, '', cb, cb2, device)
            out_features = layer.glq_out_features

        if bias is not None:
            y = y + bias.unsqueeze(0).to(y.dtype)

        return y.reshape(*orig_shape[:-1], out_features)
