"""GLQ linear method for vLLM — fused dequant+matmul (Phase 2).

Keeps compressed GLQ indices (Qidxs, SU, SV) in GPU memory.
Dequant on-the-fly in apply() using GLQ's CUDA C and Triton kernels.

Supports:
- Standard linear layers (ColumnParallel, RowParallel, Replicated)
- Fused QKV layers (QKVParallelLinear) — per-shard GLQ buffers
"""

import math
import os
from typing import Callable

import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import BasevLLMParameter

from glq.codebook import E8ShellCodebook
from glq.inference_kernel import glq_dequant_matmul, _try_load_cuda_ext, _glq_cuda


# Shared codebook singleton — moved to GPU on first use
_codebook = None
_codebook2_small = None
_codebook_device = None


def _ensure_codebook(device, max_bpw: int = 2):
    """Lazy-load codebook and move to target device (once)."""
    global _codebook, _codebook2_small, _codebook_device

    if _codebook is not None and _codebook_device == device:
        return _codebook, _codebook2_small

    cb_path = os.path.join(os.path.dirname(__file__), "..", "glq", "e8_codebook.pt")
    if os.path.exists(cb_path):
        cb = E8ShellCodebook.load(cb_path, device="cpu")
    else:
        cb = E8ShellCodebook(device="cpu", verbose=False)

    cb2 = None
    if max_bpw >= 4:
        cb2 = cb
    elif max_bpw >= 3:
        cb2 = cb.make_small(256)

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


def _register_glq_buffers(layer, prefix, out_size, in_size):
    """Register one set of GLQ compressed buffers on the layer."""
    m_pad = _glq_pad(out_size)
    n_pad = _glq_pad(in_size)
    n_blocks = n_pad // 8
    p = prefix
    setattr(layer, f'Qidxs{p}', torch.nn.Parameter(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16), requires_grad=False))
    setattr(layer, f'SU{p}', torch.nn.Parameter(
        torch.ones(m_pad, dtype=torch.float16), requires_grad=False))
    setattr(layer, f'SV{p}', torch.nn.Parameter(
        torch.ones(n_pad, dtype=torch.float16), requires_grad=False))
    setattr(layer, f'Wscale{p}', torch.nn.Parameter(
        torch.ones((), dtype=torch.float32), requires_grad=False))
    setattr(layer, f'Qidxs2{p}', torch.nn.Parameter(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16), requires_grad=False))
    setattr(layer, f'inv_resid_scale{p}', torch.nn.Parameter(
        torch.zeros((), dtype=torch.float32), requires_grad=False))
    return m_pad, n_pad


class GLQShardedParameter(BasevLLMParameter):
    """Parameter that stores per-shard GLQ buffers for fused QKV layers.

    vLLM's stacked_params_mapping renames q_proj.Qidxs → qkv_proj.Qidxs
    and calls load_qkv_weight with shard_id="q"|"k"|"v". This parameter
    routes each shard to a separate internal buffer (different padded sizes
    because GLQ pads to power-of-2 and each shard has independent RHT).
    """

    def __init__(self, shard_sizes: list[int], inner_dim: int,
                 dtype: torch.dtype, weight_loader: Callable, **kwargs):
        # Store a small dummy tensor as the parameter data
        # (vLLM requires Parameter to have .data)
        data = torch.zeros(1, dtype=dtype)
        super().__init__(data=data, weight_loader=weight_loader)
        self._shard_sizes = shard_sizes
        self._inner_dim = inner_dim
        self._dtype = dtype
        # Allocate per-shard buffers with power-of-2 padding
        self._shard_data = []
        for sz in shard_sizes:
            m_pad = _glq_pad(sz) if sz > 1 else 1
            if inner_dim > 0:
                self._shard_data.append(torch.zeros(m_pad, inner_dim, dtype=dtype))
            else:
                # Scalar per shard (Wscale, inv_resid_scale)
                self._shard_data.append(torch.zeros((), dtype=dtype))

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        idx = self._shard_id_as_int(shard_id)
        if idx < len(self._shard_data):
            self._shard_data[idx].copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        if shard_id is not None:
            idx = self._shard_id_as_int(shard_id)
        else:
            idx = 0
        if idx < len(self._shard_data):
            self._shard_data[idx].copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        if len(self._shard_data) == 1:
            self._shard_data[0].copy_(loaded_weight)

    def get_shard(self, idx: int) -> torch.Tensor:
        return self._shard_data[idx]

    @property
    def num_shards(self) -> int:
        return len(self._shard_data)

    def to(self, *args, **kwargs):
        """Move all internal shard buffers to device."""
        result = super().to(*args, **kwargs)
        result._shard_data = [s.to(*args, **kwargs) for s in result._shard_data]
        return result

    def cuda(self, device=None):
        result = super().cuda(device)
        result._shard_data = [s.cuda(device) for s in result._shard_data]
        return result


def _glq_apply_shard(x, device, cb, cb2, Qidxs, SU, SV, wscale,
                     has_stage2, inv_rs, Qidxs2, out_features, in_features,
                     m_pad, n_pad, log_n, log_m):
    """Run input RHT → dequant+matmul → output RHT for explicit tensors."""
    dtype = x.dtype
    B = x.shape[0]

    # Input RHT
    x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=device)
    if n_pad <= 16384 and _glq_cuda is not None:
        _glq_cuda.glq_input_rht_cuda(
            x.half().contiguous(), SV, x_rht,
            in_features, in_features,
            1.0 / math.sqrt(n_pad), n_pad, log_n)
    else:
        from glq.quantized_linear import _input_rht_kernel
        _input_rht_kernel[(B,)](
            x, SV, x_rht, in_features, x.stride(0),
            1.0 / math.sqrt(n_pad), N=n_pad, LOG_N=log_n, num_warps=8)

    # Dequant + matmul
    cb_packed = getattr(cb, 'codebook_packed', None)
    cb2_half = cb2.codebook_half if has_stage2 and cb2 is not None else None

    y_rht = glq_dequant_matmul(
        x_rht, Qidxs, cb.codebook_half, wscale,
        Qidxs2=Qidxs2, codebook2=cb2_half,
        inv_resid_scale=inv_rs, codebook_packed=cb_packed)

    # Output RHT
    if m_pad <= 16384 and _glq_cuda is not None:
        y = torch.empty(B, out_features, dtype=torch.float16, device=device)
        _glq_cuda.glq_output_rht_cuda(
            y_rht, SU, y, out_features, m_pad,
            log_m, 1.0 / math.sqrt(m_pad))
        if dtype != torch.float16:
            y = y.to(dtype)
    else:
        from glq.quantized_linear import _output_rht_kernel
        output_fp16 = (dtype == torch.float16)
        y = torch.empty(B, out_features, dtype=dtype, device=device)
        _output_rht_kernel[(B,)](
            y_rht, SU, y, out_features, y_rht.stride(0), y.stride(0),
            1.0 / math.sqrt(m_pad),
            OUTPUT_FP16=output_fp16, M=m_pad, LOG_M=log_m, num_warps=8)
    return y


def _glq_apply_single(x, layer, prefix, cb, cb2, device):
    """Run input RHT → dequant+matmul → output RHT for one set of GLQ buffers."""
    dtype = x.dtype
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

    B = x.shape[0]

    # Input RHT
    x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=device)
    if n_pad <= 16384 and _glq_cuda is not None:
        _glq_cuda.glq_input_rht_cuda(
            x.half().contiguous(), SV, x_rht,
            in_features, in_features,
            1.0 / math.sqrt(n_pad), n_pad, log_n)
    else:
        from glq.quantized_linear import _input_rht_kernel
        _input_rht_kernel[(B,)](
            x, SV, x_rht, in_features, x.stride(0),
            1.0 / math.sqrt(n_pad), N=n_pad, LOG_N=log_n, num_warps=8)

    # Dequant + matmul
    cb_packed = getattr(cb, 'codebook_packed', None)
    Qidxs2 = getattr(layer, f'Qidxs2{prefix}') if has_stage2 else None
    cb2_half = cb2.codebook_half if has_stage2 and cb2 is not None else None

    y_rht = glq_dequant_matmul(
        x_rht, Qidxs, cb.codebook_half, wscale,
        Qidxs2=Qidxs2, codebook2=cb2_half,
        inv_resid_scale=inv_rs, codebook_packed=cb_packed)

    # Output RHT
    if m_pad <= 16384 and _glq_cuda is not None:
        y = torch.empty(B, out_features, dtype=torch.float16, device=device)
        _glq_cuda.glq_output_rht_cuda(
            y_rht, SU, y, out_features, m_pad,
            log_m, 1.0 / math.sqrt(m_pad))
        if dtype != torch.float16:
            y = y.to(dtype)
    else:
        from glq.quantized_linear import _output_rht_kernel
        output_fp16 = (dtype == torch.float16)
        y = torch.empty(B, out_features, dtype=dtype, device=device)
        _output_rht_kernel[(B,)](
            y_rht, SU, y, out_features, y_rht.stride(0), y.stride(0),
            1.0 / math.sqrt(m_pad),
            OUTPUT_FP16=output_fp16, M=m_pad, LOG_M=log_m, num_warps=8)

    return y


class GLQLinearMethod(LinearMethodBase):
    """GLQ fused dequant+matmul linear method for vLLM.

    Weights stay compressed (int16 indices + sign vectors).
    Supports standard linear layers AND fused QKV (per-shard buffers).
    """

    def __init__(self, quant_config, bpw: int = 2):
        self.quant_config = quant_config
        self.bpw = bpw

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
        is_fused = len(output_partition_sizes) > 1
        layer.glq_is_fused = is_fused
        layer.glq_in_features = input_size_per_partition
        layer.glq_bpw = self.bpw

        weight_loader = extra_weight_attrs.get("weight_loader")

        if is_fused:
            # Fused QKV: use GLQShardedParameter for per-shard storage
            layer.glq_shard_sizes = output_partition_sizes
            layer.glq_num_shards = len(output_partition_sizes)
            n_pad = _glq_pad(input_size_per_partition)
            n_blocks = n_pad // 8

            # Each GLQ buffer is a GLQShardedParameter that routes shard_id
            layer.Qidxs = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16, weight_loader=weight_loader)
            layer.SU = GLQShardedParameter(
                output_partition_sizes, 0, torch.float16, weight_loader=weight_loader)
            layer.SV = torch.nn.Parameter(
                torch.ones(n_pad, dtype=torch.float16), requires_grad=False)
            layer.Wscale = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32, weight_loader=weight_loader)
            layer.Qidxs2 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16, weight_loader=weight_loader)
            layer.inv_resid_scale = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32, weight_loader=weight_loader)

            layer.glq_n_pad = n_pad
        else:
            # Standard: single set of GLQ buffers (no suffix)
            out_sz = sum(output_partition_sizes)
            layer.glq_out_features = out_sz
            m_pad, n_pad = _register_glq_buffers(
                layer, '', out_sz, input_size_per_partition)
            layer.glq_m_pad = m_pad
            layer.glq_n_pad = n_pad

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Set up codebook and cache scalars. NO dequantization."""
        device = next(layer.parameters()).device
        bpw = getattr(layer, 'glq_bpw', 2)

        # Ensure shared codebook is on the right device
        # Determine max_bpw across all shards
        max_bpw = 2
        if getattr(layer, 'glq_is_fused', False):
            for i in range(layer.glq_num_shards):
                inv_rs = layer.inv_resid_scale.get_shard(i).item()
                if inv_rs != 0.0:
                    max_bpw = max(max_bpw, bpw)
        else:
            inv_rs = layer.inv_resid_scale.item()
            if inv_rs != 0.0:
                max_bpw = bpw

        _ensure_codebook(device, max_bpw=max_bpw)
        _try_load_cuda_ext()

        # Cache scalars for all shards
        if getattr(layer, 'glq_is_fused', False):
            layer._glq_shard_meta = []
            n_pad = layer.glq_n_pad
            for i in range(layer.glq_num_shards):
                out_sz = layer.glq_shard_sizes[i]
                m_pad = _glq_pad(out_sz)
                inv_rs_val = layer.inv_resid_scale.get_shard(i).item()
                layer._glq_shard_meta.append({
                    'wscale': layer.Wscale.get_shard(i).item(),
                    'has_stage2': inv_rs_val != 0.0,
                    'inv_rs': inv_rs_val,
                    'out': out_sz,
                    'in': layer.glq_in_features,
                    'm_pad': m_pad,
                    'n_pad': n_pad,
                    'log_n': int(math.log2(n_pad)),
                    'log_m': int(math.log2(m_pad)),
                })
        else:
            inv_rs = layer.inv_resid_scale.item()
            layer._glq_wscale = layer.Wscale.item()
            layer._glq_has_stage2 = inv_rs != 0.0
            layer._glq_inv_rs = inv_rs
            layer._glq_m_pad = layer.glq_m_pad
            layer._glq_n_pad = layer.glq_n_pad
            layer._glq_log_n = int(math.log2(layer.glq_n_pad))
            layer._glq_log_m = int(math.log2(layer.glq_m_pad))
            layer._glq_out = layer.glq_out_features
            layer._glq_in = layer.glq_in_features

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        in_features = layer.glq_in_features
        x = x.reshape(-1, in_features)
        device = x.device
        cb, cb2 = _codebook, _codebook2_small

        if getattr(layer, 'glq_is_fused', False):
            # Fused QKV: dequant each shard independently, concatenate
            shard_outputs = []
            for i in range(layer.glq_num_shards):
                meta = layer._glq_shard_meta[i]
                y_shard = _glq_apply_shard(
                    x, device, cb, cb2,
                    Qidxs=layer.Qidxs.get_shard(i),
                    SU=layer.SU.get_shard(i),
                    SV=layer.SV,
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
