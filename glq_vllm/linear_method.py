"""GLQ linear method for vLLM — fused dequant+matmul (Phase 2).

Keeps compressed GLQ indices (Qidxs, SU, SV) in GPU memory.
Dequant on-the-fly in apply() using GLQ's CUDA C and Triton kernels.
This gives both vLLM inference speed AND GLQ VRAM compression.
"""

import math
import os

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

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

    # Move to GPU
    cb._move_to_device(device)
    if cb2 is not None and cb2 is not cb:
        cb2._move_to_device(device)

    _codebook = cb
    _codebook2_small = cb2
    _codebook_device = device
    return cb, cb2


class GLQLinearMethod(LinearMethodBase):
    """GLQ fused dequant+matmul linear method for vLLM.

    Weights stay compressed (int16 indices + sign vectors).
    apply() does: input RHT → fused dequant+matmul → output RHT.
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
        output_size_per_partition = sum(output_partition_sizes)

        m_pad = 1 << (output_size_per_partition - 1).bit_length()
        n_pad = 1 << (input_size_per_partition - 1).bit_length()
        n_blocks = n_pad // 8

        # Store dims for apply()
        layer.glq_out_features = output_size_per_partition
        layer.glq_in_features = input_size_per_partition
        layer.glq_m_pad = m_pad
        layer.glq_n_pad = n_pad
        layer.glq_bpw = self.bpw

        # Compressed weight buffers — names match safetensor keys
        layer.Qidxs = torch.nn.Parameter(
            torch.zeros(m_pad, n_blocks, dtype=torch.int16), requires_grad=False)
        layer.SU = torch.nn.Parameter(
            torch.ones(m_pad, dtype=torch.float16), requires_grad=False)
        layer.SV = torch.nn.Parameter(
            torch.ones(n_pad, dtype=torch.float16), requires_grad=False)
        layer.Wscale = torch.nn.Parameter(
            torch.ones((), dtype=torch.float32), requires_grad=False)
        layer.Qidxs2 = torch.nn.Parameter(
            torch.zeros(m_pad, n_blocks, dtype=torch.int16), requires_grad=False)
        layer.inv_resid_scale = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float32), requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Set up codebook and cache scalars. NO dequantization."""
        device = layer.Qidxs.device
        bpw = getattr(layer, 'glq_bpw', 2)

        # Ensure shared codebook is on the right device
        inv_rs = layer.inv_resid_scale.item()
        has_stage2 = inv_rs != 0.0
        max_bpw = bpw if has_stage2 else 2
        _ensure_codebook(device, max_bpw=max_bpw)

        # Cache scalars to avoid .item() calls in hot path
        layer.glq_wscale = layer.Wscale.item()
        layer.glq_has_stage2 = has_stage2
        layer.glq_inv_rs = inv_rs
        layer.glq_log_n = int(math.log2(layer.glq_n_pad))
        layer.glq_log_m = int(math.log2(layer.glq_m_pad))

        # Ensure CUDA extension is loaded
        _try_load_cuda_ext()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        orig_shape = x.shape
        x = x.reshape(-1, layer.glq_in_features)
        B = x.shape[0]
        n_pad = layer.glq_n_pad
        m_pad = layer.glq_m_pad
        log_n = layer.glq_log_n
        log_m = layer.glq_log_m
        in_features = layer.glq_in_features
        out_features = layer.glq_out_features

        # --- 1. Input RHT: pad + SV signs + FHT ---
        x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=x.device)
        if n_pad <= 16384 and _glq_cuda is not None:
            _glq_cuda.glq_input_rht_cuda(
                x.half().contiguous(), layer.SV, x_rht,
                in_features, in_features,
                1.0 / math.sqrt(n_pad), n_pad, log_n)
        else:
            from glq.quantized_linear import _input_rht_kernel
            _input_rht_kernel[(B,)](
                x, layer.SV, x_rht,
                in_features, x.stride(0),
                1.0 / math.sqrt(n_pad),
                N=n_pad, LOG_N=log_n, num_warps=8)

        # --- 2. Fused dequant+matmul in RHT domain ---
        cb, cb2 = _codebook, _codebook2_small
        has_stage2 = layer.glq_has_stage2
        cb_packed = getattr(cb, 'codebook_packed', None)

        y_rht = glq_dequant_matmul(
            x_rht, layer.Qidxs,
            cb.codebook_half,
            layer.glq_wscale,
            Qidxs2=layer.Qidxs2 if has_stage2 else None,
            codebook2=cb2.codebook_half if has_stage2 and cb2 is not None else None,
            inv_resid_scale=layer.glq_inv_rs,
            codebook_packed=cb_packed,
        )

        # --- 3. Output RHT: FHT + SU signs + unpad ---
        if m_pad <= 16384 and _glq_cuda is not None:
            y = torch.empty(B, out_features, dtype=torch.float16, device=x.device)
            _glq_cuda.glq_output_rht_cuda(
                y_rht, layer.SU, y,
                out_features, m_pad, log_m,
                1.0 / math.sqrt(m_pad))
            if dtype != torch.float16:
                y = y.to(dtype)
        else:
            from glq.quantized_linear import _output_rht_kernel
            output_fp16 = (dtype == torch.float16)
            y = torch.empty(B, out_features, dtype=dtype, device=x.device)
            _output_rht_kernel[(B,)](
                y_rht, layer.SU, y,
                out_features, y_rht.stride(0), y.stride(0),
                1.0 / math.sqrt(m_pad),
                OUTPUT_FP16=output_fp16,
                M=m_pad, LOG_M=log_m, num_warps=8)

        if bias is not None:
            y = y + bias.unsqueeze(0).to(dtype)

        return y.reshape(*orig_shape[:-1], out_features)
