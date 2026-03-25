"""GLQ linear method for vLLM — dequant at load time (Phase 1).

Phase 1 (current): Load GLQ compressed buffers (Qidxs, SU, SV, etc.),
dequantize to dense fp16 in process_weights_after_loading(), then serve
as standard unquantized weights. Gets full vLLM inference speed but
uses bf16 VRAM.

Phase 2 (future): Keep compressed format, dequant on-the-fly in apply()
using GLQ CUDA C kernels for full compression + vLLM speed.
"""

import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import LinearMethodBase

from glq.codebook import E8ShellCodebook
from glq.hadamard import fast_hadamard_transform


# Shared codebook singleton
_codebook = None
_codebook2_small = None  # 256-entry for 3bpw
_codebook2_full = None   # 65536-entry for 4bpw


def _get_codebook():
    global _codebook
    if _codebook is None:
        cb_path = os.path.join(os.path.dirname(__file__), "..", "glq", "e8_codebook.pt")
        if os.path.exists(cb_path):
            _codebook = E8ShellCodebook.load(cb_path, device="cpu")
        else:
            _codebook = E8ShellCodebook(device="cpu", verbose=False)
    return _codebook


def _get_codebook2(bpw: int):
    global _codebook2_small, _codebook2_full
    cb = _get_codebook()
    if bpw >= 4:
        if _codebook2_full is None:
            _codebook2_full = cb
        return _codebook2_full
    elif bpw >= 3:
        if _codebook2_small is None:
            _codebook2_small = cb.make_small(256)
        return _codebook2_small
    return None


def dequantize_glq_weight(
    Qidxs, SU, SV, Wscale, codebook,
    Qidxs2=None, inv_resid_scale=0.0, codebook2=None,
    out_features=None, in_features=None,
):
    """Dequantize GLQ indices to a dense weight matrix."""
    m_pad, n_blocks = Qidxs.shape
    n_pad = n_blocks * 8

    W_rht = codebook.decode(Qidxs.long().reshape(-1)).reshape(m_pad, n_pad).float()

    if Qidxs2 is not None and inv_resid_scale != 0.0 and codebook2 is not None:
        W_rht2 = codebook2.decode(Qidxs2.long().reshape(-1)).reshape(m_pad, n_pad).float()
        W_rht = W_rht + W_rht2 * inv_resid_scale

    W_rht = W_rht * Wscale.float()

    # Inverse RHT (must match E8RHTLinear.dequantize exactly)
    W = fast_hadamard_transform(W_rht.clone())     # FHT along columns
    W = W * SV.float().unsqueeze(0)                # SV signs
    W = fast_hadamard_transform(W.T.clone()).T      # FHT along rows
    W = W * SU.float().unsqueeze(1)                # SU signs

    if out_features is not None and in_features is not None:
        W = W[:out_features, :in_features]

    return W.half()


class GLQLinearMethod(LinearMethodBase):
    """GLQ dequant-at-load linear method for vLLM.

    create_weights: registers GLQ buffer params (Qidxs, SU, SV, etc.)
    process_weights_after_loading: dequantizes to dense weight, deletes buffers
    apply: standard F.linear
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

        # Padded dimensions (power of 2) matching GLQ's E8RHTLinear
        m_pad = 1 << (output_size_per_partition - 1).bit_length()
        n_pad = 1 << (input_size_per_partition - 1).bit_length()
        n_blocks = n_pad // 8

        # Store original dims for unpadding during dequant
        layer.glq_out_features = output_size_per_partition
        layer.glq_in_features = input_size_per_partition
        layer.glq_bpw = self.bpw

        # Register GLQ buffers — names must match safetensor keys
        layer.Qidxs = torch.nn.Parameter(
            torch.zeros(m_pad, n_blocks, dtype=torch.int16), requires_grad=False)
        layer.SU = torch.nn.Parameter(
            torch.ones(m_pad, dtype=torch.float16), requires_grad=False)
        layer.SV = torch.nn.Parameter(
            torch.ones(n_pad, dtype=torch.float16), requires_grad=False)
        layer.Wscale = torch.nn.Parameter(
            torch.ones((), dtype=torch.float32), requires_grad=False)

        # Stage 2 buffers (3/4bpw)
        layer.Qidxs2 = torch.nn.Parameter(
            torch.zeros(m_pad, n_blocks, dtype=torch.int16), requires_grad=False)
        layer.inv_resid_scale = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float32), requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dequantize GLQ buffers → dense weight, then delete buffers."""
        cb = _get_codebook()
        bpw = getattr(layer, 'glq_bpw', 2)
        inv_rs = layer.inv_resid_scale.item() if hasattr(layer, 'inv_resid_scale') else 0.0
        has_stage2 = inv_rs != 0.0

        cb2 = _get_codebook2(bpw) if has_stage2 else None

        # Move GLQ buffers to CPU for dequantization (codebook is on CPU)
        weight = dequantize_glq_weight(
            Qidxs=layer.Qidxs.data.cpu(),
            SU=layer.SU.data.cpu(),
            SV=layer.SV.data.cpu(),
            Wscale=layer.Wscale.data.cpu(),
            codebook=cb,
            Qidxs2=layer.Qidxs2.data.cpu() if has_stage2 else None,
            inv_resid_scale=inv_rs,
            codebook2=cb2,
            out_features=layer.glq_out_features,
            in_features=layer.glq_in_features,
        )

        # Replace GLQ buffers with a single dense weight
        device = layer.Qidxs.device
        layer.weight = torch.nn.Parameter(
            weight.to(device), requires_grad=False)

        # Clean up GLQ buffers to free memory
        del layer.Qidxs, layer.SU, layer.SV, layer.Wscale
        del layer.Qidxs2, layer.inv_resid_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight.to(x.dtype), bias)
