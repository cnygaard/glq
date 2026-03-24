"""GLQ linear method for vLLM — dequant+matmul in RHT domain.

Phase 1 (current): Dequantize at load time into standard bf16 weights.
Gets GLQ models serving through vLLM immediately at full vLLM speed,
but uses bf16 VRAM (no compression benefit).

Phase 2 (future): Store compressed indices, dequant on-the-fly in apply()
using GLQ CUDA C kernels. Full compression + vLLM speed.
"""

import math
import os

import torch
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from glq.codebook import E8ShellCodebook


# Shared codebook (loaded once, reused by all layers)
_codebook = None
_codebook2 = None


def _get_codebooks(max_bpw: int = 2):
    """Lazy-load shared codebook instances."""
    global _codebook, _codebook2
    if _codebook is None:
        cb_path = os.path.join(os.path.dirname(__file__), "..", "glq", "e8_codebook.pt")
        if os.path.exists(cb_path):
            _codebook = E8ShellCodebook.load(cb_path, device="cpu")
        else:
            _codebook = E8ShellCodebook(device="cpu", verbose=False)

    if _codebook2 is None and max_bpw >= 3:
        if max_bpw >= 4:
            _codebook2 = _codebook
        else:
            _codebook2 = _codebook.make_small(256)

    return _codebook, _codebook2


def dequantize_glq_weight(
    Qidxs: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
    Wscale: torch.Tensor,
    codebook: E8ShellCodebook,
    Qidxs2: torch.Tensor | None = None,
    inv_resid_scale: float = 0.0,
    codebook2: E8ShellCodebook | None = None,
    out_features: int | None = None,
    in_features: int | None = None,
) -> torch.Tensor:
    """Dequantize GLQ indices back to a dense fp16 weight matrix."""
    from glq.hadamard import fast_hadamard_transform

    m_pad, n_blocks = Qidxs.shape
    n_pad = n_blocks * 8

    # Decode primary codebook
    W_rht = codebook.decode(Qidxs.long().reshape(-1))
    W_rht = W_rht.reshape(m_pad, n_pad).float()

    # Two-stage residual
    if Qidxs2 is not None and inv_resid_scale != 0.0 and codebook2 is not None:
        W_rht2 = codebook2.decode(Qidxs2.long().reshape(-1))
        W_rht2 = W_rht2.reshape(m_pad, n_pad).float()
        W_rht = W_rht + W_rht2 * inv_resid_scale

    W_rht = W_rht * Wscale.float()

    # Inverse RHT: W = Had @ diag(SU) @ W_rht @ diag(SV) @ Had
    su = SU.float()
    sv = SV.float()
    W = fast_hadamard_transform(W_rht) * su.unsqueeze(1)
    W = fast_hadamard_transform(W.T).T * sv.unsqueeze(0)

    # Unpad
    if out_features is not None and in_features is not None:
        W = W[:out_features, :in_features]

    return W.half()


class GLQLinearMethod(UnquantizedLinearMethod):
    """GLQ quantized linear method for vLLM.

    Phase 1: Inherits UnquantizedLinearMethod — weights are dequantized
    to fp16 during model loading and served as standard dense weights.
    vLLM handles all inference optimization (batching, scheduling, etc.).
    """

    def __init__(self, quant_config: QuantizationConfig, bpw: int = 2):
        super().__init__()
        self.quant_config = quant_config
        self.bpw = bpw
