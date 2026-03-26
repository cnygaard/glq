"""GLQ dequantization helpers — no vLLM dependency.

Used by tests and by linear_method.py for weight dequantization.
"""

import os

import torch
from glq.codebook import E8ShellCodebook
from glq.hadamard import fast_hadamard_transform


def get_codebook():
    """Get primary codebook on CPU."""
    cb_path = os.path.join(os.path.dirname(__file__), "..", "glq", "e8_codebook.pt")
    if os.path.exists(cb_path):
        return E8ShellCodebook.load(cb_path, device="cpu")
    return E8ShellCodebook(device="cpu", verbose=False)


def get_codebook2(bpw: int):
    """Get secondary codebook on CPU."""
    cb = get_codebook()
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
    """Dequantize GLQ indices to a dense weight matrix."""
    m_pad, n_blocks = Qidxs.shape
    n_pad = n_blocks * 8

    W_rht = codebook.decode(Qidxs.long().reshape(-1)).reshape(m_pad, n_pad).float()

    if Qidxs2 is not None and inv_resid_scale != 0.0 and codebook2 is not None:
        W_rht2 = codebook2.decode(Qidxs2.long().reshape(-1)).reshape(m_pad, n_pad).float()
        W_rht = W_rht + W_rht2 * inv_resid_scale

    W_rht = W_rht * Wscale.float()

    # Inverse RHT (must match E8RHTLinear.dequantize exactly)
    W = fast_hadamard_transform(W_rht.clone())
    W = W * SV.float().unsqueeze(0)
    W = fast_hadamard_transform(W.T.clone()).T
    W = W * SU.float().unsqueeze(1)

    if out_features is not None and in_features is not None:
        W = W[:out_features, :in_features]

    return W.half()
