"""QTIP trellis-coded quantization wired into GLQ's LDLQ + RHT (Option-B scratch harness).

Reuses GLQ's `block_LDL` + LDLQ error-feedback loop (proven equivalent to QTIP's
BlockLDLQ: both use identity-diagonal-block L, and the feedback reads only rows strictly
below the current block, so QTIP's extra diagonal-zeroing is a no-op) with a **tile-based
trellis codebook** replacing the E8 codebook. Bitrate = K bits/weight (K=2 → 2 bpw).

QTIP is GPL-3.0 (same as GLQ). The trellis codebook itself is `benchmarks/_qtip_trellis.py`
(extracted `bitshift_codebook`). See plan: ~/.claude/plans/hazy-churning-shannon.md
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # _qtip_trellis
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # glq
import _qtip_trellis as qt          # noqa: E402
from glq.ldlq import block_LDL      # noqa: E402
from glq.rht import RHT             # noqa: E402

TD = 16                             # QTIP tile dim td_x = td_y = 16 → 256-element tiles


class TrellisCodebook:
    """GLQ-duck-typed (codesz=16) wrapper around QTIP's bitshift trellis codebook.

    `quantize((m,16))` groups 16 consecutive output-rows × 16 input-cols into 256-element
    tiles (== QTIP's `WXWX.T.reshape(-1,256)`), trellis-quantizes each as a tail-biting
    Viterbi sequence, and reshapes back to (m,16).
    """

    def __init__(self, variant="3inst", K=2, device="cpu"):
        if variant == "3inst":                     # lookup-free computed, V=1
            self.cb = qt.bitshift_codebook(L=16, K=K, V=1, tlut_bits=0, decode_mode="3inst")
        elif variant == "hyb":                     # QTIP's HYB: kmeans LUT, V=2, Q=9
            self.cb = qt.bitshift_codebook(L=16, K=K, V=2, tlut_bits=9, decode_mode="quantlut_sym")
        else:
            raise ValueError(variant)
        self.cb = self.cb.to(device)
        self.cb.fakeinf = self.cb.fakeinf.to(device)   # plain attr, not moved by .to()
        self.variant = variant
        self.codesz = TD
        self.device = device
        # QTIP scale: Wr /= rms(Wr)/(rms(lut)*0.9)  ⇔  GLQ Wscale = rms(W)*opt_scale
        self.opt_scale = 1.0 / (self.cb.lut.double().square().mean().sqrt().item() * 0.9)

    def quantize(self, WXWX):
        """(m, 16) fp32 → (hatW (m,16), None). m must be a multiple of 16."""
        m = WXWX.shape[0]
        tiles = WXWX.reshape(m // TD, TD * TD).contiguous()   # (m/16, 256): 16 rows × 16 cols/tile
        hat, _ = self.cb.quantize(tiles)
        return hat.reshape(m, TD).to(WXWX.dtype), None


def trellis_ldlq(W, H, cb, Wscale=None):
    """LDLQ (Hessian error feedback) with the trellis codebook, block size 16.

    Copy of GLQ's `quantize_ldlq_codebook` CPU loop (glq/ldlq.py:161-171) with b=16 and no
    index bookkeeping (the bf16-dequant gate needs only W_hat). Returns hatW in the same
    (already-RHT'd, scaled-back) domain as the input W.
    """
    dev = cb.device
    W = W.float().to(dev)
    H = H.float().to(dev)
    m, n = W.shape
    b = TD
    assert n % b == 0 and m % b == 0, f"trellis needs m,n %16 (got {m}x{n})"

    damp = 0.01 * torch.diag(H).mean()
    L, _ = block_LDL(H + damp * torch.eye(n, device=dev), block_size=b)

    if Wscale is None:
        Wscale = W.pow(2).mean().sqrt().item() * cb.opt_scale
    Wr = W / Wscale
    hatWr = torch.zeros_like(Wr)
    R = Wr.clone()
    for k in reversed(range(n // b)):
        kb, ke = k * b, (k + 1) * b
        feedback = R[:, ke:] @ L[ke:, kb:ke] if ke < n else 0.0
        WXWX = Wr[:, kb:ke] + feedback
        hatWr[:, kb:ke], _ = cb.quantize(WXWX)
        R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]
    return hatWr * Wscale


def quantize_layer_trellis_rht(W, H, cb, block_diagonal=True):
    """RHT + trellis-LDLQ, mirroring `quantize_layer_e8_shell_rht` (glq/quantize_model.py:361).

    Double-damps H (pre-RHT + in-LDLQ) exactly like the e8p path for fair comparison, uses
    GLQ's RHT (QTIP's own RHT is unavailable on the box; `_qtip_mse` confirmed GLQ-RHT+trellis
    wins), and inverse-RHTs so W_hat lands in original weight space.
    """
    dev = cb.device
    m, n = W.shape
    Wf = W.float().to(dev)
    Hf = H.float().to(dev)
    damp = 0.01 * torch.mean(torch.diag(Hf))         # pre-RHT damp (match :392-394)
    Hf = Hf + damp * torch.eye(n, device=dev)
    rht = RHT(m, n, device=dev, block_diagonal=block_diagonal, apply_left=True, e8p=False)
    Wt = rht.transform_weights(Wf)
    Ht = rht.transform_hessian(Hf)
    assert Wt.shape == (m, n), f"trellis needs no RHT padding; got {tuple(Wt.shape)} for {m}x{n}"
    W_hat_t = trellis_ldlq(Wt, Ht, cb)               # in-LDLQ damp too → double-damp like e8p
    return rht.inverse_transform_weights(W_hat_t)
