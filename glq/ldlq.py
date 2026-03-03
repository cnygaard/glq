"""Block LDL decomposition and LDLQ codebook quantization."""

import torch
from typing import Tuple, Optional


def block_LDL(H: torch.Tensor, block_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block LDL decomposition: H = L @ blkdiag(D) @ L^T.

    L is block unit-lower-triangular, D is block-diagonal with b x b blocks.

    Args:
        H: (n, n) symmetric positive-definite matrix
        block_size: b, must divide n

    Returns:
        (L, D): L is (n, n), D is (n//b, b, b)
    """
    n = H.shape[0]
    b = block_size
    assert n % b == 0, f"n={n} must be divisible by block_size={b}"
    m = n // b

    L_chol = torch.linalg.cholesky(H)

    DL = torch.diagonal(L_chol.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = DL @ DL.permute(0, 2, 1)

    DL_inv = torch.linalg.inv(DL)
    L = L_chol.reshape(n, m, b).clone()
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL_inv[i]
    L = L.reshape(n, n)

    return L, D


def quantize_ldlq_codebook(
    W: torch.Tensor,
    H: torch.Tensor,
    codebook,
    tune_iters: int = 0,
    Wscale: Optional[float] = None,
):
    """
    LDLQ (Hessian-aware sequential VQ) using a finite codebook.

    Args:
        W: (m, n) weight matrix. n must be divisible by codebook.codesz.
        H: (n, n) Hessian, symmetric positive-definite.
        codebook: E8ShellCodebook instance (or duck-type compatible).
        tune_iters: extra refinement passes.
        Wscale: global scale. If None, computed from rms(W)/opt_scale.

    Returns:
        dict with W_hat, indices, Wscale, bpw, quant_mse, proxy_loss.
    """
    device = codebook.device
    m, n = W.shape
    b = codebook.codesz
    assert n % b == 0, f"n={n} must be divisible by {b}"
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    damp = 0.01 * torch.diag(H).mean()
    H_reg = H + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    if Wscale is None:
        W_rms = W.pow(2).mean().sqrt().item()
        Wscale = W_rms / codebook.opt_scale if W_rms > 1e-10 else 1.0

    Wr = W / Wscale
    hatWr = torch.zeros_like(Wr)
    all_indices = torch.zeros(m, num_blocks, dtype=torch.long, device=device)

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b
        if ke < n:
            feedback = (Wr[:, ke:] - hatWr[:, ke:]) @ L[ke:, kb:ke]
        else:
            feedback = 0.0
        WXWX = Wr[:, kb:ke] + feedback
        hatWr[:, kb:ke], all_indices[:, k] = codebook.quantize(WXWX)

    for _ in range(tune_iters):
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            residual = Wr - hatWr
            H_col = H_reg[:, kb:ke]
            H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
            feedback = residual @ H_col @ H_diag_inv
            WXWX = hatWr[:, kb:ke] + feedback
            hatWr[:, kb:ke], all_indices[:, k] = codebook.quantize(WXWX)

    W_hat = hatWr * Wscale

    quant_mse = ((W - W_hat) ** 2).mean().item()
    diff = W - W_hat
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()
    bpw = 16.0 / b + 32.0 / (m * n)

    return {
        'W_hat': W_hat,
        'indices': all_indices,
        'Wscale': Wscale,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }


def quantize_ldlq_codebook_rvq(
    W: torch.Tensor,
    H: torch.Tensor,
    codebook,
    tune_iters: int = 0,
    Wscale: Optional[float] = None,
):
    """
    LDLQ with two-stage RVQ codebook — 4 bpw (32 bits / 8 dims).
    """
    device = codebook.device
    m, n = W.shape
    b = codebook.codesz
    assert n % b == 0
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    damp = 0.01 * torch.diag(H).mean()
    H_reg = H + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    if Wscale is None:
        W_rms = W.pow(2).mean().sqrt().item()
        Wscale = W_rms / codebook.opt_scale if W_rms > 1e-10 else 1.0

    Wr = W / Wscale
    hatWr = torch.zeros_like(Wr)

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b
        if ke < n:
            feedback = (Wr[:, ke:] - hatWr[:, ke:]) @ L[ke:, kb:ke]
        else:
            feedback = 0.0
        WXWX = Wr[:, kb:ke] + feedback
        hatWr[:, kb:ke], _ = codebook.quantize_rvq(WXWX)

    for _ in range(tune_iters):
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            residual = Wr - hatWr
            H_col = H_reg[:, kb:ke]
            H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
            feedback = residual @ H_col @ H_diag_inv
            WXWX = hatWr[:, kb:ke] + feedback
            hatWr[:, kb:ke], _ = codebook.quantize_rvq(WXWX)

    W_hat = hatWr * Wscale

    quant_mse = ((W - W_hat) ** 2).mean().item()
    diff = W - W_hat
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()
    bpw = 32.0 / b + 32.0 / (m * n)

    return {
        'W_hat': W_hat,
        'Wscale': Wscale,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }
