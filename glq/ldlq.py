"""Block LDL decomposition and LDLQ codebook quantization."""

import math
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
    # L_chol reshaped: (n, m, b). Multiply each block column by DL_inv[i].
    # L[:, i, :] @ DL_inv[i] for all i simultaneously via einsum.
    L = torch.einsum('nib,ibj->nij', L_chol.reshape(n, m, b), DL_inv).reshape(n, n)

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

    use_cuda = torch.device(device).type == 'cuda'
    if use_cuda:
        # Keep everything in fp16 to avoid per-block dtype conversions.
        # L_half for feedback matmul (Tensor Core), R_half tracks residual.
        L_half = L.half()
        Wr_half = Wr.half()
        R_half = Wr_half.clone()
        hatWr_half = torch.zeros_like(Wr_half)
        # Pre-allocate reusable buffers to avoid per-block allocation
        WXWX_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        dec_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        idx_buf = torch.empty(m, dtype=torch.int64, device=device)

        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            if ke < n:
                torch.mm(R_half[:, ke:], L_half[ke:, kb:ke], out=WXWX_buf)
                WXWX_buf.add_(Wr_half[:, kb:ke])
            else:
                WXWX_buf.copy_(Wr_half[:, kb:ke])
            # Fused NN + decode: single kernel, no separate gather
            codebook.quantize_fast(WXWX_buf,
                                   decoded_out=dec_buf, idx_out=idx_buf)
            all_indices[:, k] = idx_buf
            hatWr_half[:, kb:ke] = dec_buf
            R_half[:, kb:ke] = Wr_half[:, kb:ke] - dec_buf

        # Recover fp32 hatWr for downstream use
        hatWr = hatWr_half.float()
    else:
        R = Wr.clone()
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            if ke < n:
                feedback = R[:, ke:] @ L[ke:, kb:ke]
            else:
                feedback = 0.0
            WXWX = Wr[:, kb:ke] + feedback
            hatWr[:, kb:ke], all_indices[:, k] = codebook.quantize(WXWX)
            R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]

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


def quantize_ldlq_codebook_2stage(
    W: torch.Tensor,
    H: torch.Tensor,
    codebook1,
    codebook2,
    resid_scale: float,
    tune_iters: int = 0,
    Wscale: Optional[float] = None,
):
    """
    Two-stage LDLQ: primary codebook + secondary codebook on residual.

    For 4bpw: codebook1 = codebook2 = E8 full (65536 entries), 32 bits / 8 dims.
    For 3bpw: codebook1 = E8 full, codebook2 = E8 small (256 entries), 24 bits / 8 dims.

    Args:
        W: (m, n) weight matrix. n must be divisible by codebook1.codesz.
        H: (n, n) Hessian.
        codebook1: primary codebook (E8ShellCodebook, 65536 entries).
        codebook2: secondary codebook for residual.
        resid_scale: scale factor for residual before stage-2 quantization.
        tune_iters: extra refinement passes.
        Wscale: global scale. If None, computed from rms(W)/opt_scale.
    """
    device = codebook1.device
    m, n = W.shape
    b = codebook1.codesz
    assert n % b == 0
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    damp = 0.01 * torch.diag(H).mean()
    H_reg = H + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    if Wscale is None:
        W_rms = W.pow(2).mean().sqrt().item()
        Wscale = W_rms / codebook1.opt_scale if W_rms > 1e-10 else 1.0

    Wr = W / Wscale
    all_indices1 = torch.zeros(m, num_blocks, dtype=torch.long, device=device)
    all_indices2 = torch.zeros(m, num_blocks, dtype=torch.long, device=device)

    use_cuda = torch.device(device).type == 'cuda'
    # hatWr is allocated inside each branch: the CUDA path builds hatWr_half
    # in the loop and converts to fp32 at the end; the CPU path allocates
    # zeros upfront. Avoiding the upfront fp32 allocation on CUDA saves
    # m*n*4 bytes (~1.4 GB for the 123B down_proj shape).
    if use_cuda:
        L_half = L.half()
        Wr_half = Wr.half()
        R_half = Wr_half.clone()
        hatWr_half = torch.zeros_like(Wr_half)
        # Pre-allocate reusable buffers
        target_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        resid_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        dec1_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        dec2_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        idx1_buf = torch.empty(m, dtype=torch.int64, device=device)
        idx2_buf = torch.empty(m, dtype=torch.int64, device=device)
        resid_scale_half = torch.tensor(resid_scale, dtype=torch.float16,
                                        device=device)
        inv_resid_scale_half = torch.tensor(1.0 / resid_scale,
                                            dtype=torch.float16, device=device)

        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            if ke < n:
                torch.mm(R_half[:, ke:], L_half[ke:, kb:ke], out=target_buf)
                target_buf.add_(Wr_half[:, kb:ke])
            else:
                target_buf.copy_(Wr_half[:, kb:ke])

            # Fused NN + decode for stage 1
            codebook1.quantize_fast(target_buf,
                                    decoded_out=dec1_buf, idx_out=idx1_buf)
            all_indices1[:, k] = idx1_buf

            torch.sub(target_buf, dec1_buf, out=resid_buf)
            resid_buf.mul_(resid_scale_half)

            # Fused NN + decode for stage 2
            codebook2.quantize_fast(resid_buf,
                                    decoded_out=dec2_buf, idx_out=idx2_buf)
            all_indices2[:, k] = idx2_buf

            hatWr_half[:, kb:ke] = dec1_buf + dec2_buf * inv_resid_scale_half
            R_half[:, kb:ke] = Wr_half[:, kb:ke] - hatWr_half[:, kb:ke]

        # Free the large CUDA-loop temporaries before the final fp32
        # conversion. For a 12288x28672 weight (123B Devstral-2 down_proj),
        # L_half, Wr_half, R_half and the per-block buffers together hold
        # ~4-5 GB that's no longer needed once the loop exits; freeing it
        # avoids OOM on the hatWr_half.float() allocation (~1.4 GB fresh).
        del L_half, Wr_half, R_half
        del target_buf, resid_buf, dec1_buf, dec2_buf, idx1_buf, idx2_buf
        del resid_scale_half, inv_resid_scale_half
        # L, D, H_reg are only needed by the tune_iters refinement pass.
        if tune_iters == 0:
            del L, D, H_reg
        torch.cuda.empty_cache()

        hatWr = hatWr_half.float()
        del hatWr_half
        torch.cuda.empty_cache()
    else:
        hatWr = torch.zeros_like(Wr)
        R = Wr.clone()
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            if ke < n:
                feedback = R[:, ke:] @ L[ke:, kb:ke]
            else:
                feedback = 0.0
            target = Wr[:, kb:ke] + feedback

            dec1, idx1 = codebook1.quantize(target)
            all_indices1[:, k] = idx1

            residual = target - dec1
            dec2, idx2 = codebook2.quantize(residual * resid_scale)
            all_indices2[:, k] = idx2

            hatWr[:, kb:ke] = dec1 + dec2 / resid_scale
            R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]

    for _ in range(tune_iters):
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            residual_full = Wr - hatWr
            H_col = H_reg[:, kb:ke]
            H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
            feedback = residual_full @ H_col @ H_diag_inv
            target = hatWr[:, kb:ke] + feedback

            dec1, idx1 = codebook1.quantize(target)
            all_indices1[:, k] = idx1

            residual = target - dec1
            dec2, idx2 = codebook2.quantize(residual * resid_scale)
            all_indices2[:, k] = idx2

            hatWr[:, kb:ke] = dec1 + dec2 / resid_scale

    W_hat = hatWr * Wscale

    quant_mse = ((W - W_hat) ** 2).mean().item()
    diff = W - W_hat
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()

    cb2_size = codebook2.codebook.shape[0]
    bits2 = math.ceil(math.log2(cb2_size)) if cb2_size > 1 else 0
    bpw = (16 + bits2) / b + 32.0 / (m * n)

    return {
        'W_hat': W_hat,
        'indices1': all_indices1,
        'indices2': all_indices2,
        'Wscale': Wscale,
        'resid_scale': resid_scale,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }


def quantize_ldlq_codebook_nstage(
    W: torch.Tensor,
    H: torch.Tensor,
    codebooks: list,
    resid_scales: list[float],
    tune_iters: int = 0,
    Wscale: Optional[float] = None,
):
    """
    N-stage LDLQ: chain of codebooks on successive residuals.

    For 5bpw: [E8_full, E8_full, E8_small_256] with 2 resid_scales → 16+16+8 = 40 bits / 8 dims.
    For 6bpw: [E8_full, E8_full, E8_full] with 2 resid_scales → 16+16+16 = 48 bits / 8 dims.
    For 8bpw: [E8_full]*4 with 3 resid_scales → 64 bits / 8 dims.

    Args:
        W: (m, n) weight matrix. n must be divisible by codebooks[0].codesz.
        H: (n, n) Hessian.
        codebooks: list of codebooks [cb1, cb2, ..., cbN].
        resid_scales: list of N-1 scale factors [rs1, rs2, ..., rs_{N-1}].
        tune_iters: extra refinement passes.
        Wscale: global scale. If None, computed from rms(W)/opt_scale.
    """
    n_stages = len(codebooks)
    assert len(resid_scales) == n_stages - 1, \
        f"Need {n_stages - 1} resid_scales for {n_stages} stages, got {len(resid_scales)}"

    device = codebooks[0].device
    m, n = W.shape
    b = codebooks[0].codesz
    assert n % b == 0
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    damp = 0.01 * torch.diag(H).mean()
    H_reg = H + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    if Wscale is None:
        W_rms = W.pow(2).mean().sqrt().item()
        Wscale = W_rms / codebooks[0].opt_scale if W_rms > 1e-10 else 1.0

    Wr = W / Wscale

    # Index storage: one (m, num_blocks) tensor per stage
    all_indices = [
        torch.zeros(m, num_blocks, dtype=torch.long, device=device)
        for _ in range(n_stages)
    ]

    # Precompute cumulative inverse resid_scales for reconstruction:
    # stage 1: weight 1.0
    # stage 2: weight 1/rs1
    # stage 3: weight 1/(rs1*rs2)
    # ...
    cum_inv_rs = [1.0]
    for rs in resid_scales:
        cum_inv_rs.append(cum_inv_rs[-1] / rs)

    use_cuda = torch.device(device).type == 'cuda'

    if use_cuda:
        L_half = L.half()
        Wr_half = Wr.half()
        R_half = Wr_half.clone()
        hatWr_half = torch.zeros_like(Wr_half)

        target_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        resid_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        dec_buf = torch.empty(m, b, dtype=torch.float16, device=device)
        idx_buf = torch.empty(m, dtype=torch.int64, device=device)

        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            if ke < n:
                torch.mm(R_half[:, ke:], L_half[ke:, kb:ke], out=target_buf)
                target_buf.add_(Wr_half[:, kb:ke])
            else:
                target_buf.copy_(Wr_half[:, kb:ke])

            # Accumulate reconstruction in fp16
            recon = torch.zeros(m, b, dtype=torch.float16, device=device)

            for s in range(n_stages):
                if s == 0:
                    # Stage 1: quantize target directly
                    codebooks[s].quantize_fast(target_buf,
                                               decoded_out=dec_buf, idx_out=idx_buf)
                    all_indices[s][:, k] = idx_buf
                    recon.add_(dec_buf)
                    # resid_buf = (target - dec) * resid_scale
                    torch.sub(target_buf, dec_buf, out=resid_buf)
                    if s < n_stages - 1:
                        resid_buf.mul_(resid_scales[s])
                else:
                    # Stage s: quantize scaled residual
                    codebooks[s].quantize_fast(resid_buf,
                                               decoded_out=dec_buf, idx_out=idx_buf)
                    all_indices[s][:, k] = idx_buf
                    # Reconstruction weight = 1 / (rs1 * rs2 * ... * rs_s)
                    recon.add_(dec_buf, alpha=cum_inv_rs[s])
                    # Compute next residual
                    if s < n_stages - 1:
                        torch.sub(resid_buf, dec_buf, out=resid_buf)
                        resid_buf.mul_(resid_scales[s])

            hatWr_half[:, kb:ke] = recon
            R_half[:, kb:ke] = Wr_half[:, kb:ke] - hatWr_half[:, kb:ke]

        del L_half, Wr_half, R_half, target_buf, resid_buf, dec_buf, idx_buf
        if tune_iters == 0:
            del L, D, H_reg
        torch.cuda.empty_cache()

        hatWr = hatWr_half.float()
        del hatWr_half
        torch.cuda.empty_cache()
    else:
        hatWr = torch.zeros_like(Wr)
        R = Wr.clone()

        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            if ke < n:
                feedback = R[:, ke:] @ L[ke:, kb:ke]
            else:
                feedback = 0.0
            target = Wr[:, kb:ke] + feedback

            recon = torch.zeros(m, b, dtype=W.dtype, device=device)
            residual = target.clone()

            for s in range(n_stages):
                if s == 0:
                    dec, idx = codebooks[s].quantize(target)
                    all_indices[s][:, k] = idx
                    recon += dec
                    residual = (target - dec) * resid_scales[s] if s < n_stages - 1 else None
                else:
                    dec, idx = codebooks[s].quantize(residual)
                    all_indices[s][:, k] = idx
                    recon += dec * cum_inv_rs[s]
                    if s < n_stages - 1:
                        residual = (residual - dec) * resid_scales[s]

            hatWr[:, kb:ke] = recon
            R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]

    W_hat = hatWr * Wscale

    quant_mse = ((W - W_hat) ** 2).mean().item()
    diff = W - W_hat
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()

    # Compute effective bpw
    total_bits = sum(
        math.ceil(math.log2(cb.codebook.shape[0])) if cb.codebook.shape[0] > 1 else 0
        for cb in codebooks
    )
    bpw = total_bits / b + 32.0 / (m * n)

    return {
        'W_hat': W_hat,
        'all_indices': all_indices,  # list of N tensors
        'Wscale': Wscale,
        'resid_scales': resid_scales,
        'cum_inv_rs': cum_inv_rs,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }
