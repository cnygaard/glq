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


def block_LDL_batched(H: torch.Tensor, block_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched block LDL: ``block_LDL`` over a leading expert dim ``E``.

    Identical math to :func:`block_LDL` with an ``E`` dim prepended to every
    tensor (``torch.mm``/diagonal/inv/einsum all carry the batch). The one
    difference is failure handling: ``torch.linalg.cholesky`` aborts the whole
    batch if *any* matrix is non-PD, so this uses ``cholesky_ex`` and re-damps
    only the failing experts (0.1·mean-diag, then identity) — the per-expert
    analogue of the single path's 3-tier ``_LinAlgError`` fallback in
    ``quantize_layer_e8_shell_rht``. The caller is expected to have already
    applied the base 0.01·mean-diag damp (as the single N-stage path does).

    Args:
        H: (E, n, n) symmetric matrices, each ~PD after the caller's base damp.
        block_size: b, must divide n.

    Returns:
        (L, D): L is (E, n, n) block unit-lower-triangular, D is (E, n//b, b, b).
    """
    assert H.dim() == 3, f"expected (E, n, n), got {tuple(H.shape)}"
    E, n, _ = H.shape
    b = block_size
    assert n % b == 0, f"n={n} must be divisible by block_size={b}"
    m = n // b

    eye = torch.eye(n, device=H.device, dtype=H.dtype)
    Hc = H.clone()
    # Progressive masked damping: factor as-is, then add 0.1·mean-diag to the
    # failing experts, then fall back to identity for any that still fail.
    L_chol, info = torch.linalg.cholesky_ex(Hc)
    for tier in range(2):
        bad = info > 0
        if not bad.any():
            break
        if tier == 0:
            extra = 0.1 * torch.diagonal(Hc[bad], dim1=-2, dim2=-1).mean(dim=-1)
            Hc[bad] = Hc[bad] + extra[:, None, None] * eye
        else:
            Hc[bad] = eye                       # last resort: identity (always PD)
        L_chol, info = torch.linalg.cholesky_ex(Hc)

    DL = torch.diagonal(L_chol.reshape(E, m, b, m, b), dim1=1, dim2=3).permute(0, 3, 1, 2)
    D = DL @ DL.transpose(-1, -2)
    DL_inv = torch.linalg.inv(DL)
    # Per block-column i and per expert e: right-multiply L_chol's block column
    # by DL_inv[e, i] so the diagonal block becomes identity.
    L = torch.einsum('enib,eibj->enij', L_chol.reshape(E, n, m, b), DL_inv).reshape(E, n, n)

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
        Wscale = W_rms * codebook.opt_scale if W_rms > 1e-10 else 1.0

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
        Wscale = W_rms * codebook1.opt_scale if W_rms > 1e-10 else 1.0

    # ``resid_scale`` must be tuned for the actual (codebook1, codebook2)
    # pair when they differ. Callers should pass the output of
    # ``codebook1.compute_paired_resid_scale(codebook2)``. For canonical
    # recipes that share a codebook (4/6/8 bpw uniform), this is just
    # ``codebook1.resid_scale``.
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

    if tune_iters > 0:
        # Reusable fp16 buffers for the fused-NN+decode path. Same shapes
        # as the main loop's per-block buffers, allocated once outside the
        # iter/block loops.
        if use_cuda:
            tune_target = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_resid = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_dec1 = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_dec2 = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_idx1 = torch.empty(m, dtype=torch.int64, device=device)
            tune_idx2 = torch.empty(m, dtype=torch.int64, device=device)
            resid_scale_h = torch.tensor(resid_scale, dtype=torch.float16, device=device)
            inv_rs_h = torch.tensor(1.0 / resid_scale, dtype=torch.float16, device=device)

        for _ in range(tune_iters):
            for k in reversed(range(num_blocks)):
                kb, ke = k * b, (k + 1) * b
                residual_full = Wr - hatWr
                H_col = H_reg[:, kb:ke]
                H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
                feedback = residual_full @ H_col @ H_diag_inv
                target = hatWr[:, kb:ke] + feedback

                if use_cuda:
                    tune_target.copy_(target)  # fp32 → fp16 cast
                    codebook1.quantize_fast(tune_target,
                                            decoded_out=tune_dec1, idx_out=tune_idx1)
                    all_indices1[:, k] = tune_idx1

                    torch.sub(tune_target, tune_dec1, out=tune_resid)
                    tune_resid.mul_(resid_scale_h)
                    codebook2.quantize_fast(tune_resid,
                                            decoded_out=tune_dec2, idx_out=tune_idx2)
                    all_indices2[:, k] = tune_idx2

                    hatWr[:, kb:ke] = (tune_dec1.float()
                                       + tune_dec2.float() / resid_scale)
                else:
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
        Wscale = W_rms * codebooks[0].opt_scale if W_rms > 1e-10 else 1.0

    # ``resid_scales[k]`` must be tuned for the actual
    # (codebooks[k], codebooks[k+1]) pair. Callers should pass
    # ``codebooks[k].compute_paired_resid_scale(codebooks[k+1])`` per
    # transition. The saved cumulative inverse scales feed inference
    # untouched, so the only thing that changes is which optimum is
    # picked at quant time.
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

    # N-stage tune_iters: refine the allocation by re-running the per-block
    # quantization with Hessian-weighted feedback computed from the current
    # reconstruction. Mirrors the 2-stage tune loop. ``quantize_fast`` keeps
    # this cheap on CUDA; the CPU fallback uses the slower ``quantize``.
    if tune_iters > 0:
        if use_cuda:
            tune_target = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_resid = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_dec = torch.empty(m, b, dtype=torch.float16, device=device)
            tune_idx = torch.empty(m, dtype=torch.int64, device=device)
            resid_scales_h = [
                torch.tensor(rs, dtype=torch.float16, device=device)
                for rs in resid_scales
            ]

        for _ in range(tune_iters):
            for k in reversed(range(num_blocks)):
                kb, ke = k * b, (k + 1) * b
                residual_full = Wr - hatWr
                H_col = H_reg[:, kb:ke]
                H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
                feedback = residual_full @ H_col @ H_diag_inv
                target = hatWr[:, kb:ke] + feedback

                if use_cuda:
                    tune_target.copy_(target)  # fp32 → fp16 cast
                    recon = torch.zeros(m, b, dtype=torch.float32, device=device)
                    for s in range(n_stages):
                        src = tune_target if s == 0 else tune_resid
                        codebooks[s].quantize_fast(src,
                                                   decoded_out=tune_dec,
                                                   idx_out=tune_idx)
                        all_indices[s][:, k] = tune_idx
                        recon.add_(tune_dec.float(), alpha=cum_inv_rs[s])
                        if s < n_stages - 1:
                            torch.sub(src, tune_dec, out=tune_resid)
                            tune_resid.mul_(resid_scales_h[s])
                    hatWr[:, kb:ke] = recon
                else:
                    recon = torch.zeros(m, b, dtype=W.dtype, device=device)
                    residual = target.clone()
                    for s in range(n_stages):
                        src = target if s == 0 else residual
                        dec, idx = codebooks[s].quantize(src)
                        all_indices[s][:, k] = idx
                        recon += dec * cum_inv_rs[s]
                        if s < n_stages - 1:
                            residual = (src - dec) * resid_scales[s]
                    hatWr[:, kb:ke] = recon

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


def quantize_ldlq_codebook_nstage_batched(
    W: torch.Tensor,
    H: torch.Tensor,
    codebooks: list,
    resid_scales: list[float],
    tune_iters: int = 0,
    Wscale: Optional[torch.Tensor] = None,
):
    """Batched N-stage LDLQ: :func:`quantize_ldlq_codebook_nstage` over a leading
    expert dim ``E``.

    Quantizes E independent, identically shaped weight matrices in one stacked
    pass. Each expert's LDLQ feedback stays inside its own (m, n) slice, so this
    is mathematically per-expert-identical to looping the single-expert function
    — but the heavy ops run batched (one GPU-filling kernel instead of E tiny
    ones), which is the point: the per-expert MoE loop is launch-overhead-bound.

    The feedback matmul becomes ``torch.bmm`` and the per-block codebook NN runs
    on the flattened ``(E*m, b)`` view (the NN is per-row, so flattening the
    expert and row axes is exact). Determinism is preserved: the block sweep is
    the same fixed reversed order with no atomics.

    Args:
        W: (E, m, n) weight matrices. n divisible by codebooks[0].codesz.
        H: (E, n, n) Hessians.
        codebooks: list of codebooks [cb1, ..., cbN] (shared across experts).
        resid_scales: list of N-1 scale factors.
        tune_iters: must be 0 (batched refinement not implemented; e8p RVQ uses 0).
        Wscale: optional (E,) global scales. If None, computed per-expert from rms.

    Returns:
        dict with W_hat (E,m,n), all_indices (list of N (E,m,num_blocks) longs),
        Wscale (E,), resid_scales, cum_inv_rs, bpw, quant_mse (E,), proxy_loss (E,).
    """
    assert tune_iters == 0, "batched N-stage LDLQ does not support tune_iters>0"
    n_stages = len(codebooks)
    assert len(resid_scales) == n_stages - 1, \
        f"Need {n_stages - 1} resid_scales for {n_stages} stages, got {len(resid_scales)}"

    device = codebooks[0].device
    assert W.dim() == 3 and H.dim() == 3, "W,H must be (E,m,n)/(E,n,n)"
    E, m, n = W.shape
    b = codebooks[0].codesz
    assert n % b == 0
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    # Per-expert diagonal damping (0.01·mean-diag), then batched block-LDL with
    # the per-expert PD retry living inside block_LDL_batched.
    damp = 0.01 * torch.diagonal(H, dim1=-2, dim2=-1).mean(dim=-1)        # (E,)
    eye = torch.eye(n, device=device)
    H_reg = H + damp[:, None, None] * eye
    L, D = block_LDL_batched(H_reg, block_size=b)

    if Wscale is None:
        W_rms = W.pow(2).mean(dim=(-2, -1)).sqrt()                        # (E,)
        opt_scale = codebooks[0].opt_scale
        Wscale = torch.where(W_rms > 1e-10, W_rms * opt_scale,
                             torch.ones_like(W_rms))                      # (E,)
    Wscale_t = Wscale.to(device).view(E, 1, 1)
    Wr = W / Wscale_t                                                     # (E, m, n)

    # Index storage: one (E, m, num_blocks) tensor per stage.
    all_indices = [
        torch.zeros(E, m, num_blocks, dtype=torch.long, device=device)
        for _ in range(n_stages)
    ]

    # Cumulative inverse resid_scales for reconstruction (stage k weight 1/Πrs).
    cum_inv_rs = [1.0]
    for rs in resid_scales:
        cum_inv_rs.append(cum_inv_rs[-1] / rs)

    # fp16 working precision on CUDA (matches the single CUDA path); fp32 on CPU
    # (the bit-exact parity reference).
    work_dtype = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
    L_w = L.to(work_dtype)
    Wr_w = Wr.to(work_dtype)
    R_w = Wr_w.clone()
    hatWr_w = torch.zeros_like(Wr_w)

    # Per-block reusable buffers; flat views feed the per-row codebook NN
    # (quantize_fast needs a 2D (R, b) input, which (E*m, b) is).
    target_buf = torch.empty(E, m, b, dtype=work_dtype, device=device)
    resid_buf = torch.empty(E, m, b, dtype=work_dtype, device=device)
    dec_buf = torch.empty(E, m, b, dtype=work_dtype, device=device)
    idx_buf = torch.empty(E, m, dtype=torch.int64, device=device)
    target_flat = target_buf.view(E * m, b)
    resid_flat = resid_buf.view(E * m, b)
    dec_flat = dec_buf.view(E * m, b)
    idx_flat = idx_buf.view(E * m)

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b
        if ke < n:
            torch.bmm(R_w[:, :, ke:], L_w[:, ke:, kb:ke], out=target_buf)
            target_buf.add_(Wr_w[:, :, kb:ke])
        else:
            target_buf.copy_(Wr_w[:, :, kb:ke])

        recon = torch.zeros(E, m, b, dtype=work_dtype, device=device)

        for s in range(n_stages):
            if s == 0:
                codebooks[s].quantize_fast(target_flat,
                                           decoded_out=dec_flat, idx_out=idx_flat)
                all_indices[s][:, :, k] = idx_buf
                recon.add_(dec_buf)
                torch.sub(target_buf, dec_buf, out=resid_buf)
                if s < n_stages - 1:
                    resid_buf.mul_(resid_scales[s])
            else:
                codebooks[s].quantize_fast(resid_flat,
                                           decoded_out=dec_flat, idx_out=idx_flat)
                all_indices[s][:, :, k] = idx_buf
                recon.add_(dec_buf, alpha=cum_inv_rs[s])
                if s < n_stages - 1:
                    torch.sub(resid_buf, dec_buf, out=resid_buf)
                    resid_buf.mul_(resid_scales[s])

        hatWr_w[:, :, kb:ke] = recon
        R_w[:, :, kb:ke] = Wr_w[:, :, kb:ke] - hatWr_w[:, :, kb:ke]

    hatWr = hatWr_w.float()
    W_hat = hatWr * Wscale_t

    diff = W - W_hat
    quant_mse = (diff ** 2).mean(dim=(-2, -1))                           # (E,)
    # proxy_loss per expert = mean_i diag(diff @ H @ diff^T)_i, via bmm to avoid
    # the huge (E,m,n,n) einsum intermediate.
    diffH = torch.bmm(diff, H)                                           # (E, m, n)
    proxy_loss = (diffH * diff).sum(dim=-1).mean(dim=-1)                 # (E,)

    total_bits = sum(
        math.ceil(math.log2(cb.codebook.shape[0])) if cb.codebook.shape[0] > 1 else 0
        for cb in codebooks
    )
    bpw = total_bits / b + 32.0 / (m * n)

    return {
        'W_hat': W_hat,
        'all_indices': all_indices,  # list of N (E, m, num_blocks) tensors
        'Wscale': Wscale.to(device),
        'resid_scales': resid_scales,
        'cum_inv_rs': cum_inv_rs,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }
