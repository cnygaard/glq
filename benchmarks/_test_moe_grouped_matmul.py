"""Parity test for the GLQ grouped-GEMM MoE matmul (Stage 3 increment 2).

Compares glq_moe_grouped_matmul (gather + per-expert tensor-core batched GEMM,
m_indices-routed) against a kernel-independent PyTorch reference: dequantize each
expert's weight in fp32 (codebook gather + RVQ residual), then for each real
grouped slot do x[token] @ W[expert]^T. Covers NS=1 (single codebook) and NS=2
(residual + smem-staged cb2). fp16 MMA vs fp32 ref -> tolerance, not bit-exact.
"""
import numpy as np
import torch

import glq.inference_kernel as ik

TILE = 16
DEV = "cuda"


def dequant_W(q, cb, q2, cb2, inv_rs, wscale):
    """q (M,NB) int16 -> W (M, NB*8) fp32."""
    M, NB = q.shape
    W = cb[q.long()].float()                       # (M, NB, 8)
    if q2 is not None:
        W = W + inv_rs * cb2[q2.long()].float()
    return (W.reshape(M, NB * 8) * wscale)


def run(num_stages):
    torch.manual_seed(num_stages)
    E, M, NB, nt, top_k = 4, 64, 8, 8, 2
    N = NB * 8
    K = 256
    cb = (torch.randn(K, 8, device=DEV) * 0.1).half()
    q = torch.randint(0, K, (E, M, NB), dtype=torch.int16, device=DEV)
    ws = (torch.rand(E, device=DEV) + 0.5).float()
    x = (torch.randn(nt, N, device=DEV) * 0.5).half()
    topk = torch.randint(0, E, (nt, top_k), dtype=torch.int64, device=DEV)

    ei16 = torch.empty(0, dtype=torch.int16, device=DEV)
    eh = torch.empty(0, dtype=torch.float16, device=DEV)
    ef = torch.empty(0, dtype=torch.float32, device=DEV)
    if num_stages >= 2:
        cb2 = (torch.randn(K, 8, device=DEV) * 0.05).half()
        q2 = torch.randint(0, K, (E, M, NB), dtype=torch.int16, device=DEV)
        irs = (torch.rand(E, device=DEV) * 0.3 + 0.1).float()
    else:
        cb2, q2, irs = eh, ei16, ef

    off, mi, stk = ik._glq_cuda.glq_moe_build_grouping(topk, E, top_k, TILE)
    y = ik._glq_cuda.glq_moe_grouped_matmul(
        x, stk.int(), mi.int(), top_k, q, cb, q2, cb2, ws, irs, ei16, eh, ef, num_stages)
    y = y.cpu().float().numpy()
    mi_c, stk_c, off_c = mi.cpu().numpy(), stk.cpu().numpy(), off.cpu().numpy()

    xf = x.cpu().float()
    Wf = [dequant_W(q[e], cb, (q2[e] if num_stages >= 2 else None), cb2,
                    (float(irs[e]) if num_stages >= 2 else 0.0), float(ws[e])).cpu().numpy()
          for e in range(E)]

    M_sum = int(off_c[E])
    maxerr, nchk = 0.0, 0
    for slot in range(M_sum):
        e, r = int(mi_c[slot]), int(stk_c[slot])
        if r < 0:
            continue
        ref = xf[r // top_k].numpy() @ Wf[e].T
        maxerr = max(maxerr, float(np.abs(y[slot] - ref).max()))
        nchk += 1
    print("NS=%d: checked %d rows, max abs err = %.4e" % (num_stages, nchk, maxerr), flush=True)
    assert nchk > 0, "no real rows checked"
    assert maxerr < 5e-2, "NS=%d err %.4e too large" % (num_stages, maxerr)


def main():
    assert ik._try_load_cuda_ext(), "glq CUDA ext failed to build/load"
    assert hasattr(ik._glq_cuda, "glq_moe_grouped_matmul"), "grouped_matmul not in ext"
    for ns in [1, 2]:
        run(ns)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
