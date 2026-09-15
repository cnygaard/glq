"""Diagnostic: long-context + bf16 (mirrors MMLU prod shapes).

Phase 2.2 fixture is fp16 + Tk=74. Production uses bf16 + Tk up to ~3000.
This test exercises the gap to surface bf16/long-context-specific bugs.

Test matrix:
- dtype ∈ {fp16, bf16}
- (head_size, sliding_window) ∈ {(256, 512), (512, 0)}  # Gemma-4 sliding + full layer shapes
- num_q=8, num_kv=2 (Gemma-4 GQA)
- Tk ∈ {512, 1024, 2048}  # exercises sliding-window mask + long context
"""
import math
import sys
from itertools import product

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


def _quantizer():
    from glq.codebook import E8ShellCodebook
    from glq.kv_e8 import E8KVQuantizer
    cb = E8ShellCodebook(verbose=False)
    for n in ("codebook", "codebook_norms", "codebook_half",
              "codebook_half_t", "codebook_norms_half", "codebook_packed"):
        setattr(cb, n, getattr(cb, n).to("cuda"))
    cb.device = "cuda"
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


def _pack_paged(qt, Tk, num_kv, n_groups, dtype, num_blocks, block_size):
    """Reshape [Tk*num_kv, n_groups] → paged [nb, bs, num_kv, n_groups]."""
    flat = qt.reshape(Tk, num_kv, n_groups).contiguous()
    paged = torch.zeros(
        num_blocks, block_size, num_kv, n_groups,
        dtype=dtype, device="cuda",
    )
    for t in range(Tk):
        paged[t // block_size, t % block_size] = flat[t]
    return paged


@pytest.mark.parametrize(
    "dtype,head_size,sliding_window,Tk",
    [
        (dt, hs, sw, tk)
        for dt, (hs, sw), tk in product(
            [torch.float16, torch.bfloat16],
            [(256, 512), (512, 0)],
            [512, 1024, 2048],
        )
    ],
    ids=lambda v: str(v).replace("torch.", "").replace(" ", ""),
)
def test_long_context(dtype, head_size, sliding_window, Tk):
    quant = _quantizer()
    from glq_vllm.e8_paged_cache import _get_hadamard
    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v2_1,
    )
    H_mat = _get_hadamard(torch.float32, "cuda")
    codebook = quant.codebook.codebook_half

    num_q = 8
    num_kv = 2
    n_groups = head_size // 8
    block_size = 16
    num_blocks = (Tk + block_size - 1) // block_size
    Tq = Tk

    g = torch.Generator(device="cuda").manual_seed(
        hash((dtype, head_size, sliding_window, Tk)) & 0xFFFF)
    # Use small-magnitude K/V (post-norm range) like the real captured fixture.
    q = torch.randn(Tq, num_q, head_size, device="cuda", dtype=dtype, generator=g) * 0.3
    K = torch.randn(Tk, num_kv, head_size, device="cuda", dtype=dtype, generator=g) * 0.1
    V = torch.randn(Tk, num_kv, head_size, device="cuda", dtype=dtype, generator=g) * 1.0

    pad = num_blocks * block_size - Tk
    if pad > 0:
        K = torch.cat([K, torch.zeros(pad, num_kv, head_size, device="cuda", dtype=dtype)])
        V = torch.cat([V, torch.zeros(pad, num_kv, head_size, device="cuda", dtype=dtype)])

    qk = quant.quantize(K)
    qv = quant.quantize(V)

    k_i1 = _pack_paged(qk["idx1"], num_blocks * block_size, num_kv, n_groups, torch.int16, num_blocks, block_size)
    k_i2 = _pack_paged(qk["idx2"], num_blocks * block_size, num_kv, n_groups, torch.int16, num_blocks, block_size)
    v_i1 = _pack_paged(qv["idx1"], num_blocks * block_size, num_kv, n_groups, torch.int16, num_blocks, block_size)
    v_i2 = _pack_paged(qv["idx2"], num_blocks * block_size, num_kv, n_groups, torch.int16, num_blocks, block_size)
    k_sc = _pack_paged(qk["scale"].to(dtype), num_blocks * block_size, num_kv, n_groups, dtype, num_blocks, block_size)
    v_sc = _pack_paged(qv["scale"].to(dtype), num_blocks * block_size, num_kv, n_groups, dtype, num_blocks, block_size)

    rs = float(quant.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    # Reference: slice qk/qv back to Tk and run reference_e8_attention.
    qk_ref = {
        "idx1": qk["idx1"][:Tk * num_kv],
        "idx2": qk["idx2"][:Tk * num_kv],
        "scale": qk["scale"][:Tk * num_kv],
        "shape": (Tk, num_kv, head_size), "dtype": qk["dtype"],
    }
    qv_ref = {
        "idx1": qv["idx1"][:Tk * num_kv],
        "idx2": qv["idx2"][:Tk * num_kv],
        "scale": qv["scale"][:Tk * num_kv],
        "shape": (Tk, num_kv, head_size), "dtype": qv["dtype"],
    }
    sys.path.insert(0, "/opt/dlami/nvme/work/glq_repo")
    from tests.test_e8_attention_reference import reference_e8_attention
    out_ref = reference_e8_attention(
        q[:Tq], qk_ref, qv_ref, quant,
        causal=True, softmax_scale=softmax_scale,
        sliding_window=(sliding_window if sliding_window > 0 else None),
    )

    out = torch.zeros_like(q[:Tq])
    cu = torch.tensor([0, Tq], device="cuda", dtype=torch.int32)
    su = torch.tensor([Tk], device="cuda", dtype=torch.int32)
    bt = torch.arange(num_blocks, device="cuda", dtype=torch.int32).reshape(1, num_blocks)
    unified_attention_e8_v2_1(
        q=q[:Tq],
        k_idx1=k_i1, k_idx2=k_i2, k_scale=k_sc,
        v_idx1=v_i1, v_idx2=v_i2, v_scale=v_sc,
        codebook=codebook, H_mat=H_mat,
        out=out, cu_seqlens_q=cu, seqused_k=su,
        softmax_scale=softmax_scale, resid_scale=rs,
        block_table=bt, sliding_window=sliding_window,
    )
    mx = (out_ref.float() - out.float()).abs().max().item()
    rmse = (out_ref.float() - out.float()).pow(2).mean().sqrt().item()
    print(f"  dt={str(dtype).split('.')[-1]:8s} H={head_size} SW={sliding_window:>3} Tk={Tk:>4}: max-abs={mx:.2e} RMSE={rmse:.2e}")
    # Looser tolerance for bf16 (3 fewer mantissa bits) + long context
    tol = 5e-3 if dtype == torch.float16 else 5e-2
    assert mx < tol, f"max-abs={mx:.2e} >= {tol} (RMSE={rmse:.2e})"
