"""Fused CPU MoE decode — gated against a per-expert oracle.

The oracle deliberately reuses the *shipped* dense CPU bracket
(`glq_fused_linear_trellis_3inst_cpu`), because the decode, the block-diagonal FHT and the
RVQ-2 composition are already gated bit-exactly elsewhere (tests/test_trellis_3inst_cpu_
kernel.py). What is new here — and all these tests are about — is the MoE wrapper:
grouping routings by expert, indexing per-expert weights, the gated activation over
[gate|up], and the weighted reduce.

Semantics mirror the CUDA op (`glq_fused_moe_trellis_3inst_cuda`, glq_cuda.cu:5042):

  * `SU` and `Wscale` are **per expert**; `SV` and the block-diag metas are **shared**.
  * `w13` holds gate and up concatenated — **gate is the first half** of `w13_out`
    (glq_cuda.cu:4543-4559), i.e. `h = act(y[:, :inter]) * y[:, inter:]`.
  * activation ids: 0 silu, 1 gelu-tanh, 2 relu-squared — gated only.
  * the reduce is `out[t] = sum_k topk_weights[t,k] * expert_out(t,k)` in **fixed k order**,
    which is what makes the GPU path deterministic and must hold here too.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import trellis as gt  # noqa: E402
from glq.inference_kernel_cpu import _try_load_cpu_ext  # noqa: E402
from glq.hadamard import _block_decompose  # noqa: E402


def _ext():
    if not _try_load_cpu_ext():
        pytest.skip("CPU extension not available")
    from glq import inference_kernel_cpu as ikc
    return ikc._glq_cpu


def _pack(K, m, n, seed):
    """One expert's projection, quantized on CPU in KERNEL layout."""
    torch.manual_seed(seed)
    cb = gt.TrellisCodebook(variant="3inst", K=K, device="cpu")
    W = (torch.randn(m, n) * 0.05).float()
    _, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=True)
    return gt.pack_layer(cb, Qidxs, m, n, has_kernel=True)


def _meta(dim):
    """Block-diagonal RHT metadata, the (nblocks, 4) int32 form the CPU FHT reads."""
    blocks = _block_decompose(dim)
    rows, off = [], 0
    for bs in blocks:
        rows.append([off, bs, int(bs).bit_length() - 1, 0])
        off += bs
    return torch.tensor(rows, dtype=torch.int32)


class MoEWeights:
    """A tiny synthetic MoE layer: E experts, gated w13 -> w2, all CPU."""

    def __init__(self, E=3, hidden=64, inter=64, K=4, seed=0):
        self.E, self.hidden, self.inter = E, hidden, inter
        self.w13_out = 2 * inter
        self.w13_packed = torch.stack(
            [_pack(K, self.w13_out, hidden, seed + e) for e in range(E)])
        self.w2_packed = torch.stack(
            [_pack(K, hidden, inter, seed + 100 + e) for e in range(E)])
        g = torch.Generator().manual_seed(seed + 7)
        sign = lambda n: torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).half()
        self.w13_SU = torch.stack([sign(self.w13_out) for _ in range(E)])
        self.w2_SU = torch.stack([sign(hidden) for _ in range(E)])
        self.w13_SV = sign(hidden)          # shared across experts, as on the GPU
        self.w2_SV = sign(inter)
        self.w13_Wscale = torch.rand(E, generator=g).float() + 0.5
        self.w2_Wscale = torch.rand(E, generator=g).float() + 0.5
        self.meta_n_w13, self.meta_m_w13 = _meta(hidden), _meta(self.w13_out)
        self.meta_n_w2, self.meta_m_w2 = _meta(inter), _meta(hidden)


def _activate(y, inter, activation):
    gate, up = y[:, :inter], y[:, inter:]
    if activation == 0:
        gate = torch.nn.functional.silu(gate.float())
    elif activation == 1:
        gate = torch.nn.functional.gelu(gate.float(), approximate="tanh")
    else:
        gate = torch.nn.functional.relu(gate.float()) ** 2
    return (gate * up.float()).float()


#: Rows per fused decode-GEMM call. Above it the kernel — like the dense op — decompresses
#: once and uses a GEMM, because the SIMD tiers hold a fixed 8-deep per-row accumulator.
#: That changes the accumulation order, so bit-exactness is only claimed against an oracle
#: batched the same way (see moe_oracle(grouped=...)).
BATCH_MAX = int(os.environ.get("GLQ_TRELLIS_CPU_BATCH_MAX", "8"))


def moe_oracle_grouped(w: MoEWeights, x, topk_ids, topk_weights, activation=1):
    """Reference that batches exactly as the kernel does: one dense-bracket call per
    expert over all of that expert's routings, then the same fixed-k-order reduce. This
    isolates the MoE wrapper (grouping, indexing, activation, reduce) from the kernel's
    batch-mode boundary."""
    ext = _ext()
    T, topk = topk_ids.shape
    scratch = torch.zeros(T * topk, w.hidden, dtype=torch.float32)
    for e in range(w.E):
        rs = [r for r in range(T * topk) if int(topk_ids.flatten()[r]) == e]
        if not rs:
            continue
        rows = x[[r // topk for r in rs]].float()
        y = ext.glq_fused_linear_trellis_3inst_cpu(
            rows, w.w13_SV, w.w13_SU[e], w.w13_packed[e], w.meta_n_w13, w.meta_m_w13,
            float(w.w13_Wscale[e]), w.hidden, w.w13_out, w.hidden, w.w13_out)
        h = _activate(y, w.inter, activation)
        z = ext.glq_fused_linear_trellis_3inst_cpu(
            h, w.w2_SV, w.w2_SU[e], w.w2_packed[e], w.meta_n_w2, w.meta_m_w2,
            float(w.w2_Wscale[e]), w.inter, w.hidden, w.inter, w.hidden)
        for i, r in enumerate(rs):
            scratch[r] = z[i]
    return (scratch.view(T, topk, w.hidden) * topk_weights.unsqueeze(-1)).sum(dim=1)


def moe_oracle(w: MoEWeights, x, topk_ids, topk_weights, activation=1):
    """Per-expert, per-token reference: the shipped dense bracket per projection, then the
    documented activation and fixed-order weighted reduce."""
    ext = _ext()
    T = x.shape[0]
    out = torch.zeros(T, w.hidden, dtype=torch.float32)
    for t in range(T):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            xt = x[t:t + 1].float()
            y = ext.glq_fused_linear_trellis_3inst_cpu(
                xt, w.w13_SV, w.w13_SU[e], w.w13_packed[e],
                w.meta_n_w13, w.meta_m_w13, float(w.w13_Wscale[e]),
                w.hidden, w.w13_out, w.hidden, w.w13_out)
            h = _activate(y, w.inter, activation)
            z = ext.glq_fused_linear_trellis_3inst_cpu(
                h, w.w2_SV, w.w2_SU[e], w.w2_packed[e],
                w.meta_n_w2, w.meta_m_w2, float(w.w2_Wscale[e]),
                w.inter, w.hidden, w.inter, w.hidden)
            out[t] += float(topk_weights[t, k]) * z[0]
    return out


def _route(T, E, topk, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(T)]).long()
    wts = torch.rand(T, topk, generator=g).float()
    return ids, wts / wts.sum(dim=1, keepdim=True)


def _fused(w: MoEWeights, x, ids, wts, activation=1):
    ext = _ext()
    if not hasattr(ext, "glq_fused_moe_trellis_3inst_cpu"):
        pytest.skip("fused CPU MoE op not built yet")
    return ext.glq_fused_moe_trellis_3inst_cpu(
        x.float(), ids, wts, w.w13_packed, w.w13_SU, w.w13_SV, w.w13_Wscale,
        w.w2_packed, w.w2_SU, w.w2_SV, w.w2_Wscale,
        w.hidden, w.inter, w.w13_out,
        w.meta_n_w13, w.meta_m_w13, w.meta_n_w2, w.meta_m_w2, activation)


TIERS = ("scalar", "avx2", "avx512", "avx512fp16")


@pytest.fixture(params=TIERS)
def isa(request):
    """Run the gates on every tier this CPU and build support. The MoE wrapper delegates
    decode to the same vtable the dense path uses, so a tier that decodes wrong here would
    already fail there — but the wrapper's own indexing is tier-independent only if the
    per-expert slicing is right, which is exactly what a tier sweep catches."""
    ext = _ext()
    if not ext.glq_cpu_isa_available(request.param):
        pytest.skip(f"tier {request.param} not available on this CPU/build")
    ext.glq_cpu_set_isa(request.param)
    yield request.param
    ext.glq_cpu_set_isa("auto")


def test_every_available_tier_matches_the_oracle(isa):
    w = MoEWeights(E=3)
    x = torch.randn(5, w.hidden)
    ids, wts = _route(5, w.E, 2, seed=21)
    assert torch.equal(_fused(w, x, ids, wts), moe_oracle(w, x, ids, wts)), isa


# ---- the gate: fused == per-expert oracle, bit for bit ----------------------------------

@pytest.mark.parametrize("T,topk", [(1, 1), (1, 2), (4, 2)])
def test_fused_matches_the_per_expert_oracle(T, topk):
    """Small batches: every expert stays under the fused batch cap, so the kernel and a
    completely separate per-token decomposition must agree BIT for bit."""
    w = MoEWeights(E=3)
    x = torch.randn(T, w.hidden)
    ids, wts = _route(T, w.E, topk)
    assert max(torch.bincount(ids.flatten(), minlength=w.E)).item() <= BATCH_MAX
    got, ref = _fused(w, x, ids, wts), moe_oracle(w, x, ids, wts)
    assert torch.equal(got, ref), (got - ref).abs().max()


@pytest.mark.parametrize("T,topk", [(17, 3), (33, 2)])
def test_fused_matches_the_grouped_oracle_above_the_batch_cap(T, topk):
    """Prefill-sized batches take the decompress-once + GEMM path. Bit-exactness still
    holds — against an oracle batched the same way."""
    w = MoEWeights(E=3)
    x = torch.randn(T, w.hidden)
    ids, wts = _route(T, w.E, topk)
    assert max(torch.bincount(ids.flatten(), minlength=w.E)).item() > BATCH_MAX
    got, ref = _fused(w, x, ids, wts), moe_oracle_grouped(w, x, ids, wts)
    assert torch.equal(got, ref), (got - ref).abs().max()


def test_the_batch_mode_boundary_is_only_a_rounding_difference():
    """Crossing the cap changes accumulation order, not results: the two oracles must
    agree to fp32 rounding. If this ever grows, the batched path has a real bug rather
    than a reassociation difference."""
    w = MoEWeights(E=3)
    x = torch.randn(17, w.hidden)
    ids, wts = _route(17, w.E, 3)
    a, b = moe_oracle(w, x, ids, wts), moe_oracle_grouped(w, x, ids, wts)
    scale = float(b.abs().max())
    assert float((a - b).abs().max()) < 1e-5 * scale, "not a rounding-level difference"


@pytest.mark.parametrize("activation", [0, 1, 2])
def test_every_gated_activation_matches(activation):
    w = MoEWeights(E=2)
    x = torch.randn(3, w.hidden)
    ids, wts = _route(3, w.E, 2, seed=5)
    got = _fused(w, x, ids, wts, activation)
    ref = moe_oracle(w, x, ids, wts, activation)
    assert torch.equal(got, ref)


def test_gate_is_the_first_half_of_w13():
    """If the halves were swapped the output would still look plausible — this pins the
    convention the CUDA kernel uses (glq_cuda.cu:4543-4559)."""
    w = MoEWeights(E=1, inter=64)
    x = torch.randn(2, w.hidden)
    ids, wts = _route(2, 1, 1)
    swapped = moe_oracle(w, x, ids, wts)
    # recompute the oracle with the halves reversed; it must NOT match the kernel
    orig = _activate

    def reversed_act(y, inter, activation):
        return orig(torch.cat([y[:, inter:], y[:, :inter]], dim=1), inter, activation)

    globals()["_activate"] = reversed_act
    try:
        wrong = moe_oracle(w, x, ids, wts)
    finally:
        globals()["_activate"] = orig
    assert not torch.equal(swapped, wrong), "fixture cannot distinguish the halves"
    assert torch.equal(_fused(w, x, ids, wts), swapped)


def test_an_expert_nobody_routes_to_contributes_nothing():
    """Experts absent from topk_ids must be skipped, not decoded into the sum."""
    w = MoEWeights(E=4)
    x = torch.randn(3, w.hidden)
    ids = torch.zeros(3, 1, dtype=torch.long)          # everyone routes to expert 0
    wts = torch.ones(3, 1)
    assert torch.equal(_fused(w, x, ids, wts), moe_oracle(w, x, ids, wts))


def test_the_reduce_order_is_fixed_not_arrival_order():
    """Two routings of the same token must sum in k order, so the result cannot depend on
    which expert finished first — the determinism contract the dense path already holds."""
    w = MoEWeights(E=3)
    x = torch.randn(2, w.hidden)
    ids = torch.tensor([[2, 0], [0, 2]]).long()
    wts = torch.tensor([[0.25, 0.75], [0.75, 0.25]]).float()
    assert torch.equal(_fused(w, x, ids, wts), moe_oracle(w, x, ids, wts))


def test_repeated_runs_and_thread_counts_agree():
    w = MoEWeights(E=3)
    x = torch.randn(9, w.hidden)
    ids, wts = _route(9, w.E, 2, seed=11)
    base = _fused(w, x, ids, wts)
    threads = torch.get_num_threads()
    try:
        for n in (1, 2, 4):
            torch.set_num_threads(n)
            assert torch.equal(_fused(w, x, ids, wts), base), f"differs at {n} threads"
    finally:
        torch.set_num_threads(threads)


def test_single_expert_equals_the_dense_bracket():
    """The MoE op reimplements the bracket orchestration; this pins it against the dense
    op so the two cannot drift."""
    ext = _ext()
    w = MoEWeights(E=1)
    x = torch.randn(1, w.hidden)
    y = ext.glq_fused_linear_trellis_3inst_cpu(
        x.float(), w.w13_SV, w.w13_SU[0], w.w13_packed[0], w.meta_n_w13, w.meta_m_w13,
        float(w.w13_Wscale[0]), w.hidden, w.w13_out, w.hidden, w.w13_out)
    h = _activate(y, w.inter, 1)
    z = ext.glq_fused_linear_trellis_3inst_cpu(
        h, w.w2_SV, w.w2_SU[0], w.w2_packed[0], w.meta_n_w2, w.meta_m_w2,
        float(w.w2_Wscale[0]), w.inter, w.hidden, w.inter, w.hidden)
    ids, wts = torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)
    assert torch.equal(_fused(w, x, ids, wts), z)


def test_the_op_runs_under_inference_mode():
    """Every vLLM forward runs inside ``torch.inference_mode()``, and tensors the op
    allocates there are inference tensors — but ATen's parallel workers do not inherit that
    TLS, so an in-place ATen write from one of them raised "Inplace update to inference
    tensor outside InferenceMode" and took the engine down on the first token. The unit
    tests never saw it because they run in normal mode, where the same tensor is ordinary.

    Routed experts must be at least the thread count, or the op stays on its serial branch
    and the parallel one goes untested."""
    E = max(8, torch.get_num_threads())
    w = MoEWeights(E=E)
    x = torch.randn(E, w.hidden)
    ids = torch.arange(E, dtype=torch.long).unsqueeze(1)      # every expert gets a routing
    wts = torch.ones(E, 1)
    with torch.inference_mode():
        out = _fused(w, x, ids, wts)
    assert out.shape == (E, w.hidden) and torch.isfinite(out).all()
    assert torch.equal(out, moe_oracle(w, x, ids, wts)), \
        "inference mode must not change the numbers, only the tensor flags"


def test_a_shape_the_kernel_cannot_take_raises_instead_of_computing_garbage():
    """The kernel consumes k in 64-wide groups and splits rows as ``m / 32``; the dense
    entry checks both, the MoE entry did not — and hidden=96 (which packs fine, and is a
    perfectly legal MoE width) came back finite, plausible and wrong. A wrong answer that
    looks right is the worst outcome this op has, so it must refuse the shape."""
    w = MoEWeights(E=2, hidden=96, inter=64)
    x = torch.randn(2, w.hidden)
    ids, wts = _route(2, w.E, 2)
    with pytest.raises(RuntimeError, match=r"m % 32|k % 64"):
        _fused(w, x, ids, wts)
