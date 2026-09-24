"""Fused GatedDeltaNet decode on CPU — gated against a transcription of the reference.

36 of Qwen3.8-Flash-Next's 48 layers are `linear_attention`, and on CPU transformers runs
`torch_recurrent_gated_delta_rule`, a readable reference that walks the K x V state five
times per token and allocates four temporaries of that size. The kernel fuses that into two
passes with no allocation. Nothing about the algorithm changes.

The oracle here is a transcription of `modeling_qwen4_exp.py:440-453` rather than an import,
so these tests run anywhere; `test_the_oracle_matches_transformers` pins the transcription
against the real function whenever the installed transformers carries it.

It is NOT bit-exact and cannot be: torch reduces over dim=-2 of a (K, V) tensor in its own
order, the kernel accumulates along i. So every comparison is a tolerance sized from the
measured gap, and two sensitivity tests prove that tolerance is not vacuous.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch", reason="torch not installed")

from glq.inference_kernel_cpu import _try_load_cpu_ext  # noqa: E402

B, H, K, V = 2, 4, 64, 64


def _ext():
    if not _try_load_cpu_ext():
        pytest.skip("CPU extension not available")
    from glq import inference_kernel_cpu as ikc
    if not hasattr(ikc._glq_cpu, "glq_gdn_recurrent_step_cpu"):
        pytest.skip("glq_gdn_recurrent_step_cpu not in this build")
    return ikc._glq_cpu


def oracle_step(S, q, k, v, g, beta):
    """One token of torch_recurrent_gated_delta_rule, transcribed.

    Functional, like the reference: returns a NEW state rather than mutating, which is also
    what makes it a fair check on the kernel's in-place update.
    """
    S = S * g.exp()[..., None, None]
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (S * q.unsqueeze(-1)).sum(dim=-2)
    return out, S


def _inputs(seed=0, b=B, h=H, k=K, v=V):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    S = r(b, h, k, v) * 0.1
    q, kk = r(b, h, k), r(b, h, k)
    # match the caller: q is l2-normalised and scaled by 1/sqrt(K) before the rule
    q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6) / (k ** 0.5)
    kk = torch.nn.functional.normalize(kk, dim=-1, eps=1e-6)
    vv = r(b, h, v)
    decay = -torch.rand(b, h, generator=g) * 0.5          # g is negative in the model
    beta = torch.rand(b, h, generator=g)                  # beta = b.sigmoid() in (0, 1)
    return S, q, kk, vv, decay, beta


#: Sized from the measured agreement on this fixture (~1e-6 absolute against outputs of
#: order 1). The sensitivity tests below pin that it still catches a permuted head mapping
#: and a stale-state read.
TOL = dict(atol=2e-5, rtol=1e-5)

TIERS = ("scalar", "avx2", "avx512", "avx512fp16")


@pytest.fixture(params=TIERS)
def isa(request):
    """Every tier this CPU and build support. The body is plain fp32 FMA compiled once per
    target pragma, so a tier that vectorises wrong shows up here and nowhere else."""
    ext = _ext()
    if not ext.glq_cpu_isa_available(request.param):
        pytest.skip(f"tier {request.param} unavailable on this CPU/build")
    ext.glq_cpu_set_isa(request.param)
    yield request.param
    ext.glq_cpu_set_isa("auto")


# ---- the oracle itself ---------------------------------------------------------------

def test_the_oracle_matches_transformers():
    """Pin the transcription against upstream, so a transformers change that alters the
    rule is caught here rather than showing up as a kernel bug."""
    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            torch_recurrent_gated_delta_rule as ref,
        )
    except Exception:
        pytest.skip("this transformers has no qwen4_exp")

    S, q, k, v, g, beta = _inputs(seed=5)
    # upstream takes (B, S, H, D) and does its own transpose/normalise/scale
    raw_q = torch.nn.functional.normalize(torch.randn(B, 1, H, K), dim=-1) * (K ** 0.5)
    out_ref, state_ref = ref(
        raw_q, k.unsqueeze(1), v.unsqueeze(1), g=g.unsqueeze(1), beta=beta.unsqueeze(1),
        initial_state=S.clone(), output_final_state=True, use_qk_l2norm_in_kernel=True)
    q_eff = torch.nn.functional.normalize(raw_q.transpose(1, 2).squeeze(2), dim=-1,
                                          eps=1e-6) / (K ** 0.5)
    out_mine, state_mine = oracle_step(S.clone(), q_eff, k, v, g, beta)
    assert torch.allclose(out_ref.squeeze(1), out_mine, atol=1e-5, rtol=1e-4)
    assert torch.allclose(state_ref, state_mine, atol=1e-5, rtol=1e-4)


# ---- one step, every tier ------------------------------------------------------------

def test_one_step_matches_the_oracle(isa):
    ext = _ext()
    S, q, k, v, g, beta = _inputs(seed=1)
    want_out, want_state = oracle_step(S.clone(), q, k, v, g, beta)
    got_state = S.clone()
    got_out = ext.glq_gdn_recurrent_step_cpu(got_state, q, k, v, g, beta)
    assert torch.allclose(got_out, want_out, **TOL), \
        (isa, (got_out - want_out).abs().max().item())
    assert torch.allclose(got_state, want_state, **TOL), \
        (isa, (got_state - want_state).abs().max().item())


def test_the_state_is_updated_in_place_and_returned_by_the_caller():
    """The whole point is eliminating the reference's four per-token temporaries, so the
    kernel writes through the caller's tensor. Pin that, because a version that quietly
    copied would be correct and slow, and nothing else here would notice."""
    ext = _ext()
    S, q, k, v, g, beta = _inputs(seed=2)
    before = S.data_ptr()
    _, want_state = oracle_step(S.clone(), q, k, v, g, beta)
    ext.glq_gdn_recurrent_step_cpu(S, q, k, v, g, beta)
    assert S.data_ptr() == before, "state was reallocated instead of updated in place"
    assert torch.allclose(S, want_state, **TOL)


# ---- the recurrence, not just one step -----------------------------------------------

def test_sixty_four_steps_do_not_drift(isa):
    """A single step proves little: the state accumulates, so a small per-step bias
    compounds. 64 steps is a real decode's worth."""
    ext = _ext()
    S0, _, _, _, _, _ = _inputs(seed=3)
    S_ker, S_ref = S0.clone(), S0.clone()
    for t in range(64):
        _, q, k, v, g, beta = _inputs(seed=100 + t)
        want_out, S_ref = oracle_step(S_ref, q, k, v, g, beta)
        got_out = ext.glq_gdn_recurrent_step_cpu(S_ker, q, k, v, g, beta)
        assert torch.allclose(got_out, want_out, atol=1e-4, rtol=1e-4), (isa, t)
    assert torch.allclose(S_ker, S_ref, atol=1e-4, rtol=1e-4), \
        (isa, (S_ker - S_ref).abs().max().item())


# ---- the tolerance is not vacuous ----------------------------------------------------

def test_a_permuted_head_mapping_is_caught():
    """48 value heads against 16 key heads means q/k are repeat_interleaved before the
    rule. Getting that mapping wrong yields finite, plausible, wrong output -- the same
    failure class as the gate/up swap. If the tolerance cannot see a head permutation it
    cannot see that either."""
    ext = _ext()
    S, q, k, v, g, beta = _inputs(seed=4)
    want_out, _ = oracle_step(S.clone(), q, k, v, g, beta)
    got = ext.glq_gdn_recurrent_step_cpu(S.clone(), q, k.roll(1, dims=1), v, g, beta)
    assert not torch.allclose(got, want_out, **TOL)


def test_the_output_reads_the_updated_state_not_the_stale_one():
    """The reference computes core_attn_out from the state AFTER the rank-1 update. Fusing
    the update into the output reduction is only valid in that order; doing the reduction
    first is an easy and silent transposition."""
    ext = _ext()
    S, q, k, v, g, beta = _inputs(seed=6)
    stale = (S * g.exp()[..., None, None] * q.unsqueeze(-1)).sum(dim=-2)
    got = ext.glq_gdn_recurrent_step_cpu(S.clone(), q, k, v, g, beta)
    assert not torch.allclose(got, stale, **TOL), "output was computed pre-update"


# ---- guards --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["dtype", "rank", "mismatch", "noncontig"])
def test_bad_inputs_raise_rather_than_compute_garbage(bad):
    ext = _ext()
    S, q, k, v, g, beta = _inputs(seed=7)
    if bad == "dtype":
        S = S.to(torch.float64)
    elif bad == "rank":
        S = S.reshape(B, H, K * V)
    elif bad == "mismatch":
        q = q[:, :, : K // 2].contiguous()
    else:
        S = S.transpose(2, 3)
    with pytest.raises(Exception):
        ext.glq_gdn_recurrent_step_cpu(S, q, k, v, g, beta)


def test_batch_and_head_counts_are_independent():
    """Indexing is bh-major; a kernel that folded B and H wrongly still works at B=1."""
    ext = _ext()
    for b, h in ((1, 1), (1, 7), (3, 1), (3, 5)):
        S, q, k, v, g, beta = _inputs(seed=8, b=b, h=h)
        want_out, want_state = oracle_step(S.clone(), q, k, v, g, beta)
        got_state = S.clone()
        got = ext.glq_gdn_recurrent_step_cpu(got_state, q, k, v, g, beta)
        assert torch.allclose(got, want_out, **TOL), (b, h)
        assert torch.allclose(got_state, want_state, **TOL), (b, h)


def test_thread_count_does_not_change_the_result():
    """(b, h) slabs are disjoint, so no element's accumulation order may depend on how the
    work is partitioned."""
    ext = _ext()
    S, q, k, v, g, beta = _inputs(seed=9)
    prev = torch.get_num_threads()
    results = []
    try:
        for n in (1, 2, 4):
            torch.set_num_threads(n)
            s = S.clone()
            results.append((ext.glq_gdn_recurrent_step_cpu(s, q, k, v, g, beta), s))
    finally:
        torch.set_num_threads(prev)
    for out, st in results[1:]:
        assert torch.equal(out, results[0][0])
        assert torch.equal(st, results[0][1])
