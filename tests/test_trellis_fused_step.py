"""Stage-6 gate: the hand-fused Triton ACS step must be BIT-EXACT to the compiled update.

`glq.trellis_step_kernel.fused_update` replaces the two inductor kernels per Viterbi step
with ONE Triton kernel (min over candidates + backpointer store + state_err + cost update).
The reference is `bitshift_codebook.update` — the shipping @torch.compile path, itself
pinned to the frozen gather-form ACS by test_trellis_cudagraph.py. torch.equal on BOTH the
new cost and the backpointer row is the whole safety story: if it holds for every
(variant, K, B, masked) combination, the produced checkpoint is byte-identical.

The compiled reference is re-compiled per combination (torch._dynamo.reset) so parity is
always against inductor's output, never a silent eager fallback — the fp-contraction
choice inductor makes is exactly what the fused kernel must reproduce.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_TLUT = (torch.randn(2 ** 9, 2, generator=torch.Generator().manual_seed(0))
         * 0.9682458365518543).to(torch.float16)


def _cb(K, variant):
    tlut = _TLUT.clone() if variant == "hyb" else None
    return gt.TrellisCodebook(variant=variant, K=K, tlut=tlut, device="cuda").cb


def _fused_update():
    from glq.trellis_step_kernel import fused_update
    return fused_update


def _init_cost(cb, X, masked, seed):
    """The REAL viterbi cost init (+ the REAL overlap head-mask when masked=True)."""
    cost = (cb.recons_state - X[:cb.V].unsqueeze(-1)).square().sum(dim=0)
    if masked:
        B = X.shape[1]
        g = torch.Generator(device="cuda").manual_seed(seed)
        overlap = torch.randint(0, 2 ** (cb.L - cb.K * cb.V), (B,), generator=g,
                                device="cuda", dtype=cb.idx_dtype)
        mask = torch.ones(B, 2 ** cb.L, device=X.device) * cb.fakeinf
        allow = (overlap << (cb.K * cb.V)).unsqueeze(-1) + cb._kv_arange
        mask.scatter_(1, allow[0], 0)
        cost = torch.min(cost + mask, cb.fakeinf)
    return cost


def _both(cb, cost, thing):
    """Run compiled reference and fused kernel on clones; return (prev, cost) pairs."""
    ngroup = 2 ** (cb.L - cb.K * cb.V)
    B = cost.shape[0]
    prev_ref = torch.empty(B, ngroup, dtype=torch.int32, device="cuda")
    cost_ref = cb.update(cost.clone(), thing, prev_ref)
    prev_fus = torch.empty(B, ngroup, dtype=torch.int32, device="cuda")
    cost_fus = _fused_update()(cb, cost.clone(), thing, prev_fus)
    return (prev_ref, cost_ref), (prev_fus, cost_fus)


# ---------------------------------------------------------------------------
# THE gate: fused == compiled, bit-exact, across variant / K / B / masked
# ---------------------------------------------------------------------------
# K=1 is the stacked-RVQ residual stage of a 5 bpw checkpoint (recipe 4+1), so it runs on
# every such quantization — but it was covered only by a claim in _CFG's comment, never by
# this gate. K=5..8 stay out deliberately: they have no _CFG entry and fall back loudly.
@pytest.mark.parametrize("variant", ["hyb", "3inst"])
@pytest.mark.parametrize("K", [1, 2, 3, 4])
@pytest.mark.parametrize("B", [12, 20, 36, 60, 128, 256])
@pytest.mark.parametrize("masked", [False, True])
def test_fused_step_equiv(variant, K, B, masked):
    if variant == "hyb" and K == 1:
        # `variant` and `K` are independent axes, so widening K to 1 for the 3INST residual
        # also generated HYB K=1 — a configuration the product forbids. K=1 arises ONLY as
        # the stacked-RVQ residual, and stacked RVQ is 3INST-only: linear_method refuses HYB
        # at bpw>=5 and `_trellis_linear_apply` raises on HYB+stage2. A native primary stage
        # is 2-4. So _CFG has no (V=2, kv=2) entry by design, and asserting bit-exactness on
        # a combination that cannot be quantized would pin behaviour nothing relies on.
        pytest.skip("HYB K=1 is unreachable: K=1 is the RVQ residual and RVQ is 3INST-only")
    torch._dynamo.reset()                       # fresh inductor reference per combo
    cb = _cb(K, variant)
    torch.manual_seed(10_000 * K + B)
    X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
    cost = _init_cost(cb, X, masked, seed=K * 31 + B)
    thing = X[cb.V:2 * cb.V]
    (prev_r, cost_r), (prev_f, cost_f) = _both(cb, cost, thing)
    assert torch.equal(prev_f, prev_r), f"{variant} K={K} B={B} masked={masked}: prev"
    assert torch.equal(cost_f, cost_r), f"{variant} K={K} B={B} masked={masked}: cost"


def test_fused_step_tie_break():
    """Duplicate values everywhere → prev equality proves the strict lowest-k tie-break."""
    torch._dynamo.reset()
    cb = _cb(4, "3inst")
    torch.manual_seed(5)
    B = 36
    cost = torch.randint(0, 3, (B, 2 ** cb.L), device="cuda").float()
    thing = (torch.randn(cb.V, B, device="cuda") * 0.5).to(torch.float16)
    (prev_r, cost_r), (prev_f, cost_f) = _both(cb, cost, thing)
    assert torch.equal(prev_f, prev_r), "tie-break diverged from inductor"
    assert torch.equal(cost_f, cost_r)


def test_fused_step_nan_semantics():
    """NaN in cost / in x must propagate exactly like the compiled path. torch.equal is
    False on any NaN tensor by definition → assert on prev (int32, always comparable),
    the isnan masks, and nan_to_num'd values."""
    torch._dynamo.reset()
    cb = _cb(4, "3inst")
    torch.manual_seed(6)
    B = 20
    X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
    cost = _init_cost(cb, X, False, 0)
    cost[3, ::4097] = float("nan")              # scattered NaNs across candidate groups
    thing = X[cb.V:2 * cb.V].clone()
    thing[0, 7] = float("nan")                  # NaN input weight
    (prev_r, cost_r), (prev_f, cost_f) = _both(cb, cost, thing)
    assert torch.equal(prev_f, prev_r), "NaN handling changed backpointers"
    assert torch.equal(cost_f.isnan(), cost_r.isnan()), "NaN placement differs"
    assert torch.equal(torch.nan_to_num(cost_f, 0.0), torch.nan_to_num(cost_r, 0.0))


@pytest.mark.parametrize("variant", ["hyb", "3inst"])
def test_full_path_fused_ab(variant):
    """Whole-encoder A/B: trellis_ldlq with the fused step on vs off is torch.equal."""
    torch.manual_seed(7)
    W = (torch.randn(576, 576, device="cuda") * 0.05).float()
    Xc = torch.randn(512, 576, device="cuda")
    H = (Xc.T @ Xc) / 512

    def run(enabled):
        torch._dynamo.reset()
        gt._GLQ_TRELLIS_FUSED_STEP_ENABLED = enabled
        tlut = _TLUT.clone() if variant == "hyb" else None
        cb = gt.TrellisCodebook(variant=variant, K=4, tlut=tlut, device="cuda")
        return gt.trellis_ldlq(W, H, cb, for_kernel=True)

    try:
        h_on, q_on, s_on = run(True)
        h_off, q_off, s_off = run(False)
    finally:
        gt._GLQ_TRELLIS_FUSED_STEP_ENABLED = True
    assert torch.equal(q_on, q_off), "Qidxs differ fused vs compiled"
    assert torch.equal(h_on, h_off), "hatWr differ fused vs compiled"
    assert abs(s_on - s_off) == 0.0, "Wscale differ"


def test_fused_step_is_one_kernel(monkeypatch):
    """Mechanism: exactly ONE kernel per fused step AND it is the Triton kernel by name
    (a bit-exact fallback passing the parity tests proves nothing). Kill-switch
    counterpart: env off → the compiled 2-kernel path with no fused kernel name."""
    from torch.profiler import ProfilerActivity, profile

    def kernels(fn):
        # kineto occasionally returns an EMPTY capture after long CPU-suite stretches in
        # the same process; empty is a measurement dropout, not a mechanism verdict (a
        # genuine fallback would show the two compiled kernels) — retry the session.
        for _ in range(3):
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            ks = [e.key for e in prof.key_averages()
                  if e.self_device_time_total > 0
                  and "Memcpy" not in e.key and "Memset" not in e.key]
            if ks:
                return ks
        # Provably a measurement dropout, not a fallback: a real fallback would show the
        # two compiled kernels, and viterbi-level engagement is independently pinned by
        # test_viterbi_kernel_budget (<=560 fails on the compiled path's >=765).
        pytest.skip("kineto returned an empty capture 3x (process-state quirk after "
                    "CPU-suite stretches); engagement is pinned by the kernel-budget test")

    torch._dynamo.reset()
    cb = _cb(4, "3inst")
    torch.manual_seed(8)
    B = 60
    X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
    cost = _init_cost(cb, X, False, 0)
    thing = X[cb.V:2 * cb.V]
    prev = torch.empty(B, 2 ** (cb.L - cb.K * cb.V), dtype=torch.int32, device="cuda")

    fused = _fused_update()
    fused(cb, cost.clone(), thing, prev)        # warm/compile
    ks = kernels(lambda: fused(cb, cost.clone(), thing, prev))
    ks = [k for k in ks if "DtoD" not in k]     # the cost.clone() inside the region
    assert len(ks) == 1 and "_viterbi_acs_step" in ks[0], \
        f"fused step ran as {len(ks)} kernels: {ks}"

    # kill-switch: viterbi with env off must use the compiled path (no fused name)
    monkeypatch.setenv("GLQ_TRELLIS_FUSED_STEP", "0")
    assert gt._fused_step_on() is False
    cb.viterbi(X)                               # warm compiled path
    ks = kernels(lambda: cb.viterbi(X))
    assert not any("_viterbi_acs_step" in k for k in ks), \
        "kill-switch did not disable the fused kernel"


def test_fused_kernel_no_spills():
    """Register-pressure guard across all six (variant, K) specializations."""
    from glq.trellis_step_kernel import _viterbi_acs_step
    for variant in ("3inst", "hyb"):
        for K in (2, 3, 4):
            cb = _cb(K, variant)
            B = 12
            X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
            cost = _init_cost(cb, X, False, 0)
            prev = torch.empty(B, 2 ** (cb.L - cb.K * cb.V),
                               dtype=torch.int32, device="cuda")
            _fused_update()(cb, cost, X[cb.V:2 * cb.V], prev)
    torch.cuda.synchronize()
    spills = []
    for key, kern in getattr(_viterbi_acs_step, "cache", {}).items() \
            if isinstance(getattr(_viterbi_acs_step, "cache", None), dict) else []:
        n = getattr(kern, "n_spills", None)
        if n:
            spills.append((key, n))
    assert not spills, f"register spills in fused kernel: {spills}"
