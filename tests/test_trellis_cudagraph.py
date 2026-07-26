"""S4-speed gate: the encoder-side Viterbi CUDA-graph must be BIT-EXACT to eager (GPU-only).

`bitshift_codebook._viterbi_graphed` captures the whole `viterbi` into a CUDA graph keyed by
(T, B, has_overlap) and replays it — a pure launch-overhead optimization for offline quant.
Graph replay re-runs identical kernels on a deterministic Viterbi (min/argmin/gather, no
atomics), so the output states must be `torch.equal` to the eager `viterbi`. That equality is
the whole safety story: if it holds, the produced checkpoint is byte-identical.

Covers the native rate K∈{2,3,4}, batch B∈{12,36,256} (the distinct per-layer widths + the
bs cap), and both tail-biting passes (overlap None / tensor).
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


def _cb(K, variant="hyb"):
    tlut = _TLUT.clone() if variant == "hyb" else None
    return gt.TrellisCodebook(variant=variant, K=K, tlut=tlut, device="cuda").cb


# ---------------------------------------------------------------------------
# THE gate: graphed viterbi == eager viterbi, bit-exact, across variant / K / B
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["hyb", "3inst"])
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("B", [12, 36, 256])
@pytest.mark.parametrize("has_overlap", [False, True])
def test_viterbi_cudagraph_bitexact_vs_eager(variant, K, B, has_overlap):
    cb = _cb(K, variant)
    torch.manual_seed(1000 * K + B)
    X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
    overlap = None
    if has_overlap:
        # valid overlap ∈ [0, 2^(L-KV)) — same tensor feeds both paths, so parity holds
        overlap = torch.randint(0, 2 ** (cb.L - cb.K * cb.V), (B,),
                                device="cuda", dtype=cb.idx_dtype)
    eager = cb.viterbi(X, overlap)                      # `viterbi` is always the eager body
    graphed = cb._viterbi_graphed(X, overlap)           # capture-on-miss then replay
    assert torch.equal(eager, graphed), \
        f"K={K} B={B} overlap={has_overlap}: max|Δ|={(eager - graphed).abs().max().item()}"


def test_viterbi_graph_replay_is_deterministic():
    cb = _cb(4)
    torch.manual_seed(7)
    X = (torch.randn(256, 36, device="cuda") * 0.5).to(torch.float16)
    a = cb._viterbi_graphed(X)                          # captures + replays
    b = cb._viterbi_graphed(X)                          # replays again
    assert torch.equal(a, b)


def test_graphed_differs_from_a_different_input():
    # sanity: the replay actually reflects the copied-in input (not a stale captured result)
    cb = _cb(4)
    torch.manual_seed(8)
    x1 = (torch.randn(256, 36, device="cuda") * 0.5).to(torch.float16)
    x2 = (torch.randn(256, 36, device="cuda") * 0.5).to(torch.float16)
    y1 = cb._viterbi_graphed(x1).clone()
    y2 = cb._viterbi_graphed(x2)
    assert not torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# The strongest gate: the WHOLE encoder (trellis_ldlq) is bit-exact graphs-on vs -off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["hyb", "3inst"])
@pytest.mark.parametrize("shape", [(576, 576), (192, 576), (1536, 576)])
def test_full_layer_trellis_ldlq_graph_parity(variant, shape):
    m, n = shape
    torch.manual_seed(3)
    W = (torch.randn(m, n, device="cuda") * 0.05).float()
    Xc = torch.randn(512, n, device="cuda")
    H = (Xc.T @ Xc) / 512

    def run(enabled):
        gt._GLQ_TRELLIS_CUDAGRAPH_ENABLED = enabled
        tlut = _TLUT.clone() if variant == "hyb" else None
        cb = gt.TrellisCodebook(variant=variant, K=4, tlut=tlut, device="cuda")
        return gt.trellis_ldlq(W, H, cb, for_kernel=True)

    try:
        h_on, q_on, s_on = run(True)
        h_off, q_off, s_off = run(False)
    finally:
        gt._GLQ_TRELLIS_CUDAGRAPH_ENABLED = True
    assert torch.equal(q_on, q_off), f"Qidxs differ {shape}"
    assert torch.equal(h_on, h_off), f"hatWr differ {shape}"
    assert abs(s_on - s_off) == 0.0, "Wscale differ"


@pytest.mark.parametrize("variant", ["hyb", "3inst"])
def test_graph_actually_captured_both_passes(variant):
    """Closes the silent-fallback blind spot: a graphed `trellis_ldlq` must ACTUALLY capture a
    real graph for BOTH tail-biting passes (overlap False AND True) — not fall back to eager.
    Pre-fix, the overlap=True pass raised 'CPU→CUDA copy during capture' and got a None sentinel;
    this asserts both keys hold a live _VitGraph. B = min(256, m//16) = 36 for 576×576."""
    gt._GLQ_TRELLIS_CUDAGRAPH_ENABLED = True
    tlut = _TLUT.clone() if variant == "hyb" else None
    cb = gt.TrellisCodebook(variant=variant, K=4, tlut=tlut, device="cuda")
    torch.manual_seed(2)
    W = (torch.randn(576, 576, device="cuda") * 0.05).float()
    Xc = torch.randn(512, 576, device="cuda")
    H = (Xc.T @ Xc) / 512
    gt.trellis_ldlq(W, H, cb, for_kernel=True)
    graphs = cb.cb._vit_graphs
    for has_overlap in (False, True):
        key = (256, 36, has_overlap)
        assert key in graphs and graphs[key] is not None, \
            f"{key} not captured (None sentinel = eager fallback) — graph did not engage"


# ---------------------------------------------------------------------------
# ACS update equivalence: `update` must stay value-identical to the ORIGINAL
# gather-form ACS (inlined here as the frozen reference). Guards the view-min
# restructure: identical candidate ordering ⇒ identical min tie-breaks ⇒
# bit-identical prev_state AND cost.
# ---------------------------------------------------------------------------
def _update_reference(cb, cost, thing):
    """The original QTIP gather-form ACS, frozen (do not refactor with trellis.py)."""
    state_err = (cb.recons_state - thing.unsqueeze(-1)).square().sum(dim=0)
    cand_cost = torch.gather(
        cost.unsqueeze(-2).expand(-1, cb.state_cand.shape[1], -1), -1,
        cb.state_cand.expand(len(cost), -1, 2 ** (cb.K * cb.V)))
    best = torch.min(cand_cost, dim=-1)
    new_cost = state_err + best.values.unsqueeze(-1).expand(
        -1, -1, 2 ** (cb.K * cb.V)).reshape(state_err.shape)
    prev_state = torch.gather(
        cb.state_cand.expand(thing.shape[1], -1, -1), -1,
        best.indices.unsqueeze(-1))[..., 0]
    return prev_state, new_cost


@pytest.mark.parametrize("variant", ["hyb", "3inst"])
@pytest.mark.parametrize("K", [2, 3, 4])
def test_trellis_update_equiv(variant, K):
    cb = _cb(K, variant)
    torch.manual_seed(100 * K)
    B = 36
    # a realistic cost state: run the real init so magnitudes/duplicate-ties match prod
    X = (torch.randn(256, B, device="cuda") * 0.5).to(torch.float16)
    cost = (cb.recons_state - X[:cb.V].unsqueeze(-1)).square().sum(dim=0)
    thing = X[cb.V:2 * cb.V]
    prev_ref, cost_ref = _update_reference(cb, cost.clone(), thing)
    prev_new, cost_new = cb.update(cost.clone(), thing)
    assert torch.equal(prev_new.to(torch.int64), prev_ref.to(torch.int64)), \
        f"{variant} K={K}: prev_state diverged from the gather-form reference"
    assert torch.equal(cost_new, cost_ref), \
        f"{variant} K={K}: cost diverged from the gather-form reference"


def test_env_kill_switch_forces_eager(monkeypatch):
    monkeypatch.setenv("GLQ_TRELLIS_NO_CUDAGRAPH", "1")
    assert gt._trellis_cudagraph_on() is False
    # with the env set, quantize_seq routes to eager → the codebook captures no graphs
    cb = gt.TrellisCodebook(variant="hyb", K=4, tlut=_TLUT.clone(), device="cuda")
    torch.manual_seed(9)
    W = (torch.randn(96, 128, device="cuda") * 0.05).float()
    gt.trellis_ldlq(W, torch.eye(128, device="cuda"), cb, for_kernel=True)
    assert cb.cb._vit_graphs == {}
