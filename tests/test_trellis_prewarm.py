"""Capture CUDA graphs before the expert ThreadPool starts, not lazily inside it.

The failure this removes, observed live on a Blackwell immediately after
`quantizing 1024 experts (8 workers) ...`:

    trellis pair CUDA-graph capture failed for ('pair', 256, 80);
      CUDA error: operation failed due to a previous error during capture
    Search for `cudaErrorStreamCaptureInvalidated'

cudaErrorStreamCaptureInvalidated means another thread's kernel invalidated the capture.
CUDA-graph capture is process-global and exclusive, and `_vit_lock` only serialises threads
ENTERING _pair_graphed — the other 7 workers are inside quantize_layer_e8_shell_rht running
RHT/LDLQ kernels and hold nothing.

Every expert in a group is identically shaped, so one capture serves all ~512 of them.
Capturing them up front, single-threaded, means the pool only ever REPLAYS, which is what
_vit_lock actually suffices for.

CPU-only: the capture itself is injected. The real thing needs a GPU and lives in
test_trellis_cudagraph.py.
"""
from __future__ import annotations

import threading

import pytest
import torch

from glq import trellis as T


def _cb(monkeypatch, *, enabled=True):
    obj = T.bitshift_codebook.__new__(T.bitshift_codebook)
    torch.nn.Module.__init__(obj)
    obj._vit_graphs, obj._vit_graph_pool = {}, None
    obj._vit_lock = threading.Lock()
    obj.V, obj.K, obj.L = 1, 2, 16
    monkeypatch.setattr(T, "_GLQ_TRELLIS_CUDAGRAPH_ENABLED", enabled, raising=False)
    return obj


@pytest.fixture(autouse=True)
def _no_cuda_alloc(monkeypatch):
    """CPU-only CI has no CUDA build, so the probe tensor cannot be allocated for real.
    The shapes still flow through, which is what these tests assert."""
    real_empty = torch.empty

    def empty(*a, **kw):
        if str(kw.get("device", "")) .startswith("cuda"):
            kw = {k: v for k, v in kw.items() if k != "device"}
        return real_empty(*a, **kw)
    monkeypatch.setattr(torch, "empty", empty)


def _fake_capture(calls):
    def cap(self, X, warmup_iters=3):
        calls.append(tuple(X.shape))
        return T._VitGraph("graph", X, None, torch.zeros(1))
    return cap


def test_prewarm_populates_the_graph_cache(monkeypatch):
    """After prewarm the key is present, so the pool takes the replay fast path."""
    cb = _cb(monkeypatch)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "_chunk_b", lambda self, dev: 256)

    cb.prewarm_shape(256, 80, device=torch.device("cuda"), dtype=torch.float16)
    assert ("pair", 256, 80) in cb._vit_graphs
    assert cb._vit_graphs[("pair", 256, 80)] is not None
    assert calls == [(256, 80)]


def test_prewarm_is_idempotent(monkeypatch):
    """Called per (shape, bpw) group, it must not recapture a shape it already holds."""
    cb = _cb(monkeypatch)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "_chunk_b", lambda self, dev: 256)

    for _ in range(3):
        cb.prewarm_shape(256, 80, device=torch.device("cuda"), dtype=torch.float16)
    assert len(calls) == 1, f"recaptured: {calls}"


def test_prewarm_is_a_no_op_on_cpu(monkeypatch):
    cb = _cb(monkeypatch)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))
    cb.prewarm_shape(256, 80, device=torch.device("cpu"), dtype=torch.float16)
    assert calls == [] and cb._vit_graphs == {}


def test_prewarm_respects_the_kill_switch(monkeypatch):
    cb = _cb(monkeypatch, enabled=False)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: False)
    cb.prewarm_shape(256, 80, device=torch.device("cuda"), dtype=torch.float16)
    assert calls == []


def test_prewarm_skips_shapes_lazy_capture_would_also_skip(monkeypatch):
    """The gate must match quantize()'s: NO <= min(_chunk_b, MAX_B). If prewarm captured a
    shape the lazy path rejects, it would hold a graph nothing ever replays; if it skipped
    one the lazy path accepts, that shape still captures inside the pool — the bug."""
    cb = _cb(monkeypatch)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "_chunk_b", lambda self, dev: 128)
    monkeypatch.setattr(T, "_GLQ_TRELLIS_CUDAGRAPH_MAX_B", 256, raising=False)

    cb.prewarm_shape(256, 200, device=torch.device("cuda"), dtype=torch.float16)
    assert calls == [], "B=200 exceeds _chunk_b=128; lazy capture would skip it too"
    cb.prewarm_shape(256, 64, device=torch.device("cuda"), dtype=torch.float16)
    assert calls == [(256, 64)]


def test_a_failed_prewarm_recovers_like_a_failed_capture(monkeypatch):
    """Prewarm must not be a second, weaker error path: it goes through the same recovery
    (pool dropped, capture disabled process-wide) that PR113 added."""
    cb = _cb(monkeypatch)
    cb._vit_graph_pool = "POOL"
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair",
                        lambda self, X, warmup_iters=3: (_ for _ in ()).throw(
                            RuntimeError("capture invalidated")))
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "_chunk_b", lambda self, dev: 256)
    monkeypatch.setattr(T, "_recover_cuda_context", lambda: True)

    cb.prewarm_shape(256, 80, device=torch.device("cuda"), dtype=torch.float16)
    assert cb._vit_graph_pool is None
    assert T._GLQ_TRELLIS_CUDAGRAPH_ENABLED is False


def test_prewarm_covers_every_rvq_stage(monkeypatch):
    """4bpw builds several codebooks (rvq_stages) and each has its OWN graph cache and lock.
    Prewarming only the primary leaves the other stages capturing inside the pool."""
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "_chunk_b", lambda self, dev: 256)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))

    stages = []
    for _ in range(2):
        tc = T.TrellisCodebook.__new__(T.TrellisCodebook)
        tc.cb = _cb(monkeypatch)
        stages.append(tc)
    primary = stages[0]
    primary.rvq_stages = stages

    T.prewarm_codebook(primary, 256, 80, device=torch.device("cuda"),
                       dtype=torch.float16)
    assert len(calls) == 2, f"expected one capture per stage, got {calls}"
    for st in stages:
        assert ("pair", 256, 80) in st.cb._vit_graphs


# ---- the wiring: prewarm must precede the pool, with the right shapes ------------------

def test_expert_prewarm_uses_the_ldlq_tile_shape(monkeypatch):
    """The captured shape comes from LDLQ's TILES, not the weight.

    trellis_ldlq does `tiles = WXWX.reshape(m // TD, TD * TD)` with TD=16, so capture keys
    are (T=256, B=rows//16). Traced live: a 1280-row expert captures ('pair', 256, 80) and
    a 2560-row one ('pair', 256, 160) — neither is the weight's transpose. An earlier
    version of this wiring enumerated weight shapes, prewarmed keys nothing replayed, and
    left the real shapes capturing inside the pool.
    """
    import torch.nn as nn
    from glq import quantize_model as qm

    seen = []
    monkeypatch.setattr("glq.trellis.prewarm_codebook",
                        lambda cb, t, b, device, dtype=None: seen.append((t, b)))

    linears = {
        "mlp.experts.0.gate_up_proj": nn.Linear(2560, 1280, bias=False),   # 1280 rows
        "mlp.experts.1.gate_up_proj": nn.Linear(2560, 1280, bias=False),   # same
        "mlp.experts.0.down_proj": nn.Linear(640, 2560, bias=False),       # 2560 rows
    }
    qm._prewarm_expert_graphs(linears, list(linears), object(), "cuda")

    assert (256, 80) in seen, f"1280 rows -> (256, 1280//16): {seen}"
    assert (256, 160) in seen, f"2560 rows -> (256, 2560//16): {seen}"
    assert len(seen) == 2, f"equal tile counts must capture once, got {seen}"


def test_expert_prewarm_is_a_no_op_on_cpu(monkeypatch):
    import torch.nn as nn
    from glq import quantize_model as qm
    seen = []
    monkeypatch.setattr("glq.trellis.prewarm_codebook",
                        lambda cb, t, b, device, dtype=None: seen.append((t, b)))
    qm._prewarm_expert_graphs({"e.0": nn.Linear(8, 8, bias=False)}, ["e.0"], object(), "cpu")
    assert seen == []


def test_prewarm_accepts_a_device_string(monkeypatch):
    """Callers pass "cuda", torch.device("cuda"), or a tensor's .device. The original guard
    used getattr(device, "type", None), which is None for a string — so prewarm silently did
    nothing while appearing to run, and every shape still captured inside the pool."""
    cb = _cb(monkeypatch)
    calls = []
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", _fake_capture(calls))
    monkeypatch.setattr(T, "_trellis_cudagraph_on", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "_chunk_b", lambda self, dev: 256)

    cb.prewarm_shape(256, 80, device="cuda", dtype=torch.float16)      # string, not device
    assert calls == [(256, 80)], f"string device was ignored: {calls}"
