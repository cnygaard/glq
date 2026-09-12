"""What happens after a CUDA-graph capture fails.

A 335 GiB quantize died like this:

    RuntimeWarning: trellis pair CUDA-graph capture failed for ('pair', 256, 80);
                    using the per-pass path. AcceleratorError: CUDA error:
                    operation failed due to a previous error during capture
    torch.AcceleratorError: CUDA error: device-side assert triggered

The warning is the code catching the capture failure and promising a fallback. The assert on
the next line is that fallback being unable to run: a failed capture leaves the CUDA context
in an error state, so the "per-pass path" launches into a broken context and dies three
frames from the cause.

So the recovery was never implemented — only announced. It was never tested either:
tests/test_trellis_cudagraph.py is entirely GPU-gated, so on CI the whole file skips and the
failure path had no coverage at all. These tests inject the failure instead of provoking it,
which is what makes the contract checkable without a GPU.

They assert the CONTRACT, not the cause: capture failing for ('pair', 256, 80) specifically
is a separate, still-open question that needs hardware to diagnose.
"""
from __future__ import annotations

import pytest
import torch

from glq import trellis as T


@pytest.fixture
def cb(monkeypatch):
    """A codebook with capture state, without constructing the real (heavy) one."""
    obj = T.bitshift_codebook.__new__(T.bitshift_codebook)
    # __new__ skips nn.Module.__init__, which monkeypatch-free attribute setting needs.
    torch.nn.Module.__init__(obj)
    obj._vit_graphs = {}
    obj._vit_graph_pool = "POOL-SENTINEL"
    import threading
    obj._vit_lock = threading.Lock()
    # Capture is normally gated on a real GPU; these tests drive the handler directly.
    monkeypatch.setattr(T, "_GLQ_TRELLIS_CUDAGRAPH_ENABLED", True, raising=False)
    return obj


def test_a_failed_capture_releases_the_shared_pool(cb, monkeypatch):
    """Both capture paths share _vit_graph_pool, so a pool poisoned by one failure would
    taint the other. It must not be reused."""
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair",
                        lambda self, X: (_ for _ in ()).throw(RuntimeError("capture boom")))
    monkeypatch.setattr(T, "_recover_cuda_context", lambda: True)
    X = torch.zeros(8, 4)
    assert cb._pair_graphed(X) is None          # signals "use the per-pass path"
    assert cb._vit_graph_pool is None, "poisoned pool was kept"


def test_a_failed_capture_disables_capture_for_the_process(cb, monkeypatch):
    """The None sentinel is per-shape, but a poisoned context is not: retrying capture on
    the next shape risks compounding the failure. One failure disables the feature."""
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair",
                        lambda self, X: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(T, "_recover_cuda_context", lambda: True)
    assert cb._pair_graphed(torch.zeros(8, 4)) is None
    assert T._GLQ_TRELLIS_CUDAGRAPH_ENABLED is False
    assert T._trellis_cudagraph_on() is False


def test_no_second_capture_attempt_after_a_failure(cb, monkeypatch):
    calls = []

    def boom(self, X):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair", boom)
    monkeypatch.setattr(T, "_recover_cuda_context", lambda: True)
    cb._pair_graphed(torch.zeros(8, 4))
    cb._pair_graphed(torch.zeros(16, 4))        # a DIFFERENT shape
    assert len(calls) == 1, f"capture retried after a failure ({len(calls)} attempts)"


def test_an_unrecoverable_context_raises_instead_of_promising_a_fallback(cb, monkeypatch):
    """The actual bug. If the context cannot be recovered, the per-pass path cannot run, and
    returning None sends the caller into `device-side assert triggered` far from the cause.
    Fail here, naming capture."""
    monkeypatch.setattr(T.bitshift_codebook, "_capture_pair",
                        lambda self, X: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(T, "_recover_cuda_context", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-graph capture"):
        cb._pair_graphed(torch.zeros(8, 4))


def test_the_viterbi_path_recovers_the_same_way(cb, monkeypatch):
    """_viterbi_graphed promises 'using eager for this shape' and has the same defect."""
    monkeypatch.setattr(T.bitshift_codebook, "_capture_viterbi",
                        lambda self, X, o: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(T, "_recover_cuda_context", lambda: True)
    monkeypatch.setattr(T.bitshift_codebook, "viterbi",
                        lambda self, X, overlap=None: torch.full((2, X.shape[1]), 7))
    out = cb._viterbi_graphed(torch.zeros(8, 4))
    assert out.shape == (2, 4), "eager fallback did not run"
    assert cb._vit_graph_pool is None
    assert T._GLQ_TRELLIS_CUDAGRAPH_ENABLED is False


def test_recovery_helper_is_safe_without_cuda():
    """On a CPU-only box the helper must report success rather than raise: there is no
    context to poison, and the caller's error path must not itself fail."""
    assert T._recover_cuda_context() is True


def test_the_fused_step_does_not_swallow_its_error_during_capture(monkeypatch):
    """The third instance of the same flaw.

    viterbi's inner loop catches a fused-step failure and falls back to `self.update(...)`.
    That is right in normal execution, but inside a graph capture the fallback launches a
    kernel into a region that is already failing — so the capture handler never sees a clean
    error and the context is poisoned twice over. While capturing, re-raise and let the
    capture handler own recovery.
    """
    assert T._fused_step_fallback_allowed(capturing=False) is True
    assert T._fused_step_fallback_allowed(capturing=True) is False
