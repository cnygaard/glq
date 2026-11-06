"""Routed experts quantized on several GPUs at once, behind `--expert-devices`.

GLM-5.3-Flash spends ~98% of each MoE layer in the expert pool (layer 3: 2,027 s of 2,067 s)
while the box's second GPU idles. The split must be:

* **off unless asked.** Two independent quantizes pinned with CUDA_VISIBLE_DEVICES, one per
  card, are an equally valid way to use the box; nothing may widen a job onto a card it was
  not given. So: a flag, no environment variable, and `auto` only sees visible devices.
* **byte-identical** to one device, which is what lets the resume run key ignore it.
* **capture-free inside the pool.** CUDA-graph capture is process-global, so a capture on one
  card is exposed to the other card's workers; every shape is captured before any pool runs.

CPU tests pin the device resolution, the runner and the codebook copy. The GPU tests need two
identical cards and run on the box.
"""
from __future__ import annotations

import inspect
import threading
import time

import pytest
import torch
import torch.nn as nn

from glq import quantize_model as qm
from glq import trellis as T

GiB = 1 << 30


# ------------------------------------------------------------------------- resolving

@pytest.fixture
def two_cards(monkeypatch):
    """Two visible CUDA devices; tests edit `names` and `used` to vary them."""
    state = {"names": ["NVIDIA RTX PRO 6000", "NVIDIA RTX PRO 6000"], "used": [0, 0]}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(state["names"]))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: state["names"][i])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda i=0: (95 * GiB - state["used"][i], 95 * GiB))
    return state


def test_without_the_flag_experts_stay_on_the_main_device(two_cards):
    """Two cards visible, nothing asked: today's behaviour, the main device alone."""
    assert qm._resolve_expert_devices(None, "cuda", "trellis") == ["cuda"]
    assert qm._resolve_expert_devices(None, "cpu", "e8_shell") == ["cpu"]


def test_no_environment_variable_turns_the_split_on(two_cards, monkeypatch):
    """An inherited variable would put both of two pinned jobs on each other's card."""
    for name in ("GLQ_QUANT_EXPERT_DEVICES", "GLQ_EXPERT_DEVICES"):
        monkeypatch.setenv(name, "auto")
    assert qm._resolve_expert_devices(None, "cuda", "trellis") == ["cuda"]
    assert inspect.signature(qm.quantize).parameters["expert_devices"].default is None
    assert qm.build_parser().parse_args(["--model", "m", "--output", "o"]).expert_devices is None


def test_auto_takes_every_visible_card(two_cards, capsys):
    assert qm._resolve_expert_devices("auto", "cuda", "trellis") == ["cuda:0", "cuda:1"]
    assert "expert devices: cuda:0 (95.0 GiB free), cuda:1 (95.0 GiB free)" in capsys.readouterr().out


def test_auto_stays_inside_the_visible_set(two_cards):
    """Under CUDA_VISIBLE_DEVICES=1 the process sees one card, and `auto` is just that one."""
    two_cards["names"] = two_cards["names"][:1]
    assert qm._resolve_expert_devices("auto", "cuda", "trellis") == ["cuda:0"]


def test_auto_leaves_out_a_different_model(two_cards):
    two_cards["names"][1] = "NVIDIA GeForce RTX 4090"
    assert qm._resolve_expert_devices("auto", "cuda", "trellis") == ["cuda:0"]


def test_a_list_is_taken_in_order(two_cards):
    assert qm._resolve_expert_devices(" cuda:1 , cuda:0 ", "cuda", "trellis") == ["cuda:1", "cuda:0"]


@pytest.mark.parametrize("spec,device,codebook,match", [
    ("cuda:1,cuda:1", "cuda", "trellis", "twice"),
    ("cuda:2", "cuda", "trellis", "not visible"),
    ("auto", "cpu", "trellis", "--device cuda"),
    ("cpu", "cuda", "trellis", "CUDA"),
    ("auto", "cuda", "e8p", "trellis"),
    ("auto", "cuda", "e8_shell", "trellis"),
])
def test_refusals(two_cards, spec, device, codebook, match):
    with pytest.raises(ValueError, match=match):
        qm._resolve_expert_devices(spec, device, codebook)


def test_a_listed_card_of_another_model_is_refused(two_cards):
    """Different cuBLAS kernels and L2 sizes could change the encoded bytes depending on
    which card took an expert; the checkpoint must not depend on scheduling."""
    two_cards["names"][1] = "NVIDIA GeForce RTX 4090"
    with pytest.raises(ValueError, match="same GPU model"):
        qm._resolve_expert_devices("cuda:0,cuda:1", "cuda", "trellis")


def test_one_card_needs_no_split_so_any_codebook_may_ask(two_cards):
    two_cards["names"] = two_cards["names"][:1]
    assert qm._resolve_expert_devices("auto", "cuda", "e8p") == ["cuda:0"]


def test_a_busy_card_is_named_but_not_refused(two_cards, capsys):
    """Another job on a requested card is the user's call; say so plainly."""
    two_cards["used"][1] = 5 * GiB
    assert qm._resolve_expert_devices("auto", "cuda", "trellis") == ["cuda:0", "cuda:1"]
    out = capsys.readouterr().out
    assert "WARNING" in out and "cuda:1 has 5.0 GiB in use" in out


# ------------------------------------------------------------------------- the runner

def _run(names, devices, workers, work=None, collect=None):
    seen, binding, collected = [], {}, []
    lock = threading.Lock()
    main = threading.get_ident()

    def quantize_one(name, dev):
        if work is not None:
            work(name)
        with lock:
            seen.append(name)
            binding.setdefault(threading.get_ident(), set()).add(dev)
        return name, dev

    def _collect(name, dev):
        assert threading.get_ident() == main, "collect must run on the calling thread"
        collected.append((name, dev))
        if collect is not None:
            collect(name)

    counts = qm._quantize_experts_parallel(names, quantize_one, _collect, devices, workers)
    return seen, binding, collected, counts


def test_every_expert_runs_once_and_is_collected_on_the_calling_thread():
    names = [f"e{i}" for i in range(64)]
    seen, _, collected, counts = _run(names, ["dev0", "dev1"], 3,
                                      work=lambda n: time.sleep(0.002))
    assert sorted(seen) == sorted(names) and len(seen) == len(set(seen))
    assert sorted(n for n, _ in collected) == sorted(names)
    assert sum(counts.values()) == len(names)


def test_each_thread_serves_one_device_and_both_devices_work():
    names = [f"e{i}" for i in range(64)]
    _, binding, _, counts = _run(names, ["dev0", "dev1"], 3, work=lambda n: time.sleep(0.002))
    assert all(len(devs) == 1 for devs in binding.values()), binding
    assert set(counts) == {"dev0", "dev1"} and min(counts.values()) > 0, counts
    per_dev = {}
    for devs in binding.values():
        d = next(iter(devs))
        per_dev[d] = per_dev.get(d, 0) + 1
    assert max(per_dev.values()) <= 3, per_dev


def test_a_worker_error_stops_the_pool_and_reaches_the_caller():
    names = [f"e{i}" for i in range(200)]

    def work(name):
        time.sleep(0.005)
        if name == "e3":
            raise RuntimeError("expert e3 failed")

    before = threading.active_count()
    with pytest.raises(RuntimeError, match="expert e3 failed"):
        _run(names, ["dev0", "dev1"], 2, work=work)
    assert threading.active_count() == before, "worker threads outlived the error"


def test_a_collect_error_also_stops_the_pool():
    names = [f"e{i}" for i in range(200)]

    def collect(name):
        raise ValueError("writeback failed")

    before = threading.active_count()
    with pytest.raises(ValueError, match="writeback failed"):
        _run(names, ["dev0"], 4, work=lambda n: time.sleep(0.005), collect=collect)
    assert threading.active_count() == before


def test_results_are_not_held_after_collection():
    """The futures dict held every expert's fp32 W_hat until the layer ended; collected
    results must be freeable at once."""
    import gc
    import weakref

    refs = []

    class Big:
        pass

    def quantize_one(name, dev):
        return (Big(),)

    def collect(obj):
        refs.append(weakref.ref(obj))

    qm._quantize_experts_parallel([f"e{i}" for i in range(16)], quantize_one, collect,
                                  ["dev0"], 2)
    gc.collect()
    assert all(r() is None for r in refs)


# ------------------------------------------------------------------------- the codebook copy

def _rvq(*ks):
    stages = [T.TrellisCodebook(variant="3inst", K=k, device="cpu") for k in ks]
    stages[0].rvq_stages = stages
    return stages[0]


def test_a_copy_matches_every_buffer_and_has_its_own_graph_cache():
    cb = _rvq(4)
    cp = T.replicate_trellis_codebook(cb, "cpu")
    assert cp is not cb and cp.cb is not cb.cb
    for k, v in cb.cb.state_dict().items():
        assert torch.equal(v, cp.cb.state_dict()[k]), k
    assert (cp.variant, cp.K, cp.V, cp.L, cp.opt_scale, cp.has_kernel) == \
           (cb.variant, cb.K, cb.V, cb.L, cb.opt_scale, cb.has_kernel)
    assert cp.cb._vit_graphs is not cb.cb._vit_graphs
    assert cp.cb._vit_lock is not cb.cb._vit_lock
    assert cp.rvq_stages == [cp]


def test_every_rvq_stage_is_copied():
    cb = _rvq(4, 2)
    cp = T.replicate_trellis_codebook(cb, "cpu")
    assert [s.K for s in cp.rvq_stages] == [4, 2]
    assert cp.rvq_stages[0] is cp
    for a, b in zip(cb.rvq_stages, cp.rvq_stages):
        assert b is not a
        for k, v in a.cb.state_dict().items():
            assert torch.equal(v, b.cb.state_dict()[k]), k


def test_a_hyb_copy_reuses_the_fitted_tlut(monkeypatch):
    """HYB fits its tlut by k-means; a copy must carry the original's, never refit."""
    pytest.importorskip("scipy", reason="the original HYB codebook fits its tlut with scipy")
    cb = T.TrellisCodebook(variant="hyb", K=2, device="cpu")
    cb.rvq_stages = [cb]

    def refit(*a, **k):
        raise AssertionError("the copy refitted the tlut")
    monkeypatch.setattr(T, "_make_hyb_tlut", refit)
    cp = T.replicate_trellis_codebook(cb, "cpu")
    assert torch.equal(cp.tlut, cb.tlut)
    assert torch.equal(cp.cb.lut, cb.cb.lut)


def test_a_copy_that_differs_is_refused():
    cb = _rvq(4)
    cp = T.replicate_trellis_codebook(cb, "cpu")
    cp.cb.lut = cp.cb.lut + 1
    with pytest.raises(RuntimeError, match="lut"):
        T._check_replica(cb, cp)


# ------------------------------------------------------------------------- on two GPUs

needs_two = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2
    or len({torch.cuda.get_device_name(i) for i in range(min(2, torch.cuda.device_count()))}) != 1,
    reason="needs two identical CUDA devices")


def _expert(rows, cols, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(rows, cols, generator=g) * 0.02).to(torch.bfloat16)
    X = torch.randn(2 * cols, cols, generator=g)
    return W, X.T @ X / X.shape[0]


def _pair():
    from glq.trellis_recipe import trellis_rvq_recipe
    stages = [T.TrellisCodebook(variant="3inst", K=k, device="cuda:0")
              for k in trellis_rvq_recipe(4)]
    stages[0].rvq_stages = stages
    return {"cuda:0": stages[0], "cuda:1": T.replicate_trellis_codebook(stages[0], "cuda:1")}


@needs_two
@pytest.mark.parametrize("rows,cols", [(4096, 2048), (4096, 4096), (1280, 2560)])
def test_an_expert_encodes_identically_on_either_card(rows, cols):
    """GLM-5.3's expert shapes (4096 rows: the chunked Viterbi path) and a 1280-row one
    (the pair-graph path). W stays on cuda:0, as in the driver.

    Both cards must have encoded through their graphs: a failed capture falls back to eager
    on every card, which is also identical. cuda:1 capturing after cuda:0 is the order that
    failed while torch's class-wide capture stream was in use."""
    cbs = _pair()
    W, H = _expert(rows, cols)
    W = W.to("cuda:0")
    out = {}
    for dev, cb in cbs.items():
        with T.cuda_device_scope(dev):
            T.prewarm_codebook(cb, T.TD * T.TD, rows // T.TD, device=dev)
            W_hat, arts, _ = qm.quantize_layer_e8_shell_rht(W, H.to(dev), cb, bpw=4)
            out[dev] = (W_hat.cpu(), {k: v.cpu() for k, v in arts.items()})
    assert T._GLQ_TRELLIS_CUDAGRAPH_ENABLED, "a capture failed and the cards ran eager"
    for dev, cb in cbs.items():
        assert any(g is not None for g in cb.cb._vit_graphs.values()), f"no graph on {dev}"
    (w0, a0), (w1, a1) = out["cuda:0"], out["cuda:1"]
    assert torch.equal(w0, w1)
    assert a0.keys() == a1.keys()
    for k in a0:
        assert torch.equal(a0[k], a1[k]), k


@needs_two
@pytest.mark.parametrize("rows", [4096, 1280])
def test_after_prewarm_the_pool_only_replays(monkeypatch, rows):
    """The mechanism, not just the output: no capture happens while both cards' workers run,
    every graph key the pool replays was captured by the prewarm, and each worker ran with
    its own card current."""
    cbs = _pair()
    captured, replayed = [], []
    for meth in ("_capture_viterbi", "_capture_pair"):
        real = getattr(T.bitshift_codebook, meth)

        def spy(self, *a, _real=real, _m=meth, **k):
            captured.append((_m, threading.current_thread().name))
            return _real(self, *a, **k)
        monkeypatch.setattr(T.bitshift_codebook, meth, spy)
    for meth in ("_viterbi_graphed", "_pair_graphed"):
        real = getattr(T.bitshift_codebook, meth)

        def spy(self, X, *a, _real=real, **k):
            replayed.append((id(self), int(X.shape[0]), int(X.shape[1])))
            return _real(self, X, *a, **k)
        monkeypatch.setattr(T.bitshift_codebook, meth, spy)

    names = [f"mlp.experts.{i}.down_proj" for i in range(6)]
    linears = {}
    for i, n in enumerate(names):
        W, _ = _expert(rows, 1024, seed=i)
        lin = nn.Linear(1024, rows, bias=False, device="cuda:0", dtype=torch.bfloat16)
        lin.weight.data = W.to("cuda:0")
        linears[n] = lin
    _, H = _expert(rows, 1024)
    for dev, cb in cbs.items():
        qm._prewarm_expert_graphs(linears, names, cb, dev)
    n_prewarm = len(captured)
    assert n_prewarm > 0, "prewarm captured nothing: the test would prove nothing"

    current = []

    def quantize_one(name, dev):
        current.append((dev, torch.cuda.current_device()))
        W_hat, arts, _ = qm.quantize_layer_e8_shell_rht(
            linears[name].weight.data, H.to(dev), cbs[dev], bpw=4)
        return name, W_hat.to("cuda:0")

    counts = qm._quantize_experts_parallel(names, quantize_one, lambda *r: None,
                                           ["cuda:0", "cuda:1"], 2)
    assert captured[n_prewarm:] == [], f"captured inside the pool: {captured[n_prewarm:]}"
    assert replayed, "nothing replayed a graph"
    assert min(counts.values()) > 0 and set(counts) == {"cuda:0", "cuda:1"}, counts
    assert all(int(dev.split(":")[1]) == cur for dev, cur in current), current
