"""The S0 routing trace: what `apply` records so a cache can be sized before it is built.

Strata sized its cache by scoring a run's routing offline rather than guessing
(`Memory/cache_allocation.py`), and that is the cheapest way to learn whether a GLQ expert
cache pays at all. This records `(layer, topk_ids)` per MoE forward behind an env flag and
changes nothing else.

Pure logic — no GPU, no torch device.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq_vllm.expert_trace import RoutingTrace, layer_key  # noqa: E402


# ---------------------------------------------------------------------------
# Layer identity
# ---------------------------------------------------------------------------

def test_the_decoder_and_the_mtp_head_are_DIFFERENT_layers():
    """The collision that would silently corrupt a residency table. vLLM renumbers the draft
    block to `mtp.layers.48`, but a checkpoint names it `mtp.layers.0` — and decoder layer 0
    also exists. Keying residency on the bare index would make them share a row, so two
    unrelated expert sets would evict each other."""
    a = layer_key("model.language_model.layers.0.mlp.experts")
    b = layer_key("mtp.layers.0.mlp.experts")
    assert a != b, "decoder layer 0 and MTP layer 0 must not share a key"


def test_the_same_layer_always_gets_the_same_key():
    p = "model.language_model.layers.7.mlp.experts"
    assert layer_key(p) == layer_key(p)


def test_a_missing_prefix_still_yields_a_usable_key():
    """`get_quant_method` always has the prefix, but a caller that predates the change (or a
    test) may not pass one. A trace with one lumped layer is less useful than a correct one,
    so this must not crash — it must be visibly degraded instead."""
    k = layer_key(None)
    assert isinstance(k, str) and k


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def test_records_are_appended_per_forward():
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [3, 1, 2])
    t.record("layers.1.mlp.experts", [0, 5])
    t.record("layers.0.mlp.experts", [3, 1, 2])
    assert len(t) == 3
    assert t.n_layers == 2


def test_ids_are_stored_verbatim_not_sorted():
    """The routed set's ORDER is the k-order the GPU reduces in. A trace that sorted ids
    would still score the same hit rate, but would be useless for anything that later needs
    the reduction order — so do not quietly normalise it."""
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [9, 2, 7])
    assert list(t.records()[0][1]) == [9, 2, 7]


def test_a_torch_tensor_of_ids_is_accepted():
    torch = pytest.importorskip("torch")
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", torch.tensor([[4, 1], [2, 3]]))
    # one record per forward, flattened across the batch's tokens
    assert len(t) == 2
    assert sorted(int(x) for x in t.records()[0][1]) == [1, 4]


# ---------------------------------------------------------------------------
# Round trip, and the scorer's input contract
# ---------------------------------------------------------------------------

def test_save_then_load_round_trips(tmp_path):
    t = RoutingTrace()
    for step in range(3):
        for li in range(4):
            t.record(f"layers.{li}.mlp.experts", [step, li, (step + li) % 8])
    p = tmp_path / "trace.npz"
    t.save(str(p))

    loaded = RoutingTrace.load(str(p))
    assert len(loaded) == len(t)
    assert loaded.n_layers == t.n_layers
    assert [list(ids) for _, ids in loaded.records()] == [list(ids) for _, ids in t.records()]


def test_the_loaded_trace_feeds_the_scorer_directly(tmp_path):
    """The two halves of S0 must compose without glue: the recorder's output is the scorer's
    input. If this needs a conversion step, the conversion is where they will drift apart."""
    from glq_vllm.expert_cache import score_trace

    t = RoutingTrace()
    for step in range(20):
        for li in range(4):
            t.record(f"layers.{li}.mlp.experts", [(step + li) % 6, (step * 2 + li) % 6])
    p = tmp_path / "trace.npz"
    t.save(str(p))

    loaded = RoutingTrace.load(str(p))
    r = score_trace(loaded.scored_records(), n_layers=loaded.n_layers,
                    n_experts=loaded.n_experts, slots=4 * 4, per_layer=True)
    assert r.lookups == len(t) * 2
    assert 0.0 <= r.hit_rate <= 1.0


def test_the_saved_trace_records_its_own_provenance(tmp_path):
    """A hit rate quoted without knowing which model and which run produced the trace is not
    a number anyone can act on."""
    t = RoutingTrace(meta={"model": "flash-next-3bpw", "note": "256-step decode"})
    t.record("layers.0.mlp.experts", [1, 2])
    p = tmp_path / "trace.npz"
    t.save(str(p))
    meta = json.loads(str(RoutingTrace.load(str(p)).meta_json))
    assert meta["model"] == "flash-next-3bpw"
    assert "n_records" in meta and meta["n_records"] == 1


def test_an_empty_trace_saves_and_loads_without_special_casing(tmp_path):
    t = RoutingTrace()
    p = tmp_path / "empty.npz"
    t.save(str(p))
    loaded = RoutingTrace.load(str(p))
    assert len(loaded) == 0
    assert loaded.n_layers == 0


# ---------------------------------------------------------------------------
# Phase: separating decode from prefill, and from vLLM's dummy profiling run
# ---------------------------------------------------------------------------

def test_each_record_remembers_how_many_tokens_its_forward_carried():
    """The width of a forward is the only thing that distinguishes a decode step from a
    prefill, and the scorer cannot be trusted without it — see the test below for what it
    costs to get this wrong."""
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [[1, 2], [3, 4], [5, 6]])     # a 3-token prefill
    t.record("layers.0.mlp.experts", [7, 8])                        # a 1-token decode step
    assert list(t.widths) == [3, 3, 3, 1]


def test_decode_and_prefill_can_be_scored_separately():
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [[1, 1], [1, 1], [1, 1], [1, 1]])
    t.record("layers.0.mlp.experts", [2, 3])
    assert [ids for _, ids in t.records(phase="decode")] == [[2, 3]]
    assert len(t.records(phase="prefill")) == 4
    assert len(t.records(phase="all")) == 5


def test_an_unknown_phase_is_refused_not_silently_treated_as_all():
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [1])
    with pytest.raises(ValueError, match="phase"):
        t.records(phase="warmup")


def test_vllm_PROFILING_RUN_dominates_a_trace_and_inflates_the_hit_rate():
    """★ The trap this phase split exists for, measured on the real L4 trace.

    vLLM runs a synthetic max-model-len forward at startup to size its KV pool. On
    Flash-Next at `--max-model-len 4096` that is ONE 4096-token forward per layer —
    **196,608 of 220,211 records, 89.3% of the trace** — and because its input is dummy data
    the routing is near-degenerate: it touched **495** distinct (layer, expert) pairs out of
    24,576, where real decode touched 14,109.

    Any cache nails a 495-pair working set, so including it moved the reported hit rate at
    115 slots/layer from a real **0.5263** to a meaningless **0.9400**. The arithmetic below
    reproduces the mechanism in miniature: a wide degenerate burst plus diverse decode, where
    scoring everything reports roughly double the decode truth."""
    from glq_vllm.expert_cache import score_trace

    t = RoutingTrace()
    # the profiling run: one very wide forward, always the same two experts
    t.record("layers.0.mlp.experts", [[0, 1]] * 400)
    # real decode: narrow forwards across a wide, shifting expert set
    for step in range(100):
        t.record("layers.0.mlp.experts", [step % 50, (step * 7 + 3) % 50])

    kw = dict(n_layers=1, n_experts=64, slots=4, per_layer=True, eviction=None)
    all_rate = score_trace(t.scored_records(phase="all"), **kw).hit_rate
    dec_rate = score_trace(t.scored_records(phase="decode"), **kw).hit_rate
    assert all_rate > dec_rate + 0.3, (
        f"the degenerate wide burst must visibly inflate the combined rate "
        f"(all={all_rate:.4f} decode={dec_rate:.4f}); if it does not, this fixture no "
        f"longer reproduces the trap it was written for")


def test_a_legacy_trace_without_widths_derives_them_and_says_so(tmp_path):
    """Traces already recorded (the L4 run took ~10 min of GPU time) carry no width array.
    They are still usable, because a forward is a contiguous run of one layer — but the
    derivation is an INFERENCE, so it must be flagged rather than presented as recorded
    fact. A 1-token prompt would make it wrong, and silence would hide that."""
    import numpy as np

    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [[1, 2], [3, 4]])
    t.record("layers.1.mlp.experts", [[1, 2], [3, 4]])
    t.record("layers.0.mlp.experts", [5, 6])
    p = tmp_path / "legacy.npz"
    t.save(str(p))

    # strip the width array, as a pre-existing trace has none
    with np.load(str(p), allow_pickle=False) as z:
        keep = {k: z[k] for k in z.files if k != "width"}
    assert "width" not in keep
    np.savez(str(p), **keep)

    loaded = RoutingTrace.load(str(p))
    assert loaded.widths_were_derived is True
    assert list(loaded.widths) == [2, 2, 2, 2, 1]
    assert [ids for _, ids in loaded.records(phase="decode")] == [[5, 6]]


def test_a_compressed_trace_is_byte_equal_and_much_smaller(tmp_path):
    """The flush path stays uncompressed because it re-saves every `flush_every` records
    inside the serving process, where CPU time is not free. An artifact kept in the repo is
    written once and read many times, and measured **9.70 MB -> 0.35 MB (28x)** on the real
    L4 trace — so the option exists, rather than a numpy incantation in a README."""
    t = RoutingTrace()
    for step in range(400):
        for li in range(8):
            t.record(f"layers.{li}.mlp.experts", [step % 17, (step * 3 + li) % 17])
    plain, packed = tmp_path / "plain.npz", tmp_path / "packed.npz"
    t.save(str(plain))
    t.save(str(packed), compress=True)

    assert packed.stat().st_size < plain.stat().st_size
    a, b = RoutingTrace.load(str(plain)), RoutingTrace.load(str(packed))
    assert a.records() == b.records()
    assert a.widths == b.widths


def test_DERIVED_widths_stay_flagged_after_a_resave(tmp_path):
    """Re-saving a legacy trace (to compress it for the repo) writes the derived widths into
    the file. Without this, they would come back looking RECORDED — the resave would launder
    an inference into a fact, and the next reader could not tell that a 1-token prompt would
    have made them wrong. The flag has to survive the round trip."""
    import numpy as np

    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [[1, 2], [3, 4]])
    t.record("layers.0.mlp.experts", [5, 6])
    legacy = tmp_path / "legacy.npz"
    t.save(str(legacy))
    with np.load(str(legacy), allow_pickle=False) as z:
        keep = {k: z[k] for k in z.files if k != "width"}
    np.savez(str(legacy), **keep)

    loaded = RoutingTrace.load(str(legacy))
    assert loaded.widths_were_derived is True

    resaved = tmp_path / "resaved.npz"
    loaded.save(str(resaved), compress=True)
    assert RoutingTrace.load(str(resaved)).widths_were_derived is True, (
        "a resave laundered derived widths into recorded ones")


def test_a_recorded_width_is_NOT_flagged_as_derived(tmp_path):
    t = RoutingTrace()
    t.record("layers.0.mlp.experts", [[1, 2], [3, 4]])
    p = tmp_path / "fresh.npz"
    t.save(str(p))
    loaded = RoutingTrace.load(str(p))
    assert loaded.widths_were_derived is False
    assert list(loaded.widths) == [2, 2]


# ---------------------------------------------------------------------------
# The env gate — the recorder must be inert by default
# ---------------------------------------------------------------------------

def test_the_recorder_is_inert_unless_the_env_var_is_set(monkeypatch):
    from glq_vllm.expert_trace import active_trace

    monkeypatch.delenv("GLQ_MOE_TRACE_ROUTING", raising=False)
    assert active_trace() is None, "tracing must cost nothing when it is not asked for"


def test_setting_the_env_var_activates_one_shared_trace(monkeypatch, tmp_path):
    from glq_vllm import expert_trace
    from glq_vllm.expert_trace import active_trace

    monkeypatch.setattr(expert_trace, "_TRACE", None)
    monkeypatch.setenv("GLQ_MOE_TRACE_ROUTING", str(tmp_path / "t.npz"))
    a = active_trace()
    b = active_trace()
    assert a is not None
    assert a is b, "every layer must append to the SAME trace, not one per layer"


# ---------------------------------------------------------------------------
# Surviving vLLM's shutdown
# ---------------------------------------------------------------------------

def test_the_trace_flushes_periodically_not_only_at_exit(tmp_path):
    """Measured on the L4: the run finished RUN_OK and wrote NOTHING, because `apply` runs in
    vLLM's EngineCore **subprocess** and the shutdown path *"force killing remaining process
    EngineCore"* — `atexit` does not survive SIGKILL. A trace held only in memory is a trace
    that is lost exactly when a long run ends.

    So the recorder re-saves atomically every `flush_every` records. Re-saving the whole file
    is wasteful in principle and irrelevant in practice (a few hundred KB, a handful of
    times), and atomicity means a kill mid-write cannot leave a partial file."""
    p = tmp_path / "flushing.npz"
    t = RoutingTrace(path=str(p), flush_every=4)
    for i in range(10):
        t.record("layers.0.mlp.experts", [i % 5, (i + 1) % 5])
    # 10 records with flush_every=4 => flushed at 4 and 8, so the file exists and holds >= 8
    assert p.exists(), "nothing was flushed; a killed process would lose the whole trace"
    on_disk = RoutingTrace.load(str(p))
    assert len(on_disk) >= 8


def test_a_flush_is_atomic_so_a_kill_cannot_leave_a_partial_file(tmp_path):
    """The reader is the scorer, and half an .npz is not loadable."""
    p = tmp_path / "atomic.npz"
    t = RoutingTrace(path=str(p), flush_every=2)
    for i in range(6):
        t.record("layers.0.mlp.experts", [i, i + 1])
    assert not list(tmp_path.glob("*.tmp*")), "a temp file was left behind"
    RoutingTrace.load(str(p))          # loads, i.e. was never partial


def test_no_path_means_no_flush_and_no_crash(tmp_path):
    """The in-memory use (tests, the scorer's own fixtures) must not need a path."""
    t = RoutingTrace(flush_every=2)
    for i in range(5):
        t.record("layers.0.mlp.experts", [i])
    assert len(t) == 5
