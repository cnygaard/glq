"""Sizing a checkpoint that is already on disk.

`glq-chat` and `glq-code` plan four things from a checkpoint's declared metadata — the
context window, PLE offload, expert offload and the expert arena — and every one of them came
from an HTTP call to huggingface.co. For `--model /path/to/checkpoint` that call could never
succeed: `checkpoint_offload_bytes` interpolates the path straight into a URL, producing
`https://huggingface.co//opt/dlami/nvme/.../config.json`. All three lookups swallow failure,
so the four plans collapsed together in silence.

Measured on an L4 before the fix (Qwen3.8-Flash-Next 3bpw, real snapshot directory):

    repo id    weights 73.66 GiB  ple 23.84  expert 42.63  ->  arena 3 GiB, window 131072
    local path weights None       ple 0.00   expert 0.00   ->  arena 0,     window 16384

and the serve env carried no `GLQ_MOE_EXPERT_CACHE` at all, so the server logged
`GLQ_MOE_EXPERT_CACHE unset: expert arena disabled` and decoded at 6.4 tok/s where the same
box with the arena does 15.2.

A token does NOT fix this half: the local arm is byte-identical with and without one, because
no amount of authorization makes that URL resolve.

The last test here is the one whose absence let this ship: every existing arena-env test calls
`child_env(expert_arena_gib=6, ...)` by hand, so nothing asserted that the supervisor passes
what it PLANNED. This one reads the env off the `spawn` seam instead.

Sparse files (`truncate`) give real `st_size` for a 77 GiB checkpoint at no disk cost, which is
what lets the fixture be the actual size the planner reacts to.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

GIB = 1024 ** 3

#: Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4, read live from the Hub 2026-10-12. The
#: planner's behaviour is specific to these magnitudes — 23.84 GiB of PLE beside 42.63 GiB of
#: experts is what makes a 22.49 GiB card need offload at all — so the fixture uses the real
#: declaration rather than round numbers.
PLE_BYTES = 25_598_029_824
EXPERT_BYTES = 45_768_458_240
NONTEXT_BYTES = 5_218_762_752
WEIGHTS_BYTES = 83_279_953_072
MAX_POS = 262_144


def _write_checkpoint(root: Path, *, shards=2, config=None) -> Path:
    """A checkpoint directory shaped like an HF cache snapshot: safetensors as symlinks into
    a sibling blobs/ directory, which is how `hf download` lays one out. Following those links
    is load-bearing — `lstat` would report the size of the link, not of the weights."""
    root.mkdir(parents=True, exist_ok=True)
    blobs = root.parent / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    per_shard, remainder = divmod(WEIGHTS_BYTES, shards)
    for i in range(shards):
        size = per_shard + (remainder if i == 0 else 0)
        blob = blobs / f"blob{i}"
        with open(blob, "wb") as fh:          # sparse: real st_size, no disk
            fh.truncate(size)
        (root / f"model-{i + 1:05d}-of-{shards:05d}.safetensors").symlink_to(blob)
    cfg = config if config is not None else {
        "max_position_embeddings": MAX_POS,
        "quantization_config": {
            "quant_method": "glq",
            "ple_offload_bytes": PLE_BYTES,
            "expert_offload_bytes": EXPERT_BYTES,
            "nontext_bytes": NONTEXT_BYTES,
        },
    }
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    # Noise a real snapshot carries, none of which is weights.
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "layer_metrics.json").write_text("{}", encoding="utf-8")
    return root


@pytest.fixture
def checkpoint(tmp_path):
    return _write_checkpoint(tmp_path / "snapshots" / "deadbeef")


@pytest.fixture
def no_network(monkeypatch):
    """Any HTTP call is a failure of this fix, not a slow test. The whole point is that the
    bytes are already on disk, so a lookup that reaches the Hub for a cached checkpoint is
    itself the bug — even when it happens to succeed.

    Patches `urlopen` as well as the module attribute, because `repo_size_bytes`,
    `model_max_len` and friends bind `fetch=_fetch_json` as a DEFAULT ARGUMENT, evaluated at
    import. Replacing `discovery._fetch_json` alone leaves those defaults pointing at the real
    one, so a test that believed it had cut the network would quietly make HTTP calls.
    """
    import urllib.request

    import glq.installer.discovery as D

    def boom(url, *a, **k):
        raise AssertionError(f"reached the network for a local checkpoint: {url}")

    monkeypatch.setattr(D, "_fetch_json", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    return boom


# ---------------------------------------------------------------------------
# local_fetcher: the seam
# ---------------------------------------------------------------------------

def test_the_local_fetcher_answers_the_tree_shape_with_real_sizes(checkpoint):
    """`repo_size_bytes` sums `.safetensors` entries from a tree listing. Answering that shape
    locally reuses its parsing instead of duplicating the sum — the two must not drift."""
    from glq.installer.discovery import local_fetcher

    tree = local_fetcher(checkpoint)("https://x/api/models/any/tree/main?recursive=true")
    shards = [f for f in tree if f["path"].endswith(".safetensors")]
    assert len(shards) == 2
    assert sum(f["size"] for f in shards) == WEIGHTS_BYTES


def test_the_local_fetcher_follows_symlinks_into_the_blob_store(checkpoint):
    """An HF cache snapshot is all symlinks. `lstat` would report a few dozen bytes per
    shard and the planner would conclude the model fits in VRAM."""
    from glq.installer.discovery import local_fetcher

    tree = local_fetcher(checkpoint)("https://x/api/models/any/tree/main?recursive=true")
    biggest = max(f["size"] for f in tree)
    assert biggest > 1 * GIB, f"read link sizes, not weight sizes: {biggest}"


def test_the_local_fetcher_answers_a_resolve_url_from_the_directory(checkpoint):
    from glq.installer.discovery import local_fetcher

    cfg = local_fetcher(checkpoint)("https://huggingface.co/any/resolve/main/config.json")
    assert cfg["max_position_embeddings"] == MAX_POS
    assert cfg["quantization_config"]["expert_offload_bytes"] == EXPERT_BYTES


def test_the_local_fetcher_REFUSES_a_url_it_cannot_serve(checkpoint):
    """Loudly, so a new call site that passes a collections or revisions URL fails in a test
    rather than silently receiving a None the caller reads as "unknown"."""
    from glq.installer.discovery import local_fetcher

    with pytest.raises(Exception):
        local_fetcher(checkpoint)("https://huggingface.co/api/collections/x")


def test_a_missing_file_is_an_error_not_an_empty_answer(tmp_path):
    """A directory with no config.json must not read as a checkpoint declaring nothing —
    that is exactly the silent collapse this fix exists to remove."""
    from glq.installer.discovery import local_fetcher

    (tmp_path / "empty").mkdir()
    with pytest.raises(Exception):
        local_fetcher(tmp_path / "empty")("https://h/any/resolve/main/config.json")


# ---------------------------------------------------------------------------
# The three lookups, on a directory, with the network poisoned
# ---------------------------------------------------------------------------

def test_checkpoint_bytes_reads_a_local_directory(checkpoint, no_network):
    from glq.chat import _checkpoint_bytes

    assert _checkpoint_bytes(str(checkpoint)) == WEIGHTS_BYTES


def test_checkpoint_offload_bytes_reads_a_local_directory(checkpoint, no_network):
    """★ The one that mattered most: `expert_offload_bytes` of 0 is what made
    `plan_arena_and_offload` return at its first guard."""
    from glq.chat import checkpoint_offload_bytes

    assert checkpoint_offload_bytes(str(checkpoint)) == (
        PLE_BYTES, EXPERT_BYTES, NONTEXT_BYTES)


def test_model_max_len_reads_a_local_directory(checkpoint, no_network):
    """The window clamp. Without it `glq-code` fell back to its 16384 floor on a checkpoint
    declaring 262144."""
    from glq.chat import _model_max_len

    assert _model_max_len(str(checkpoint)) == MAX_POS


def test_a_multimodal_wrapper_declares_its_window_on_text_config(tmp_path, no_network):
    """gemma-4 and Qwen3.5 put the text limits on `text_config`; `model_max_len` already
    knows that, and answering locally must go through it rather than re-reading the key."""
    from glq.chat import _model_max_len

    root = _write_checkpoint(tmp_path / "snapshots" / "beef", config={
        "text_config": {"max_position_embeddings": 131_072},
        "quantization_config": {"expert_offload_bytes": EXPERT_BYTES},
    })
    assert _model_max_len(str(root)) == 131_072


def test_a_repo_id_still_goes_to_the_hub(monkeypatch, tmp_path):
    """The local read must not capture the normal path. A repo id is not a directory, so it
    keeps using the injected fetcher."""
    import glq.installer.discovery as D

    seen = []

    def fetch(url, *a, **k):
        seen.append(url)
        return {"max_position_embeddings": 4096}

    monkeypatch.setattr(D, "_fetch_json", fetch)
    from glq.chat import _model_max_len

    assert _model_max_len("xv0y5ncu/some-repo") == 4096
    assert seen and "huggingface.co" in seen[0]


def test_a_path_that_does_not_exist_is_still_unknown_not_an_error(no_network):
    """`--model /typo` must degrade to today's behaviour, not raise out of a start. The
    network is poisoned here, which is the offline case too."""
    from glq.chat import _checkpoint_bytes, checkpoint_offload_bytes

    assert _checkpoint_bytes("/nonexistent/checkpoint") is None
    assert checkpoint_offload_bytes("/nonexistent/checkpoint") == (0, 0, 0)


# ---------------------------------------------------------------------------
# ★ The missing link: a local checkpoint reaches the server's env
# ---------------------------------------------------------------------------

def _serve_env(model, *, vram_gib=22.49, ram_gib=121.0):
    """Build the supervisor the way `glq-code` does, from the REAL sizing helpers, and return
    the env handed to `spawn`.

    Reading the env off `spawn` is the point. Every pre-existing arena-env test calls
    `child_env(expert_arena_gib=6, ...)` directly, so all of them passed while `glq-code`
    planned 0 and set nothing — and both end-to-end checks on the box had the same hole (one
    exported `GLQ_MOE_EXPERT_CACHE=3` in the shell, the other hand-fed `child_env`).
    """
    from glq.chat import checkpoint_offload_bytes, _checkpoint_bytes, _model_max_len
    from glq.code import DEFAULT_CODE_MAX_MODEL_LEN, DEFAULT_CODE_MAX_NUM_SEQS
    from glq.supervisor import WINDOW_OFFLOAD_MAX_EXTRA_GIB, VllmSupervisor

    captured = {}
    probes = {"n": 0}

    class _Proc:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def spawn(argv, **kw):
        captured["env"] = kw.get("env") or {}
        captured["argv"] = argv
        return _Proc()

    def probe(_base_url):
        # Unhealthy on the first probe so `start()` actually spawns, healthy afterwards so it
        # stops waiting. A probe that is healthy immediately makes the supervisor ATTACH to a
        # supposed existing server and never spawn — the env would then never be captured and
        # the assertions below would read an empty dict.
        probes["n"] += 1
        return probes["n"] > 1

    ple, expert, nontext = checkpoint_offload_bytes(model)
    sup = VllmSupervisor(
        model=model, port=8000,
        spawn=spawn, probe=probe,
        sleep=lambda s: None, monotonic=lambda: 0.0,
        log_path=Path(tempfile.mkdtemp()) / "vllm.log",
        device="cuda",
        out=io.StringIO(),
        weights_bytes=_checkpoint_bytes(model),
        vram_bytes=int(vram_gib * GIB),
        ram_bytes=int(ram_gib * GIB),
        model_max_len=_model_max_len(model),
        max_model_len_floor=DEFAULT_CODE_MAX_MODEL_LEN,
        max_num_seqs=DEFAULT_CODE_MAX_NUM_SEQS,
        window_concurrency=DEFAULT_CODE_MAX_NUM_SEQS,
        ple_offload_bytes=ple, expert_offload_bytes=expert, nontext_bytes=nontext,
        window_offload_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB,
    )
    started = sup.start()
    assert started is not False, "attached instead of spawning; no env was captured"
    assert "env" in captured, "start() never reached the spawn seam"
    return sup, captured


def test_a_local_checkpoint_PLANS_an_arena_and_the_server_gets_it(checkpoint, no_network):
    """★★ The regression test for the whole defect. A checkpoint directory on disk, a 22.49
    GiB card and a 121 GiB host — the configuration measured at 15.234 tok/s with the arena
    and 6.352 without — must reach `vllm serve` with the arena budget set.
    """
    sup, captured = _serve_env(str(checkpoint))

    assert sup.expert_arena_gib > 0, "planned no arena from a local checkpoint"
    env = captured["env"]
    assert env.get("GLQ_MOE_EXPERT_CACHE") == str(sup.expert_arena_gib), (
        f"planned {sup.expert_arena_gib} GiB but the server env says "
        f"{env.get('GLQ_MOE_EXPERT_CACHE')!r}")
    assert env.get("GLQ_MOE_ARENA_MAX_TOKENS") == str(sup.max_num_seqs)
    # The other three plans the same lookup feeds, all of which collapsed together.
    assert env.get("VLLM_PLE_CPU_OFFLOAD") == "1"
    assert sup.expert_offload_gib > 0
    assert sup.max_model_len > 16384, "window stayed at the floor"


def test_the_FAILING_CONTROL_an_unsizable_model_sets_nothing(no_network):
    """The control that makes the test above mean something: the same code path on a model it
    cannot size must still plan nothing and set nothing. Without this, a `child_env` that
    unconditionally exported a budget would satisfy the assertions above."""
    sup, captured = _serve_env("/nonexistent/checkpoint")

    assert sup.expert_arena_gib == 0
    assert "GLQ_MOE_EXPERT_CACHE" not in captured["env"]
    assert "GLQ_MOE_ARENA_MAX_TOKENS" not in captured["env"]


def test_an_exported_budget_still_wins_over_a_local_plan(checkpoint, no_network, monkeypatch):
    """`GLQ_MOE_EXPERT_CACHE=0` stays the documented opt-out once the plan is non-zero."""
    monkeypatch.setenv("GLQ_MOE_EXPERT_CACHE", "0")
    sup, captured = _serve_env(str(checkpoint))

    assert sup.expert_arena_gib > 0, "the plan itself should be unchanged"
    assert captured["env"]["GLQ_MOE_EXPERT_CACHE"] == "0"
