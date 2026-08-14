"""Per-layer resume checkpoints for `glq-quantize` (glq/resume.py).

A 30B quantize is ~2h and writes nothing until the final save_file(), so a spot reclaim at
1h50m loses every layer. ResumeStore persists each layer's artifacts as they complete,
locally and (optionally) to a private S3 bucket that outlives the box.

The tests that matter here are the *failure* ones. A resume store that works when
everything succeeds is easy; the value is entirely in what happens when the box dies
mid-upload, because that is the case it exists for. Two invariants:

  * a layer is marked done in the manifest ONLY after its shard is fully stored — a crash
    mid-upload must leave it not-done, never half-done, or resume silently continues from a
    truncated layer and the checkpoint is quietly corrupt;
  * a shard whose sha256 does not match the manifest is treated as missing rather than
    loaded — same reasoning.

No AWS, no GPU, no network: the S3 backend is injected.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Shards are safetensors files, and safetensors lives in the `quantize` extra — CI installs
# only torch + glq[hub], so skip rather than fail. resume.py imports it lazily, so the
# module below still imports fine; every test here writes a shard, so skip the whole file.
pytest.importorskip("safetensors")  # noqa: E402

from glq import resume as R  # noqa: E402


class _FakeBackend:
    """In-memory stand-in for the S3 backend, with a controllable failure point."""

    def __init__(self, fail_on=None):
        self.objects: dict[str, bytes] = {}
        self.fail_on = fail_on          # substring of key that should raise on put
        self.puts: list[str] = []

    def put(self, local_path, key):
        if self.fail_on and self.fail_on in key:
            raise RuntimeError(f"simulated upload failure for {key}")
        with open(local_path, "rb") as f:
            self.objects[key] = f.read()
        self.puts.append(key)

    def get(self, key, local_path):
        if key not in self.objects:
            raise FileNotFoundError(key)
        with open(local_path, "wb") as f:
            f.write(self.objects[key])

    def exists(self, key):
        return key in self.objects


def _arts(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "model.layers.3.self_attn.q_proj": {
            "Qidxs": torch.randint(0, 255, (8, 16), generator=g, dtype=torch.uint8),
            "SU": torch.randn(16, generator=g),
        },
        "model.layers.3.mlp.down_proj": {
            "Qidxs": torch.randint(0, 255, (4, 32), generator=g, dtype=torch.uint8),
            "SV": torch.randn(32, generator=g),
        },
    }


def _store(tmp_path, backend=None, run_key="rk-abc"):
    return R.ResumeStore(local_dir=str(tmp_path), run_key=run_key,
                         bucket=("b" if backend else None), prefix="p", backend=backend)


def test_layer_round_trip_is_bit_identical(tmp_path):
    """Artifacts must come back exactly — a resumed run writes these straight into the
    final checkpoint, so any drift is silent corruption."""
    s = _store(tmp_path)
    arts = _arts()
    s.save_layer(3, arts, metrics={"m": 1.0}, proxy_losses={"p": 2.0})
    got_arts, got_metrics, got_losses = s.load_layer(3)
    assert set(got_arts) == set(arts)
    for prefix, tensors in arts.items():
        for k, t in tensors.items():
            assert torch.equal(got_arts[prefix][k], t), f"{prefix}.{k} changed"
    assert got_metrics == {"m": 1.0}
    assert got_losses == {"p": 2.0}


def test_completed_layers_starts_empty_and_tracks_saves(tmp_path):
    s = _store(tmp_path)
    assert s.completed_layers() == set()
    assert s.next_layer() == 0
    s.save_layer(0, _arts(0))
    s.save_layer(1, _arts(1))
    assert s.completed_layers() == {0, 1}
    assert s.next_layer() == 2


def test_next_layer_stops_at_the_first_gap(tmp_path):
    """Resume replays 0..k-1 sequentially, so a gap means everything after it must be
    redone — returning max()+1 would skip a layer entirely."""
    s = _store(tmp_path)
    for i in (0, 1, 3, 4):
        s.save_layer(i, _arts(i))
    assert s.next_layer() == 2


def test_layer_not_marked_done_when_upload_fails(tmp_path):
    """THE invariant: the manifest is written after the shard is stored. A box dying
    mid-upload must leave the layer not-done so it is re-quantized, not resumed from a
    truncated object."""
    backend = _FakeBackend(fail_on="layer_0005")
    s = _store(tmp_path, backend)
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        s.save_layer(5, _arts(5))
    assert 5 not in s.completed_layers()
    # and a fresh store over the same dir agrees (the manifest on disk is authoritative)
    assert 5 not in _store(tmp_path, backend).completed_layers()


def test_uploads_precede_the_manifest_write(tmp_path):
    """Ordering is the mechanism behind the invariant above; assert it directly rather
    than trusting the happy path."""
    backend = _FakeBackend()
    s = _store(tmp_path, backend)
    s.save_layer(2, _arts(2))
    manifest_idx = backend.puts.index("p/manifest.json")
    shard_idx = backend.puts.index("p/layer_0002.safetensors")
    assert shard_idx < manifest_idx, "shard must be uploaded before the manifest"


def test_corrupt_shard_is_treated_as_missing(tmp_path):
    """A truncated shard that still parses would produce a wrong checkpoint. sha256 in the
    manifest turns that into a re-quantize instead."""
    s = _store(tmp_path)
    s.save_layer(7, _arts(7))
    shard = os.path.join(str(tmp_path), "layer_0007.safetensors")
    with open(shard, "r+b") as f:          # corrupt a byte in the payload
        f.seek(os.path.getsize(shard) - 1)
        f.write(b"\x00")
    s2 = _store(tmp_path)
    assert 7 not in s2.completed_layers()


def test_run_key_mismatch_refuses_to_resume(tmp_path):
    """Resuming a 4bpw manifest into a 3bpw run would blend two configurations into one
    checkpoint with no error anywhere. Refuse loudly."""
    _store(tmp_path, run_key="rk-4bpw").save_layer(0, _arts(0))
    with pytest.raises(R.ResumeKeyMismatch):
        _store(tmp_path, run_key="rk-3bpw").completed_layers()


def test_run_key_covers_the_settings_that_change_output():
    """Two runs that would produce different weights must not share a manifest."""
    base = dict(model="m", codebook="trellis", bpw=4, nsamples=128, seqlen=2048,
                variant="3inst", sd_prefix="model.layers", version="0.8.2", bpw_map=None)
    k = R.compute_run_key(**base)
    assert k == R.compute_run_key(**base)                      # deterministic
    for field, other in [("bpw", 3), ("codebook", "e8p"), ("nsamples", 64),
                         ("seqlen", 4096), ("variant", "hyb"), ("model", "n")]:
        assert R.compute_run_key(**{**base, field: other}) != k, f"{field} not in run_key"


def test_local_only_store_needs_no_backend(tmp_path):
    """No bucket configured must still checkpoint locally — that alone survives a process
    crash or OOM kill, just not loss of the instance."""
    s = R.ResumeStore(local_dir=str(tmp_path), run_key="rk")
    s.save_layer(0, _arts(0))
    assert s.completed_layers() == {0}


def test_missing_local_shard_is_fetched_from_the_bucket(tmp_path):
    """The whole point: a NEW box has an empty local dir and must recover from S3."""
    backend = _FakeBackend()
    _store(tmp_path, backend).save_layer(4, _arts(4))
    fresh = tmp_path / "other-box"
    fresh.mkdir()
    s2 = R.ResumeStore(local_dir=str(fresh), run_key="rk-abc", bucket="b", prefix="p",
                       backend=backend)
    assert s2.completed_layers() == {4}
    got, _, _ = s2.load_layer(4)
    assert torch.equal(got["model.layers.3.mlp.down_proj"]["Qidxs"],
                       _arts(4)["model.layers.3.mlp.down_proj"]["Qidxs"])


def test_manifest_is_valid_json_with_the_run_key(tmp_path):
    s = _store(tmp_path)
    s.save_layer(0, _arts(0))
    with open(os.path.join(str(tmp_path), "manifest.json")) as f:
        m = json.load(f)
    assert m["run_key"] == "rk-abc"
    assert "0" in m["layers"] and "sha256" in m["layers"]["0"]
