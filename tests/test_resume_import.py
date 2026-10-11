"""Rebuilding a resume store from a finished GLQ checkpoint.

Why this exists: a 77 GiB Flash-Next quantize is hours of GPU time, and the layers are
already published on HuggingFace. Re-sharding them into the resume layout lets a new run
replay what is done and quantize only what is new — adding the MTP draft head, say — instead
of starting at layer 0.

The shard layout is deliberately the same flattening the final checkpoint uses
(`state_dict[f"{layer_prefix}.{key}"]`), which is what makes this a re-sharding job rather
than a conversion: the tensors go across untouched.

Local safetensors throughout, no network and no GPU.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.resume import ResumeStore, shard_from_checkpoint  # noqa: E402

PREFIX = "model.language_model.layers"


def _write_checkpoint(d):
    """A two-layer checkpoint shaped like a real one: quantized artifacts, per-expert
    prefixes, plus unquantized tensors (norms, embeddings) that must NOT be sharded."""
    from safetensors.torch import save_file

    quantized = {}
    for layer in (0, 1):
        quantized[f"{PREFIX}.{layer}.self_attn.q_proj"] = ("Qidxs", "SV", "Wscale")
        for e in range(2):
            quantized[f"{PREFIX}.{layer}.mlp.experts.{e}.gate_proj"] = ("Qidxs", "SV")
            quantized[f"{PREFIX}.{layer}.mlp.experts.{e}.down_proj"] = ("Qidxs", "SV")

    tensors = {}
    for p, keys in quantized.items():
        for k in keys:
            tensors[f"{p}.{k}"] = torch.ones(4, 2, dtype=torch.int16)
    # Not quantized: these live in the final checkpoint but belong to no resume shard.
    tensors[f"{PREFIX}.0.input_layernorm.weight"] = torch.ones(4)
    tensors["model.language_model.embed_tokens.weight"] = torch.ones(4, 4)

    save_file(tensors, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "layer_metrics.json"), "w") as f:
        json.dump({p: {"sqnr": 60.0, "bpw": 3, "Wscale": 1.0, "proxy_loss": 0.5}
                   for p in quantized}, f)
    return sorted(quantized)


def test_the_rebuilt_store_reports_the_layers_as_done(tmp_path):
    """The whole point: `next_layer()` must skip what the checkpoint already contains, so a
    resumed run starts where the published work ended."""
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    store = ResumeStore(local_dir=str(tmp_path / "store"), run_key="rk")
    written = shard_from_checkpoint(checkpoint_dir=str(src), store=store,
                                    sd_prefix=PREFIX)
    assert written == {0, 1}
    assert store.completed_layers() == {0, 1}
    assert store.next_layer() == 2


def test_artifacts_round_trip_to_the_prefixes_replay_asks_for(tmp_path):
    """Replay looks artifacts up by exact prefix — `_restore_replayed_weights` does
    `arts_map.get(f"{prefix}.{name}")` for linears and `...experts.{e}.gate_proj` for stacked
    experts. A prefix that does not round-trip restores nothing, silently."""
    src = tmp_path / "ckpt"
    src.mkdir()
    expected = _write_checkpoint(str(src))
    store = ResumeStore(local_dir=str(tmp_path / "store"), run_key="rk")
    shard_from_checkpoint(checkpoint_dir=str(src), store=store, sd_prefix=PREFIX)

    arts, metrics, losses = store.load_layer(0)
    assert sorted(arts) == [p for p in expected if p.startswith(f"{PREFIX}.0.")]
    assert sorted(arts[f"{PREFIX}.0.self_attn.q_proj"]) == ["Qidxs", "SV", "Wscale"]
    assert torch.equal(arts[f"{PREFIX}.0.self_attn.q_proj"]["Qidxs"],
                       torch.ones(4, 2, dtype=torch.int16))
    assert metrics[f"{PREFIX}.0.self_attn.q_proj"]["bpw"] == 3
    assert losses[f"{PREFIX}.0.self_attn.q_proj"] == 0.5


def test_unquantized_tensors_are_not_sharded(tmp_path):
    """Norms and embeddings are copied from the source on save, not replayed. Putting them
    in a shard would make `load_layer` hand replay a prefix with no module behind it."""
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    store = ResumeStore(local_dir=str(tmp_path / "store"), run_key="rk")
    shard_from_checkpoint(checkpoint_dir=str(src), store=store, sd_prefix=PREFIX)
    arts, _, _ = store.load_layer(0)
    assert not [p for p in arts if "layernorm" in p or "embed_tokens" in p]


def test_a_wrong_sd_prefix_is_an_error_not_an_empty_store(tmp_path):
    """The failure this is most likely to hit in practice — `model.layers` against a wrapped
    `model.language_model.layers` checkpoint. Writing zero shards and returning quietly would
    look like success and send a re-quantize back to layer 0 at full cost."""
    import pytest
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    store = ResumeStore(local_dir=str(tmp_path / "store"), run_key="rk")
    with pytest.raises(ValueError, match="sd_prefix"):
        shard_from_checkpoint(checkpoint_dir=str(src), store=store,
                              sd_prefix="model.layers")


def _write_configs(d, *, bpw=3, variant="3inst", wrapped=True):
    """The two files a published GLQ checkpoint ships, with the fields measured off
    xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4."""
    with open(os.path.join(d, "quantize_config.json"), "w") as f:
        json.dump({"quant_method": "glq", "codebook": "trellis", "bpw": bpw,
                   "nsamples": 128, "seqlen": 2048,
                   "source_model": "Qwen/Qwen3.8-Flash-Next"}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"quantization_config": {"codebook": "trellis", "bpw": bpw,
                                           "variant": variant}}, f)


def test_the_sd_prefix_is_detected_from_the_checkpoint(tmp_path):
    """Asking the user for it invites the one mistake that matters — `model.layers` against a
    wrapped checkpoint — and the answer is already in the file."""
    from glq.resume import detect_sd_prefix
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    assert detect_sd_prefix(str(src)) == PREFIX


def test_the_run_key_is_derived_from_the_checkpoints_own_settings(tmp_path):
    """Every field `compute_run_key` hashes is recorded in the checkpoint except the glq
    version, which must be the version of the run about to RESUME — not the older one that
    produced the published weights. Deriving beats asking: a mistyped nsamples yields a store
    the resuming run refuses, after the download."""
    from glq.resume import compute_run_key, derive_run_key
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    _write_configs(str(src))
    key = derive_run_key(str(src), version="0.8.26")
    assert key == compute_run_key(
        model="Qwen/Qwen3.8-Flash-Next", codebook="trellis", bpw=3, nsamples=128,
        seqlen=2048, variant="3inst", sd_prefix=PREFIX, version="0.8.26", bpw_map=None)


def test_a_fractional_bpw_checkpoint_is_refused(tmp_path):
    """Mixed-precision bpw comes from the sensitivity allocator, and `compute_run_key` hashes
    the *input* map — which the checkpoint does not record, only the resulting per-layer
    assignment. Guessing it would produce a plausible key that silently addresses the wrong
    manifest, so refuse and say why."""
    import pytest
    from glq.resume import derive_run_key
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    _write_configs(str(src), bpw=3.5)
    with pytest.raises(ValueError, match="mixed|fractional"):
        derive_run_key(str(src), version="0.8.26")


def test_the_cli_carries_the_flags_the_workflow_needs(tmp_path):
    """Asserted on the parser: the point of the tool is to avoid hand-entering settings that
    must match exactly, so `--checkpoint` and `--store-dir` are the required pair and the
    rest exist only as escape hatches."""
    from glq.resume import build_import_parser
    p = build_import_parser()
    dests = {a.dest for a in p._actions}
    assert {"checkpoint", "store_dir", "bucket", "prefix", "run_key",
            "sd_prefix", "layers"} <= dests
    args = p.parse_args(["--checkpoint", "x", "--store-dir", "y"])
    assert args.run_key is None and args.sd_prefix is None   # derived, not demanded


def test_a_layer_subset_can_be_restricted(tmp_path):
    """Re-sharding 48 layers of a 77 GiB checkpoint is a lot of I/O to repeat while getting
    the run key right, so the tool can do a couple of layers as a dry run."""
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    store = ResumeStore(local_dir=str(tmp_path / "store"), run_key="rk")
    assert shard_from_checkpoint(checkpoint_dir=str(src), store=store,
                                 sd_prefix=PREFIX, layers=[0]) == {0}
    assert store.completed_layers() == {0}
    assert store.next_layer() == 1


def test_a_store_can_be_built_for_a_run_that_will_add_an_mtp_head(tmp_path):
    """The composition of items 2 and 3, which does not work without this.

    `compute_run_key` includes `mtp_bpw`, so a run launched with `--mtp-bpw 3` derives a
    different key than this tool's default — and `ResumeStore` then REFUSES the store. That
    breaks the one workflow the tool exists for: reuse the published decoder layers in order
    to add a draft head without re-quantizing 48 layers.

    Note what the key is actually gating. The shards hold decoder layers only; the MTP
    post-pass runs after the loop and is never checkpointed or replayed, so `mtp_bpw` does
    not change a single stored byte. Declaring the intended head bpw here is therefore
    recording intent, not content."""
    from glq.resume import compute_run_key, derive_run_key
    src = tmp_path / "ckpt"
    src.mkdir()
    _write_checkpoint(str(src))
    _write_configs(str(src))
    key = derive_run_key(str(src), version="0.8.26", mtp_bpw=3)
    assert key == compute_run_key(
        model="Qwen/Qwen3.8-Flash-Next", codebook="trellis", bpw=3, nsamples=128,
        seqlen=2048, variant="3inst", sd_prefix=PREFIX, version="0.8.26",
        bpw_map=None, mtp_bpw=3)
    assert key != derive_run_key(str(src), version="0.8.26")


def test_the_cli_can_declare_the_intended_mtp_bpw(tmp_path):
    from glq.resume import build_import_parser
    args = build_import_parser().parse_args(
        ["--checkpoint", "x", "--store-dir", "y", "--mtp-bpw", "3"])
    assert args.mtp_bpw == 3
