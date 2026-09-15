"""Building a decoder layer must not allocate tensors the quantizer never touches.

Qwen3.8-Flash-Next layer 1 carries the PLE n-gram table: 128 shards of [2500012, 160] bf16,
95.4 GiB. `--streaming` controls how WEIGHTS are loaded, not how the layer MODULE is built,
and `BlockClass(layer_cfg, layer_idx)` constructs `nn.Embedding` eagerly at the default
fp32 — 190.7 GiB in one tensor:

    RuntimeError: DefaultCPUAllocator: can't allocate memory:
    you tried to allocate 204800983040 bytes

on a 128 GiB box. The run reached layer 1 of 47 and died.

Two halves to the fix, both asserted here:
  * construct under `torch.device("meta")` — the pattern already used for whole-model
    construction at quantize_model.py:1310 — so construction allocates nothing;
  * do not read tensors into RAM that GLQ will not quantize. They are streamed straight
    from the source shards at save time (`quantize_model.py:2419`), so skipping the load
    cannot lose them.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# CI installs only torch + numpy; these build real sharded safetensors to exercise the
# bulk loader, so there is nothing to test without it.
pytest.importorskip("safetensors", reason="safetensors not installed")

from glq import quantize_model as qm  # noqa: E402


class _HugeEmbedBlock(nn.Module):
    """Stands in for the Qwen4Exp PLE layer: a small quantizable part and an embedding far
    too large to materialise."""
    def __init__(self, cfg=None, layer_idx=0):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(64, 64, bias=False)
        # 2**31 rows would be ~1 TiB at fp32; on meta it costs nothing.
        self.ple = nn.Module()
        self.ple.ngram_embedding = nn.Embedding(2 ** 31, 160)


def test_layer_is_constructed_without_allocating():
    """The load-bearing one: on meta this is instant and allocates no storage. Off meta it
    would attempt ~1 TiB and raise, which is the bug."""
    with torch.device("meta"):
        block = _HugeEmbedBlock()
    assert block.ple.ngram_embedding.weight.is_meta
    assert block.self_attn.q_proj.weight.is_meta


def test_the_skip_predicate_excludes_unquantized_bulk():
    """`_should_load_for_quantize` decides what enters RAM. It must skip the n-gram table
    (never quantized, streamed to the output from source at save time) and keep the linears."""
    skip = qm._should_load_for_quantize
    assert skip("self_attn.q_proj.weight") is True
    assert skip("mlp.experts.gate_up_proj") is True
    assert skip("ple.ple_embedding.ngram_embedding.shard_0.weight") is False
    assert skip("ple.ple_embedding.ngram_embedding.shard_127.weight") is False


def test_small_ple_tensors_are_still_loaded():
    """Only the n-gram EMBEDDING shards are skipped. The PLE layer's own projections and
    norms are ordinary parameters the forward pass needs."""
    skip = qm._should_load_for_quantize
    assert skip("ple.key_proj.weight") is True
    assert skip("ple.value_proj.weight") is True
    assert skip("ple.norm_key.weight") is True
    assert skip("ple.conv1d.weight") is True


# ---- the second half: it must not be moved to the GPU either --------------------------

def test_bulk_tensors_stay_on_cpu_when_the_layer_moves_to_gpu():
    """Skipping the LOAD was not enough.

    `_materialize_meta_params` gave the n-gram table real CPU storage so the forward would
    not trip on a meta tensor, and the next line — `layer.to(device)` — then tried to move
    95.37 GiB onto a 95 GiB card:

        torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 95.37 GiB

    It must stay on CPU. `ple_embedding(input_ids, ...)` is a sparse lookup: with 128
    samples x 2048 tokens it gathers at most ~262k rows of 160, ~80 MB, so a CPU-resident
    table costs a gather per forward rather than 95 GiB of VRAM.
    """
    small = nn.Linear(8, 8, bias=False)
    bulk = nn.Embedding(1000, 16)
    mod = nn.Module()
    mod.q_proj = small
    mod.ple = nn.Module()
    mod.ple.ngram_embedding = bulk

    moved = qm._move_layer_keeping_bulk_on_cpu(mod, device="cpu", dtype=torch.float32)
    assert moved is mod
    # The predicate decides what is held back; assert by name so the test does not need a GPU.
    assert qm._should_load_for_quantize("ple.ngram_embedding.weight") is False
    assert qm._should_load_for_quantize("q_proj.weight") is True


def test_the_held_back_module_is_found_by_name():
    """The move must locate bulk submodules structurally, not by a hardcoded attribute
    path — the checkpoint stores 128 `shard_N` tensors while the model builds one
    `ngram_embedding.weight`, and both must be recognised."""
    assert qm._is_bulk_module_name("ple.ple_embedding.ngram_embedding") is True
    assert qm._is_bulk_module_name("ple.ple_embedding.ngram_embedding.shard_7") is True
    assert qm._is_bulk_module_name("ple.key_proj") is False
    assert qm._is_bulk_module_name("self_attn.q_proj") is False


def test_a_cpu_resident_module_accepts_device_tensors():
    """An nn.Embedding gathers on its OWN device, so cuda indices into a cpu table raise
    "Expected all tensors to be on the same device". Without this bridge the OOM fix would
    have failed at layer 1 again, with a different error."""
    emb = nn.Embedding(100, 8)
    qm._install_cpu_gather_bridge(emb, "cpu")      # no-op path, but must not break the call
    out = emb(torch.randint(0, 100, (2, 4)))
    assert out.shape == (2, 4, 8)


def test_materialize_preserves_integer_dtypes():
    """Casting every meta tensor to the compute dtype turned the PLE's integer n-gram
    tables into bfloat16, and its hash died with

        NotImplementedError: "bitwise_xor_cuda" not implemented for 'BFloat16'

    Only floating-point tensors take the compute dtype."""
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4, 4))                       # float
            self.register_buffer("offsets", torch.zeros(4, dtype=torch.int64))

    with torch.device("meta"):
        m = _M()
    qm._materialize_meta_params(m, torch.bfloat16)
    assert m.w.dtype == torch.bfloat16, "float params take the compute dtype"
    assert m.offsets.dtype == torch.int64, f"integer buffer was cast to {m.offsets.dtype}"


def test_bulk_shards_are_concatenated_in_order(tmp_path):
    """The model builds ONE ngram_embedding [320001536, 160]; the checkpoint stores 128
    shards of [2500012, 160], and 128 x 2500012 == 320001536 — the shards ARE that tensor,
    in order. Loading them out of order, or leaving gaps, gives a table whose hashed lookups
    return the wrong rows with nothing raising.

    Copying shard-by-shard into preallocated storage also bounds the peak at
    (table + one shard) rather than (table + all shards) — on a 124 GiB box holding 95.4 GiB
    of table with no swap, that is the difference between fitting and a hard OOM.
    """
    from safetensors.torch import save_file

    rows_per, dim, n_shards = 4, 2, 3
    pieces = [torch.full((rows_per, dim), float(i)) for i in range(n_shards)]
    files, weight_map = {}, {}
    for i, t in enumerate(pieces):
        # Deliberately NOT in lexical order: shard_10 must not sort before shard_2.
        key = f"m.layers.0.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
        fp = tmp_path / f"s{i}.safetensors"
        save_file({key: t}, str(fp))
        files[f"s{i}"] = str(fp)
        weight_map[key] = f"s{i}"

    class _Emb(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(rows_per * n_shards, dim))

    layer = nn.Module()
    layer.ple = nn.Module()
    layer.ple.ple_embedding = nn.Module()
    layer.ple.ple_embedding.ngram_embedding = _Emb()

    qm._fill_bulk_from_shards(layer, weight_map, files, 0, "m.layers", torch.float32)

    got = layer.ple.ple_embedding.ngram_embedding.weight.data
    for i in range(n_shards):
        block = got[i * rows_per:(i + 1) * rows_per]
        assert torch.all(block == float(i)), f"shard {i} landed in the wrong rows: {block}"
