"""Reading PLE rows without holding the table in RAM.

The existing loader calls ``safe_open(...).get_tensor(key)`` — the whole tensor. That is
fine for gemma-4's ~4 GB PLE and fatal for Qwen4Exp's, which is **95.4 GiB across 128
shards** on boxes with 62–128 GiB of RAM. An earlier run died exactly this way, and with no
swap the machine became unreachable rather than raising.

So the quantizer reads row ranges instead, and the interesting case is a range that spans a
shard boundary — get it wrong and rows are silently duplicated or dropped, which shows up
as a quality regression nobody can locate rather than as an error.

Real safetensors files on disk, because the bug this guards lives in the shard-offset
arithmetic, not in a mock.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _ple_row_reader  # noqa: E402

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
WIDTH = 160


@pytest.fixture
def sharded(tmp_path):
    """4 shards of uneven length — the last shard being short is the real layout."""
    from safetensors.torch import save_file
    torch.manual_seed(0)
    sizes = [7, 7, 7, 3]
    full = torch.randn(sum(sizes), WIDTH, dtype=torch.float32)
    weight_map, shard_paths, row = {}, {}, 0
    for i, n in enumerate(sizes):
        key = f"{PREFIX}.shard_{i}.weight"
        fn = f"shard{i}.safetensors"
        path = tmp_path / fn
        save_file({key: full[row:row + n].contiguous()}, str(path))
        weight_map[key] = fn
        shard_paths[fn] = str(path)
        row += n
    return weight_map, shard_paths, full


def test_reports_the_tables_true_shape(sharded):
    weight_map, shard_paths, full = sharded
    rows, width, _ = _ple_row_reader(weight_map, shard_paths,
                                     {"prefix": PREFIX, "shards": 4})
    assert (rows, width) == tuple(full.shape)


def test_a_range_inside_one_shard_matches(sharded):
    weight_map, shard_paths, full = sharded
    _, _, read = _ple_row_reader(weight_map, shard_paths,
                                 {"prefix": PREFIX, "shards": 4})
    assert torch.equal(read(2, 5), full[2:5])


def test_a_range_spanning_a_shard_boundary_matches(sharded):
    """The case the arithmetic exists for."""
    weight_map, shard_paths, full = sharded
    _, _, read = _ple_row_reader(weight_map, shard_paths,
                                 {"prefix": PREFIX, "shards": 4})
    assert torch.equal(read(5, 10), full[5:10]), "boundary 7 crossed wrongly"


def test_a_range_spanning_three_shards_matches(sharded):
    weight_map, shard_paths, full = sharded
    _, _, read = _ple_row_reader(weight_map, shard_paths,
                                 {"prefix": PREFIX, "shards": 4})
    assert torch.equal(read(3, 18), full[3:18])


def test_reading_every_chunk_in_sequence_reconstructs_the_table(sharded):
    """What the quantizer actually does: walk the table in fixed chunks. Any off-by-one in
    the offsets shows up here as a row that is duplicated or missing."""
    weight_map, shard_paths, full = sharded
    n, _, read = _ple_row_reader(weight_map, shard_paths,
                                 {"prefix": PREFIX, "shards": 4})
    for chunk in (1, 4, 5, 24):
        got = torch.cat([read(r, min(r + chunk, n)) for r in range(0, n, chunk)])
        assert torch.equal(got, full), f"chunk={chunk}"


def test_an_unsharded_table_reads_through_the_same_interface(tmp_path):
    """gemma-4 stores one tensor. It must go through the same lazy path rather than a second
    code path that loads it whole."""
    from safetensors.torch import save_file
    key = "model.language_model.embed_tokens_per_layer.weight"
    full = torch.randn(11, WIDTH)
    save_file({key: full}, str(tmp_path / "m.safetensors"))
    n, width, read = _ple_row_reader(
        {key: "m.safetensors"}, {"m.safetensors": str(tmp_path / "m.safetensors")},
        {"prefix": "model.language_model.embed_tokens_per_layer"})
    assert (n, width) == (11, WIDTH)
    assert torch.equal(read(4, 9), full[4:9])
