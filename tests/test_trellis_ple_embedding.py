"""Trellis row-decode for a gathered embedding table.

Trellis codes a sequence. `quantize(X)` transposes, so laying a PLE table out with each
**row** as its own length-`embedding_dim` tail-biting sequence makes rows independent — and
that independence is the whole point: a gather touches only the rows it asked for, with no
tile shared between neighbours.

The contract these pin:

  1. decoding a packed row reproduces exactly what the encoder produced in memory, so the
     stored checkpoint and the in-memory W_hat cannot drift;
  2. decoding a SUBSET of rows equals decoding all of them and slicing — the property a
     gather depends on, and the one that silently breaks if a tile ever spans rows.

Pure torch on CPU with fixed seeds, so the eventual fused kernel has an exact oracle.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantized_linear import _dequant_embedding_rows_trellis  # noqa: E402
from glq.quantize_model import _quantize_ple_chunk_trellis  # noqa: E402

WIDTH = 160          # Qwen4Exp PLE width: not a power of two, which is the point


def _table(rows=64, width=WIDTH, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, width) * 0.02


@pytest.mark.parametrize("K", [2, 3, 4])
def test_decoding_a_packed_row_reproduces_the_encoder(K):
    """Storage round-trip: what comes back is what the encoder computed, exactly."""
    W = _table()
    arts, w_hat = _quantize_ple_chunk_trellis(W, bpw=K, device="cpu")
    got = _dequant_embedding_rows_trellis(
        torch.arange(W.shape[0]), arts["trellis_packed"], arts["SV"],
        arts["Wscale"], arts["_codebook"], arts["_blocks_n"], WIDTH)
    assert torch.allclose(got.float(), w_hat.float(), atol=1e-3), (
        f"K={K}: decode does not match the encoder's W_hat")


def test_a_gathered_subset_equals_the_full_decode_sliced():
    """The property the whole row-wise layout exists for. If a trellis tile ever spanned
    16 rows (the weight-matrix layout), this would fail — gathering one row would depend on
    its neighbours."""
    W = _table(rows=64)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    args = (arts["trellis_packed"], arts["SV"], arts["Wscale"],
            arts["_codebook"], arts["_blocks_n"], WIDTH)

    full = _dequant_embedding_rows_trellis(torch.arange(64), *args)
    ids = torch.tensor([7, 3, 61, 3, 0])
    subset = _dequant_embedding_rows_trellis(ids, *args)
    assert torch.equal(subset, full[ids]), "a gathered row depends on its neighbours"


def test_repeated_ids_decode_identically():
    """A hashed n-gram table gathers the same row many times in one forward."""
    W = _table(rows=32)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    args = (arts["trellis_packed"], arts["SV"], arts["Wscale"],
            arts["_codebook"], arts["_blocks_n"], WIDTH)
    out = _dequant_embedding_rows_trellis(torch.tensor([5, 5, 5]), *args)
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_output_shape_follows_the_input_ids():
    """vLLM and HF both pass [...]-shaped ids and expect [..., embedding_dim] back."""
    W = _table(rows=32)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    ids = torch.randint(0, 32, (2, 5))
    out = _dequant_embedding_rows_trellis(
        ids, arts["trellis_packed"], arts["SV"], arts["Wscale"],
        arts["_codebook"], arts["_blocks_n"], WIDTH)
    assert out.shape == (2, 5, WIDTH)


def test_storage_is_the_advertised_size():
    """60 B/row at K=3 for a 160-wide row is what makes the table 17.9 GiB instead of 95.4.
    Shell would need a full Hadamard, padding 160 -> 256."""
    W = _table(rows=16)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    packed = arts["trellis_packed"]
    assert packed.dtype == torch.int16
    assert packed.numel() * 2 / W.shape[0] == 60.0
    # per-row scale, fp16: fp32 would cost 1.19 GiB over 320M rows
    assert arts["Wscale"].dtype == torch.float16
    assert arts["Wscale"].shape == (W.shape[0],)


def test_the_rht_does_not_pad_a_160_wide_row():
    """Guards the 1.6x footprint regression: a full Hadamard would pad to 256, and that is
    also the layout the trellis path cannot use."""
    W = _table(rows=16)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    assert sum(arts["_blocks_n"]) == WIDTH, arts["_blocks_n"]
    assert arts["SV"].numel() == WIDTH


def test_quantization_actually_preserves_the_table():
    """A decode that returns noise would still satisfy every shape assertion above."""
    W = _table(rows=256)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    got = _dequant_embedding_rows_trellis(
        torch.arange(256), arts["trellis_packed"], arts["SV"], arts["Wscale"],
        arts["_codebook"], arts["_blocks_n"], WIDTH).float()
    sqnr = 10 * torch.log10(W.pow(2).sum() / (W - got).pow(2).sum())
    assert sqnr > 12.0, f"SQNR {sqnr:.1f} dB — decode is not reconstructing the table"
