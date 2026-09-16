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


# ---- the nn.Module wrapper ----------------------------------------------------------------

def _build_module(rows=64, bpw=3):
    """Quantize a table, then load it into a fresh module the way a checkpoint would."""
    from glq.quantized_linear import TrellisRHTEmbedding
    W = _table(rows=rows)
    arts, w_hat = _quantize_ple_chunk_trellis(W, bpw=bpw, device="cpu")
    mod = TrellisRHTEmbedding(rows, WIDTH)
    sd = {k: v for k, v in arts.items() if not k.startswith("_")}
    sd["rht_blocks"] = torch.tensor(arts["_blocks_n"], dtype=torch.int32)
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    return mod, W, w_hat, missing, unexpected


def test_the_module_loads_a_checkpoint_shaped_state_dict():
    mod, _, _, missing, unexpected = _build_module()
    assert not unexpected, f"checkpoint keys the module does not declare: {unexpected}"
    assert not [m for m in missing if not m.startswith("_")], f"missing: {missing}"


def test_the_module_reproduces_the_encoder():
    """Module forward and the encoder's W_hat must agree, or the checkpoint and the
    in-memory result have quietly diverged."""
    mod, _, w_hat, _, _ = _build_module()
    out = mod(torch.arange(64))
    assert torch.allclose(out.float(), w_hat.float(), atol=1e-3)


def test_the_module_infers_the_rate_from_the_packed_shape():
    """cols == ceil(width*K/16) is the only record of K in a checkpoint. Guessing it wrong
    decodes garbage that still has the right shape."""
    for bpw in (2, 3, 4):
        mod, _, _, _, _ = _build_module(bpw=bpw)
        assert mod.K == bpw, f"inferred K={mod.K} for a {bpw} bpw table"


def test_a_gather_through_the_module_matches_a_full_decode():
    mod, _, _, _, _ = _build_module()
    ids = torch.tensor([9, 2, 63, 2])
    assert torch.equal(mod(ids), mod(torch.arange(64))[ids])


def test_embed_scale_is_applied():
    """Scaled-word-embedding archs multiply lookups inside their own forward; a substituted
    module has to reproduce that or every downstream activation is off by sqrt(dim)."""
    from glq.quantized_linear import TrellisRHTEmbedding
    mod, _, _, _, _ = _build_module()
    plain = mod(torch.arange(8)).float()
    mod2 = TrellisRHTEmbedding(64, WIDTH, embed_scale=2.0)
    mod2.load_state_dict(mod.state_dict(), strict=False)
    mod2.set_codebook(mod.codebook)
    assert torch.allclose(mod2(torch.arange(8)).float(), plain * 2.0, atol=1e-3)


@pytest.mark.parametrize("n", [160, 256, 320, 2560, 96, 1536])
def test_the_modules_default_block_layout_matches_the_rht(n):
    """`_pow2_blocks` is the module's pre-load default for the inverse transform. If it ever
    disagreed with RHT's own decomposition — in composition OR order — a module used before
    a checkpoint overwrote `rht_blocks` would inverse-transform against the wrong block
    structure and return plausible-looking noise."""
    from glq.rht import RHT
    from glq.quantized_linear import _pow2_blocks
    rht = RHT(8, n, device="cpu", block_diagonal=True, apply_left=False, e8p=False)
    assert list(rht.blocks_n) == _pow2_blocks(n)


# ---- HF substitution ----------------------------------------------------------------------

def test_a_trellis_payload_selects_the_trellis_module(tmp_path):
    """Which module replaces an nn.Embedding is decided by what the checkpoint actually
    stores, not by the run's --codebook.

    A checkpoint records its PLE payload as `.trellis_packed` or `.Qidxs`, and picking the
    wrong module is not a graceful failure: the shell module would look for Qidxs that is
    not there, or size a full-Hadamard buffer for an unpadded row.
    """
    # glq.hf_integration imports transformers at module scope (it registers the
    # quantizer), so this cannot run in the torch-only CI environment.
    pytest.importorskip("transformers", reason="not installed in CI")
    import torch.nn as nn
    from glq.hf_integration import replace_with_glq_embedding
    from glq.quantized_linear import E8RHTEmbedding, TrellisRHTEmbedding

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ple = nn.Embedding(64, WIDTH)
            self.other = nn.Embedding(64, WIDTH)

    m = _M()
    replace_with_glq_embedding(m, quantized_layers={"ple", "other"},
                               trellis_layers={"ple"})
    assert isinstance(m.ple, TrellisRHTEmbedding), type(m.ple)
    assert isinstance(m.other, E8RHTEmbedding), "non-trellis payload must stay on shell"


def test_no_trellis_set_keeps_every_embedding_on_shell():
    """Back-compat: existing gemma-4 checkpoints pass no trellis set and must be untouched."""
    # glq.hf_integration imports transformers at module scope (it registers the
    # quantizer), so this cannot run in the torch-only CI environment.
    pytest.importorskip("transformers", reason="not installed in CI")
    import torch.nn as nn
    from glq.hf_integration import replace_with_glq_embedding
    from glq.quantized_linear import E8RHTEmbedding

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ple = nn.Embedding(32, 256)

    m = _M()
    replace_with_glq_embedding(m, quantized_layers={"ple"})
    assert isinstance(m.ple, E8RHTEmbedding)


def test_the_rate_survives_assign_style_loading():
    """K must be read from the buffer, not cached at load time.

    transformers assigns checkpoint tensors straight onto the module (`assign=True`), which
    swaps the buffer and never calls `_load_from_state_dict`. A rate recovered only in that
    hook then keeps the constructor's default while the buffer holds a different rate, and
    the unpack reshapes to the wrong width — observed live as
    "shape '[10, 26880]' is invalid for input of size 358400" when a K=4 table met K=3.
    """
    from glq.quantized_linear import TrellisRHTEmbedding
    W = _table(rows=32)
    arts, w_hat = _quantize_ple_chunk_trellis(W, bpw=4, device="cpu")

    mod = TrellisRHTEmbedding(32, WIDTH, bpw=3)          # deliberately the WRONG default
    mod.trellis_packed = arts["trellis_packed"]          # assign, as transformers does
    mod.Wscale = arts["Wscale"]
    mod.SV = arts["SV"]
    mod.rht_blocks = torch.tensor(arts["_blocks_n"], dtype=torch.int32)

    assert mod.K == 4, f"rate not recovered from the buffer: got {mod.K}"
    out = mod(torch.arange(32))
    assert torch.allclose(out.float(), w_hat.float(), atol=1e-3)


@pytest.mark.parametrize("bpw", [2, 3, 4, 6])
def test_the_rate_is_read_from_the_buffer_at_every_rate(bpw):
    from glq.quantized_linear import TrellisRHTEmbedding
    arts, _ = _quantize_ple_chunk_trellis(_table(rows=16), bpw=bpw, device="cpu")
    mod = TrellisRHTEmbedding(16, WIDTH, bpw=3)
    mod.trellis_packed = arts["trellis_packed"]
    assert mod.K == bpw


def test_the_decode_returns_a_contiguous_tensor():
    """The op's real output must match what its fake/meta kernel promises.

    The fake returns `input_ids.new_empty((*shape, dim))` — contiguous. The real decode
    builds its result via `.reshape(dim, B).T`, which is stride (1, dim), and `.to(dtype)`
    preserves memory format by default, so the transposed layout reached the return.
    torch.compile compares the two and refuses:

        assert_size_stride(buf26, (s72, 10752), (10752, 1),
                           'torch.ops.glq.embedding_dequant_trellis.default')
        AssertionError: expected ... stride 1==10752 at dim=0

    Eager execution never checks this, which is why an `enforce_eager=True` serve test
    passed while vLLM's compiled path could not start at all.
    """
    W = _table(rows=64)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    out = _dequant_embedding_rows_trellis(
        torch.arange(64), arts["trellis_packed"], arts["SV"], arts["Wscale"],
        arts["_codebook"], arts["_blocks_n"], WIDTH)
    assert out.is_contiguous(), f"non-contiguous output, strides {out.stride()}"


def test_the_decode_matches_its_fake_kernels_shape_and_stride():
    """Pin the contract directly against the fake, since that is what torch.compile uses."""
    W = _table(rows=32)
    arts, _ = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    ids = torch.randint(0, 32, (5,))
    real = _dequant_embedding_rows_trellis(
        ids, arts["trellis_packed"], arts["SV"], arts["Wscale"],
        arts["_codebook"], arts["_blocks_n"], WIDTH)
    fake = ids.new_empty((*ids.shape, WIDTH), dtype=real.dtype)
    assert real.shape == fake.shape, (real.shape, fake.shape)
    assert real.stride() == fake.stride(), (real.stride(), fake.stride())


def test_the_decode_takes_block_sizes_as_ints_not_a_tensor():
    """Block sizes are static metadata, and reading them from a tensor forces a host sync.

    `blocks_n.tolist()` inside the op is a device-to-host copy, which CUDA-graph capture
    forbids outright:

        RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
                      unless the CPU tensor is pinned

    vLLM captures the embedding lookup, so the op has to be free of host syncs. Taking a
    list of ints keeps the sizes in the schema, where they belong, and removes the copy.
    """
    import inspect
    from glq.quantized_linear import _dequant_embedding_rows_trellis_fn as fn
    sig = inspect.signature(fn)
    assert "blocks_n" in sig.parameters
    W = _table(rows=16)
    arts, hat = _quantize_ple_chunk_trellis(W, bpw=3, device="cpu")
    # a plain list must work — no tensor, nothing to sync
    out = fn(torch.arange(16), arts["trellis_packed"], arts["SV"], arts["Wscale"],
             arts["_codebook"].cb.lut, [int(b) for b in arts["_blocks_n"]],
             WIDTH, int(arts["_codebook"].cb.L), int(arts["_codebook"].cb.K),
             int(arts["_codebook"].cb.V), 1.0, None)
    assert torch.allclose(out.float(), hat.float(), atol=1e-3)


def test_the_decode_performs_no_host_sync():
    """Guard the property directly: nothing in the decode may move data off-device.

    Asserted by counting `.tolist()`/`.item()`/`.cpu()` reachable in the function source —
    crude, but it fails loudly if someone reintroduces a sync, which otherwise only shows
    up as a CUDA-graph capture error on a GPU box nobody runs tests on.
    """
    import ast
    import inspect
    from glq import quantized_linear as ql

    tree = ast.parse(inspect.getsource(ql._dequant_embedding_rows_trellis_fn).lstrip())
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]          # drop the docstring: it *mentions* .tolist() by design
    calls = {n.func.attr for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    offenders = calls & {"tolist", "item", "cpu", "numpy"}
    assert not offenders, f"{sorted(offenders)} in the decode force a host sync"
