"""Trellis gate_up split: the packed buffer must survive being cut in half.

Gemma-4 routed experts are quantized JOINTLY as one ``[gate; up]`` matrix, then split
back under the ``gate_proj``/``up_proj`` keys. For E8/e8p artifacts axis 0 is rows (or
rows/16 mma tiles), so the split is a plain slice. ``trellis_packed`` is neither: it is
``[(m//16)*(n//16), ...]`` — a FLATTENED (row-block, col-block) index — and on top of that
``kernel_tile_flip`` permutes bytes with

    (m//16//2, 2, n//16//2, 2, 32, K) -> permute(0, 2, 4, 3, 1, 5)

which groups TWO 16-row blocks into one self-contained unit. So a cut at a 16-but-not-32
row boundary interleaves gate and up bytes, and would corrupt every expert silently — the
weights would still load and still decode, just to the wrong numbers.

These tests pin the decode, not the shapes: each half must decode to exactly the rows the
fused buffer decodes to.
"""
from __future__ import annotations

import pytest
import torch

trellis = pytest.importorskip("glq.trellis")


def _fused_artifacts(m, n, K=4, variant="3inst", for_kernel=True):
    """Pack a valid random trellis for an (m, n) layer, as pack_layer would.

    States are generated directly in the TILE domain (that is the domain pack_trellis
    validates) as a random tail-biting walk: each state shifts the previous one in by
    K*V bits, which is exactly the overlap invariant pack_trellis asserts. Random
    integers do not satisfy it and are rejected.
    """
    cb = trellis.TrellisCodebook(variant=variant, K=K, device="cpu")
    torch.manual_seed(0)
    num_tiles = (m // trellis.TD) * (n // trellis.TD)
    T, KV, L = trellis.TD * trellis.TD // cb.V, cb.K * cb.V, cb.L
    states = torch.empty(num_tiles, T, dtype=torch.int64)
    states[:, 0] = torch.randint(0, 2 ** L, (num_tiles,), dtype=torch.int64)
    for i in range(1, T):
        fresh = torch.randint(0, 2 ** KV, (num_tiles,), dtype=torch.int64)
        states[:, i] = ((states[:, i - 1] << KV) | fresh) & ((1 << L) - 1)
    packed = cb.pack_trellis(states)
    if for_kernel:
        packed = trellis.kernel_tile_flip(packed, m, n, cb.K, forward=True)
    return cb, packed


def _split(packed, gate_rows, up_rows, n):
    """The split under test, mirroring quantize_model._split_gate_up_arts."""
    from glq.quantize_model import split_trellis_packed
    return split_trellis_packed(packed, gate_rows, up_rows, n)


@pytest.mark.parametrize("for_kernel", [True, False])
def test_halves_decode_to_the_fused_rows(for_kernel):
    """The gate for silent corruption: decode(half) == decode(fused)[rows], exactly."""
    m, n, gate = 128, 64, 64
    cb, packed = _fused_artifacts(m, n, for_kernel=for_kernel)
    g, u = _split(packed, gate, m - gate, n)

    full = trellis.decode_layer(cb, packed, m, n, has_kernel=for_kernel)
    dg = trellis.decode_layer(cb, g, gate, n, has_kernel=for_kernel)
    du = trellis.decode_layer(cb, u, m - gate, n, has_kernel=for_kernel)

    assert torch.equal(dg, full[:gate]), "gate half does not decode to the fused rows"
    assert torch.equal(du, full[gate:]), "up half does not decode to the fused rows"


def test_split_is_exhaustive():
    """No bytes dropped or duplicated — the two halves must tile the buffer."""
    m, n, gate = 96, 32, 32
    _, packed = _fused_artifacts(m, n)
    g, u = _split(packed, gate, m - gate, n)
    assert g.shape[0] + u.shape[0] == packed.shape[0]
    assert g.shape[1] == packed.shape[1] == u.shape[1]


def test_refuses_a_cut_inside_a_row_block_pair():
    """48 rows is 16-aligned but NOT 32-aligned, so it lands inside an mma pair. This must
    raise: producing a plausible-looking buffer here is the corrupting outcome."""
    m, n = 128, 64
    _, packed = _fused_artifacts(m, n)
    with pytest.raises(AssertionError, match="32"):
        _split(packed, 48, 80, n)


def test_refuses_a_column_count_that_does_not_tile_the_buffer():
    """n is recovered from the shared SV artifact at the call site; if it disagrees with
    the packed shape the split silently mis-slices, so the size check has to be an
    assertion rather than a comment."""
    m, n = 128, 64
    _, packed = _fused_artifacts(m, n)
    with pytest.raises(AssertionError):
        _split(packed, 64, 64, n * 2)
