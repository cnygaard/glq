"""Gate 0 for YAQA 2-sided LDLQ (arXiv 2505.22988) on GLQ's trellis codebook.

YAQA replaces LDLQ's single input-side Hessian with a Kronecker-factored approximation of
the full-model KL Hessian, giving two factors — ``Lin`` (n×n) and ``Lout`` (m×m) — so error
feedback runs along BOTH axes. Reference: ``yaqa-quantization/lib/algo/ldlq.py:16``.

These are CPU-only and need no model or calibration data. They exist because the two ways
this port can go wrong are both silent:

* it can quietly stop being a generalization of the code that produced every shipped
  checkpoint (caught by the ``Lout=0`` bit-exactness test), and
* the 2D traversal can read a tile that has not been quantized yet, which does not raise —
  it just makes the output slightly worse (caught by the ordering test).

The third test is the go/no-go: on a Hessian that is *exactly* Kronecker, 2-sided feedback
must beat 1-sided. If it cannot win there, no amount of real calibration will save it.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import trellis as gt  # noqa: E402


# ---------------------------------------------------------------------------
# traversal: the ordering invariant is the whole correctness argument
# ---------------------------------------------------------------------------
def test_antidiag_order_covers_every_tile_exactly_once():
    for M, N in [(1, 1), (1, 4), (4, 1), (3, 5), (6, 4), (8, 8)]:
        order = [t for diag in gt._antidiag_tile_order(M, N) for t in diag]
        assert sorted(order) == sorted((i, j) for i in range(M) for j in range(N)), \
            f"tile coverage wrong for {M}x{N}"
        assert len(order) == len(set(order)) == M * N, \
            f"{M}x{N}: a tile was visited twice (YAQA's own starts list duplicates one " \
            f"anti-diagonal; ours must not)"


def test_antidiag_order_only_reads_already_quantized_tiles():
    """Feedback for tile (jm,jn) reads rows >= jm and cols >= jn, so every such tile must
    already be done. Violating this is silent: it degrades quality without erroring."""
    for M, N in [(3, 5), (6, 4), (8, 8)]:
        order = [t for diag in gt._antidiag_tile_order(M, N) for t in diag]
        pos = {t: k for k, t in enumerate(order)}
        for (jm, jn), k in pos.items():
            for jm2 in range(jm, M):
                for jn2 in range(jn, N):
                    if (jm2, jn2) == (jm, jn):
                        continue
                    assert pos[(jm2, jn2)] < k, (
                        f"{M}x{N}: tile {(jm, jn)} at step {k} reads {(jm2, jn2)} "
                        f"which is quantized later at step {pos[(jm2, jn2)]}")


def test_antidiag_tiles_within_a_diagonal_are_independent():
    """Tiles on one anti-diagonal are quantized in a single batched codebook call, so none
    of them may depend on another: no two may share a (row>=, col>=) relation."""
    for M, N in [(3, 5), (6, 4), (8, 8)]:
        for diag in gt._antidiag_tile_order(M, N):
            for a in diag:
                for b in diag:
                    if a is b:
                        continue
                    assert not (b[0] >= a[0] and b[1] >= a[1]), \
                        f"{a} and {b} on one diagonal but {b} feeds {a}"


# ---------------------------------------------------------------------------
# the regression gate: Lout=0 must reproduce the shipped one-sided path exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (48, 48)])
def test_lout_zero_is_bit_identical_to_one_sided(shape):
    """`Lout=0` reduces YAQA's update to classic LDLQ. Byte-identical output is the
    CLAUDE.md exactness rule: every published checkpoint came from the one-sided path."""
    cb = gt.TrellisCodebook(variant="3inst", K=2, device="cpu")
    torch.manual_seed(0)
    m, n = shape
    W = torch.randn(m, n)
    X = torch.randn(4 * n, n)
    H = (X.T @ X) / (4 * n)

    hat_1, idx_1, s_1 = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    hat_2, idx_2, s_2 = gt.trellis_ldlq_2hess(W, H, None, cb, for_kernel=True)

    assert s_1 == s_2
    assert torch.equal(idx_1, idx_2), "trellis states differ with Lout=0"
    assert torch.equal(hat_1, hat_2), "reconstruction differs with Lout=0"


# ---------------------------------------------------------------------------
# the go/no-go: does two-sided feedback actually help?
# ---------------------------------------------------------------------------
def _yaqa_proxy(W, W_hat, Hin, Hout):
    """YAQA's objective: tr(Hout · E · Hin · Eᵀ) with E = W - Ŵ (Kronecker Hessian)."""
    E = (W - W_hat).double()
    return torch.trace(Hout.double() @ E @ Hin.double() @ E.T).item()


def test_two_sided_beats_one_sided_on_exact_kronecker_hessian():
    """The gate. Build a Hessian that IS a Kronecker product, so YAQA's approximation is
    exact by construction, and check 2-sided feedback lowers the objective it optimizes.

    If this fails the port is wrong or the idea does not transfer to a trellis codebook —
    either way there is no point collecting gradient Hessians for real models.
    """
    cb = gt.TrellisCodebook(variant="3inst", K=2, device="cpu")
    torch.manual_seed(3)
    m, n = 64, 64
    W = torch.randn(m, n)

    # anisotropic on BOTH axes — with an isotropic Hout the output-side term is ~0 and the
    # two paths trivially coincide, which would make this test vacuous.
    Xi = torch.randn(4 * n, n) @ torch.diag(torch.linspace(0.2, 3.0, n))
    Hin = (Xi.T @ Xi) / (4 * n)
    Xo = torch.randn(4 * m, m) @ torch.diag(torch.linspace(0.2, 3.0, m))
    Hout = (Xo.T @ Xo) / (4 * m)

    hat_1, _, s1 = gt.trellis_ldlq_2hess(W, Hin, None, cb, for_kernel=False)
    hat_2, _, s2 = gt.trellis_ldlq_2hess(W, Hin, Hout, cb, for_kernel=False)

    p1 = _yaqa_proxy(W, hat_1 * s1, Hin, Hout)
    p2 = _yaqa_proxy(W, hat_2 * s2, Hin, Hout)
    assert p2 < p1, f"2-sided did not beat 1-sided on an exact Kronecker Hessian: {p2} vs {p1}"
