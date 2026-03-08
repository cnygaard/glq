"""Tests for the Fast Walsh-Hadamard Transform.

Note: the FHT implementation is IN-PLACE — it modifies the input tensor.
All tests must .clone() inputs before calling fast_hadamard_transform.
"""

import math

import pytest
import torch

from glq.hadamard import fast_hadamard_transform


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 256])
def test_fht_is_involution(n):
    """Applying FHT twice returns the original vector (FHT is its own inverse)."""
    x = torch.randn(n)
    x_orig = x.clone()
    y = fast_hadamard_transform(x.clone())
    x_roundtrip = fast_hadamard_transform(y.clone())
    torch.testing.assert_close(x_roundtrip, x_orig, atol=1e-5, rtol=1e-5)


def test_fht_known_values():
    """Check FHT against the 2x2 Hadamard matrix manually."""
    x = torch.tensor([3.0, 1.0])
    y = fast_hadamard_transform(x.clone())
    # H_2 = [[1,1],[1,-1]] / sqrt(2)
    expected = torch.tensor([4.0, 2.0]) / math.sqrt(2)
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)


def test_fht_orthogonality():
    """FHT preserves norms (Parseval's theorem)."""
    x = torch.randn(128)
    x_norm = x.norm()
    y = fast_hadamard_transform(x.clone())
    torch.testing.assert_close(
        x_norm, y.norm(), atol=1e-4, rtol=1e-4
    )


def test_fht_batched():
    """FHT works on 2D input (batch of vectors)."""
    x = torch.randn(32, 16)
    y = fast_hadamard_transform(x.clone())
    assert y.shape == (32, 16)
    # Each row should be the FHT of the corresponding input row
    for i in range(32):
        y_i = fast_hadamard_transform(x[i].clone())
        torch.testing.assert_close(y[i], y_i, atol=1e-5, rtol=1e-5)


def test_fht_rejects_non_power_of_2():
    """FHT should reject dimensions that are not a power of 2."""
    x = torch.randn(3)
    with pytest.raises(AssertionError):
        fast_hadamard_transform(x)
