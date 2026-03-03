"""Fast Walsh-Hadamard Transform."""

import math
import torch


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    In-place Fast Walsh-Hadamard transform along the last dimension.
    O(n log n) instead of O(n^2) matrix multiply.
    Input last dim must be a power of 2.
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Last dim {n} must be power of 2"

    h = 1
    while h < n:
        x_reshaped = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_reshaped[..., 0, :].clone()
        b = x_reshaped[..., 1, :].clone()
        x_reshaped[..., 0, :] = a + b
        x_reshaped[..., 1, :] = a - b
        x = x_reshaped.reshape(*x.shape)
        h *= 2

    return x / math.sqrt(n)
