"""E8P TC-pack (`maybe_pack_idxs`) speedup gate.

The int64 sign-pack was a 4x8 Python bit-loop (CPU-serial, run once per expert per RVQ
stage during quantization). It's now vectorized. This pins the vectorized output
bit-for-bit against the original loop on random 16-bit E8P indices, across shapes incl.
the gemma-4-26B-A4B expert gate_up (1408, 352) and down (2816, 88).
"""
import torch

from glq.codebook_e8p import E8PCodebook


def _old_pack_loop(idxs):
    """Original loop implementation of maybe_pack_idxs (reference)."""
    m, n = idxs.shape
    idxs = idxs.view(m // 2, 2, (n * 8) // 16, 2).transpose(1, 2).contiguous()
    abs32 = (idxs[:, :, 0, 0] >> 8) + ((idxs[:, :, 1, 0] >> 8) << 8) + \
        ((idxs[:, :, 0, 1] >> 8) << 16) + ((idxs[:, :, 1, 1] >> 8) << 24)
    sign32 = torch.zeros(abs32.shape, dtype=abs32.dtype, device=abs32.device)
    for i in range(4):
        wt = idxs[:, :, i % 2, i // 2]
        for j in range(8):
            sign32 += ((wt >> j) & 1) << (4 * j + i)
    out = (sign32 << 32) + abs32
    return out.reshape(m // 16, 8, n // 8, 4).transpose(1, 2).contiguous().view(m, n // 4)


def test_maybe_pack_idxs_vectorized_bit_identical():
    # (m mult of 16, n8 mult of 8); n8 = in_features/8. Includes the 26B-A4B experts.
    shapes = [(16, 8), (32, 16), (1408, 352), (2816, 88)]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        for m, n8 in shapes:
            torch.manual_seed(m * 1000 + n8)
            idxs = torch.randint(0, 65536, (m, n8), dtype=torch.int64, device=dev)
            ref = _old_pack_loop(idxs.clone())
            # maybe_pack_idxs uses no `self` state -> call unbound with self=None.
            got = E8PCodebook.maybe_pack_idxs(None, idxs.clone())
            assert got.shape == (m, n8 // 4), f"bad shape {tuple(got.shape)} for ({m},{n8})"
            assert torch.equal(got, ref), f"vectorized pack != loop on {dev} ({m},{n8})"


if __name__ == "__main__":
    test_maybe_pack_idxs_vectorized_bit_identical()
    print("OK: maybe_pack_idxs vectorized is bit-identical to the loop")
