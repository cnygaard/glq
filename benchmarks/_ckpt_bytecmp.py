"""Per-tensor byte comparison of two safetensors checkpoints.

Usage: python benchmarks/_ckpt_bytecmp.py <ckpt_dir_a> <ckpt_dir_b>

Prints per-tensor sha256 verdicts: BYTE-IDENTICAL, or the differing tensor names.
Handles every dtype incl. bf16 and 0-dim scalars (hashes a flattened uint8 view —
.numpy() rejects bf16 and .view(uint8) rejects 0-dim). This is the file-level
exactness gate used by the trellis quantization-speed work: any encode-path change
must reproduce the reference checkpoint byte-for-byte.
"""
import hashlib
import sys

import torch
from safetensors import safe_open


def _sha256(t: torch.Tensor) -> str:
    t = t.contiguous().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main(a: str, b: str) -> int:
    fa = safe_open(a + "/model.safetensors", "pt")
    fb = safe_open(b + "/model.safetensors", "pt")
    ka, kb = set(fa.keys()), set(fb.keys())
    print(f"keys equal: {ka == kb} ({len(ka)} vs {len(kb)} tensors)")
    for only, name in ((ka - kb, a), (kb - ka, b)):
        if only:
            print(f"  only in {name}: {sorted(only)[:8]}")
    diff = [k for k in sorted(ka & kb)
            if _sha256(fa.get_tensor(k)) != _sha256(fb.get_tensor(k))]
    if not diff and ka == kb:
        print("BYTE-IDENTICAL")
        return 0
    print(f"DIFFERS ({len(diff)}): {diff[:10]}" + (" ..." if len(diff) > 10 else ""))
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
