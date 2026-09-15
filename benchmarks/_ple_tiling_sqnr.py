"""Phase 0 gate: is row-tiling actually better than 16x16 for an embedding table?

Trellis codes a *sequence*. `quantize(X)` transposes and takes `T, NO = X.shape`, so T is the
sequence length and each column is an independent tail-biting sequence. QTIP tiles weights
16x16 (=256 elements) so decoded lanes land in tensor-core fragment order — a matmul
concern. An embedding is gathered by row, not multiplied, so the tiling is free to change.

The hypothesis this script exists to FALSIFY: a 16x16 tile couples 16 *unrelated vocab
rows* into one trellis sequence. For a weight matrix adjacent output rows are correlated;
for a hashed n-gram table they are not, so the trellis's context-based coding gain is being
spent on noise. Row-tiling should therefore be BETTER on SQNR, not merely cheaper to gather.

If it is not better, the plan built on it stops here — the read amplification it was
supposed to avoid measures 0.04% of per-token traffic, so there is no fallback argument.

Three arms separate two effects that would otherwise be confounded:

    A  16x16 tiled, scalar Wscale   — the status quo path
    B  row-tiled,   scalar Wscale   — isolates TILING alone
    C  row-tiled,   per-row Wscale  — the proposed design

All three share one right-side-only RHT (apply_left=False), which is what keeps rows
independent in the first place, so the transform cannot explain a difference between them.

    python benchmarks/_ple_tiling_sqnr.py --rows 16384 [--K 3] [--synthetic]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.rht import RHT                                          # noqa: E402
from glq.trellis import TrellisCodebook                          # noqa: E402

MODEL = "Qwen/Qwen3.8-Flash-Next"
PLE_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float()
    err = (ref - got).pow(2).sum()
    return float(10 * torch.log10(ref.pow(2).sum() / err.clamp_min(1e-30)))


def load_rows(n_rows: int, synthetic: bool) -> torch.Tensor:
    """Real PLE rows from ONE 0.75 GiB shard — not the 335 GiB checkpoint."""
    if synthetic:
        # Shaped like the real thing, for validating the script itself. A Gaussian has
        # none of the outlier structure that decides this question, so a synthetic result
        # is NOT evidence either way.
        print("  !! synthetic rows: validates the harness, proves nothing about quality")
        return torch.randn(n_rows, 160) * 0.02

    import json
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))
    shard = idx["weight_map"][PLE_KEY]
    path = hf_hub_download(MODEL, shard)
    with safe_open(path, framework="pt") as f:
        # get_slice: read only the rows we need, never the whole 0.75 GiB tensor.
        return f.get_slice(PLE_KEY)[:n_rows, :].float()


def quantize_16x16(Wt, tcb, scale):
    """Status quo: for each 16-column block, fold 16 rows into one 256-element tile."""
    m, n = Wt.shape
    Wr = Wt / scale
    hat = torch.zeros_like(Wr)
    for k in range(n // 16):
        kb, ke = k * 16, (k + 1) * 16
        tiles = Wr[:, kb:ke].reshape(m // 16, 256)
        hatX, _ = tcb.cb.quantize(tiles)          # for_kernel=False: no _PERMUTE
        hat[:, kb:ke] = hatX.reshape(m, 16)
    return hat * scale


def quantize_rowwise(Wt, tcb, scale):
    """Proposed: each row is its own tail-biting sequence of length n."""
    Wr = Wt / scale
    hatX, _ = tcb.cb.quantize(Wr)                 # (m,n) -> T=n, NO=m independent seqs
    return hatX * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--variant", default="3inst")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.rows % 16 == 0, "16x16 tiling needs rows %16 == 0"

    W = load_rows(args.rows, args.synthetic).to(dev)
    m, n = W.shape
    print(f"rows {m} x {n} on {dev} | variant={args.variant} K={args.K}")
    print(f"  input rms {W.pow(2).mean().sqrt():.5f}  max|w| {W.abs().max():.5f}")

    tcb = TrellisCodebook(variant=args.variant, K=args.K, device=dev)

    # One right-side-only RHT, shared by every arm: apply_left=False is what keeps rows
    # independent, and using the same object means it cannot explain an A/B/C difference.
    rht = RHT(m, n, device=dev, block_diagonal=True, apply_left=False, e8p=False)
    Wt = rht.transform_weights(W)
    assert Wt.shape == (m, n), f"RHT padded to {tuple(Wt.shape)}; block_diagonal failed"

    scalar = Wt.pow(2).mean().sqrt().item() * tcb.opt_scale
    per_row = Wt.pow(2).mean(dim=1, keepdim=True).sqrt() * tcb.opt_scale

    arms = {
        "A  16x16, scalar scale ": quantize_16x16(Wt, tcb, scalar),
        "B  row-tiled, scalar   ": quantize_rowwise(Wt, tcb, scalar),
        "C  row-tiled, per-row  ": quantize_rowwise(Wt, tcb, per_row),
    }
    print()
    base = None
    for label, hat_t in arms.items():
        got = rht.inverse_transform_weights(hat_t)
        s = sqnr_db(W, got)
        if base is None:
            base = s
        print(f"  {label}  SQNR {s:6.2f} dB   ({s - base:+.2f} vs A)")

    print(f"\n  storage/row @K={args.K}: {n * args.K / 8:.0f} B "
          f"+ 4 B per-row scale (arm C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
