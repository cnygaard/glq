"""Is a trellis embedding row-decode cheap enough to sit on the per-token path?

`unpack_trellis` walks the tail-biting state recursion with a carried dependency, so it
costs T//V sequential steps. For the PLE row layout (T=160) and the 3inst K=3 codebook
(V=1, measured) that is **160 steps**, every one a tiny elementwise kernel. Whether that is
free or fatal depends entirely on launch overhead, which is what this measures.

The comparison that matters is not "is it fast" in the abstract — it is against the shell
path (`_dequant_embedding_rows`), because that is the fallback. Shell costs a 1.6x footprint
on this tensor (full Hadamard pads 160 -> 256) but decodes with two index_selects and no
sequential loop. If trellis is dramatically slower per lookup, the 11 GiB it saves is not
worth it.

Three arms per batch size:

    shell        two index_selects + FHT            — the fallback, no loop
    trellis      unpack_trellis + recons + FHT      — eager, 160 launches
    trellis+cg   the same under a CUDA graph        — launches collapse to one replay

A CUDA graph is legitimate here: at decode the gather shape is static.

    python benchmarks/_ple_decode_cost.py [--rows 65536] [--K 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.hadamard import block_diagonal_fht          # noqa: E402
from glq.rht import RHT                              # noqa: E402
from glq.trellis import TrellisCodebook              # noqa: E402

PLE_DIM = 160


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters      # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=65536, help="table rows to gather from")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--variant", default="3inst")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU: launch overhead is the whole question")
        return 1
    dev = "cuda"
    torch.manual_seed(0)

    tcb = TrellisCodebook(variant=args.variant, K=args.K, device=dev)
    cb = tcb.cb
    print(f"codebook {args.variant} K={cb.K} V={cb.V} L={cb.L} "
          f"-> {PLE_DIM // cb.V} sequential steps at T={PLE_DIM}")

    # --- build a small quantized table in each format -------------------------------------
    W = (torch.randn(args.rows, PLE_DIM, device=dev) * 0.02)
    rht = RHT(args.rows, PLE_DIM, device=dev, block_diagonal=True,
              apply_left=False, e8p=False)
    Wt = rht.transform_weights(W)
    per_row = Wt.pow(2).mean(dim=1, keepdim=True).sqrt() * tcb.opt_scale
    _, state = cb.quantize(Wt / per_row)
    packed = cb.pack_trellis(state)                       # [rows, ceil(T*K/16)] int16
    wscale = per_row.squeeze(-1).to(torch.float16)
    sv = rht.sv.to(torch.float16)
    blocks_n = rht.blocks_n
    print(f"  packed {tuple(packed.shape)} {packed.dtype} "
          f"= {packed.numel() * 2 / args.rows:.0f} B/row  (+2 B fp16 scale)")

    # Shell-shaped stand-in at the SAME bpw, padded 160 -> 256 as the full-Hadamard path
    # forces. Only the decode shape matters for timing, so the contents are irrelevant.
    n_pad = 1 << (PLE_DIM - 1).bit_length()
    qidxs = torch.randint(-32768, 32767, (args.rows, n_pad // 8),
                          dtype=torch.int16, device=dev)
    shell_cb = torch.randn(65536, 8, device=dev)
    shell_ws = torch.randn(args.rows, device=dev).abs()
    shell_sv = torch.randn(n_pad, device=dev)
    print(f"  shell  {tuple(qidxs.shape)} int16 = {qidxs.numel() * 2 / args.rows:.0f} B/row "
          f"(n_pad {PLE_DIM} -> {n_pad})")

    def shell_decode(ids):
        rows = qidxs.index_select(0, ids)
        idx = rows.reshape(-1).long() & 0xFFFF
        deq = shell_cb.index_select(0, idx).reshape(ids.shape[0], n_pad).float()
        deq = deq * shell_ws.index_select(0, ids).unsqueeze(-1)
        from glq.hadamard import fast_hadamard_transform
        return (fast_hadamard_transform(deq) * shell_sv)[..., :PLE_DIM]

    def trellis_decode(ids):
        rows = packed.index_select(0, ids)
        st = cb.unpack_trellis(rows, PLE_DIM)
        deq = cb.recons(st).float().reshape(ids.shape[0], PLE_DIM)
        deq = deq * wscale.index_select(0, ids).unsqueeze(-1).float()
        return block_diagonal_fht(deq, blocks_n) * sv.float()

    print(f"\n{'B':>7} {'shell ms':>10} {'trellis ms':>12} {'trellis+cg ms':>14} "
          f"{'cg vs shell':>12}")
    for B in (8, 64, 512, 4096, 32768):
        ids = torch.randint(0, args.rows, (B,), device=dev)
        t_shell = _time(lambda: shell_decode(ids))
        t_tr = _time(lambda: trellis_decode(ids))

        # CUDA graph: static shape, static input buffer — what decode actually looks like.
        t_cg = float("nan")
        try:
            static_ids = ids.clone()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    trellis_decode(static_ids)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                trellis_decode(static_ids)
            t_cg = _time(lambda: g.replay())
        except Exception as e:                                     # noqa: BLE001
            print(f"  (cuda graph capture failed at B={B}: {type(e).__name__}: {e})")

        ratio = f"{t_cg / t_shell:.1f}x" if t_cg == t_cg else "-"
        print(f"{B:>7} {t_shell:>10.3f} {t_tr:>12.3f} {t_cg:>14.3f} {ratio:>12}")

    print("\nRead it against the budget: the PLE is looked up on layer 1 only, so this cost "
          "lands once per forward.\nAt ~90 tok/s decode a token is ~11 ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
