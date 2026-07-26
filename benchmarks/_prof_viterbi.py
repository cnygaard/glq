"""Isolated nsys target + B-sweep for the trellis Viterbi (3INST, SmolLM2-360M shapes).

The trellis quant hot path is `trellis_ldlq`'s per-block Viterbi. Its kernel mix is identical
regardless of model/#layers, so quantizing ONE pass of each distinct linear shape
characterizes exactly what the full-model quant spends its time on — without the model
download + calibration that kept nsys's capture window landing on CPU/network work.

Two modes:
  default        — per-shape trellis_ldlq wall times + cudaProfilerApi anchor (nsys target)
  --sweep        — pair-graph quantize timing over a batch sweep: fits T(B) = F + c*B and
                   prints the Stage-5 (qkv lockstep) GO/NO-GO inputs. Run on EVERY target
                   GPU class (Blackwell AND sm_86/89) before adopting lockstep batching.

SmolLM2-360M: hidden=960, intermediate=2560, GQA (15 q heads / 5 kv heads, head_dim 64).
Env: GLQ_TRELLIS_NO_CUDAGRAPH / GLQ_TRELLIS_CUDAGRAPH_MAX_B pass through to glq.trellis.
"""
import sys

import torch

import glq.trellis as gt

torch.manual_seed(0)
dev = "cuda"
cb = gt.TrellisCodebook(variant="3inst", K=4, device=dev)   # the production 4bpw variant

# (out, in) for the distinct linear shapes in a SmolLM2-360M decoder layer.
SHAPES = [
    ("q_proj",   960, 960),
    ("kv_proj",  320, 960),   # GQA: 5 kv heads * 64
    ("o_proj",   960, 960),
    ("gate/up",  2560, 960),
    ("down",     960, 2560),
]


def one(m, n):
    W = (torch.randn(m, n, device=dev) * 0.05).float()
    X = torch.randn(512, n, device=dev)
    H = (X.T @ X) / 512
    torch.cuda.synchronize()
    gt.trellis_ldlq(W, H, cb, for_kernel=True)
    torch.cuda.synchronize()


def sweep():
    """Time the pair-graph quantize (both tail-biting passes) per batch width B.
    B = m/16 tile-stripes; qkv lockstep would merge 60+20+20 -> 100 on the 360M."""
    reps = 50
    rows = []
    for B in (20, 60, 100, 128, 160, 192, 224, 256):
        torch.manual_seed(B)
        tiles = (torch.randn(B, 256, device=dev) * 0.5).float()
        for _ in range(3):
            cb.quantize_tiles(tiles)                       # capture + warm
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(reps):
            cb.quantize_tiles(tiles)
        en.record()
        torch.cuda.synchronize()
        us = st.elapsed_time(en) / reps * 1000
        rows.append((B, us))
        print(f"B={B:4d}  {us:9.0f} us/quantize  {us / B:7.1f} us/row", flush=True)
    # least-squares fit T(B) = F + c*B over the L2-safe range (B <= 192)
    fit = [(b, t) for b, t in rows if b <= 192]
    n = len(fit)
    sb = sum(b for b, _ in fit); st_ = sum(t for _, t in fit)
    sbb = sum(b * b for b, _ in fit); sbt = sum(b * t for b, t in fit)
    c = (n * sbt - sb * st_) / (n * sbb - sb * sb)
    F = (st_ - c * sb) / n
    t = dict(rows)
    ratio = (t[100] / 100) / (t[60] / 60)
    print(f"\nfit: T(B) = {F:.0f} us + {c:.1f} us/row   (B<=192)")
    print(f"Stage-5 GO rule: F >= 1300 us -> {'GO' if F >= 1300 else 'NO-GO'} (F={F:.0f})")
    print(f"                 us/row(100)/us/row(60) <= 1.05 -> "
          f"{'GO' if ratio <= 1.05 else 'NO-GO'} (ratio={ratio:.3f})")


if "--sweep" in sys.argv:
    sweep()
    sys.exit(0)

# warm (resolve torch.compile of `update`/`_tb_step`, cache allocator) — OUTSIDE the timed
# region ideally, but nsys profiles the whole process; the compile kernels are a one-time
# blip vs the Viterbi storm.
one(960, 960)

torch.cuda.profiler.start()          # capture-range anchor (nsys --capture-range=cudaProfilerApi)
for name, m, n in SHAPES:
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record(); one(m, n); en.record()
    torch.cuda.synchronize()
    print(f"{name:9s} ({m}x{n})  {st.elapsed_time(en):8.1f} ms", flush=True)
torch.cuda.profiler.stop()
