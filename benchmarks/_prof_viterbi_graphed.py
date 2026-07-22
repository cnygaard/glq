"""nsys target: STEADY-STATE (replay) Viterbi launch profile, graphs ON.

Warms all SmolLM2-135M shapes first (captures every graph), THEN marks the capture range around
a pure replay pass — so the nsys report shows the replay launch profile, not the one-time capture.
Compare `cudaLaunchKernel`/`cuLaunchKernel`/`cudaMemcpyAsync` counts against the eager baseline
(_prof_viterbi.py: ~256k/123k/63k for 5 layers). Expect a collapse to a few hundred cudaGraphLaunch.
"""
import torch

import glq.trellis as gt

DEV = "cuda"
gt._GLQ_TRELLIS_CUDAGRAPH_ENABLED = True
tlut = (torch.randn(2 ** 9, 2, generator=torch.Generator().manual_seed(0))
        * 0.9682458365518543).to(torch.float16)
cb = gt.TrellisCodebook(variant="hyb", K=4, tlut=tlut, device=DEV)

SHAPES = [("q_proj", 576, 576), ("kv_proj", 192, 576), ("o_proj", 576, 576),
          ("gate_up", 1536, 576), ("down", 576, 1536)]


def one(m, n, seed):
    torch.manual_seed(seed)
    W = (torch.randn(m, n, device=DEV) * 0.05).float()
    X = torch.randn(512, n, device=DEV)
    H = (X.T @ X) / 512
    gt.trellis_ldlq(W, H, cb, for_kernel=True)
    torch.cuda.synchronize()


# Warm: capture every (B, has_overlap) graph so the profiled pass is pure replay.
for name, m, n in SHAPES:
    one(m, n, seed=1)
    one(m, n, seed=2)

torch.cuda.profiler.start()          # nsys --capture-range=cudaProfilerApi
for name, m, n in SHAPES:
    one(m, n, seed=3)                # replay only
torch.cuda.profiler.stop()
