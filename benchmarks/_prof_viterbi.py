"""Isolated nsys target: run the trellis Viterbi on SmolLM2-135M's real layer shapes.

The trellis quant hot path is `trellis_ldlq`'s per-block Viterbi. Its kernel mix is identical
regardless of model/#layers, so quantizing ONE pass of each distinct SmolLM2-135M linear shape
characterizes exactly what the full-model quant spends its time on — without the model download
+ calibration that kept nsys's capture window landing on CPU/network work.

SmolLM2-135M: hidden=576, intermediate=1536, GQA (9 q heads / 3 kv heads, head_dim 64).
"""
import torch
import glq.trellis as gt

torch.manual_seed(0)
dev = "cuda"
tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
cb = gt.TrellisCodebook(variant="hyb", K=4, tlut=tlut, device=dev)   # 4bpw, K-independent cost

# (out, in) for the distinct linear shapes in a SmolLM2-135M decoder layer.
SHAPES = [
    ("q_proj",   576, 576),
    ("kv_proj",  192, 576),   # GQA: 3 kv heads * 64
    ("o_proj",   576, 576),
    ("gate/up",  1536, 576),
    ("down",     576, 1536),
]

def one(m, n):
    W = (torch.randn(m, n, device=dev) * 0.05).float()
    X = torch.randn(512, n, device=dev)
    H = (X.T @ X) / 512
    torch.cuda.synchronize()
    gt.trellis_ldlq(W, H, cb, for_kernel=True)
    torch.cuda.synchronize()

# warm (resolve torch.compile of `update`, cache allocator) — OUTSIDE the timed region ideally,
# but nsys profiles the whole process; the compile kernels are a one-time blip vs the Viterbi storm.
one(576, 576)

torch.cuda.profiler.start()          # capture-range anchor (nsys --capture-range=cudaProfilerApi)
for name, m, n in SHAPES:
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record(); one(m, n); en.record()
    torch.cuda.synchronize()
    print(f"{name:9s} ({m}x{n})  {st.elapsed_time(en):8.1f} ms", flush=True)
torch.cuda.profiler.stop()
