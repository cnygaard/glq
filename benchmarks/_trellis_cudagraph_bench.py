"""Wall-clock A/B for the Viterbi CUDA-graph: `trellis_ldlq` graphs ON vs OFF.

Times the REAL per-layer encode unit (`trellis_ldlq` = RHT + block_LDL + the LDLQ sweep that
calls the Viterbi ~72×/layer) on SmolLM2-135M's linear shapes, K=4. One shared codebook per arm
(mirrors the real quant, where one codebook serves every layer, so a width is captured once and
replayed thereafter).

Reports, per shape:
  - eager (graphs OFF) steady per-layer ms
  - graphed capture ms  = first call (warmup+compile+capture+replay), the one-time cost
  - graphed steady ms   = replay (median of subsequent calls) — the amortized per-layer time
  - speedup             = eager_steady / graphed_steady
Plus a projected full-SmolLM2-135M Viterbi-quant time (30 layers × {q,k,v,o,gate,up,down}),
ON vs OFF, and a bit-exact parity assert as a belt-and-suspenders check.
"""
import statistics
import sys

import torch

import glq.trellis as gt

DEV = "cuda"
K = 4
REPS = 6  # calls per shape: [0]=capture, [1:]=replay steady-state

# SmolLM2-135M: hidden 576, GQA 9 q / 3 kv heads (head_dim 64), intermediate 1536.
SHAPES = [
    ("q_proj",   576, 576),
    ("k_proj",   192, 576),
    ("v_proj",   192, 576),
    ("o_proj",   576, 576),
    ("gate_proj", 1536, 576),
    ("up_proj",   1536, 576),
    ("down_proj", 576, 1536),
]


def _cb():
    torch.manual_seed(0)
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    return gt.TrellisCodebook(variant="hyb", K=K, tlut=tlut, device=DEV)


def _inputs(m, n, seed):
    torch.manual_seed(seed)
    W = (torch.randn(m, n, device=DEV) * 0.05).float()
    X = torch.randn(512, n, device=DEV)
    return W, (X.T @ X) / 512


def _time(cb, W, H):
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    st.record()
    out = gt.trellis_ldlq(W, H, cb, for_kernel=True)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en), out


def run(enabled):
    """Return {shape: [ms per call]} and the last-shape Qidxs (for parity)."""
    gt._GLQ_TRELLIS_CUDAGRAPH_ENABLED = enabled
    res, last_q = {}, None
    for name, m, n in SHAPES:
        cb = _cb()  # fresh per shape → capture cost lands on call 0 of that width
        ts = []
        for r in range(REPS):
            W, H = _inputs(m, n, seed=1000 + r)
            t, out = _time(cb, W, H)
            ts.append(t)
        res[name] = ts
        last_q = out[1]
    return res, last_q


def main():
    off, q_off = run(False)   # eager baseline
    on, q_on = run(True)      # graphed
    gt._GLQ_TRELLIS_CUDAGRAPH_ENABLED = True

    assert torch.equal(q_on, q_off), "PARITY FAIL: graphed Qidxs != eager"
    print("parity: graphed == eager (torch.equal) ✓\n", flush=True)

    hdr = f"{'shape':10s} {'eager ms':>9s} {'capture ms':>11s} {'graph ms':>9s} {'speedup':>8s}"
    print(hdr); print("-" * len(hdr))
    tot_eager = tot_graph = tot_cap = 0.0
    # per-layer counts in a SmolLM2-135M decoder layer (all 7 are distinct calls)
    for name, m, n in SHAPES:
        e = statistics.median(off[name][1:])       # eager steady
        cap = on[name][0]                          # graphed first call (capture)
        g = statistics.median(on[name][1:])        # graphed steady (replay)
        print(f"{name:10s} {e:9.1f} {cap:11.1f} {g:9.1f} {e / g:7.2f}x", flush=True)
        tot_eager += e; tot_graph += g; tot_cap += cap

    L = 30  # SmolLM2-135M decoder layers
    print("-" * len(hdr))
    print(f"{'1 layer':10s} {tot_eager:9.1f} {'':>11s} {tot_graph:9.1f} {tot_eager / tot_graph:7.2f}x")
    # full model: L layers replay + one capture per distinct width (amortized once)
    full_off = tot_eager * L
    full_on = tot_graph * L + tot_cap    # capture paid once per width, replay for all L
    print(f"\nProjected SmolLM2-135M Viterbi-quant (30 layers):")
    print(f"  eager   : {full_off / 1000:6.1f} s")
    print(f"  graphed : {full_on / 1000:6.1f} s  (incl. {tot_cap / 1000:.1f} s one-time capture)"
          f"  → {full_off / full_on:.2f}× faster", flush=True)


if __name__ == "__main__":
    sys.exit(main())
