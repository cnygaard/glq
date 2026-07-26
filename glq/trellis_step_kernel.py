"""Hand-fused Triton kernel for the trellis Viterbi ACS step (Stage 6).

Replaces the two inductor kernels `bitshift_codebook.update` compiles per step with ONE
kernel: candidate-min (strict lowest-k tie-break) + int32 backpointer store + state_err
from the recons lut + cost update. Bit-exact to the compiled path by construction — the
min combine is the minimum of a total order on (value, index) pairs (NaN below
everything, index breaks ties), so the sequential ascending-k loop agrees with
inductor's `minimum_with_index` reduction for every input, including the +inf overlap
masks; all f32 arithmetic keeps the reference operand order. Gated by
tests/test_trellis_fused_step.py (torch.equal vs the compiled update across
variant x K x B x masked + tie/NaN torture + whole-encoder A/B).

B is a RUNTIME argument: one compilation per (variant, K) covers every batch width —
no per-B specialization, so CUDA-graph warmup compiles it once and capture replays
plain kernel launches. Portable Triton only (sm_86/89/120): loads/stores/where/
static_range — none of the version-drifting reduce/argmin APIs.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                             # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _viterbi_acs_step(
        cost_ptr,        # (B, NSTATES) f32  in : cost from step i-1
        x_ptr,           # (V, B)       f16  in : X[i*V:(i+1)*V] slice (contiguous)
        lut_ptr,         # (V, 1, NSTATES) f32 in : cb.recons_state (contiguous)
        cost_out_ptr,    # (B, NSTATES) f32  out
        prev_ptr,        # (B, NGROUP)  i32  out: from_state[i] row
        B,               # runtime x row stride — NOT constexpr (one compile per config)
        V: tl.constexpr, NSTATES: tl.constexpr, NCAND: tl.constexpr,
        NGROUP: tl.constexpr, SHIFT: tl.constexpr,
        TILE_J: tl.constexpr, KCHUNK: tl.constexpr,
    ):
        pid_j = tl.program_id(0)
        b = tl.program_id(1)
        js = pid_j * TILE_J + tl.arange(0, TILE_J)
        row = cost_ptr + b * NSTATES

        # Phase 1: running min over candidates k ascending. `take` is inductor's
        # minimum_with_index combine specialized to ascending k: strict < keeps the
        # lowest k on ties; a NaN candidate displaces any non-NaN accumulator; a NaN
        # accumulator is never displaced (IEEE `<` is false). The +inf/k0 sentinel
        # init is provably identical to initializing from k=0.
        best = tl.full([TILE_J], float("inf"), tl.float32)
        bestk = tl.zeros([TILE_J], tl.int32)
        for kc in range(0, NCAND, KCHUNK):
            for u in tl.static_range(KCHUNK):
                k = kc + u
                v = tl.load(row + k * NGROUP + js)
                take = (v < best) | ((v != v) & (best == best))
                best = tl.where(take, v, best)
                bestk = tl.where(take, k, bestk)

        # predecessor state = j + bestk*2^(L-KV); values < 2^16 so int32-native is
        # bit-identical to the reference's int64 arithmetic + .to(int32).
        tl.store(prev_ptr + b * NGROUP + js, js + (bestk << SHIFT))

        # Phase 2: cost_new[s] = sum_v (lut[v,s] - x[v,b])^2 + best[j], for the
        # contiguous output span s in [j0*NCAND, (j0+TILE_J)*NCAND). Ascending-v sum
        # starting from 0.0 is exact (squares are never -0.0); operand order matches
        # the reference (lut - x, then err + best).
        rs = tl.arange(0, NCAND)
        s = js[:, None] * NCAND + rs[None, :]
        err = tl.zeros((TILE_J, NCAND), tl.float32)
        for vi in tl.static_range(V):
            xv = tl.load(x_ptr + vi * B + b).to(tl.float32)
            d = tl.load(lut_ptr + vi * NSTATES + s) - xv
            err += d * d
        tl.store(cost_out_ptr + b * NSTATES + s, err + best[:, None])


# (V, KV) -> (TILE_J, KCHUNK, num_warps). TILE_J = NGROUP//16 makes the output block a
# uniform 4096-f32 (16 KB) contiguous span for every config; grid is (16, B). Static —
# no autotune (capture-time re-benchmarking is illegal under CUDA graphs).
_CFG = {
    (1, 2): (1024, 4, 4), (1, 3): (512, 8, 4), (1, 4): (256, 16, 4),
    (2, 4): (256, 16, 4), (2, 6): (64, 16, 4), (2, 8): (16, 16, 4),
}


def fused_update(cb, cost, thing, out_row):
    """Drop-in for `bitshift_codebook.update(cost, thing, out_row)`: one Triton kernel.
    Returns the new cost tensor; writes the int32 backpointer row into out_row."""
    B = cost.shape[0]
    kv = cb.K * cb.V
    tile_j, kchunk, warps = _CFG[(cb.V, kv)]
    cost_new = torch.empty_like(cost)
    _viterbi_acs_step[((1 << (cb.L - kv)) // tile_j, B)](
        cost, thing, cb.recons_state, cost_new, out_row, B,
        V=cb.V, NSTATES=1 << cb.L, NCAND=1 << kv, NGROUP=1 << (cb.L - kv),
        SHIFT=cb.L - kv, TILE_J=tile_j, KCHUNK=kchunk, num_warps=warps)
    return cost_new
