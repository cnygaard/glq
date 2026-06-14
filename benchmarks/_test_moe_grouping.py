"""Parity test for the GLQ MoE token-grouping kernels (Stage 3 increment 1).

Builds the device grouping (count -> padded cumsum -> scatter) and checks it
against a numpy reference that groups routings by expert. Slot order WITHIN an
expert is atomic-nondeterministic, so per-expert sorted_tk is compared as a SET.
"""
import numpy as np
import torch

import glq.inference_kernel as ik

TILE = 16


def main():
    assert ik._try_load_cuda_ext(), "glq CUDA ext failed to load/build"
    cuda = ik._glq_cuda
    assert hasattr(cuda, "glq_moe_build_grouping"), "glq_moe_build_grouping not in ext"

    for nt in [1, 32, 64, 256]:
        E, top_k = 128, 8
        torch.manual_seed(nt)
        topk = torch.randint(0, E, (nt, top_k), dtype=torch.int64, device="cuda")
        off, mi, stk = cuda.glq_moe_build_grouping(topk, E, top_k, TILE)
        off = off.cpu().numpy()
        mi = mi.cpu().numpy()
        stk = stk.cpu().numpy()
        flat = topk.reshape(-1).cpu().numpy()  # (R,) expert per routing
        R = nt * top_k

        cnt = np.bincount(flat, minlength=E)
        pad = ((cnt + TILE - 1) // TILE) * TILE
        ref_off = np.concatenate([[0], np.cumsum(pad)]).astype(np.int32)
        M_sum = int(off[E])

        assert np.array_equal(off, ref_off), f"nt={nt}: offset mismatch\n{off}\n{ref_off}"
        assert M_sum == int(ref_off[E]), f"nt={nt}: M_sum {M_sum} != {ref_off[E]}"

        for e in range(E):
            s, n = int(ref_off[e]), int(cnt[e])
            pe = int(ref_off[e + 1])
            real = stk[s:s + n]
            ref_routes = set(np.where(flat == e)[0].tolist())
            assert set(real.tolist()) == ref_routes, f"nt={nt} e={e}: routes mismatch"
            assert np.all(mi[s:s + n] == e), f"nt={nt} e={e}: m_indices != e"
            assert np.all(stk[s + n:pe] == -1), f"nt={nt} e={e}: sorted_tk padding != -1"
            assert np.all(mi[s + n:pe] == -1), f"nt={nt} e={e}: m_indices padding != -1"
        # every real routing placed exactly once
        placed = stk[stk >= 0]
        assert sorted(placed.tolist()) == list(range(R)), f"nt={nt}: not a permutation of [0,R)"
        print(f"nt={nt}: PASS  (R={R}, M_sum={M_sum}, active_experts={int((cnt>0).sum())})", flush=True)

    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
