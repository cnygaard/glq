"""★ The bit-exactness gate: the arena path must produce byte-identical output.

This is the claim S1 rests on, and it is the one that cannot be argued from the source alone.
The reasoning says it holds — `glq_cuda.cu` reaches a per-expert weight as
``base + eidx * estride``, and the final reduce is a *"fixed-order k loop per output element
-> deterministic (no atomicAdd)"*, so substituting slot ids changes only *where* a routing's
weights are read from — but a reduce whose grouping depends on the number of experts is
exactly the kind of thing that is right in the comment and wrong in the binary. The arena
makes the op see `n_slots` experts instead of `n_experts`, which changes `M_sum_max` and the
per-expert padding of the grouped buffer. So: run it.

Two properties, not one:

* **off == on.** Same tokens, same weights, byte-identical output.
* **cold == warm.** Output must not depend on which experts happen to be resident. If it
  does, exactness has been lost somewhere and the cache is unsafe to ship whatever the
  throughput says — the same prompt would answer differently depending on cache history.

Needs a GPU and the glq CUDA extension.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Small but legal for the fused op: m_pad % 32 == 0, n_pad % 64 == 0, stage-1 rate in 2..4.
HIDDEN, INTER = 128, 64
W13_OUT = 2 * INTER          # gated
BPW, N_EXPERTS, N_SLOTS = 3, 16, 8


def _ext_available():
    """Load the extension *and* register the torch ops.

    Loading the CUDA extension is not enough: `torch.ops.glq.fused_moe_trellis_3inst` is a
    `torch.library` entry declared in `glq_vllm/custom_ops.py` and only bound by
    `_ensure_registered()`. Measured on a fresh box, `_try_load_cuda_ext()` returned True
    while the op was still absent — which is precisely the state in which the tests below
    would all skip and appear to pass.
    """
    from glq import inference_kernel as ik
    ok = bool(ik._try_load_cuda_ext())
    try:
        from glq_vllm.custom_ops import _ensure_registered
        _ensure_registered()
    except Exception:                                 # noqa: BLE001 - reported by the guard
        return False
    return (ok and hasattr(torch.ops, "glq")
            and hasattr(torch.ops.glq, "fused_moe_trellis_3inst"))


def _build_layer(device="cuda", seed=0):
    """A synthetic trellis MoE layer: real shapes, real metadata, random bytes.

    The decode need not be *meaningful* — both arms decode the same bytes with the same
    kernel, so any self-consistent weights test the substitution. What must be real is the
    shape and metadata, since those drive the grouping this test exists to stress.
    """
    from glq.hadamard import _block_decompose as _bd
    from glq.quantized_linear import _pack_block_meta as _pbm
    from glq_vllm.moe_shapes import trellis_moe_shapes

    g = torch.Generator(device="cpu").manual_seed(seed)
    shapes = trellis_moe_shapes(N_EXPERTS, HIDDEN, INTER, W13_OUT, BPW)
    t = {}
    for name, shape in shapes.items():
        if "trellis_packed" in name:
            t[name] = torch.randint(-32768, 32767, shape, dtype=torch.int16,
                                    generator=g).to(device)
        elif name.endswith("SU") or name.endswith("SV"):
            # +/-1 RHT signs, as the quantizer emits.
            t[name] = (torch.randint(0, 2, shape, generator=g).to(torch.float16) * 2 - 1
                       ).to(device)
        elif name.endswith("Wscale"):
            t[name] = (torch.rand(shape, generator=g) * 0.02 + 0.01).to(device)
        else:                                     # inv_resid_scale2: 0 at 3 bpw
            t[name] = torch.zeros(shape, dtype=torch.float32, device=device)

    meta = {}
    for pfx, n_pad, m_pad in (("w13", HIDDEN, W13_OUT), ("w2", INTER, HIDDEN)):
        bn, bm = _bd(n_pad), _bd(m_pad)
        meta[pfx] = dict(
            n_pad=n_pad, m_pad=m_pad,
            _bn=torch.tensor(bn, dtype=torch.int64, device="cpu"),
            _bm=torch.tensor(bm, dtype=torch.int64, device="cpu"),
            _bnm=_pbm(bn).to(device), _bmm=_pbm(bm).to(device))
    return t, meta


def _call(weights, meta, x, ids, topk_weights):
    m13, m2 = meta["w13"], meta["w2"]
    return torch.ops.glq.fused_moe_trellis_3inst(
        x.half().contiguous(), ids.to(torch.int64), topk_weights.float().contiguous(),
        weights["w13_trellis_packed"], weights["w13_trellis_packed2"],
        weights["w13_SU"], weights["w13_SV"], weights["w13_Wscale"],
        weights["w13_inv_resid_scale2"],
        weights["w2_trellis_packed"], weights["w2_trellis_packed2"],
        weights["w2_SU"], weights["w2_SV"], weights["w2_Wscale"],
        weights["w2_inv_resid_scale2"],
        HIDDEN, INTER, W13_OUT,
        m13["n_pad"], m13["m_pad"], m2["n_pad"], m2["m_pad"],
        m13["_bn"], m13["_bm"], m13["_bnm"], m13["_bmm"],
        m2["_bn"], m2["_bm"], m2["_bnm"], m2["_bmm"],
        0)


def _arena_arm(weights, meta, x, ids, topk_weights, n_slots=N_SLOTS, warm=0):
    """Fill an arena from `weights`, then call the op against the arena with slot ids."""
    from glq_vllm.expert_arena import (ArenaDecision, ArenaWeights, ExpertArena,
                                       plan_step)

    per_expert = {k: v for k, v in weights.items() if v.shape[0] == N_EXPERTS}
    arena = ExpertArena(ArenaDecision(True, slots=n_slots, bytes_per_slot=1, n_layers=1,
                                      reason="test"),
                        per_expert, n_experts=N_EXPERTS)
    for _ in range(warm):                 # pre-resident, so the next step is all hits
        plan = plan_step(arena.state, ids.reshape(-1))
        arena.fill(plan)
    plan = plan_step(arena.state, ids.reshape(-1))
    arena.fill(plan)

    class _L:
        pass
    layer = _L()
    for k, v in weights.items():
        setattr(layer, k, v)
    W = ArenaWeights(arena.buffers, layer)
    wdict = {k: getattr(W, k) for k in weights}
    return _call(wdict, meta, x, plan.slot_of.view(ids.shape), topk_weights), arena


# ---------------------------------------------------------------------------

def test_the_extension_and_fused_op_are_available():
    """Guard the gate itself: a missing op would make every assertion below vacuous."""
    assert _ext_available(), (
        "glq CUDA extension or fused_moe_trellis_3inst missing — the exactness tests below "
        "would silently prove nothing")


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_the_arena_path_is_BYTE_IDENTICAL_to_the_full_tensor_path(n_tokens):
    """★ The gate. Only the weight address differs between the two arms."""
    if not _ext_available():
        pytest.skip("extension unavailable")
    torch.manual_seed(7)
    w, meta = _build_layer()
    top_k = 2
    x = torch.randn(n_tokens, HIDDEN, device="cuda", dtype=torch.float16)
    ids = torch.stack([torch.randperm(N_EXPERTS, device="cuda")[:top_k]
                       for _ in range(n_tokens)])
    tw = torch.rand(n_tokens, top_k, device="cuda")

    out_full = _call(w, meta, x, ids, tw)
    out_arena, arena = _arena_arm(w, meta, x, ids, tw)

    assert arena.state.fills > 0, "the arena served the step without filling anything"
    assert torch.equal(out_full, out_arena), (
        f"arena output differs from the full-tensor path: max|delta| = "
        f"{(out_full.float() - out_arena.float()).abs().max().item():.3e}. The reduce is a "
        f"fixed-order k loop, so this means the substitution changed more than the address.")


def test_output_does_not_depend_on_CACHE_WARMTH():
    """★ Cold vs warm. If output moved with residency, the same prompt would answer
    differently depending on history — which breaks reproducibility and every A/B
    comparison far more seriously than a few ULPs would."""
    if not _ext_available():
        pytest.skip("extension unavailable")
    torch.manual_seed(11)
    w, meta = _build_layer()
    n_tokens, top_k = 2, 2
    x = torch.randn(n_tokens, HIDDEN, device="cuda", dtype=torch.float16)
    ids = torch.stack([torch.randperm(N_EXPERTS, device="cuda")[:top_k]
                       for _ in range(n_tokens)])
    tw = torch.rand(n_tokens, top_k, device="cuda")

    cold, a_cold = _arena_arm(w, meta, x, ids, tw, warm=0)
    warm, a_warm = _arena_arm(w, meta, x, ids, tw, warm=3)
    assert a_warm.state.hits > 0, "the warm arm did not actually hit"
    assert torch.equal(cold, warm), "output changed with cache warmth"


def test_a_SMALLER_arena_still_matches_the_full_path():
    """The arena makes the op see `n_slots` experts rather than `n_experts`, which changes
    `M_sum_max` and the grouped buffer's per-expert padding. If the reduce were sensitive to
    that grouping, a different slot count would show it."""
    if not _ext_available():
        pytest.skip("extension unavailable")
    torch.manual_seed(13)
    w, meta = _build_layer()
    n_tokens, top_k = 3, 2
    x = torch.randn(n_tokens, HIDDEN, device="cuda", dtype=torch.float16)
    ids = torch.stack([torch.randperm(N_EXPERTS, device="cuda")[:top_k]
                       for _ in range(n_tokens)])
    tw = torch.rand(n_tokens, top_k, device="cuda")
    out_full = _call(w, meta, x, ids, tw)
    for n_slots in (6, 8, 16):
        out_arena, _ = _arena_arm(w, meta, x, ids, tw, n_slots=n_slots)
        assert torch.equal(out_full, out_arena), f"n_slots={n_slots} diverged"


def test_a_CONTROL_that_the_gate_can_fail():
    """A byte-identity test that cannot fail proves nothing. Corrupt one arena slot and the
    comparison must break — otherwise the arena is not the tensor the op is reading."""
    if not _ext_available():
        pytest.skip("extension unavailable")
    torch.manual_seed(17)
    w, meta = _build_layer()
    x = torch.randn(1, HIDDEN, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[2, 5]], device="cuda")
    tw = torch.rand(1, 2, device="cuda")
    out_full = _call(w, meta, x, ids, tw)

    from glq_vllm.expert_arena import (ArenaDecision, ArenaWeights, ExpertArena,
                                       plan_step)
    per_expert = {k: v for k, v in w.items() if v.shape[0] == N_EXPERTS}
    arena = ExpertArena(ArenaDecision(True, slots=8, bytes_per_slot=1, n_layers=1,
                                      reason="test"), per_expert, n_experts=N_EXPERTS)
    plan = plan_step(arena.state, ids.reshape(-1))
    arena.fill(plan)
    arena.buffers["w13_trellis_packed"][int(plan.slot_of[0])] += 1      # corrupt one slot

    class _L:
        pass
    layer = _L()
    for k, v in w.items():
        setattr(layer, k, v)
    W = ArenaWeights(arena.buffers, layer)
    bad = _call({k: getattr(W, k) for k in w}, meta, x, plan.slot_of.view(ids.shape), tw)
    assert not torch.equal(out_full, bad), (
        "corrupting an arena slot did not change the output — the op is not reading the "
        "arena, so the exactness tests above are vacuous")
