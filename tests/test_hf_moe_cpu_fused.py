"""HF's stacked gated experts, wired to the fused CPU MoE op.

`glq_fused_moe_trellis_3inst_cpu` shipped in 0.8.17 and has been serving vLLM-CPU since;
`GLQStackedGatedExperts` (the HF container) kept running a Python loop of `E8RHTLinear`
calls beside it. This connects the two, so HF-CPU and vLLM-CPU run one MoE path.

The op wants stacked `(E, tiles, 16R)` tensors and GLQ stores per-expert buffers. The CUDA
container solves that with `torch.stack`, which here would duplicate 42 GiB of expert
weights on Qwen3.8-Flash-Next. So the build *re-homes* instead: allocate the stacked buffer
once, copy each expert in, and point the per-expert linear at `stacked[e]`. The Python loop
keeps working unchanged on those views, which is what makes it available as the A/B oracle.

Two things these tests are careful about:

* **No bit-exactness is claimed.** The loop rounds the `gate_up` output to the activation
  dtype before the gated multiply and reduces with `index_add_` in expert order; the op
  stays fp32 throughout and reduces in fixed k order. The comparison is a tolerance, and
  the tightest form of it is float32 activations.
* **Refusals are asserted by their reason**, not just by "it fell back". A silent drop to
  the loop reads as "GLQ is slow on CPU" rather than as a missing gate.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch", reason="torch not installed")

from glq import trellis as gt  # noqa: E402
from glq.fused_experts import (  # noqa: E402
    GLQStackedGatedExperts,
    _gated_activation_id,
)
from glq.inference_kernel_cpu import _try_load_cpu_ext  # noqa: E402

HIDDEN, INTER, K = 64, 64, 4


def _ext():
    if not _try_load_cpu_ext():
        pytest.skip("CPU extension not available")
    from glq import inference_kernel_cpu as ikc
    if not hasattr(ikc._glq_cpu, "glq_fused_moe_trellis_3inst_cpu"):
        pytest.skip("fused CPU MoE op not in this build")
    return ikc._glq_cpu


def _pack(m, n, seed):
    """One projection's codes, quantized on CPU in kernel layout."""
    torch.manual_seed(seed)
    cb = gt.TrellisCodebook(variant="3inst", K=K, device="cpu")
    W = (torch.randn(m, n) * 0.05).float()
    _, Qidxs, _ = gt.trellis_ldlq(W, torch.eye(n), cb, for_kernel=True)
    return gt.pack_layer(cb, Qidxs, m, n, has_kernel=True)


def _signs(n, g):
    return torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).half()


def build_container(E=4, hidden=HIDDEN, inter=INTER, seed=0, shared_sv=True,
                    act=None) -> GLQStackedGatedExperts:
    """A loaded `GLQStackedGatedExperts`: fused gate_up per expert, landing pads dropped.

    Mirrors the state `_process_model_after_weight_loading` leaves behind -- buffers
    populated, `fuse_gate_up()` already run -- without needing a checkpoint on disk.
    """
    c = GLQStackedGatedExperts(E, hidden, inter, act or torch.nn.SiLU(),
                               codebook_type="trellis")
    g = torch.Generator().manual_seed(seed + 7)
    sv13, sv2 = _signs(hidden, g), _signs(inter, g)
    for e in range(E):
        pair = c[e]
        f, d = pair.gate_up_proj, pair.down_proj
        f.trellis_packed = _pack(2 * inter, hidden, seed + e)
        f.SU = _signs(2 * inter, g)
        f.SV = _signs(hidden, g) if not shared_sv else sv13.clone()
        f.Wscale = torch.tensor(0.5 + float(torch.rand(1, generator=g)))
        d.trellis_packed = _pack(hidden, inter, seed + 100 + e)
        d.SU = _signs(hidden, g)
        d.SV = sv2.clone()
        d.Wscale = torch.tensor(0.5 + float(torch.rand(1, generator=g)))
        # A fresh load leaves these unresolved; __init__ seeds 1.0, and the loop only
        # re-reads Wscale when it is None. Without this the oracle would silently use 1.0.
        f._wscale_float = None
        d._wscale_float = None
        pair.gate_proj = None
        pair.up_proj = None
        pair._fused = True
    return c


def _route(T, E, topk, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(T)]).long()
    wts = torch.rand(T, topk, generator=g).float()
    return ids, wts / wts.sum(dim=1, keepdim=True)


# ---- the activation resolver ---------------------------------------------------------

@pytest.mark.parametrize("mod,want", [
    (torch.nn.SiLU(), 0),
    (torch.nn.GELU(approximate="tanh"), 1),
    (torch.nn.Tanh(), None),          # not a gated MoE activation -> refuse
    (torch.nn.Identity(), None),
])
def test_the_activation_resolver_refuses_what_it_cannot_name(mod, want):
    """A wrong id is not a crash, it is finite plausible wrong output -- so anything not
    positively recognised must come back None and take the loop."""
    assert _gated_activation_id(mod) == want


def test_the_resolver_handles_a_bare_function():
    """`_replace_stacked_gated_experts` falls back to `F.silu` when the native container
    exposes no `act_fn` module."""
    import torch.nn.functional as F
    assert _gated_activation_id(F.silu) == 0


# ---- the re-homing build -------------------------------------------------------------

def test_the_build_repoints_per_expert_buffers_at_slices():
    """The mechanism, not the output: after the build, expert e's `trellis_packed` must BE
    row e of the stacked buffer. If it is a copy the memory doubles, and on the real model
    that is 42 GiB."""
    _ext()
    c = build_container(E=3)
    assert c._build_stacked_cpu() is None, c._build_stacked_cpu()
    for e in range(3):
        assert c._w13_packed[e].data_ptr() == c[e].gate_up_proj.trellis_packed.data_ptr()
        assert c._w2_packed[e].data_ptr() == c[e].down_proj.trellis_packed.data_ptr()


def test_the_build_does_not_duplicate_the_weights():
    _ext()
    c = build_container(E=3)

    def packed_bytes():
        seen, total = set(), 0
        for e in range(3):
            for lin in (c[e].gate_up_proj, c[e].down_proj):
                t = lin.trellis_packed
                if t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    total += t.numel() * t.element_size()
        return total

    before = packed_bytes()
    c._build_stacked_cpu()
    assert packed_bytes() == before


# ---- fusing straight into the stacked buffers -----------------------------------------

def _pad(hidden, packed_rows, su, sv, wscale):
    """A gate/up landing pad as the loader leaves it: one `[I, H]` half of the pair."""
    from glq.quantized_linear import E8RHTLinear
    lin = E8RHTLinear(hidden, INTER, bias=False, block_diagonal=True,
                      codebook_type="trellis")
    lin.trellis_packed = packed_rows.clone()
    lin.SU = su.clone()
    lin.SV = sv.clone()
    lin.Wscale = wscale.clone()
    lin._wscale_float = None
    return lin


def _unfused_container(E=3, **kw):
    """A container as the loader leaves it: gate/up pads populated, nothing fused yet.

    Built by splitting an already-fused container back into halves, so the pads carry
    exactly the bytes `fuse_gate_up` would have concatenated -- which is what makes the
    byte-identity comparison meaningful rather than circular.
    """
    c = build_container(E=E, **kw)
    for e in range(E):
        p = c[e]
        f = p.gate_up_proj
        hp = f.trellis_packed.shape[0] // 2          # packed rows per half
        hs = f.SU.shape[0] // 2                      # SU is a row artifact too
        p.gate_proj = _pad(f.in_features, f.trellis_packed[:hp], f.SU[:hs], f.SV, f.Wscale)
        p.up_proj = _pad(f.in_features, f.trellis_packed[hp:], f.SU[hs:], f.SV, f.Wscale)
        f.trellis_packed = torch.zeros(0, dtype=torch.int16)
        p._fused = False
    return c


def test_fusing_into_stacked_is_byte_identical_to_fusing_then_rehoming():
    """THE gate. Both routes must land the same bytes in the stacked buffer.

    `fuse_gate_up` + `_build_stacked_cpu` allocates the expert bytes twice on the way;
    `fuse_into_stacked` copies the halves straight into slices of the destination. That is
    only valid if gate occupies rows 0:I and up rows I:2I -- and getting it wrong is the
    gate/up-swap failure: loads clean, decodes finite, emits garbage. Unlike the
    fused-vs-loop numerics this CAN be bit-exact, so assert equality, not a tolerance.
    """
    _ext()
    old = _unfused_container(E=4)
    for e in range(4):
        old[e].fuse_gate_up()
    assert old._build_stacked_cpu() is None

    new = _unfused_container(E=4)
    assert new.fuse_into_stacked() is None

    assert torch.equal(new._w13_packed, old._w13_packed), "w13 bytes differ"
    assert torch.equal(new._w2_packed, old._w2_packed), "w2 bytes differ"
    assert torch.equal(new._w13_SU, old._w13_SU)
    assert torch.equal(new._w2_SU, old._w2_SU)
    assert torch.equal(new._w13_SV, old._w13_SV)


def test_a_swapped_gate_up_order_would_not_be_byte_identical():
    """Proves the test above is load-bearing rather than vacuously true: if the two halves
    were interchangeable, swapping them would still match and the gate would prove nothing.
    """
    _ext()
    old = _unfused_container(E=3)
    for e in range(3):
        old[e].fuse_gate_up()
    old._build_stacked_cpu()
    half = old._w13_packed.shape[1] // 2
    swapped = torch.cat([old._w13_packed[:, half:], old._w13_packed[:, :half]], dim=1)
    assert not torch.equal(swapped, old._w13_packed), \
        "the halves are identical, so the byte-identity gate cannot see an order error"


def test_fusing_into_stacked_leaves_no_per_expert_intermediate():
    """The point of the change: every expert's packed codes must BE a slice of the stacked
    buffer, so the `[2I, H]` per-expert allocation never exists."""
    _ext()
    c = _unfused_container(E=4)
    assert c.fuse_into_stacked() is None
    for e in range(4):
        assert c._w13_packed[e].data_ptr() == c[e].gate_up_proj.trellis_packed.data_ptr()
        assert c._w2_packed[e].data_ptr() == c[e].down_proj.trellis_packed.data_ptr()
        assert c[e].gate_proj is None and c[e].up_proj is None
    assert c._stacked_is_live()


def test_fusing_into_stacked_is_idempotent_and_skips_the_lazy_build():
    _ext()
    c = _unfused_container(E=3)
    assert c.fuse_into_stacked() is None
    ptr = c._w13_packed.data_ptr()
    assert c.fuse_into_stacked() is None
    assert c._build_stacked_cpu() is None, "the lazy build must no-op once already stacked"
    assert c._w13_packed.data_ptr() == ptr


def test_fusing_into_stacked_declines_when_already_fused():
    """The loader calls this before fuse_gate_up; if the pads are gone it must hand back a
    reason so the caller falls through, not build from a half-populated container."""
    _ext()
    c = build_container(E=3)          # build_container already leaves them fused
    why = c.fuse_into_stacked()
    assert why is not None and "fuse_gate_up" in why, why


def test_an_unshared_sv_is_refused_by_the_direct_path_too():
    """fuse_into_stacked runs BEFORE the pads are dropped, so it cannot call the refusal
    gate (which reads the fused projection). The SV check has to be re-done in the shared
    tail, or this path would happily stack experts that do not share an RHT basis."""
    _ext()
    c = _unfused_container(E=3, shared_sv=False)
    why = c.fuse_into_stacked()
    assert why is not None and "SV" in why, why


# ---- returning the freed heap to the OS ----------------------------------------------

def test_the_build_asks_the_allocator_to_return_freed_pages(monkeypatch):
    """The mechanism, not the RSS: an RSS assertion would be flaky in CI, but a build that
    silently stopped trimming would put the ~47 GiB back and nothing else here would fail.

    Measured on a one-layer repro (512 experts, 900 MiB of codes) with other allocations
    interleaved between the experts, as a real checkpoint load does: RSS grew by the
    destination's FULL size (+900.0 MiB) and fell to +2.7 MiB after the trim.
    """
    _ext()
    from glq import fused_experts as fe
    calls = []
    monkeypatch.setattr(fe, "_return_freed_heap_to_os",
                        lambda: calls.append(1) or True)
    c = build_container(E=3)
    assert c._build_stacked_cpu() is None
    assert calls, "the re-home did not ask the allocator to return the freed pages"


def test_the_trim_helper_is_best_effort(monkeypatch):
    """glibc-only by design. musl and macOS have no malloc_trim, and a missing one costs
    footprint, never correctness -- so it must report False, not raise."""
    from glq import fused_experts as fe
    monkeypatch.setattr(fe, "_MALLOC_TRIM", None)
    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("no libc")))
    assert fe._return_freed_heap_to_os() is False
    assert fe._MALLOC_TRIM is False, "the failure should be cached, not retried per layer"
    assert fe._return_freed_heap_to_os() is False


def test_the_trim_helper_works_where_it_exists(monkeypatch):
    import platform

    from glq import fused_experts as fe
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc":
        pytest.skip("no glibc malloc_trim on this platform")
    monkeypatch.setattr(fe, "_MALLOC_TRIM", None)
    assert fe._return_freed_heap_to_os() is True


def test_the_build_is_idempotent():
    _ext()
    c = build_container(E=3)
    assert c._build_stacked_cpu() is None
    ptr = c._w13_packed.data_ptr()
    assert c._build_stacked_cpu() is None
    assert c._w13_packed.data_ptr() == ptr, "second build should be a no-op"


def test_stale_slices_are_rebuilt_rather_than_used():
    """Plain attributes do not follow a `.to()`, so a device move re-allocates the
    per-expert views and leaves the stacked buffer pointing nowhere useful. The identity
    check must notice and rebuild."""
    _ext()
    c = build_container(E=3)
    c._build_stacked_cpu()
    old = c._w13_packed.data_ptr()
    # Simulate what `.to()` does to a buffer: replace it with a fresh allocation.
    c[0].gate_up_proj.trellis_packed = c[0].gate_up_proj.trellis_packed.clone()
    assert not c._stacked_is_live()
    c._build_stacked_cpu()
    assert c._w13_packed.data_ptr() != old
    assert c._stacked_is_live()


# ---- numerics ------------------------------------------------------------------------

#: Sized from the measured agreement, not from habit. On this fixture the two paths differ
#: by 6.1e-5 on values with an RMS of ~272, so a percentage-style rtol would admit errors of
#: O(0.2) and catch nothing. The sensitivity tests below pin that this is tight enough to
#: see a wrong expert (923) and a gate/up swap (1934).
TOL = dict(atol=5e-4, rtol=1e-6)


@pytest.mark.parametrize("T,topk", [(1, 2), (1, 4), (3, 2), (12, 3)])
def test_fused_matches_the_python_loop(T, topk, monkeypatch):
    """fp32 in, fp32 out: the tightest comparison available. Not bit-exact -- the loop
    reduces with index_add_ in expert order and the op in fixed k order."""
    _ext()
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    c = build_container(E=6)
    x = torch.randn(T, HIDDEN)
    ids, wts = _route(T, 6, topk, seed=11)

    fused = c._try_fused_cpu(x, ids, wts)
    assert fused is not None, "the fused path should be eligible for this container"
    ref = c._loop_forward(x, ids, wts)
    assert torch.allclose(fused, ref, **TOL), (fused - ref).abs().max().item()


def test_the_comparison_can_tell_the_experts_apart(monkeypatch):
    """A tolerance that passes no matter which expert ran would make every numeric test
    above vacuous. Feed the op a permuted routing and require the comparison to break."""
    _ext()
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    c = build_container(E=6)
    x = torch.randn(4, HIDDEN)
    ids, wts = _route(4, 6, 3, seed=11)
    ref = c._loop_forward(x, ids, wts)
    wrong = c._try_fused_cpu(x, (ids + 1) % 6, wts)
    assert not torch.allclose(wrong, ref, **TOL), (wrong - ref).abs().max().item()


def test_the_comparison_catches_a_gate_up_swap(monkeypatch):
    """The failure that cost a day earlier in this work: gate is rows 0:I of w13 and up is
    I:2I. Swapping them loads cleanly, decodes finitely, and emits garbage."""
    _ext()
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    c = build_container(E=6)
    x = torch.randn(4, HIDDEN)
    ids, wts = _route(4, 6, 3, seed=11)
    ref = c._loop_forward(x, ids, wts)
    assert c._build_stacked_cpu() is None
    half = c._w13_packed.shape[1] // 2
    c._w13_packed[:] = torch.cat(
        [c._w13_packed[:, half:], c._w13_packed[:, :half]], dim=1)
    swapped = c._try_fused_cpu(x, ids, wts)
    assert not torch.allclose(swapped, ref, **TOL)


def test_an_unrouted_expert_contributes_nothing(monkeypatch):
    _ext()
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    c = build_container(E=5)
    x = torch.randn(2, HIDDEN)
    ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    wts = torch.full((2, 2), 0.5)
    assert torch.allclose(c._try_fused_cpu(x, ids, wts),
                          c._loop_forward(x, ids, wts), **TOL)


def test_forward_prefers_the_fused_path_when_the_flag_is_on(monkeypatch):
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    _ext()
    c = build_container(E=4)
    x = torch.randn(2, HIDDEN)
    ids, wts = _route(2, 4, 2, seed=3)
    seen = []
    orig = GLQStackedGatedExperts._loop_forward

    def _spy(self, *a):
        seen.append(1)
        return orig(self, *a)

    monkeypatch.setattr(GLQStackedGatedExperts, "_loop_forward", _spy)
    out = c(x, ids, wts)
    assert not seen, "the loop ran even though the fused path was eligible"
    assert out.shape == (2, HIDDEN)


def test_forward_falls_back_to_the_loop_when_refused(monkeypatch):
    """The other half of the same assertion: when the gate says no, the loop must be what
    produces the answer -- not an exception, and not a silently wrong tensor."""
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    _ext()
    c = build_container(E=4, shared_sv=False)      # refused: SV not shared
    x = torch.randn(2, HIDDEN)
    ids, wts = _route(2, 4, 2, seed=3)
    seen = []
    orig = GLQStackedGatedExperts._loop_forward

    def _spy(self, *a):
        seen.append(1)
        return orig(self, *a)

    monkeypatch.setattr(GLQStackedGatedExperts, "_loop_forward", _spy)
    assert c(x, ids, wts).shape == (2, HIDDEN)
    assert seen, "refused, but the loop did not run"


# ---- refusals, each naming itself ----------------------------------------------------

def test_the_flag_off_takes_the_loop(monkeypatch):
    """Default is off while the win is being measured; the loop must be what runs."""
    _ext()
    monkeypatch.delenv("GLQ_HF_MOE_CPU_FUSED", raising=False)
    c = build_container(E=3)
    x = torch.randn(1, HIDDEN)
    ids, wts = _route(1, 3, 2, seed=5)
    assert c._try_fused_cpu(x, ids, wts) is None
    assert c(x, ids, wts).shape == (1, HIDDEN)


@pytest.mark.parametrize("switch", ["GLQ_FUSED_TRELLIS_CPU", "GLQ_MOE_FORCE_FALLBACK"])
def test_the_existing_switches_still_reach_this_path(switch, monkeypatch):
    """One gate says no, not three. Both pre-existing switches must keep working."""
    _ext()
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    monkeypatch.setenv(switch, "0" if switch == "GLQ_FUSED_TRELLIS_CPU" else "1")
    c = build_container(E=3)
    assert c._glq_moe_cpu_refusal() is not None


def test_the_expensive_scan_runs_once(monkeypatch):
    """Scanning 512 experts is free at load and ruinous per forward -- and the refused
    container is exactly where it would repeat, since `_stacked_is_live()` only
    short-circuits the accepted case. The env switches must stay live, though, or the A/B
    driver could not toggle the flag between forwards on one set of weights."""
    _ext()
    c = build_container(E=4, shared_sv=False)
    calls = []
    real = torch.equal
    monkeypatch.setattr(torch, "equal", lambda *a: calls.append(1) or real(*a))
    c._glq_moe_cpu_refusal()
    first = len(calls)
    assert first > 0, "the SV scan did not run at all"
    c._glq_moe_cpu_refusal()
    c._glq_moe_cpu_refusal()
    assert len(calls) == first, "the weight scan repeated instead of caching"


def test_the_env_switches_are_not_cached_with_the_weight_facts(monkeypatch):
    """Toggling the flag between forwards is how the real-model A/B avoids a second
    140 s load, so the env half of the gate must be re-read every time."""
    _ext()
    c = build_container(E=3)
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    assert c._glq_moe_cpu_refusal() is None
    monkeypatch.setenv("GLQ_MOE_FORCE_FALLBACK", "1")
    assert "GLQ_MOE_FORCE_FALLBACK" in c._glq_moe_cpu_refusal()
    monkeypatch.setenv("GLQ_MOE_FORCE_FALLBACK", "0")
    assert c._glq_moe_cpu_refusal() is None


def test_an_unshared_sv_is_refused_and_says_so():
    """The op applies ONE SV to every expert. That holds because the RHT seed is fixed per
    layer (glq/rht.py:162) -- a property of the checkpoint, not of the op. Decoding 511
    experts in the wrong basis yields finite, plausible, wrong numbers, so check it."""
    _ext()
    c = build_container(E=3, shared_sv=False)
    why = c._glq_moe_cpu_refusal()
    assert why is not None and "SV" in why, why


def test_a_hyb_layer_is_refused_and_says_so():
    _ext()
    c = build_container(E=3)
    c[0].gate_up_proj.tlut = torch.ones(16, dtype=torch.float16)
    why = c._glq_moe_cpu_refusal()
    assert why is not None and ("3INST" in why or "HYB" in why or "tlut" in why), why


def test_a_stage2_layer_is_refused_and_says_so():
    _ext()
    c = build_container(E=3)
    c[0].down_proj.trellis_packed2 = torch.zeros(8, 16, dtype=torch.int16)
    why = c._glq_moe_cpu_refusal()
    assert why is not None and "stage-2" in why, why


def test_an_unnameable_activation_is_refused_and_says_so():
    _ext()
    c = build_container(E=3, act=torch.nn.Tanh())
    why = c._glq_moe_cpu_refusal()
    assert why is not None and "activation" in why, why


def test_a_cuda_input_never_takes_this_path(monkeypatch):
    """CPU-only by construction: the container has no fused CUDA MoE path, and handing a
    GPU tensor to a CPU op would be an error rather than a fallback."""
    _ext()
    monkeypatch.setenv("GLQ_HF_MOE_CPU_FUSED", "1")
    c = build_container(E=3)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    x = torch.randn(1, HIDDEN, device="cuda")
    ids, wts = _route(1, 3, 2, seed=5)
    assert c._try_fused_cpu(x, ids.cuda(), wts.cuda()) is None
