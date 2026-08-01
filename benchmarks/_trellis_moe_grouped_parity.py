"""Synthetic gate for the grouped 3INST trellis MoE kernel (Phase B).

Runs on the box, no checkpoint needed. Three legs, in increasing strength:

  L1  expert-index invariance — EXACT (torch.equal)
      Roll every per-expert buffer by one slot and roll the routing to match. The output
      must be bit-identical. This isolates precisely what the grouped kernel added over the
      single matmul: the ``eidx = m_indices[base]`` route, the ``eidx * w_estride`` weight
      base, and the per-expert ``wscale_dev[eidx]``. A stride in the wrong units, an int
      overflow, or a scale read from the wrong expert all break it and nothing else would.
      Everything downstream (RHT, activation, scatter-reduce) is shared with the already
      validated e8p path, so a diff here is the new code.

  L2  fused vs the Phase A per-expert loop — tolerance + SQNR + max|Δ|
      NOT asserted exact, deliberately. The two paths differ in the output-RHT kernel
      (block-diag vs grouped), the activation (torch vs the batched CUDA kernel) and the
      topk-weighted sum (Python `+=` per expert vs a fixed-order fp32 scatter-reduce), so
      demanding torch.equal here would be asserting something the code never promised. The
      max|Δ| is PRINTED so the run states how close it actually is.

  L3  determinism — EXACT across two identical calls. The grouping uses atomics for slot
      PLACEMENT, so this is worth pinning: the numeric reduce is fixed-order and must not
      inherit that nondeterminism.

The 6 bpw leg exercises the stacked-RVQ accumulate pass. No trellis MoE checkpoint above
4 bpw exists, so this synthetic run is the ONLY gate that arm has — say so anywhere its
results are reported.

    python benchmarks/_trellis_moe_grouped_parity.py            # 4 and 6 bpw
    python benchmarks/_trellis_moe_grouped_parity.py --bpw 4
"""
from __future__ import annotations

import argparse
import sys
import types

import torch


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float()
    err = (ref - got).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    if err == 0.0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(sig / err)).item()


def build_layer(bpw, num_experts, hidden, inter, device, seed=0):
    """A loaded trellis MoE layer with random-but-valid packed weights.

    Random int16 IS a valid 3INST pack: the decode is arithmetic (a uint32 hash), not a
    lookup, so every bit pattern maps to a representable fp16 pair. Only ``pack_trellis``
    on the quantize side cares about the tail-biting overlap invariant.
    """
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod

    method = GLQFusedMoEMethod.__new__(GLQFusedMoEMethod)
    method.codebook_type = "trellis"
    method.quant_config = types.SimpleNamespace(variant="3inst", bpw=bpw)
    method.moe = types.SimpleNamespace(is_act_and_mul=True)

    layer = torch.nn.Module()
    method.create_weights(layer, num_experts, hidden, inter, torch.float16,
                          weight_loader=lambda *a, **k: None)
    layer.activation = "gelu_tanh"                       # gemma-4's routed-expert activation

    g = torch.Generator(device="cpu").manual_seed(seed)
    for pfx in ("w13", "w2"):
        for name in (f"{pfx}_trellis_packed", f"{pfx}_trellis_packed2"):
            buf = getattr(layer, name)
            if buf.numel() > num_experts:                # a real stage, not the sentinel
                buf.data.copy_(torch.randint(-32768, 32767, buf.shape,
                                             generator=g, dtype=torch.int16))
        # RHT sign vectors are +-1; anything else is not a Hadamard sign flip.
        for name in (f"{pfx}_SU", f"{pfx}_SV"):
            buf = getattr(layer, name)
            signs = (torch.randint(0, 2, buf.shape, generator=g) * 2 - 1).half()
            buf.data.copy_(signs)
        getattr(layer, f"{pfx}_Wscale").data.copy_(
            0.01 + 0.002 * torch.arange(num_experts, dtype=torch.float32))
        if bpw >= 5:
            getattr(layer, f"{pfx}_inv_resid_scale2").data.copy_(
                0.2 + 0.01 * torch.arange(num_experts, dtype=torch.float32))

    for name, buf in list(layer.named_parameters()) + list(layer.named_buffers()):
        buf.data = buf.data.to(device)
    method._process_trellis(layer)
    return method, layer


def _roll_experts(layer, shift=1):
    """Rotate every per-expert buffer by `shift` along the expert axis (SV is shared)."""
    for pfx in ("w13", "w2"):
        for name in (f"{pfx}_trellis_packed", f"{pfx}_trellis_packed2",
                     f"{pfx}_SU", f"{pfx}_Wscale", f"{pfx}_inv_resid_scale2"):
            buf = getattr(layer, name)
            buf.data.copy_(torch.roll(buf.data, shifts=shift, dims=0))


def run(bpw, num_experts=8, hidden=256, inter=128, tokens=6, top_k=2, device="cuda"):
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_moe_trellis_3inst"):
        print(f"  bpw {bpw}: FAIL — fused_moe_trellis_3inst is not registered. The CUDA "
              f"extension is stale or unbuilt; rm -rf ~/.cache/torch_extensions/*/glq_cuda")
        return False

    torch.manual_seed(1234)
    method, layer = build_layer(bpw, num_experts, hidden, inter, device)
    if not layer.glq_trellis_fused_ok:
        print(f"  bpw {bpw}: FAIL — glq_trellis_fused_ok is False for a servable shape")
        return False

    x = torch.randn(tokens, hidden, device=device, dtype=torch.float16) * 0.1
    # Distinct experts per token. A duplicate routing is legal for the kernel, but it makes
    # the eager loop's mask collapse two routings into one and L2 would then compare
    # different arithmetic, not different implementations.
    ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k]
                       for _ in range(tokens)]).to(torch.int64)
    w = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    w = w / w.sum(dim=1, keepdim=True)

    fused = method.apply(layer, x, w, ids)
    ok = True

    # ---- L1: expert-index invariance, EXACT ----
    _roll_experts(layer, shift=1)
    fused_rolled = method.apply(layer, x, w, (ids + 1) % num_experts)
    _roll_experts(layer, shift=-1)                                   # restore
    if torch.equal(fused, fused_rolled):
        print(f"  bpw {bpw}  L1 expert-index invariance : EXACT")
    else:
        d = (fused.float() - fused_rolled.float()).abs().max().item()
        print(f"  bpw {bpw}  L1 expert-index invariance : FAIL  max|Δ| {d:.3e} "
              f"(SQNR {_sqnr_db(fused, fused_rolled):.1f} dB) — the per-expert weight base "
              f"or scale is indexed wrong")
        ok = False

    # ---- L2: fused vs the Phase A eager loop, tolerance ----
    loop = method._apply_trellis(layer, x, w, ids)
    d = (fused.float() - loop.float()).abs().max().item()
    s = _sqnr_db(loop, fused)
    tol_ok = torch.allclose(fused.float(), loop.float(), rtol=2e-2, atol=2e-2)
    print(f"  bpw {bpw}  L2 fused vs eager loop        : "
          f"{'PASS' if tol_ok else 'FAIL'}  max|Δ| {d:.3e}  SQNR {s:.1f} dB"
          f"{'  (bit-exact)' if d == 0.0 else ''}")
    ok = ok and tol_ok

    # ---- L3: determinism, EXACT ----
    again = method.apply(layer, x, w, ids)
    if torch.equal(fused, again):
        print(f"  bpw {bpw}  L3 determinism               : EXACT")
    else:
        print(f"  bpw {bpw}  L3 determinism               : FAIL — repeated calls differ; "
              f"the atomics used for SLOT placement have leaked into the numeric reduce")
        ok = False

    if not torch.isfinite(fused).all():
        print(f"  bpw {bpw}  non-finite values in the fused output")
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", type=int, nargs="*", default=[4, 6])
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--inter", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=6)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    print(f"grouped trellis MoE parity — E={args.experts} hidden={args.hidden} "
          f"inter={args.inter} tokens={args.tokens} on {torch.cuda.get_device_name(0)}")
    results = {b: run(b, args.experts, args.hidden, args.inter, args.tokens)
               for b in args.bpw}
    bad = [b for b, r in results.items() if not r]
    print("\nVERDICT:", "PASS" if not bad else f"FAIL at bpw {bad}")
    if 6 in results and results[6]:
        print("note: the 6 bpw leg is the ONLY gate the stacked-RVQ arm has — no trellis "
              "MoE checkpoint above 4 bpw exists.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
