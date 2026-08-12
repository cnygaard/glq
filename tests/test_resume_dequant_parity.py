"""Resume replay must reconstruct the SAME dense weight the quantizer wrote back.

The end-to-end gate on SmolLM2-135M showed the failure precisely: layers restored from
shards were byte-identical, but every layer from the first RE-quantized one onward
differed. That localises the fault to the weight reconstruction, not the store — replay
puts a slightly-wrong W_hat into the layer, the calibration forward then produces slightly
different activations, and every later layer is quantized against them. Nothing raises;
the checkpoint just quietly stops matching.

This is the isolated version of that gate: quantize ONE matrix, then rebuild it from its
own artifacts. ``quantize_layer_trellis_rht`` returns ``(W_hat, artifacts)`` — the exact
pair replay has to agree on — so ``torch.equal`` here is necessary and sufficient for the
replayed activations to match.

GPU-only: the trellis Viterbi encoder needs CUDA.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="trellis encode requires CUDA")


def _fixture(m, n, seed=0):
    """A weight plus a real (PSD) Hessian from sampled activations — an identity H would
    make LDLQ's error feedback trivial and could hide an ordering bug."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = torch.randn(m, n, generator=g, device="cuda", dtype=torch.float32)
    X = torch.randn(512, n, generator=g, device="cuda", dtype=torch.float32)
    H = (X.T @ X) / X.shape[0]
    H += torch.eye(n, device="cuda") * 1e-2      # keep it comfortably PD
    return W, H


@pytest.mark.parametrize("m,n,bpw", [(576, 576, 4), (1536, 576, 4), (576, 1536, 4),
                                     (576, 576, 5), (576, 576, 6), (576, 576, 8)])
def test_dequantize_artifacts_reproduces_quantizer_W_hat(m, n, bpw):
    """The whole correctness argument for resume, in one assertion.

    bpw >= 5 is the stacked RVQ (stage 1 K=4 + a residual stage), which replay must sum
    with the cumulative inverse residual scale; getting that wrong is invisible in the
    shapes and only shows up as drifted weights."""
    from glq.trellis import (TrellisCodebook, quantize_layer_trellis_rht,
                             trellis_rvq_recipe)
    from glq.resume import dequantize_artifacts

    cbs = [TrellisCodebook(variant="3inst", K=k, device="cuda")
           for k in trellis_rvq_recipe(bpw)]
    cb = cbs[0]
    cb.rvq_stages = cbs                      # how the driver hands stages to the quantizer
    W, H = _fixture(m, n)
    W_hat, arts = quantize_layer_trellis_rht(W, H, cb)

    W_re = dequantize_artifacts(arts, in_features=n, out_features=m,
                                codebook_type="trellis", codebook=cb)

    assert W_re.shape == W_hat.shape, f"{tuple(W_re.shape)} != {tuple(W_hat.shape)}"
    W_re = W_re.to(W_hat.device, W_hat.dtype)
    if not torch.equal(W_re, W_hat):
        d = (W_re - W_hat).abs()
        ratio = (W_hat / W_re.clamp(min=1e-12))
        pytest.fail(
            f"replay W_hat != quantizer W_hat for ({m},{n}) @ {bpw}bpw: "
            f"max|d|={d.max():.6e} mean|d|={d.mean():.6e} "
            f"rel={d.max() / W_hat.abs().max():.6e} | "
            f"ratio mean={ratio.mean():.6f} std={ratio.std():.6f} "
            f"(tight non-1.0 mean => pure scale factor; large std => wrong "
            f"transpose/block order)")
