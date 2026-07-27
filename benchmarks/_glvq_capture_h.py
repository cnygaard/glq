"""Capture real SmolLM3-3B weights + calibration Hessians for the GLVQ bake-off (Phase 1).

Reuses glq.quantize_model.HessianCapture (mean X^T X per layer, the exact object the
quantizer uses). nsamples=128 x 2048 wikitext-2-raw. Saves {name: {W, H}} for the
representative sublayers (down_proj large/sensitive n=11008; q_proj square 2048^2)
across shallow/mid/deep layers -> /opt/dlami/nvme/glvq/layers.pt.
"""
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from glq.quantize_model import HessianCapture

MODEL = os.environ.get("GLVQ_MODEL", "HuggingFaceTB/SmolLM3-3B")
LAYERS = [int(x) for x in os.environ.get("GLVQ_LAYERS", "0,18,35").split(",")]
# LDLQ/Viterbi cost is proportional to m*n, so a study that sweeps many arms wants
# uniformly-sized sublayers (down_proj is 5.4x a q_proj on SmolLM3-3B and would dominate
# the wall-clock). Override with GLVQ_SUBS to trade coverage against runtime.
SUBS = [s for s in os.environ.get(
    "GLVQ_SUBS", "self_attn.q_proj,mlp.down_proj").split(",") if s.strip()]
NSAMPLES, SEQLEN = 128, 2048
OUT = os.environ.get("GLVQ_OUT", "/opt/dlami/nvme/glvq/layers.pt")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    mods = dict(model.named_modules())
    names = [f"model.layers.{L}.{s}" for L in LAYERS for s in SUBS]
    for nm in names:
        assert nm in mods, f"missing {nm}; layer-0 keys: {[k for k in mods if 'layers.0.' in k][:10]}"
    caps = {nm: HessianCapture(mods[nm]) for nm in names}

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    ntok = ids.numel()
    random.seed(0)
    with torch.no_grad():
        for i in range(NSAMPLES):
            st = random.randint(0, ntok - SEQLEN - 1)
            model(ids[st:st + SEQLEN].unsqueeze(0).cuda())
            if (i + 1) % 32 == 0:
                print(f"calib {i + 1}/{NSAMPLES}", flush=True)

    out = {}
    for nm, cap in caps.items():
        H = cap.finalize().float().cpu()
        W = mods[nm].weight.detach().float().cpu()
        out[nm] = {"W": W, "H": H}
        print(nm, "W", tuple(W.shape), "H", tuple(H.shape),
              "Hdiag_mean", round(float(H.diagonal().mean()), 4), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(out, OUT)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
