"""P2/P3 Option-B e2e: independent-per-layer quant of e8p-2b vs trellis-2b → bf16 checkpoints.

Captures per-layer input Hessians once on the bf16 model (128 wikitext seqs), then quantizes
EVERY decoder linear independently for both arms (same H, same RHT protocol) and writes two
dequantized bf16 HF checkpoints. Independent-per-layer (no sequential propagation) is a fair,
CONSERVATIVE comparison — sequential prop would only compound a better codebook's advantage.
PPL is measured separately via _ppl_checkpoint.py. Pure scratch; no glq/ edits.
"""
import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _qtip_ldlq as qtl  # noqa: E402
from glq.codebook_e8p import E8PCodebook  # noqa: E402
from glq.quantize_model import HessianCapture, quantize_layer_e8_shell_rht  # noqa: E402

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def target_linears(model):
    return {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ".layers." in n and n.split(".")[-1] in PROJ}


def capture_hessians(model, tok, nsamples, seqlen, dev):
    caps = {n: HessianCapture(m) for n, m in target_linears(model).items()}
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    random.seed(0)
    with torch.no_grad():
        for i in range(nsamples):
            st = random.randint(0, ids.numel() - seqlen - 1)
            model(ids[st:st + seqlen].unsqueeze(0).to(dev))
            if (i + 1) % 32 == 0:
                print(f"  calib {i + 1}/{nsamples}", flush=True)
    return {n: c.finalize().float().cpu() for n, c in caps.items()}


def quant_arm(base, arm, H, out_dir, dev, variant="3inst"):
    """Load a fresh bf16 model, quantize every target linear with `arm`, save_pretrained."""
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=True)
    model.to(dev).eval()
    cb = qtl.TrellisCodebook(variant=variant, device=dev) if arm == "trellis" \
        else E8PCodebook(device=dev, verbose=False)
    lins = target_linears(model)
    for i, (name, lin) in enumerate(lins.items()):
        W = lin.weight.data.float()
        Hn = H[name].to(dev)
        if arm == "trellis":
            W_hat = qtl.quantize_layer_trellis_rht(W, Hn, cb)
        else:
            W_hat = quantize_layer_e8_shell_rht(W, Hn.clone(), cb, bpw=2, block_diagonal=True)[0]
        lin.weight.data = W_hat.to(torch.bfloat16).to(lin.weight.device)
        if (i + 1) % 40 == 0:
            print(f"  [{arm}] quantized {i + 1}/{len(lins)}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    del model
    torch.cuda.empty_cache()
    print(f"WROTE {arm} -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--out", default="/opt/dlami/nvme/qtip")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--arms", default="e8p,trellis")
    ap.add_argument("--variant", default="3inst", help="trellis variant: 3inst | hyb")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    tag = args.base.split("/")[-1]

    print(f"=== capture Hessians: {args.base} ({args.nsamples}x{args.seqlen}) ===", flush=True)
    m0 = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, trust_remote_code=True)
    m0.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.base)
    H = capture_hessians(m0, tok, args.nsamples, args.seqlen, dev)
    print(f"  captured {len(H)} layer Hessians", flush=True)
    del m0
    torch.cuda.empty_cache()

    for arm in args.arms.split(","):
        suffix = f"trellis-{args.variant}" if arm == "trellis" else arm
        print(f"\n=== quantize arm={arm} {'(' + args.variant + ')' if arm == 'trellis' else ''} ===", flush=True)
        quant_arm(args.base, arm, H, f"{args.out}/{tag}-{suffix}-2bit", dev, variant=args.variant)
    print("\nDONE. PPL both with benchmarks/_ppl_checkpoint.py", flush=True)


if __name__ == "__main__":
    main()
