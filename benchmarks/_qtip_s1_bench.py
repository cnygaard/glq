"""S1 payoff gates: does the trellis CUDA kernel deliver VRAM + decode speed on a real model?

Reports, per checkpoint: weight VRAM right after load (must track bpw, NOT dense), and B=1
decode tok/s (the fused GEMV keeps weights compressed end-to-end). Run one model per process
so the VRAM number is clean.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# REQUIRED: importing this registers the GLQ HF quantizer (@register_quantizer("glq")).
# Without it transformers silently ignores quantization_config, builds the model dense, and
# you measure bf16 VRAM + generate garbage — which is exactly what happened the first time.
import glq.hf_integration  # noqa: F401,E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    torch.cuda.synchronize()
    weights_mb = (torch.cuda.memory_allocated() - base) / 2**20

    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

    with torch.no_grad():                                   # warmup (also triggers any lazy decode)
        model.generate(ids, max_new_tokens=args.warmup, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    after_warm_mb = (torch.cuda.memory_allocated() - base) / 2**20
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=args.new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]

    lbl = args.label or args.model.split("/")[-1]
    print(f"RESULT {lbl:28s} weights={weights_mb:8.1f} MiB  "
          f"after_gen={after_warm_mb:8.1f} MiB  peak={peak_mb:8.1f} MiB  "
          f"decode_b1={n_new / dt:6.2f} tok/s", flush=True)
    print(f"  sample: {tok.decode(out[0, -12:], skip_special_tokens=True)!r}", flush=True)


if __name__ == "__main__":
    main()
