"""E2E gate for the productized E8P-RVQ path: load a glq --codebook e8p checkpoint via HF
transformers, report GPU memory after load (should match the bpw), compute WikiText-2 PPL
(same harness as the prototype benchmarks/_e8p_real_model.py: test split, n_chunks×2048), and
a coherence generation. Optional --baseline runs the same PPL on the bf16 source for a paired gap.

Usage: python _e8p_e2e_eval.py --model <glq_dir> [--baseline HuggingFaceTB/SmolLM3-3B] [--n-chunks 128]
"""
import argparse
import time

import torch

import glq.hf_integration  # noqa: F401  — registers the GLQ HF quantizer (@register_quantizer("glq"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEV = "cuda"


def wikitext_ppl(model, tok, n_chunks, seqlen=2048):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n_chunks = min(n_chunks, enc.numel() // seqlen)
    nll, ntok = 0.0, 0
    for i in range(n_chunks):
        ids = enc[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(DEV)
        with torch.no_grad():
            nll += model(ids, labels=ids).loss.item() * (seqlen - 1)
        ntok += seqlen - 1
    return float(torch.exp(torch.tensor(nll / ntok))), n_chunks


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True)
    model.train(False)
    # Touch one forward so the lazy E8P extension build + grid caching happens before timing.
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]], device=DEV))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return model, tok, load_s, mem_gb


def coherence(model, tok):
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", default=None, help="bf16 source id for a paired PPL gap")
    ap.add_argument("--n-chunks", type=int, default=128)
    args = ap.parse_args()

    print(f"\n=== E8P e2e eval: {args.model} ===")
    model, tok, load_s, mem_gb = load(args.model)
    ppl, nc = wikitext_ppl(model, tok, args.n_chunks)
    gen = coherence(model, tok)
    print(f"load {load_s:.1f}s   GPU-mem-after-load {mem_gb:.2f} GiB")
    print(f"WikiText-2 PPL ({nc}x2048): {ppl:.3f}")
    print(f"coherence: {gen!r}")

    if args.baseline:
        del model
        torch.cuda.empty_cache()
        print(f"\n=== bf16 baseline: {args.baseline} ===")
        bmodel, btok, bload, bmem = load(args.baseline)
        bppl, _ = wikitext_ppl(bmodel, btok, args.n_chunks)
        print(f"bf16 GPU-mem {bmem:.2f} GiB   WikiText-2 PPL: {bppl:.3f}")
        print(f"\nGAP e8p - bf16 = {ppl - bppl:+.3f}   (footprint {mem_gb:.1f} vs {bmem:.1f} GiB)")


if __name__ == "__main__":
    main()
