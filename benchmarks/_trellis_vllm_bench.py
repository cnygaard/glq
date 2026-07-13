"""S2 gate: serve a GLQ-trellis checkpoint on vLLM under FULL cudagraph.

No `enforce_eager` — the whole point is that the fused trellis op captures. Reports coherence
plus decode tok/s at B=1 and B=32 (B=32 is the case the batched GEMM exists for: before it,
every decode step would have decompressed the full weight).
"""
import argparse
import time

import glq_vllm  # noqa: F401 — registers the GLQ quant config + custom ops with vLLM
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.60)
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    lbl = args.label or args.model.split("/")[-1]

    t0 = time.perf_counter()
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, dtype="float16")
    load_s = time.perf_counter() - t0
    print(f"LOAD {lbl}: {load_s:.1f}s (FULL cudagraph — no enforce_eager)", flush=True)

    greedy = SamplingParams(max_tokens=24, temperature=0.0)
    out = llm.generate(["The capital of France is"], greedy)
    print(f"SAMPLE {lbl}: {out[0].outputs[0].text!r}", flush=True)

    for B in (1, 32):
        sp = SamplingParams(max_tokens=args.new_tokens, temperature=0.0, ignore_eos=True)
        prompts = ["Write a short story about a robot who learns to paint."] * B
        llm.generate(prompts, sp)                       # warm the graphs
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"RESULT {lbl:22s} B={B:<3d} total={ntok / dt:8.1f} tok/s  "
              f"per-seq={ntok / dt / B:6.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
