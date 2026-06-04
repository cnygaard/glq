#!/usr/bin/env python
"""Decode tok/s for a GLQ checkpoint on vLLM, single-stream and batched.

Two-point isolation per batch size: time a 1-token generate and an N-token
generate over the SAME set of `batch` identical prompts; (t_N - t_1) cancels
prefill + scheduler overhead, so total decode throughput is
    batch * (N - 1) / (t_N - t_1)
ignore_eos forces exactly N tokens per sequence (we measure speed, not text).
Weight-only path — no E8-KV envs — matching what the README checkpoints table
advertises. Resident weight VRAM is read from vLLM's INFO memory-profiling
line ("model weights take X GiB"); run with VLLM_LOGGING_LEVEL=INFO to capture.
"""
import argparse
import time

from vllm import LLM, SamplingParams

PROMPT = "Tell me a long and detailed story about a robot who learns to paint."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.86)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--batches", default="1", help="comma list, e.g. 1,8,32")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    llm = LLM(
        model=args.model,
        quantization="glq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=False,
    )

    def run(batch, n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        prompts = [PROMPT] * batch
        t = time.perf_counter()
        out = llm.generate(prompts, sp)
        dt = time.perf_counter() - t
        ntok = sum(len(o.outputs[0].token_ids) for o in out)
        return dt, ntok

    run(max(batches), 16)            # warmup: capture cudagraphs at the top batch
    for b in batches:
        t1, _ = run(b, 1)            # prefill + 1 decode for b seqs
        tN, nN = run(b, args.decode)
        total_tps = b * (args.decode - 1) / (tN - t1)
        print(
            f"RESULT label={args.label} model={args.model} batch={b} "
            f"total_decode_tokps={total_tps:.1f} per_seq_tokps={total_tps / b:.1f} "
            f"ttft_ms={t1 * 1000:.0f} n_tokens={nN}"
        )


if __name__ == "__main__":
    main()
