"""Generic GLQ serve smoke on vLLM: load + generate a coherent sample + decode tok/s.
Model-agnostic (dense or MoE); used to re-verify published checkpoints serve on a
given vLLM version after the glq_vllm plugin port.

python _serve_smoke.py --model /opt/dlami/nvme/<dir> --quant glq
"""
import argparse
import os
import time

import glq_vllm  # noqa: F401
from vllm import LLM, SamplingParams


def _rss_gib():
    """Resident set of this process, which on the CPU backend is where the weights live."""
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / (1 << 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="glq")
    ap.add_argument("--maxtok", type=int, default=200)
    ap.add_argument("--mm0", action="store_true", help="pass limit_mm_per_prompt=0 (gemma-4 mm archs)")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU backend: no gpu_memory_utilization (the KV pool is sized by "
                         "VLLM_CPU_KVCACHE_SPACE), eager, and report load time / TTFT / RSS")
    # gemma-4 repeats at temperature 0 — sample from the model card instead of greedily,
    # or the coherence check reads as a quality failure that is really a sampling one.
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--seqs", default="1,8", help="batch sizes for the decode sweep")
    ap.add_argument("--dectok", type=int, default=256, help="tokens per decode-sweep sequence")
    args = ap.parse_args()
    kw = dict(model=args.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=2048)
    if args.cpu:
        kw["enforce_eager"] = True
    else:
        kw["gpu_memory_utilization"] = 0.9
    if args.quant == "glq":
        kw["quantization"] = "glq"
    if args.mm0:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
    t_load = time.perf_counter()
    llm = LLM(**kw)
    t_load = time.perf_counter() - t_load
    if args.cpu:
        print(f"LOAD: {t_load:.1f}s   RSS after load: {_rss_gib():.2f} GiB "
              f"(weights + KV pool + activations)", flush=True)
        t0 = time.perf_counter()
        llm.generate(["Hello"], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
        print(f"TTFT (1-token prompt, cold): {time.perf_counter() - t0:.2f}s", flush=True)

    msgs = [{"role": "user", "content": "In one short paragraph, explain what a rainbow is."}]
    sp = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=args.maxtok,
                        **({"top_k": args.top_k} if args.top_k > 0 else {}))
    out = llm.chat([msgs], sp, use_tqdm=False)
    print("=== SAMPLE (coherence check) ===", flush=True)
    print(out[0].outputs[0].text[:800], flush=True)

    for nseq in [int(s) for s in args.seqs.split(",")]:
        llm.generate(["Hello"] * nseq, SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False)
        sp2 = SamplingParams(temperature=0.0, max_tokens=args.dectok, ignore_eos=True)
        t0 = time.perf_counter()
        o = llm.generate(["Tell me about the sea:"] * nseq, sp2, use_tqdm=False)
        dt = time.perf_counter() - t0
        tok = sum(len(x.outputs[0].token_ids) for x in o)
        print(f"DECODE B={nseq}: {tok} tok in {dt:.2f}s = {tok / dt:7.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
