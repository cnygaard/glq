"""MoE kernel parity + bench harness for the 26B-A4B GLQ model.

Modes (argv[1]):
  fallback : force the Python per-expert loop (GLQ_MOE_FORCE_FALLBACK=1) -- the
             correctness ORACLE. Independent of the C++ MoE kernel.
  fused    : use the C++ fused block-diag MoE kernel (after the Stage-1 fix).

Prints, for a fixed greedy prompt set, the first 48 output token IDs per prompt
(PARITY[i]: ...) so a later run can be diffed token-for-token, plus b1/b32 tok/s.
Eager (enforce_eager=True) for Stages 0-1; Stage 2 flips cudagraphs on separately.
"""
import os
import sys
import time


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fused"
    os.environ.pop("GLQ_MOE_FORCE_FALLBACK", None)
    os.environ.pop("GLQ_MOE_GROUPED", None)
    if mode == "fallback":
        os.environ["GLQ_MOE_FORCE_FALLBACK"] = "1"   # Python oracle
    elif mode == "grouped":
        os.environ["GLQ_MOE_GROUPED"] = "1"          # force grouped (incl. b1)
    elif mode == "blockdiag":
        os.environ["GLQ_MOE_GROUPED"] = "0"          # force block-diag (no grouped)
    # else "fused"/"default": both unset -> SHIPPED DEFAULT gate
    #   (b1 -> block-diag matvec; num_tokens >= 2 -> grouped-GEMM)
    import glq_vllm  # noqa
    from vllm import LLM, SamplingParams

    # GLQ_PARITY_MODEL overrides the model (use the on-disk 3bpw to skip a download);
    # GLQ_PARITY_EAGER=0 enables FULL cudagraph (capture + cudagraph-throughput run).
    model = os.environ.get("GLQ_PARITY_MODEL", "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw")
    eager = os.environ.get("GLQ_PARITY_EAGER", "1") == "1"
    kw = dict(model=model, quantization="glq", dtype="bfloat16", trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.90, max_num_seqs=32,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    if eager:
        kw["enforce_eager"] = True
    else:
        # GLQ_PARITY_CAPTURE tunes FULL-cudagraph capture sizes (24GB GPUs can't
        # fit the full [1..64] set with the ~15GB 26B model -> use "1,2,4,8").
        _caps = [int(s) for s in os.environ.get(
            "GLQ_PARITY_CAPTURE", "1,2,4,8,16,32,64").split(",") if s.strip()]
        kw["compilation_config"] = {"cudagraph_mode": "FULL",
                                    "cudagraph_capture_sizes": _caps}
    llm = LLM(**kw)

    # Fixed parity prompts (chat-templated) -- greedy, deterministic.
    msgs = [
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Name three primary colors."}],
        [{"role": "user", "content": "Write one sentence about the ocean."}],
        [{"role": "user", "content": "List the first five prime numbers."}],
    ]
    sp = SamplingParams(max_tokens=48, temperature=0.0)
    outs = llm.chat(msgs, sp)
    for i, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        print(f"PARITY[{mode}][{i}]: {ids}", flush=True)
        print(f"TEXT[{mode}][{i}]: {o.outputs[0].text.strip()!r}", flush=True)

    # Throughput (decode-dominated via ignore_eos).
    countries = ["Rome", "Japan", "Brazil", "Egypt", "France", "India", "Peru",
                 "Mali", "Norway", "Kenya", "Chile", "Nepal", "Cuba", "Iran",
                 "Spain", "Ghana", "Laos", "Fiji", "Oman", "Togo", "Iraq",
                 "Chad", "Niue", "Palau", "Samoa", "Tonga", "Yemen", "Aruba",
                 "Benin", "Gabon", "Haiti", "Macau"]
    base = [f"Tell me about the history of {c}." for c in countries]
    bsp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    _batches = tuple(int(s) for s in os.environ.get(
        "GLQ_PARITY_BATCHES", "1,32").split(",") if s.strip())
    for B in _batches:
        p = base[:B]
        llm.generate(p, bsp)  # warmup
        t0 = time.perf_counter()
        gen = llm.generate(p, bsp)
        dt = time.perf_counter() - t0
        tok = sum(len(o.outputs[0].token_ids) for o in gen)
        print(f"RESULT[{mode}] B={B}: {tok} tok in {dt:.2f}s -> "
              f"{tok/dt:.1f} tok/s aggregate ({tok/dt/B:.1f}/seq)", flush=True)


if __name__ == "__main__":
    main()
