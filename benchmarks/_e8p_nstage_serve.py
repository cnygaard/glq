"""Phase B vLLM serve gate: load a GLQ --codebook e8p 5/6/7/8 bpw checkpoint on vLLM
with FULL cudagraph (enforce_eager=False), generate at B=1 and B=32, assert coherence.
Exercises the glq_vllm linear_method N-stage wiring (create_weights sizing +
_setup_e8p_weights meta + _glq_apply_e8p), which the HF e2e does NOT touch.

Usage: python _e8p_nstage_serve.py <model_dir> [<model_dir> ...]
"""
import os
import sys

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/opt/dlami/nvme/torch_ext")

import glq_vllm  # noqa: F401 — registers the GLQ vLLM quantization plugin
from vllm import LLM, SamplingParams

PROMPT = "The capital of France is"


def serve_one(path):
    print(f"\n=== vLLM serve (FULL cudagraph): {path} ===", flush=True)
    llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.45,
              max_model_len=2048, enforce_eager=False, max_num_seqs=64)
    sp = SamplingParams(max_tokens=24, temperature=0.0)

    out1 = llm.generate([PROMPT], sp)
    t1 = out1[0].outputs[0].text
    print(f"B=1  : {t1!r}", flush=True)

    out32 = llm.generate([PROMPT] * 32, sp)
    texts = [o.outputs[0].text for o in out32]
    same = len(set(texts)) == 1
    print(f"B=32 : {texts[0]!r}  (all-identical={same})", flush=True)

    paris = "paris" in t1.lower()
    ok = paris and same
    print(f"  coherent(Paris)={paris}  B32-consistent={same}  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    models = sys.argv[1:]
    if not models:
        print("usage: _e8p_nstage_serve.py <model_dir> [<model_dir> ...]")
        sys.exit(2)
    results = {m: serve_one(m) for m in models}
    print("\n=== SERVE SUMMARY ===")
    for m, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {m}")
    print("=== RESULT:", "ALL PASS" if all(results.values()) else "FAILURES", "===")
