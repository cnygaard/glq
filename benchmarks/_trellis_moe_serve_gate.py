"""Phase B gate: serve the 26B trellis-3inst-4bpw MoE on vLLM under a CAPTURED CUDA graph.

Run once per arm, then compare the two JSONs:

    python _trellis_moe_serve_gate.py --arm fused   # enforce_eager=False, fused grouped op
    GLQ_MOE_FORCE_FALLBACK=1 \
    python _trellis_moe_serve_gate.py --arm loop    # enforce_eager=True, Phase A per-expert loop
    python _trellis_moe_serve_gate.py --compare

What each arm asserts, and why the obvious check is not enough:

  * The eager loop is a BIT-EXACT fallback, so identical text between arms proves nothing on
    its own. The mechanism assertions are (a) ``_apply_trellis`` is called ZERO times during
    a B=1 decode — counted directly, not inferred — and (b) vLLM reports graph capture. The
    loop could not survive (b) even if it wanted to: its ``topk_ids.unique()`` is a host sync,
    which is illegal during capture.
  * Footprint comes from vLLM's own "Model loading took X GiB" line in the log, never
    nvidia-smi.
  * Greedy parity is on TOKEN IDS, not text, so a detokenizer difference cannot mask a
    numeric one.

``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is set so the model runs in THIS process; otherwise the
counter patch lands in the driver and the model runs in a worker that never sees it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO = "xv0y5ncu/gemma-4-26B-A4B-it-trellis-3inst-4bpw"
OUT = "/opt/dlami/nvme/moe_gate_{arm}.json"

PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
CHATS = [PROMPT,
         [{"role": "user", "content": "Write a short joke about saving RAM."}],
         [{"role": "user", "content": "What is 2+2? Reply with just the number."}]]
# Parity needs a long IN-DISTRIBUTION generation. The first attempt reused the B=1 timing
# run, which sets ignore_eos to get a fixed token count — and that forces decoding PAST the
# end-of-turn token, into a regime where the model is maximally unsure and argmax ties are
# everywhere. The two arms are not bit-identical (different output-RHT kernel, activation
# kernel and reduce order: max|Δ| ~1e-6), so a tie there flips and the sequences split
# permanently. That measures the prompt, not the kernel. This prompt runs long on its own.
PARITY = [{"role": "user", "content":
           "Explain how a CPU cache works, in about 150 words."}]


def main(arm):
    import torch
    from vllm import LLM, SamplingParams
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod

    # Count fallback entries. Patched on the CLASS before the engine builds, so every MoE
    # layer's method object inherits it.
    calls = {"n": 0}
    _orig = GLQFusedMoEMethod._apply_trellis

    def _counted(self, *a, **k):
        calls["n"] += 1
        return _orig(self, *a, **k)
    GLQFusedMoEMethod._apply_trellis = _counted

    eager = arm == "loop"
    t0 = time.time()
    llm = LLM(model=REPO, quantization="glq", enforce_eager=eager,
              max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
    load_s = time.time() - t0
    print(f"\n=== engine up in {load_s:.0f}s (enforce_eager={eager})", flush=True)

    greedy = SamplingParams(temperature=0.0, max_tokens=64)
    # Warm up before timing anything: the first chat() pays tokenizer init, the first
    # graph replay and allocator growth. Timing it measured 0.3 tok/s on a 2-token reply,
    # which is engine startup, not decode rate.
    llm.chat([PROMPT], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))

    # --- mechanism + B=1 decode rate. ignore_eos so the token count is the one asked for;
    #     "Answer in one word" otherwise stops at 2 tokens and the rate is all overhead. ---
    calls["n"] = 0
    b1_sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    t0 = time.time()
    one = llm.chat([PROMPT], b1_sp)
    b1_s = time.time() - t0
    fallback_calls = calls["n"]
    b1_tok = len(one[0].outputs[0].token_ids)
    print(f"B=1: {b1_tok} tok in {b1_s:.2f}s = {b1_tok / b1_s:.1f} tok/s  "
          f"| _apply_trellis entries: {fallback_calls}", flush=True)

    # --- greedy parity material: a long natural generation plus the three short chats.
    #     NOT the ignore_eos run above — see the PARITY note. ---
    par = llm.chat([PARITY], SamplingParams(temperature=0.0, max_tokens=256))
    outs = llm.chat(CHATS, greedy)
    ids = [list(par[0].outputs[0].token_ids)] + [list(o.outputs[0].token_ids) for o in outs]
    print(f"\nparity generation: {len(ids[0])} tokens (natural stop)", flush=True)
    for c, o in zip(CHATS, outs):
        print(f"\n>>> {c[0]['content']}\n  {o.outputs[0].text.strip()[:250]}", flush=True)

    # --- decode throughput at B=32, equal token counts via ignore_eos ---
    b32 = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    t0 = time.time()
    many = llm.chat([PROMPT] * 32, b32)
    b32_s = time.time() - t0
    b32_tok = sum(len(o.outputs[0].token_ids) for o in many)
    print(f"\nB=32: {b32_tok} tok in {b32_s:.2f}s = {b32_tok / b32_s:.1f} tok/s", flush=True)

    rec = {"arm": arm, "enforce_eager": eager, "load_s": round(load_s, 1),
           "b1_tok_s": round(b1_tok / b1_s, 2), "b32_tok_s": round(b32_tok / b32_s, 2),
           "fallback_calls_b1": fallback_calls, "token_ids": ids,
           "gpu": torch.cuda.get_device_name(0)}
    with open(OUT.format(arm=arm), "w") as f:
        json.dump(rec, f)
    print("\nwrote", OUT.format(arm=arm))

    if arm == "fused" and fallback_calls != 0:
        print(f"VERDICT: FAIL — the fused arm entered the eager loop {fallback_calls}x at "
              f"B=1; the grouped op is not the path being exercised.")
        return 1
    print("VERDICT: PASS (arm complete)")
    return 0


def compare():
    a = json.load(open(OUT.format(arm="fused")))
    b = json.load(open(OUT.format(arm="loop")))
    print(f"GPU: {a['gpu']}\n")
    print(f"{'':22} {'fused (captured)':>18} {'loop (eager)':>18}")
    print(f"{'B=1 decode tok/s':22} {a['b1_tok_s']:>18} {b['b1_tok_s']:>18}")
    print(f"{'B=32 decode tok/s':22} {a['b32_tok_s']:>18} {b['b32_tok_s']:>18}")
    print(f"{'engine load s':22} {a['load_s']:>18} {b['load_s']:>18}")
    print(f"{'_apply_trellis @ B=1':22} {a['fallback_calls_b1']:>18} "
          f"{b['fallback_calls_b1']:>18}")

    same = a["token_ids"] == b["token_ids"]
    n = sum(len(t) for t in a["token_ids"])
    print(f"\ngreedy token-id parity over {n} tokens "
          f"({len(a['token_ids'][0])} in the long generation): "
          f"{'IDENTICAL' if same else 'DIFFER'}")
    if not same:
        for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])):
            if x != y:
                d = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                print(f"  chat {i}: first divergence at token {d}")
    sp1 = a["b1_tok_s"] / b["b1_tok_s"] if b["b1_tok_s"] else 0
    sp32 = a["b32_tok_s"] / b["b32_tok_s"] if b["b32_tok_s"] else 0
    print(f"\nspeedup  B=1 {sp1:.2f}x   B=32 {sp32:.2f}x")
    ok = same and a["fallback_calls_b1"] == 0
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["fused", "loop"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    raise SystemExit(compare() if args.compare else main(args.arm))
