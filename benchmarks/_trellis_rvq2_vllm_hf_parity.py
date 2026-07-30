#!/usr/bin/env python
"""vLLM vs HF-eager greedy parity for a trellis checkpoint — the stacked-RVQ serving gate.

The unit tests already prove the *arithmetic*: ``test_trellis_fused_rvq2_paths_agree`` shows
both vLLM apply paths are torch.equal to the shared HF staticmethod on fused-shard shapes,
and that stage 2 moves the output. What no unit test can cover is the **loader**: whether a
real checkpoint's ``trellis_packed2``/``inv_resid_scale2`` land in the right shard of a
merged qkv / gate_up parameter, with the right scale attached to each. A residual loaded
into the wrong shard, or a per-shard scale off by one slot, still produces fluent text.

So: same checkpoint, same prompt token ids, greedy on both runtimes, compare token ids.
Runs each arm in its own subprocess (vLLM owns the GPU and forces spawn), then reports the
common-prefix length.

Prefix length is reported, not asserted at some magic threshold. The two runtimes differ in
attention backend and reduction order, so a late divergence is expected and benign; what a
mis-loaded residual looks like is divergence within the first few tokens. Run it on a
single-stage checkpoint too — that arm is the control for how well these runtimes agree at
all on this model.

  python benchmarks/_trellis_rvq2_vllm_hf_parity.py --model <dir|repo> --ntok 64
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

PROMPT = ("The history of the printing press begins in fifteenth-century Mainz, where "
          "Johannes Gutenberg combined movable type with an adapted wine press. Within a "
          "few decades")


def _run_hf(args, out_path):
    import torch
    # MUST precede from_pretrained, or transformers ignores quantization_config and builds a
    # dense model that generates plausible text — which would make this gate pass vacuously.
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda",
        trust_remote_code=True).eval()

    ids = tok(PROMPT, return_tensors="pt").to("cuda")
    n_in = ids["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**ids, do_sample=False, max_new_tokens=args.ntok,
                             min_new_tokens=args.ntok)
    new_ids = gen[0][n_in:].tolist()
    _write(out_path, {
        "arm": "hf",
        "prompt_ids": ids["input_ids"][0].tolist(),
        "token_ids": new_ids,
        "text": tok.decode(new_ids, skip_special_tokens=True),
        "weights_gib": torch.cuda.memory_allocated() / 2 ** 30,
    })


def _run_vllm(args, out_path):
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    import glq_vllm  # noqa: F401  registers the quant method for a source checkout
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Tokenize HERE and feed vLLM ids, not text: a tokenizer difference between the two
    # runtimes would show up as a prefix mismatch and read exactly like a decode bug.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = tok(PROMPT)["input_ids"]

    llm = LLM(model=args.model, quantization="glq", dtype=args.dtype,
              trust_remote_code=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=1,
              compilation_config={"cudagraph_mode": "FULL",
                                  "cudagraph_capture_sizes": [1, 2, 4, 8]})
    sp = SamplingParams(temperature=0.0, max_tokens=args.ntok, ignore_eos=True)
    out = llm.generate([{"prompt_token_ids": prompt_ids}], sp, use_tqdm=False)
    new_ids = list(out[0].outputs[0].token_ids)

    # Same prompt B times, greedy: every sequence must decode identically, and identically to
    # B=1. This is the batched-path gate — B=1 runs the GEMV, B>1 the tensor-core GEMM, and
    # a fused layer under a replayed CUDA graph is where cross-sequence contamination or a
    # stale captured buffer would show. B=1 parity alone would never see it.
    batched = llm.generate([{"prompt_token_ids": prompt_ids}] * args.batch, sp, use_tqdm=False)
    batch_ids = [list(o.outputs[0].token_ids) for o in batched]
    n_same = sum(ids == batch_ids[0] for ids in batch_ids)

    _write(out_path, {
        "arm": "vllm",
        "prompt_ids": list(prompt_ids),
        "token_ids": new_ids,
        "text": out[0].outputs[0].text,
        "batch": args.batch,
        "batch_all_identical": n_same == len(batch_ids),
        "batch_matches_b1": batch_ids[0] == new_ids,
        "batch_common_prefix_b1": common_prefix(batch_ids[0], new_ids),
    })


def _write(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh)
    print(f"WROTE {payload['arm']} n={len(payload['token_ids'])}", flush=True)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ntok", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32,
                    help="second vLLM pass at this batch, greedy, same prompt (identity gate)")
    ap.add_argument("--dtype", default="float16",
                    help="SAME on both arms — the point is to vary only the runtime")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--label", default="")
    ap.add_argument("--_arm", choices=["hf", "vllm"], help=argparse.SUPPRESS)
    ap.add_argument("--_out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._arm:
        (_run_hf if args._arm == "hf" else _run_vllm)(args, args._out)
        return 0

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ("hf", "vllm"):                       # sequential: each arm owns the GPU
            path = os.path.join(tmp, f"{arm}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "--model", args.model,
                   "--ntok", str(args.ntok), "--dtype", args.dtype,
                   "--max-model-len", str(args.max_model_len), "--batch", str(args.batch),
                   "--gpu-mem", str(args.gpu_mem), "--_arm", arm, "--_out", path]
            print(f"\n===== ARM {arm} =====", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"FAIL arm {arm} exited {rc}", flush=True)
                print("RUN_FAILED", flush=True)
                return 1
            with open(path) as fh:
                results[arm] = json.load(fh)

    hf, vllm = results["hf"], results["vllm"]
    if hf["prompt_ids"] != vllm["prompt_ids"]:
        print("FAIL prompt token ids differ between arms — not a decode comparison")
        print("RUN_FAILED", flush=True)
        return 1

    n = common_prefix(hf["token_ids"], vllm["token_ids"])
    print(f"\nPARITY label={args.label} model={args.model} ntok={args.ntok} "
          f"common_prefix={n} identical={n == args.ntok}", flush=True)
    print(f"BATCH  b={vllm['batch']} all_identical={vllm['batch_all_identical']} "
          f"matches_b1={vllm['batch_matches_b1']} "
          f"prefix_vs_b1={vllm['batch_common_prefix_b1']}", flush=True)
    print(f"HF   {hf['text'].strip()[:300]!r}", flush=True)
    print(f"VLLM {vllm['text'].strip()[:300]!r}", flush=True)

    failed = []
    if n == 0:
        failed.append("the two runtimes disagree on the FIRST token — a mis-loaded residual "
                      "or shard looks exactly like this")
    if not vllm["batch_all_identical"]:
        failed.append(f"greedy B={vllm['batch']} sequences are not identical to each other")
    for f in failed:
        print(f"FAIL {f}", flush=True)
    print("RUN_FAILED" if failed else "RUN_OK", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
