"""Phase 5.3 Stage 3a — profile the vLLM hot path under the Stage 2c-2c
stack.

NIAH at ctx=16k gave 90% accuracy but ~3 tok/s decode. Before writing
CUDA kernels for ``gather_kv_to_paged_fp16``/dequant, we need to confirm
that's actually the bottleneck (vs codebook NN, vs Python dispatch, vs
flash_attn).

Key constraint: vLLM v1 runs the model in a separate ``EngineCore``
subprocess. A ``torch.profiler`` context in *this* process only sees
RPC poll/wait — not a single kernel. We therefore drive the in-engine
profiler via ``VLLM_TORCH_PROFILER_DIR`` + ``llm.start_profile()`` /
``llm.stop_profile()`` so the trace is captured inside the worker that
actually runs the patched K/V hot path.

We also force ``ignore_eos=True`` + ``min_tokens=n_decode`` because the
unconstrained chat template often emits EOS after 2-6 tokens, which
makes the profile too short to be informative.

Usage:
    GLQ_KV_QUANT=e8_relaxed:1 \\
    GLQ_KV_E8_SIDECAR=1 \\
    GLQ_KV_E8_SIDECAR_READ=1 \\
    GLQ_KV_E8_COMPRESSED_ALLOC=1 \\
    python benchmarks/profile_vllm_e8_hotpath.py \\
        --ctx-len 4096 --n-decode 32 \\
        --out-dir /tmp/logs/profile_e8

Without the env vars set, profiles the baseline fp16 path — useful for
A/B comparison.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# Make the sibling NIAH module importable (we reuse its prompt builder).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _variant_label() -> str:
    bits = []
    if os.environ.get("GLQ_KV_QUANT"):
        bits.append(os.environ["GLQ_KV_QUANT"])
    if os.environ.get("GLQ_KV_E8_SIDECAR") == "1":
        bits.append("sidecar")
    if os.environ.get("GLQ_KV_E8_SIDECAR_READ") == "1":
        bits.append("read")
    if os.environ.get("GLQ_KV_E8_COMPRESSED_ALLOC") == "1":
        bits.append("compressed_alloc")
    return "+".join(bits) if bits else "baseline-fp16"


def _build_prompt(tok, distractor_ids, *, ctx_len: int, seed: int):
    """Reuse NIAH's prompt builder but return a flat str (cheaper for
    repeated profiling runs — no needle bookkeeping needed)."""
    from kv_compress_niah_vllm import _build_chat_prompt  # noqa: E402
    msgs, gold = _build_chat_prompt(
        tok, distractor_ids, ctx_len=ctx_len, depth=0.5, seed=seed)
    return msgs, gold


def _decode_once(llm, msgs, *, n_decode: int):
    """Run a single chat.generate. ``ignore_eos`` + ``min_tokens`` keep
    the decode going for the full ``n_decode`` count so the profile
    actually has steady-state decode samples."""
    from vllm import SamplingParams
    out = llm.chat(
        [msgs],
        sampling_params=SamplingParams(
            max_tokens=n_decode,
            min_tokens=n_decode,
            temperature=0,
            ignore_eos=True,
        ),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    return out[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/cache/epub/6593/pg6593.txt")
    p.add_argument("--ctx-len", type=int, default=4096,
                   help="prompt token count")
    p.add_argument("--n-decode", type=int, default=32,
                   help="tokens to generate during the profiled phase")
    p.add_argument("--warmup-decode", type=int, default=4,
                   help="decode tokens before profiler starts (warm CUDA "
                        "graphs, caches, allocator)")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--out-dir", default="/tmp/logs/profile_e8")
    p.add_argument("--label", default=None)
    p.add_argument(
        "--profiler", choices=("torch", "cuda"), default="torch",
        help=("torch = vLLM's TorchProfilerWrapper (medium overhead, "
              "rich stacks/shapes; writes *.pt.trace.json.gz). "
              "cuda = vLLM's CudaProfilerWrapper (low overhead; gates "
              "cudaProfilerStart/Stop). Run *under* `nsys profile "
              "--capture-range=cudaProfilerApi --capture-range-end=stop "
              "-o trace.nsys-rep python ...` to capture an nsys report "
              "around only the decode window."))
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM

    label = args.label or _variant_label()
    print(f"=== vLLM hot-path profile — variant={label} ===", flush=True)
    print(f"  ctx_len={args.ctx_len}  warmup={args.warmup_decode}  "
          f"decode={args.n_decode}", flush=True)

    max_model_len = args.max_model_len or (args.ctx_len + 256)

    # vLLM 0.20 wires the engine-side profiler via profiler_config.
    # llm.start_profile()/stop_profile() RPC into the worker, which
    # gates the chosen backend:
    #   * "torch" -> torch.profiler trace JSON in torch_profiler_dir
    #   * "cuda"  -> cudaProfilerStart/Stop; capture via outer nsys
    if args.profiler == "torch":
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": args.out_dir,
        }
    else:
        profiler_config = {"profiler": "cuda"}
    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=max_model_len,
              gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True,
              profiler_config=profiler_config)
    print(f"  load: {time.time() - t0:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    from kv_compress_niah import read_gutenberg_text
    text = read_gutenberg_text(args.text_url)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    distractor_ids = enc[0]
    if distractor_ids.shape[-1] < args.ctx_len:
        n_reps = args.ctx_len // distractor_ids.shape[-1] + 2
        distractor_ids = distractor_ids.repeat(n_reps)
    print(f"  distractor pool: {distractor_ids.shape[-1]:,} tokens",
          flush=True)

    msgs, _gold = _build_prompt(
        tok, distractor_ids, ctx_len=args.ctx_len, seed=args.seed)

    # ---- Warmup pass (no profiler) so CUDA graphs / caches are hot ----
    if args.warmup_decode > 0:
        t0 = time.time()
        _decode_once(llm, msgs, n_decode=args.warmup_decode)
        print(f"  warmup decode ({args.warmup_decode} tok): "
              f"{time.time() - t0:.2f}s", flush=True)

    # ---- Engine-side profiler pass ----
    # vLLM v1 runs the model in a worker subprocess. start_profile()/
    # stop_profile() RPC into that worker; either a torch trace lands
    # in args.out_dir or (cuda mode) cudaProfilerStart/Stop is gated
    # and an outer `nsys profile` captures only this window.
    print(f"\n  [profile] engine-side {args.profiler} profiler pass ...",
          flush=True)
    pre_traces = set(glob.glob(os.path.join(args.out_dir, "*.pt.trace.json*")))

    llm.start_profile()
    t0 = time.time()
    out = _decode_once(llm, msgs, n_decode=args.n_decode)
    decode_dt = time.time() - t0
    llm.stop_profile()

    new_traces: list[str] = []
    if args.profiler == "torch":
        # stop_profile() returns before the worker finishes flushing —
        # poll the trace dir until something new appears.
        flush_deadline = time.time() + 60.0
        while time.time() < flush_deadline:
            new_traces = sorted(set(glob.glob(
                os.path.join(args.out_dir, "*.pt.trace.json*"))) - pre_traces)
            if new_traces:
                break
            time.sleep(0.5)

    n_out = len(out.outputs[0].token_ids)
    tps = n_out / decode_dt if decode_dt > 0 else 0.0
    print(f"  decode: {decode_dt:.2f}s for {n_out} tokens "
          f"({tps:.2f} tok/s, profiler overhead included)", flush=True)
    if args.profiler == "torch":
        if new_traces:
            print(f"  engine traces ({len(new_traces)}):", flush=True)
            for p in new_traces:
                sz = os.path.getsize(p) / (1024 * 1024)
                print(f"    {p}  ({sz:.1f} MB)", flush=True)
        else:
            print("  WARN: no engine trace appeared within 60s — check "
                  "the worker log for errors.", flush=True)
    else:
        print("  cuda profiler stopped — outer `nsys profile "
              "--capture-range=cudaProfilerApi` should have captured "
              "this window into its .nsys-rep file.", flush=True)

    # ---- Summary JSON ----
    summary = {
        "variant": label,
        "args": vars(args),
        "decode": {
            "wall_sec": decode_dt,
            "decode_tokens": n_out,
            "tok_per_sec": tps,
        },
        "engine_traces": new_traces,
    }
    json_path = os.path.join(args.out_dir, f"summary_{label}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote summary: {json_path}", flush=True)
    if args.profiler == "torch":
        print("\n  Load the trace in chrome://tracing or "
              "https://ui.perfetto.dev/ — search for "
              "'gather_kv_to_paged_fp16', 'aten::index', "
              "'triton_codebook_nn' to find the hot path.",
              flush=True)
    else:
        print("\n  Open the .nsys-rep in Nsight Systems "
              "(`nsys-ui trace.nsys-rep`) or dump CLI stats with "
              "`nsys stats --report gputrace,cudaapisum trace.nsys-rep`.",
              flush=True)


if __name__ == "__main__":
    main()
