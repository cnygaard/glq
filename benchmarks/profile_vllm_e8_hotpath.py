"""Phase 5.3 Stage 3a — profile the vLLM hot path under the Stage 2c-2c
stack.

NIAH at ctx=16k gave 90% accuracy but ~3 tok/s decode. Before writing
CUDA kernels for ``gather_kv_to_paged_fp16``/dequant, we need to confirm
that's actually the bottleneck (vs codebook NN, vs Python dispatch, vs
flash_attn).

What this script does:
    1. Boot vLLM with the same env-var stack used by the NIAH probe.
    2. Build a moderate-length prompt (default 4 k tokens — long enough
       that each decode step touches every block, short enough to keep
       the profile run under a minute).
    3. Generate ``--n-decode`` tokens with ``torch.profiler`` recording
       CPU+CUDA activity.
    4. Print a top-N table by (cuda_time, self_cpu_time) and dump a
       Chrome-trace JSON for visual drill-down.
    5. Also runs cProfile over a separate identical decode to surface
       pure-Python hotspots (the profiler picks up the C kernel layer
       cleanly but pure-Python loops show up cleaner under cProfile).

The profiler/cProfile overhead is non-zero — only treat the *relative*
ordering of hotspots as load-bearing, not the absolute decode tok/s.

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
import cProfile
import io
import json
import os
import pstats
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
    """Run a single chat.generate, no timing — caller wraps in profiler."""
    from vllm import SamplingParams
    out = llm.chat(
        [msgs],
        sampling_params=SamplingParams(
            max_tokens=n_decode, temperature=0),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    return out[0]


def _format_torch_profile(prof, *, top_n: int) -> str:
    """Pretty-print top events by CUDA self-time."""
    # Sort by self_cuda_time_total (descending). key_averages gives
    # aggregated per-op view.
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=top_n,
        max_name_column_width=80,
    )
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/gemma-4-E4B-it")
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
    p.add_argument("--skip-cprofile", action="store_true",
                   help="skip the cProfile pass (torch.profiler only)")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    label = args.label or _variant_label()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== vLLM hot-path profile — variant={label} ===", flush=True)
    print(f"  ctx_len={args.ctx_len}  warmup={args.warmup_decode}  "
          f"decode={args.n_decode}", flush=True)

    max_model_len = args.max_model_len or (args.ctx_len + 256)

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=max_model_len,
              gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True)
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

    # ---- torch.profiler pass ----
    print("\n  [1/2] torch.profiler pass ...", flush=True)
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    t0 = time.time()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        out = _decode_once(llm, msgs, n_decode=args.n_decode)
    torch_dt = time.time() - t0
    n_out = len(out.outputs[0].token_ids)
    tps = n_out / torch_dt if torch_dt > 0 else 0.0
    print(f"  torch.profiler: {torch_dt:.2f}s for {n_out} tokens "
          f"({tps:.2f} tok/s, profiler overhead included)", flush=True)

    # Chrome trace
    chrome_path = os.path.join(args.out_dir, f"trace_{label}.json")
    prof.export_chrome_trace(chrome_path)
    print(f"  wrote chrome trace: {chrome_path}", flush=True)

    # Top-N table
    table = _format_torch_profile(prof, top_n=args.top_n)
    table_path = os.path.join(args.out_dir, f"torch_top_{label}.txt")
    with open(table_path, "w") as f:
        f.write(f"variant: {label}\n")
        f.write(f"ctx_len: {args.ctx_len}\n")
        f.write(f"decode tokens: {n_out}\n")
        f.write(f"wall: {torch_dt:.2f}s ({tps:.2f} tok/s)\n")
        f.write("\n")
        f.write(table)
    print(f"  wrote torch top-{args.top_n}: {table_path}", flush=True)

    # ---- cProfile pass (separate run; profilers don't compose cleanly) ----
    if not args.skip_cprofile:
        print("\n  [2/2] cProfile pass ...", flush=True)
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
        out2 = _decode_once(llm, msgs, n_decode=args.n_decode)
        pr.disable()
        cprof_dt = time.time() - t0
        n_out2 = len(out2.outputs[0].token_ids)
        tps2 = n_out2 / cprof_dt if cprof_dt > 0 else 0.0
        print(f"  cProfile: {cprof_dt:.2f}s for {n_out2} tokens "
              f"({tps2:.2f} tok/s, cProfile overhead included)", flush=True)

        # Top-N by cumulative time, then top-N by tottime
        for sort_key, suffix in [("cumulative", "cum"), ("tottime", "tot")]:
            buf = io.StringIO()
            ps = pstats.Stats(pr, stream=buf).sort_stats(sort_key)
            ps.print_stats(args.top_n)
            cprof_path = os.path.join(
                args.out_dir, f"cprofile_{suffix}_{label}.txt")
            with open(cprof_path, "w") as f:
                f.write(f"variant: {label}\n")
                f.write(f"sort: {sort_key}\n")
                f.write(f"decode tokens: {n_out2}\n")
                f.write(f"wall: {cprof_dt:.2f}s ({tps2:.2f} tok/s)\n\n")
                f.write(buf.getvalue())
            print(f"  wrote cProfile top-{args.top_n} ({sort_key}): "
                  f"{cprof_path}", flush=True)

        # Raw stats for later drill-down
        raw_path = os.path.join(args.out_dir, f"cprofile_raw_{label}.prof")
        pr.dump_stats(raw_path)
        print(f"  wrote raw cProfile: {raw_path}  "
              f"(load via `python -m pstats {raw_path}`)", flush=True)

    # ---- Summary JSON ----
    summary = {
        "variant": label,
        "args": vars(args),
        "torch_profiler": {
            "wall_sec": torch_dt,
            "decode_tokens": n_out,
            "tok_per_sec": tps,
            "chrome_trace": chrome_path,
            "top_table": table_path,
        },
    }
    if not args.skip_cprofile:
        summary["cprofile"] = {
            "wall_sec": cprof_dt,
            "decode_tokens": n_out2,
            "tok_per_sec": tps2,
        }
    json_path = os.path.join(args.out_dir, f"summary_{label}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote summary: {json_path}", flush=True)

    # ---- Quick eyeball at the top hotspot ----
    print("\n  [eyeball] torch.profiler top-10 (self CUDA time):", flush=True)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10,
        max_name_column_width=60), flush=True)


if __name__ == "__main__":
    main()
