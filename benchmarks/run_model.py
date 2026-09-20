#!/usr/bin/env python
"""Load, smoke-test and benchmark a quantized checkpoint — the harness to reach for.

Roughly 18 near-duplicate ad-hoc drivers grew in this directory, each re-solving the same
traps and each missing a different one. This consolidates them so the fixes live in one
place:

  two-point (t_N - t_1) decode isolation, --batches, ttft_ms   <- bench_tps_vllm.py
  model-agnostic argparse, glq_vllm import, --mm0, chat smoke  <- _serve_smoke.py
  load wall-time, VRAM, per-seq tok/s                          <- _bench_decode.py
  degeneracy assertion + nonzero exit                          <- _e2b_smoke.py
  --expect substring gate                                      <- smoke_inline_dequant.py
  footprint from vLLM's own log, not a CUDA delta              <- glq/bench/runtime.py

It reports the three numbers CLAUDE.md requires of any perf claim — wall-clock, TTFT, and
GPU memory after load (weights and KV separately) — and refuses to print a tok/s number it
cannot show was measured on the intended path: no captured CUDA graphs, or a footprint that
misses the bpw budget, is a hard failure rather than a quietly wrong row.

  python benchmarks/run_model.py --model <dir|repo> --batches 1,32
  python benchmarks/run_model.py --model <dir> --runtime hf --expect-gib 0.25

Only stdlib is imported at module scope: tests/test_run_model.py exercises the pure helpers
on CPU and must import this file with neither torch nor vllm present.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import threading
import time

# GLQ MoE only stream-captures batches <= 256; above it the kernel routes to a per-expert
# Python loop with host syncs, and capture dies with cudaErrorStreamCaptureUnsupported.
MOE_CAPTURE_CAP = 256
DEFAULT_CAPTURE = (1, 2, 4, 8, 16, 32)

_WEIGHTS_RE = re.compile(r"Model loading took\s+([\d.]+)\s*GiB")
_KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_GRAPH_RE = re.compile(r"Graph capturing finished in\s+\d+\s+secs?,\s*took\s+([\d.]+)\s*GiB")

PROMPT = "Tell me a long and detailed story about a robot who learns to paint."
COHERENCE_Q = "Name three primary colors and give one short fact about each."


# --------------------------------------------------------------------------- pure helpers

def parse_weights_gib(log):
    """Resident weight memory from vLLM's own INFO line, or None if it never appeared.

    Deliberately has no fallback. `nvidia-smi` under vLLM measures the
    gpu_memory_utilization KV pool (util x total VRAM), and a parent-process
    torch.cuda.max_memory_allocated() reads 0.00 because vLLM v1 loads weights in an
    EngineCore subprocess — both have been reported as a footprint by mistake.
    With tensor parallelism each rank logs its own shard.
    """
    hits = _WEIGHTS_RE.findall(log)
    return float(hits[-1]) if hits else None


def parse_kv_tokens(log):
    """KV cache capacity in tokens. CLAUDE.md requires KV be reported alongside weights."""
    hits = _KV_RE.findall(log)
    return int(hits[-1].replace(",", "")) if hits else None


def parse_graph_gib(log):
    """GiB spent on CUDA-graph capture, or None if no capture happened."""
    hits = _GRAPH_RE.findall(log)
    return float(hits[-1]) if hits else None


def is_degenerate(text, min_words=5, min_unique_ratio=0.3):
    """True when a sample looks like broken decode rather than a short answer.

    Uses a unique-word *ratio*, not a unique count: the characteristic quantization failure
    is a repeating n-gram ("alpha beta gamma alpha beta gamma ...") which keeps the unique
    count above any small threshold while being obviously broken.
    """
    words = text.split()
    if len(words) < min_words:
        return True
    return len(set(words)) / len(words) < min_unique_ratio


def footprint_ok(actual, expected, tol=0.15):
    """Whether measured weight memory matches the bpw budget. Unmeasured is never a pass.

    This is what catches a silently-dense model: it loads, it generates plausible text, and
    only the footprint gives it away.
    """
    if actual is None:
        return False
    return abs(actual - expected) <= tol * expected


def resolve_capture_sizes(spec, batches):
    """Capture sizes to request, defaulting to a set that covers every benched batch.

    vLLM derives its default from max_num_seqs*2 — often just [1, 2] — so an unlisted B=32
    silently benches off-graph and looks slow for the wrong reason.
    """
    if spec:
        return sorted({int(s) for s in spec.split(",") if s.strip()})
    sizes = set(DEFAULT_CAPTURE) | set(batches)
    return sorted(s for s in sizes if 0 < s <= MOE_CAPTURE_CAP)


def parse_batches(spec):
    out = [int(x) for x in spec.split(",") if x.strip()]
    if not out or any(b <= 0 for b in out):
        raise ValueError(f"--batches must be positive integers, got {spec!r}")
    return out


def needs_mm0(archs):
    """Multimodal archs need limit_mm_per_prompt zeroed for a text-only serve.

    vLLM's profiling forward otherwise crashes in the processor *after* weights load, which
    reads as a GLQ load bug. All three keys must be set — capping image+audio alone leaves
    <|video|> placeholders and still crashes.
    """
    return any("ConditionalGeneration" in a for a in archs)


def local_architectures(model):
    """Arch names from a local config.json; empty for a bare HF repo id (no network here)."""
    cfg = pathlib.Path(model) / "config.json"
    if not cfg.is_file():
        return []
    try:
        data = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return []
    archs = list(data.get("architectures") or [])
    for key in ("text_config", "language_config"):
        archs += list((data.get(key) or {}).get("architectures") or [])
    return archs


# ------------------------------------------------------------------------------ log capture

class _LogTee:
    """Tee fds 1 and 2 through a pipe for the rest of the process: echo live, keep a copy.

    Capturing is the point (vLLM's footprint line is the only trustworthy source, and it is
    emitted by the EngineCore subprocess on the fd it inherits). Echoing matters just as
    much: loading is minutes of silence otherwise, which reads as a hang.

    Never uninstalled. The engine subprocess holds the write end open for the whole run, so
    tearing the reader down would block that child as soon as the pipe buffer filled.
    """

    def __init__(self):
        self._chunks = []
        self._real_out = os.dup(1)
        read_fd, write_fd = os.pipe()
        threading.Thread(target=self._pump, args=(read_fd,), daemon=True).start()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)

    def _pump(self, read_fd):
        with os.fdopen(read_fd, "rb", 0) as pipe:
            for chunk in iter(lambda: pipe.read(65536), b""):
                self._chunks.append(chunk)
                os.write(self._real_out, chunk)

    def text(self):
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.3)   # drain: subprocess writes land in the pipe asynchronously
        return b"".join(self._chunks).decode("utf-8", "replace")


# ---------------------------------------------------------------------------------- runners

def summarize(samples):
    """(median, min, max). Median because one scheduling hiccup should not move the
    headline number, min/max because a figure with no spread cannot be judged."""
    xs = sorted(samples)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0
    return med, xs[0], xs[-1]


def _two_point(generate_fn, batch, decode, repeats: int = 1, warmup: int = 16):
    """Decode throughput with prefill removed, repeated so it can report a spread.

    Time the same batch for 1 token and for N; the difference cancels prefill and scheduler
    setup, leaving (N-1) pure decode steps. A single timed generate folds prefill into the
    decode number and flatters short runs.

    The subtraction has a cost: a difference of two noisy samples carries BOTH variances, so
    with a small ``decode`` the endpoint noise can rival the signal. Relative error falls
    roughly as 1/decode, which is why 64 is a far better default than 8, and ``repeats``
    exists so the result comes with a range instead of being a bare point estimate.

    ``warmup`` is deliberately not tiny. On CPU a multi-GiB checkpoint arrives by lazy
    page-fault through mmap, and a 4-token warmup cannot fault in the working set -- the
    timed run then pays first-touch cost, which is a bias rather than noise.
    """
    if warmup:
        generate_fn(batch, warmup)
    rates, ttfts, ntok, degraded = [], [], 0, False
    for _ in range(max(1, repeats)):
        t1, _ = generate_fn(batch, 1)
        tn, ntok = generate_fn(batch, decode)
        span = tn - t1
        if span <= 0:                  # too fast/noisy to separate — fall back, say so
            rates.append(batch * decode / tn)
            degraded = True
        else:
            rates.append(batch * (decode - 1) / span)
        ttfts.append(t1 * 1000.0)
    tps, lo, hi = summarize(rates)
    return tps, summarize(ttfts)[0], ntok, degraded, (lo, hi)


def run_vllm(args, failures):
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")   # the footprint line is INFO-level
    tee = _LogTee()

    if args.quant == "glq":
        import glq_vllm  # noqa: F401  registers the quant method; the packaged entry point
        # covers installed use, but an editable/source checkout may not be registered yet.
    from vllm import LLM, SamplingParams

    batches = parse_batches(args.batches)
    caps = resolve_capture_sizes(args.capture_sizes, batches)
    archs = local_architectures(args.model)

    kw = dict(model=args.model, dtype=args.dtype, trust_remote_code=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=max(batches))
    if args.quant == "glq":
        kw["quantization"] = "glq"
    if args.cpu_offload_gb:
        kw["cpu_offload_gb"] = args.cpu_offload_gb
    if args.eager:
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"cudagraph_mode": "FULL", "cudagraph_capture_sizes": caps}
    if args.mm0 or needs_mm0(archs):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}

    print(f"CONFIG {json.dumps({k: v for k, v in kw.items() if k != 'model'}, default=str)}",
          flush=True)

    t0 = time.perf_counter()
    llm = LLM(**kw)
    load_s = time.perf_counter() - t0

    log = tee.text()
    weights_gib, kv_tokens, graph_gib = (parse_weights_gib(log), parse_kv_tokens(log),
                                         parse_graph_gib(log))
    print(f"FOOTPRINT load_s={load_s:.1f} weights_gib="
          f"{'?' if weights_gib is None else f'{weights_gib:.2f}'} "
          f"kv_tokens={kv_tokens if kv_tokens is not None else '?'} "
          f"capture_gib={'0 (eager)' if args.eager else graph_gib}", flush=True)

    if weights_gib is None:
        failures.append("no 'Model loading took X GiB' line — footprint unverified; do not "
                        "quote a VRAM number from this run")
    if args.expect_gib is not None and not footprint_ok(weights_gib, args.expect_gib):
        failures.append(f"footprint {weights_gib} GiB misses the expected {args.expect_gib} "
                        f"GiB by >15% — model may not be quantized on this path")
    # Assert the mechanism, not just the output: without this a fallback to eager would be
    # reported as a cudagraph tok/s number and nothing would complain.
    if not args.eager and graph_gib is None:
        failures.append("no CUDA graphs captured — the tok/s below is an eager number")

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.maxtok, seed=7)
    out = llm.chat([[{"role": "user", "content": COHERENCE_Q}]], sp, use_tqdm=False)
    text = out[0].outputs[0].text
    print(f"SAMPLE {text.strip()[:400]!r}", flush=True)
    if is_degenerate(text):
        failures.append("coherence sample is degenerate")
    if args.expect and args.expect.lower() not in text.lower():
        failures.append(f"coherence sample lacks expected substring {args.expect!r}")

    def gen(batch, n):
        sp2 = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t = time.perf_counter()
        o = llm.generate([PROMPT] * batch, sp2, use_tqdm=False)
        return time.perf_counter() - t, sum(len(x.outputs[0].token_ids) for x in o)

    gen(max(batches), 8)                       # warm the captured shapes before timing
    for b in batches:
        tps, ttft_ms, ntok, degraded, spread = _two_point(
            gen, b, args.decode, repeats=args.repeats, warmup=args.warmup)
        note = " (prefill NOT isolated: decode too fast to separate)" if degraded else ""
        rng = (f" tokps_range={spread[0]:.1f}-{spread[1]:.1f} n_repeats={args.repeats}"
               if args.repeats > 1 else "")
        print(f"RESULT label={args.label} model={args.model} batch={b} "
              f"total_decode_tokps={tps:.1f} per_seq_tokps={tps / b:.1f} "
              f"ttft_ms={ttft_ms:.0f} n_tokens={ntok}{rng}{note}", flush=True)


def is_cuda_device(device_map) -> bool:
    """True when ``device_map`` names a CUDA device. 'auto' counts as not-CUDA here: the
    CPU branches below are the safe default, and 'auto' on a GPU box still works because
    the model lands wherever HF puts it."""
    return str(device_map).startswith("cuda")


def hf_footprint_gib(device_map, cuda_mod=None):
    """Weight footprint after an HF load.

    On CUDA that is ``memory_allocated()``. On CPU there is no such counter, and calling
    the CUDA ones does not merely return 0 — ``reset_peak_memory_stats`` raises
    ``RuntimeError: invalid argument to reset_peak_memory_stats`` — so fall back to peak
    process RSS. RSS includes the interpreter and torch itself (~1 GiB), so it slightly
    OVERSTATES the weights; it is a footprint bound, not a weights-only figure.
    """
    if is_cuda_device(device_map):
        if cuda_mod is None:
            import torch
            cuda_mod = torch.cuda
        return cuda_mod.memory_allocated() / 2 ** 30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def wants_offload_buffers(device_map, opt_in: bool = False) -> bool:
    """Whether to pass ``offload_buffers=True``. **Opt-in only.**

    GLQ registers its quantized weights as buffers, not parameters, and that makes both
    accelerate settings bad:

    * ``False`` (accelerate's default) keeps the buffers of CPU-assigned modules on the
      execution device, so the whole offloaded portion follows the model onto the GPU --
      38.3 GiB on Qwen Next, which fills the card whatever ``max_memory`` says.
    * ``True`` streams those buffers CPU->GPU every forward, which is the weight-streaming
      cost module placement was supposed to avoid. Measured on Qwen Next: **0.2 tok/s**
      against **1.5 tok/s** for plain ``--device-map cpu``.

    So it is not defaulted on: turning an OOM into a silent 7.5x slowdown is worse than
    the OOM. Fixing this properly means storing GLQ weights as
    ``nn.Parameter(requires_grad=False)`` so accelerate leaves CPU-assigned modules on CPU.
    """
    if not opt_in:
        return False
    return not isinstance(device_map, str) or device_map in ("auto", "balanced",
                                                             "balanced_low_0", "sequential")


def parse_device_map(spec):
    """A device_map string, or a parsed dict when given JSON.

    ``auto`` places whole MODULES and cannot split one, so a single oversized module can
    still land badly. transformers accepts a module-name-keyed dict, which is the escape
    hatch for pinning it by hand.
    """
    if isinstance(spec, str) and spec.strip().startswith("{"):
        import json as _json
        return _json.loads(spec)
    return spec


def parse_max_memory(spec):
    """Parse a ``--max-memory`` JSON spec into accelerate's ``max_memory`` dict.

    Device keys are ints for GPUs and the string "cpu", which is what
    ``infer_auto_device_map`` expects; JSON can only give string keys, so digits are
    converted back.
    """
    if not spec:
        return None
    import json as _json
    raw = _json.loads(spec)
    return {(int(k) if str(k).isdigit() else k): v for k, v in raw.items()}


def resolve_hf_class(cfg, transformers_mod=None):
    """The model class to load with, from ``config.architectures[0]``.

    ``AutoModelForCausalLM`` maps the *config class*, not the architecture string. On a
    multimodal wrapper that returns the TEXT-ONLY variant: Qwen4Exp resolves to
    ``Qwen4ExpForCausalLM`` (modules ``model.layers.*``) while the checkpoint is
    ``Qwen4ExpForConditionalGeneration`` (``model.language_model.layers.*``), so every key
    mismatches and the load reports the whole GLQ payload as UNEXPECTED instead of failing.

    Falls back to ``AutoModelForCausalLM`` when the architecture is absent or not exported.
    """
    if transformers_mod is None:
        import transformers as transformers_mod
    archs = getattr(cfg, "architectures", None) or []
    if archs:
        cls = getattr(transformers_mod, archs[0], None)
        if cls is not None:
            return cls
    return transformers_mod.AutoModelForCausalLM


def run_hf(args, failures):
    import torch
    # MUST precede from_pretrained. Without it transformers silently ignores
    # quantization_config, builds a DENSE model that generates plausible text and reports
    # bf16 memory — a bench that already shipped one wrong set of numbers.
    import glq.hf_integration  # noqa: F401
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_cls = resolve_hf_class(cfg)
    print(f"HF class {model_cls.__name__} (from architectures)", flush=True)

    on_cuda = is_cuda_device(args.device_map)   # dict/auto -> False, so inputs go to cpu
                                                #    and accelerate's hooks relocate them
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _mm = parse_max_memory(getattr(args, "max_memory", None))
    _dm = parse_device_map(args.device_map)
    _extra = {"max_memory": _mm} if _mm else {}
    if wants_offload_buffers(_dm, getattr(args, "offload_buffers", False)):
        _extra["offload_buffers"] = True
    model = model_cls.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype),
        device_map=_dm, trust_remote_code=True, **_extra)
    model.eval()
    load_s = time.perf_counter() - t0
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # HF loads in-process, so unlike the vLLM path the CUDA counter is the right instrument.
    # Report where things actually landed. A hybrid run that silently put everything on
    # one device looks identical to a working split from the outside.
    dmap = getattr(model, "hf_device_map", None)
    if dmap:
        import collections as _c
        tally = _c.Counter(str(v) for v in dmap.values())
        print(f"DEVICE_MAP {dict(tally)} modules={len(dmap)}", flush=True)
        for _mod, _dev in list(dmap.items())[:4]:
            print(f"  {_dev}  {_mod}", flush=True)

    weights_gib = hf_footprint_gib(args.device_map)
    print(f"FOOTPRINT load_s={load_s:.1f} weights_gib={weights_gib:.2f} runtime=hf "
          f"device={args.device_map}"
          + ("" if on_cuda else " (peak RSS: includes interpreter+torch)"), flush=True)
    if args.expect_gib is not None and not footprint_ok(weights_gib, args.expect_gib):
        failures.append(f"footprint {weights_gib:.2f} GiB misses the expected "
                        f"{args.expect_gib} GiB by >15% — likely loaded dense (is "
                        f"glq.hf_integration imported before from_pretrained?)")

    msgs = [{"role": "user", "content": COHERENCE_Q}]
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = COHERENCE_Q
    ids = tok(prompt, return_tensors="pt").to("cuda" if on_cuda else "cpu")
    with torch.no_grad():
        gen_ids = model.generate(**ids, max_new_tokens=args.maxtok, do_sample=False)
    text = tok.decode(gen_ids[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"SAMPLE {text.strip()[:400]!r}", flush=True)
    if is_degenerate(text):
        failures.append("coherence sample is degenerate")
    if args.expect and args.expect.lower() not in text.lower():
        failures.append(f"coherence sample lacks expected substring {args.expect!r}")

    enc = tok([PROMPT], return_tensors="pt").to("cuda" if on_cuda else "cpu")

    def gen(batch, n):
        batched = {k: v.repeat(batch, 1) for k, v in enc.items()}
        if on_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            o = model.generate(**batched, max_new_tokens=n, do_sample=False,
                               min_new_tokens=n)
        if on_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        return dt, int(o.shape[0] * (o.shape[1] - batched["input_ids"].shape[1]))

    for b in parse_batches(args.batches):
        tps, ttft_ms, ntok, degraded, spread = _two_point(
            gen, b, args.decode, repeats=args.repeats, warmup=args.warmup)
        note = " (prefill NOT isolated)" if degraded else ""
        rng = (f" tokps_range={spread[0]:.1f}-{spread[1]:.1f} n_repeats={args.repeats}"
               if args.repeats > 1 else "")
        print(f"RESULT label={args.label} model={args.model} runtime=hf batch={b} "
              f"total_decode_tokps={tps:.1f} per_seq_tokps={tps / b:.1f} "
              f"ttft_ms={ttft_ms:.0f} n_tokens={ntok}{rng}{note}", flush=True)
    if on_cuda:
        print(f"PEAK_ALLOC_GIB {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}", flush=True)
    else:
        print(f"PEAK_RSS_GIB {hf_footprint_gib('cpu'):.2f}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local checkpoint dir or HF repo id")
    ap.add_argument("--runtime", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--quant", default="glq", choices=["glq", "none"],
                    help="'none' for a bf16 baseline arm")
    ap.add_argument("--batches", default="1,32", help="comma list, e.g. 1,8,32")
    ap.add_argument("--decode", type=int, default=256, help="decode steps per timed run. "
                    "Relative error falls ~1/decode, so small values are noisy")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed pairs per batch size; >1 reports a median and a range. "
                         "Without it the result is a point estimate with no error bar")
    ap.add_argument("--warmup", type=int, default=16,
                    help="warmup tokens before timing. On CPU a big checkpoint arrives by "
                         "lazy page-fault, so too short a warmup biases the first timing")
    ap.add_argument("--maxtok", type=int, default=96, help="tokens for the coherence sample")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--offload-buffers", dest="offload_buffers", action="store_true",
                    help="allow accelerate to offload BUFFERS. GLQ stores weights as "
                         "buffers, so this streams them over PCIe every forward: measured "
                         "0.2 tok/s vs 1.5 for --device-map cpu on Qwen Next. Without it a "
                         "multi-device map OOMs instead. Neither is good; see the docstring.")
    ap.add_argument("--max-memory", dest="max_memory", default=None,
                    help='accelerate max_memory budget, JSON, e.g. \'{"0": "20GiB", '
                         '"cpu": "200GiB"}\'. Caps what device_map=auto puts on each '
                         'device; needed when a single module is large enough to OOM the '
                         'card on its own.')
    ap.add_argument("--device-map", dest="device_map", default="cuda",
                    help="HF device_map: 'cuda' (default) or 'cpu' for a CPU-only run")
    ap.add_argument("--cpu-offload-gb", dest="cpu_offload_gb", type=float, default=0.0,
                    help="GiB of WEIGHTS to keep in host (CPU) RAM and stream over PCIe "
                         "each forward pass. Not free VRAM: it trades bandwidth for "
                         "capacity, so it suits prefill-bound work (perplexity) far "
                         "better than decode. 0 = off.")
    ap.add_argument("--dtype", default="bfloat16",
                    help="float16 is required by some GLQ kernels; bf16-native models "
                         "(Mistral/Ministral) NaN in fp16 on activation outliers")
    ap.add_argument("--capture-sizes", default=None,
                    help="override cudagraph capture sizes (default covers --batches)")
    ap.add_argument("--eager", action="store_true", help="skip CUDA graphs (eager numbers "
                                                         "do not generalize — see the skill)")
    ap.add_argument("--mm0", action="store_true",
                    help="force limit_mm_per_prompt=0 (auto-set for *ConditionalGeneration)")
    ap.add_argument("--expect-gib", type=float, default=None,
                    help="assert weight footprint matches this bpw budget within 15%%")
    ap.add_argument("--expect", default=None,
                    help="assert the coherence sample contains this substring")
    ap.add_argument("--label", default="")
    return ap


def main():
    args = build_parser().parse_args()

    failures = []
    (run_hf if args.runtime == "hf" else run_vllm)(args, failures)

    if failures:
        for f in failures:
            print(f"FAIL {f}", flush=True)
        print("RUN_FAILED", flush=True)
        return 1
    print("RUN_OK", flush=True)
    return 0


if __name__ == "__main__":
    # vLLM forces the spawn start method once CUDA is initialized, which re-imports this
    # module in the EngineCore child; a module-level LLM() would recursively spawn and die
    # in multiprocessing's bootstrap check.
    sys.exit(main())
