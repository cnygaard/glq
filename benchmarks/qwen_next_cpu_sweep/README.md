# Qwen-Next CPU decode: thread sweep on a 96-thread Sapphire Rapids metal box

`xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4`, 2026-09-27.
Reproduce with `benchmarks/_qwen_next_cpu_sweep.sh`.

## Box and provenance

| | |
|---|---|
| instance | **`m7i.metal-24xl`** (spot), AWS eu-north-1 |
| CPU | Intel Xeon Platinum 8488C, **Sapphire Rapids**, 48c/2t = **96 vCPU**, 1 NUMA node |
| ISA tier | **`avx512fp16`** — GLQ's fastest CPU tier, confirmed by `glq-setup --verify`: `fused CPU kernels ready (loaded (isa=avx512fp16))`, not inferred from `/proc/cpuinfo` |
| RAM / disk | 377 GB / gp3 root, **no local NVMe** |
| stack | glq 0.8.21, torch 2.13.0+**cpu**, vLLM **0.29.0+cpu**, transformers 5.17.0, accelerate 1.15.0 |
| runtime | HF transformers, `--device-map cpu`, **dtype bfloat16** (auto-resolved by `preferred_dtype` for the qwen family) |
| method | `run_model.py --runtime hf --batches 1 --decode 32 --repeats 3 --warmup 8 --expect-gib 75` |

vLLM resolved to 0.29.0+cpu on its own via `install.sh --cpu` — **no manual pin was needed**,
despite `cpu_wheel.latest_cpu_wheel_url` reading GitHub's latest release.

## Result: the knee is at 16 threads

| threads | decode tok/s | range (3 repeats) | gain over previous | TTFT ms | load_s | RSS GiB |
|---|---|---|---|---|---|---|
| 8 | 2.380 | 2.375–2.415 | — | 10,591 | 133.0 | 75.35 |
| 16 | 3.194 | 3.192–3.202 | **+34.2%** | 9,945 | 132.2 | 75.32 |
| 32 | 3.436 | 3.427–3.437 | +7.6% | 9,930 | 132.0 | 75.35 |
| 48 | 3.585 | 3.583–3.614 | +4.3% | 9,931 | 132.3 | 75.35 |
| ~~96~~ **48 (see below)** | 3.603 | 3.594–3.625 | **+0.5%** | 10,015 | 787.2 (cold) | 76.71 |

### ⚠️ CORRECTED 2026-09-29: the "96 thread" cell was a 48-thread run

The sweep set thread count via `OMP_NUM_THREADS` only, and **torch caps its intra-op pool at the
PHYSICAL core count.** Measured on this box (48c/2t = 96 logical):

```
OMP_NUM_THREADS unset      -> torch.get_num_threads() = 48
OMP_NUM_THREADS=96         -> torch.get_num_threads() = 48   <- capped
torch.set_num_threads(96)  -> torch.get_num_threads() = 96   <- the only way up
std::thread::hardware_concurrency() : 96
```

Cells 8/16/32/48 are all ≤ 48 and were honoured. **The 96 cell was therefore a duplicate of the
48 cell**, and its +0.5% is run-to-run noise between two identical configurations. The earlier
claim here that "48→96 is flat and SMT contributes nothing" is **withdrawn: SMT was never
engaged**, so this sweep says nothing about it either way.

`benchmarks/run_model.py` now takes `--threads`, which calls `torch.set_num_threads()` and
prints a `THREADS requested=… torch.get_num_threads()=…` line, so the gap cannot recur silently.

Two regimes, and reporting only one of them would mislead:

* **8 → 16 is a real +34.2%.** Eight threads is genuinely under-provisioned; cores do help here.
* **16 → 48 is +12.2% for 3x the cores.** Returns collapse immediately past 16, and 16 threads
  already captures **89%** of what 48 delivers. (Previously stated as "16 → 96 for 6x the
  cores"; the top cell was 48, so the ratio was wrong even though the conclusion holds.)

**TTFT is nearly flat across the whole 6x range** (10,591 ms at 8 threads, 9,930–10,015 for
16–48 — a 6.6% spread). Prefill is the phase that should parallelise best, and it barely moves
even over the 8→16 step where decode gains a third.

### Why — CORRECTED 2026-09-28 by hardware counters: it is NOT dispatch

**This section previously claimed the cause was "Python-level dispatch over 24,576 expert
modules". That was a hypothesis inferred from the flat curve, and it is wrong.** A T=1
hardware-counter profile on this same box (`benchmarks/qwen_next_cpu_profile/`) measured
interpreter plus pybind dispatch at **3.32% of cycles** — `python3.12` 2.53% + `libtorch_python`
0.79% — against **70.67%** inside the GLQ trellis kernel. Independently, wiring the fused CPU MoE
path, which deletes exactly that per-expert dispatch, measured **+0%** on this model. Both point
the same way: dispatch was never the binding constraint.

What the counters do establish:

* **The step is ~91% non-scaling.** A least-squares fit of `T(n) = S + P/n` over all five points
  gives **S = 252.7 ms, P = 1268.6 ms** — S is 91% of the 277.5 ms step at the top cell (48 threads). Model-free and
  assumption-free: 12x the threads removed only 143 ms of a 420 ms step.
* **The kernel is not what fails to scale.** It is 70.67% of a T=1 step (~996 ms of ~1.41 s) but
  on the order of ~20 ms of a 279 ms step at T=48 — it parallelizes. The remainder is the
  *non-GLQ* work (oneDNN bf16 GEMMs 15.19%, libtorch 7.87%, interpreter 3.32%, libc 1.95%), which
  is ~400 ms at T=1. That is *larger* than S, so part of it parallelizes too; the exact split
  needs a multi-thread profile and is **not yet measured**.
* **It is core-bound, not memory-bound**: 45.3% core-bound vs 19.3% memory-bound, at ~0.9% of
  DRAM peak. More cores cannot help work that is already serial, and bandwidth was never the
  limit.

**The cause is now measured: OpenMP barrier wait.** `libgomp` share against thread count, on a
second identical box with the stack matched:

| T | tok/s | **libgomp** | glq_cpu | IPC | useful cycles | **effective threads** |
|---|---|---|---|---|---|---|
| 1 | 0.709 (derived) | **0.03%** | 70.67% | 1.522 | 99.5% | **1.00** |
| 8 | 2.357 | **47.37%** | 34.55% | 1.163 | 49.4% | **3.95** |
| 48 | 3.412 | **82.89%** | 10.30% | 0.357 | 16.8% | **8.07** |

61.22% of all T=48 cycles are in `gomp_barrier_wait_end` alone. Per token at T=48 (293.1 ms):
**243 ms barrier wait**, 30 ms trellis kernel, 12 ms oneDNN, 5 ms libtorch, 1 ms Python.

**Effective parallelism plateaus at ~8 threads' worth of useful work however many cores are
added** — that is the knee. (The sweep's apparent 48 -> 96 flatness is not evidence for this:
that cell was a 48-thread duplicate, see the correction above.) The cause is too little work per
parallel region: with the fused MoE path off (the default), the per-expert Python loop opens
**1,262 `at::parallel_for` regions per token**, and an expert `gate_up` is m=1280 = only **40**
blocks of 32 rows, so at 48 threads most threads take one block and then wait. A zero-barrier-cost
bound puts the prize at **~6.1x (~21 tok/s)**.

Full analysis and the caveats: `benchmarks/qwen_next_cpu_profile/`.

The earlier claim that this "reproduces a previously recorded finding (4x the cores plus AVX-512
buy ~12%)" still holds as an *observation* about thread scaling. Its stated **mechanism** does
not.

### Practical consequence, which inverts the usual sizing instinct

**Size for ~16 threads, and stop.** 16 threads gets ~89% of what 48 threads deliver, so
paying for `m7i.metal-24xl` over something 6x smaller buys almost nothing — while dropping to
8 threads does cost a third of the throughput. The knee is narrow and it is at 16.

Scope: batch 1, decode 32, this checkpoint, this ISA, 3 repeats. Batch > 1 was **not** measured
and may scale differently — the per-expert loop has more to chew on with more tokens in flight,
so the flat region above 16 threads should not be assumed to hold for batched serving.

## Two corrections this run produced

**1. HF *does* honour GLQ quantization for Qwen4Exp.** RSS came in at 75.32–76.71 GiB with
`--expect-gib 75` passing; a dense bf16 load would be 157+ GiB and would have failed the gate
loudly. A prior note claiming "HF gives a dense bf16 model, substitution matches nothing" is
**wrong for this checkpoint** — recorded here because believing it would rule out a route that
works. On a box with 377 GB of RAM a dense load would have fitted happily and silently
benchmarked the wrong model, which is why the footprint gate was not optional.

**2. The 13-minute first load was disk after all.** The `Loading weights ... [00:19<00:00]`
progress bar walks mmap'd tensor metadata, not bytes off the platter, so it is not evidence
about I/O. Cold **787.2 s** vs warm **132.3 s** for identical work settles it: ~100 MB/s cold
against 75 GiB, close to **gp3's 125 MB/s default cap**. Concrete argument for the
`m7id`/`m8id` local-NVMe families — a 13-minute cold start dominated this sweep's cost far more
than any thread-count difference.

## Parked: speculative decoding needs a GPU box

Established today, so it is not re-derived:

* The GLQ checkpoint **preserves the MTP head** — 31 `mtp.*` tensors of 297,256 — so
  `method: "mtp"` applies to this checkpoint, not just the base model.
* vLLM 0.29.0 lists both `mtp` and `dspark`; Qwen4Exp ships `nvidia/mtp.py`.
* **It cannot be tested on CPU.** `vllm/models/qwen4_exp/__init__.py` dispatches CUDA/ROCm only
  ("Qwen4Exp currently supports CUDA and ROCm only"); `dspark` lives in
  `v1/worker/gpu_worker.py`; and HF — the runtime used here — has no MTP support at all, only a
  draft model or `prompt_lookup_num_tokens`, neither of which uses the checkpoint's own head.

On the next GPU box:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

Two conditions for the result to mean anything:

* **The gate is draft acceptance rate, not tok/s alone.** Rejected drafts cost throughput, so a
  speculative config can come out slower than baseline; acceptance rate is what says whether
  the head is useful on this workload.
* `run_model.py` has **no `--speculative-config` passthrough** — it needs that flag added, or
  `vllm serve` driven directly. Fourth instance of the same gap, after `--dtype` (fixed),
  `tensor_parallel_size` and `kv_transfer_config`.

## Files

| file | what |
|---|---|
| `results.tsv` | the table above, machine-readable |
| `cpu_tN.trimmed.txt` | run logs with progress bars stripped (raw logs are megabytes of `it/s` spam; these are ~7.7 KB). Each carries its `DTYPE`, `FOOTPRINT`, `RESULT` and `EXIT=0`. Named `.txt` because `.gitignore` excludes `*.log` — correctly, for transient logs; these are curated artifacts |

## See also

`benchmarks/qwen_next_cpu_profile/` — the T=1 hardware-counter profile on this same box that
corrected the "Why" section above, and that found **51.6% of per-token weight traffic sits in
layers GLQ does not quantize** (`lm_head` alone is 32.3%, larger than all routed experts in all
48 layers combined).
