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
| 96 | 3.603 | 3.594–3.625 | **+0.5%** | 10,015 | 787.2 (cold) | 76.71 |

Ranges are ±0.4–1.7%, so every step except 48→96 is outside noise; **48→96 is flat** and SMT
contributes nothing.

Two regimes, and reporting only one of them would mislead:

* **8 → 16 is a real +34.2%.** Eight threads is genuinely under-provisioned; cores do help here.
* **16 → 96 is +12.8% for 6x the cores.** Returns collapse immediately past 16, and 16 threads
  already captures **89%** of what 96 delivers.

**TTFT is nearly flat across the whole 12x range** (10,591 ms at 8 threads, 9,930–10,015 for
16–96 — a 6.6% spread). Prefill is the phase that should parallelise best, and it barely moves
even over the 8→16 step where decode gains a third.

### Why: dispatch, not compute

This checkpoint has **48 layers x 512 experts = 24,576 expert modules**. If per-step cost is
dominated by Python-level dispatch over experts rather than SIMD work inside the kernels, then
neither more cores nor wider vectors help — which is exactly what the flat TTFT and the 12.8%
decode ceiling both show. This reproduces a previously recorded CPU finding ("4x the cores plus
AVX-512 buy ~12%") on a **fourth machine, at the top ISA tier, with a different model**. The
consistency is the point: it is not a property of one box.

### Practical consequence, which inverts the usual sizing instinct

**Size for ~16 threads, and stop.** A 16-vCPU instance gets ~89% of a 96-vCPU metal box, so
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
