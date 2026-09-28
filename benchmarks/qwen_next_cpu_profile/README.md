# Qwen-Next CPU decode: hardware-counter attribution of one decode step

`xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4`, 2026-09-27/28.
Companion to `benchmarks/qwen_next_cpu_sweep/`, whose stated *cause* for the flat thread curve
this run corrects.

## Why counters rather than `torch.profiler`

The existing bucket profile (`benchmarks/_profile_cpu_decode.py`) accounts for 59% of a step and
leaves "the 248k-vocab lm_head, norms, sampling and interpreter time" unbucketed — and it runs at
~3x overhead, so its ratios are usable and its absolute ms are not. It cannot close that gap by
design: `torch.profiler` records aten ops and `record_function` scopes only, so a pybind11 call
is invisible to it and so is interpreter time. Hardware counters and a sampling profiler see
both.

## Box and provenance

| | |
|---|---|
| instance | `m7i.metal-24xl` (spot), AWS eu-north-1 |
| CPU | Intel Xeon Platinum 8488C, Sapphire Rapids, 48c/2t = 96 threads, **1 NUMA node** |
| kernel / perf | 7.0.0-1013-aws / perf **7.0.14** — kernel-matched |
| PMU | **full metal uncore**: 8 `uncore_imc` + 4 free-running, 56 `uncore_cha`; `TopdownL1`–`L6` |
| memory | 377 GB, **swap 0**, 284 GB free — the "RSS fits, swap 0" precondition for any CPU perf comparison |
| ISA | `amx_tile`, `avx512_fp16`, `avx512_bf16`; GLQ tier **`avx512fp16`** (top), read from the extension not `/proc/cpuinfo` |
| governor | `performance` |
| stack | glq 0.8.21, torch 2.13.0+cpu, transformers 5.17.0; HF runtime, `--device-map cpu`, dtype **bfloat16** |
| gate | `FOOTPRINT load_s=142.9 weights_gib=75.33` — a dense bf16 load would be 157+ GiB and fail |

**A non-metal instance could not have produced this.** `TOPDOWN.SLOTS` and the uncore IMC
counters are both hidden behind the hypervisor on non-metal `m7i`, which is what makes TMA and
DRAM bandwidth unavailable there.

## Method

**Region scoping with no code change.** Load is ~143 s against ~9 s of decode per repeat, so
whole-process counters would be ~90% model loading. `run_model.py:441` prints `FOOTPRINT`
immediately after load and before `_two_point` (`:481`), so that line is an exact phase marker:
launch detached, watch for it, settle past warmup and the 1-token call, then attach to the PID.

**One load, many windows.** Each load costs 143 s and 75 GiB, and multiplexing degrades metric
accuracy, so `--decode 512 --repeats 1` gives a long steady-state window and each metric group
gets its own ~12 s `perf stat -p PID`. Never reload per group.

**Profiled at T=1 deliberately.** At T=96, ~95 threads spinning in a barrier retire instructions
cheaply, so aggregate TMA would describe the spinners rather than the work. Since the sweep shows
~91% of the step does not scale, T=1 is representative of the bottleneck and gives a profile with
no barrier noise and no SMT confound.

**Mechanism asserted:** `OMP_NUM_THREADS=1` -> `torch.get_num_threads()=1` (so the sweep's thread
axis is real — this could have invalidated it), `glq cpu ext = loaded (isa=avx512fp16)`.
`torch.get_num_interop_threads()` stays **48** regardless — a separate pool `OMP_NUM_THREADS`
does not govern.

## Result 1 — where a decode step goes

Cycle attribution by shared object (`perf record --call-graph=lbr -F 999`, 30 s). Sums to 100%
with **0.44% unresolved**, so nothing material is hidden by `kptr_restrict`:

| DSO | share | what it is |
|---|---|---|
| `glq_cpu.so` | **70.67%** | `zmm_matvec_impl<DecodeFp16,3>`, reached via `matvec_trellis_3inst_cpu` <- `fused_linear_impl` <- pybind11 |
| `[JIT]` | 15.19% | oneDNN JIT bf16 GEMMs — the *unquantized* layers, lm_head chief among them |
| `libtorch_cpu.so` | 7.87% | elementwise (`mul_kernel`), norms, gating, the MoE loop's gather/scatter |
| `python3.12` + `libtorch_python` | **3.32%** | interpreter + pybind dispatch |
| `libc` / `libgomp` / rest | 2.95% | **`libgomp` is 0.03%** (T=1 has no fork/join) |

py-spy agrees from the Python side: 6/6 sampled stacks inside `_trellis_linear_apply`
(`glq/quantized_linear.py:897`).

## Result 2 — it is core-bound, not memory-bound

| | |
|---|---|
| backend-bound | 64.7% of slots — **45.3% core-bound vs 19.3% memory-bound** |
| memory drill | dram 10.4%, l3 4.8%, l2 1.8%, l1 2.0% (multiplexed ~25%, so ±) |
| frontend / bad-spec | 5.5% / **0.8%** |
| IPC | **1.522** (56.32 G insn / 37.01 G cycles) at ~3.08 GHz effective |
| FP width | **91.4%** of FP instructions are 512-bit |
| DRAM | **2.79 GB/s** (CAS x 64 B) and **2.88 GB/s** (perf's own metric) — **~0.9% of ~307 GB/s peak** |

The two DRAM figures come from independent windows and independent event sets and agree, which is
the precondition for quoting either. Bad-speculation at 0.8% and frontend at 5.5% are the
signature that this is *not* an interpreter-dispatch-bound profile — bytecode dispatch is
characteristically heavy on both.

## Result 3 — half of per-token weight traffic is never quantized

Derived from `config.json` (`vocab_size 248320`, `hidden_size 2560`, 48 layers, 512 experts,
`num_experts_per_tok 10`, `moe_intermediate_size 640`, `tie_word_embeddings false`):

| not quantized by GLQ | GB/token | share |
|---|---|---|
| `lm_head` 248320x2560 **bf16** — `hf_integration.py:96` `MODULES_TO_NOT_CONVERT` | 1.271 | **32.3%** |
| hyper-connection `input_mix_weight_{down,up}`, `block_inject_weight` — `quantize_model.py:337-339` | 0.633 | 16.1% |
| router `mlp.gate` — an `nn.Parameter`, so the linear walk never sees it | 0.126 | 3.2% |
| **subtotal** | **2.030** | **51.6%** |

| GLQ-quantized @ 3 bpw | GB/token | share |
|---|---|---|
| routed experts (48 x 10 x 3) | 0.885 | 22.5% |
| GDN projections (36 layers) | 0.779 | 19.8% |
| full attention (12 layers) | 0.153 | 3.9% |
| shared experts (48 layers) | 0.088 | 2.2% |
| **subtotal** | **1.905** | **48.4% — less than the unquantized half** |
| **total** | **3.935** | |

**Cross-validated against the counters, not asserted:** 2.79 GB/s / 3.935 GB = **0.71 tok/s at
T=1**, against **0.66 tok/s** from extrapolating the sweep's Amdahl fit to n=1.

`lm_head` alone is the largest single tensor touched per token — larger than all 10 routed
experts in all 48 layers combined (1.271 vs 0.885 GB) — and it is excluded by a bare constant,
with no quality rationale recorded. The hyper-connection exclusion *does* have one: its group
reconstructs worst at 15.33 dB.

## Three conclusions that contradict the obvious reading

**1. Optimizing the top of the profile would be near-useless.** The kernel is 70.67% at T=1, but
the step falls 806 -> 277 ms from T=1 to T=96 and only the kernel is large enough to supply that
drop — so at T=48 it is roughly 6-12 ms of a 279 ms step. The lever recorded in
`glq_trellis_layout.hpp:128-132` (the staging store, ~10% of the kernel) is therefore worth
**~0.4% end-to-end** at the thread count anyone deploys at. **A single-thread profile inverts the
optimization ranking**; that is the transferable lesson here.

**2. Cutting bytes will not help CPU, but should help GPU.** CPU decode is core-bound at ~1-5% of
DRAM peak, and the kernel issues ~20 instructions for every arithmetic one. GLQ trades bytes for
instructions, so on CPU that trade is against the grain — quantizing lm_head may well make CPU
*slower*. GPU B=1 decode is bandwidth-bound, which is exactly where GLQ's recorded 1.90x-on-an-L4
comes from, so the same change points the other way there.

**3. The sweep's stated cause was wrong.** `qwen_next_cpu_sweep/README.md` attributed the flat
curve to "dispatch over 48 layers x 512 experts = 24,576 expert modules". Interpreter plus pybind
dispatch is **3.32%**. Independently, wiring the fused MoE path — which deletes exactly that
dispatch — measured **+0%** on this model. Corrected in that file.

## What this run does NOT establish

Stated plainly, because the box was reclaimed before the follow-up arms could run:

* **The T=1 tok/s was never captured.** The run's `RESULT` line had not been written when results
  were rescued, and the box is gone. The 806 ms/token above is **derived** from DRAM traffic and
  cross-checked against the Amdahl extrapolation — it is not a measured rate.
* **No multi-thread profile exists.** The ~253 ms non-scaling term (least-squares `S` over the
  five sweep points, `P` = 1268.6 ms, 91% of the T=96 step) is *attributed by inference* from T=1
  shares — `[JIT]` + libtorch + python + libc = 29.3% of an 806 ms step ≈ 236 ms, which matches
  `S` closely. **That agreement is suggestive, not measured.** A T=48 arm would settle it.
* **The knee mechanism is untested.** The leading hypothesis is parallel-region granularity:
  parallelism is `at::parallel_for` over `m/32` row blocks (`glq_bindings_cpu.cpp:97`, grain 1),
  and an expert `gate_up` is m=1280 = **40 blocks**, so efficiency saturates once threads approach
  the block count — which fits a knee at 16 as smooth saturation. A *separate* mechanism exists in
  the fused path (`glq_moe_cpu.cpp:197`: `expert_parallel = used.size() >= at::get_num_threads()`,
  which with 10 routed experts flips at T=11), but that path was **off** in the sweep, so it
  cannot be the explanation for these numbers. Both need `libgomp` share at T=8/16/48 to separate.
* The traffic ledger is arithmetic over `config.json`, not a measured per-tensor trace.

## Apparatus notes, each of which cost a real failure

* **`perf_event_paranoid=4` blocks everything**, including `task-clock`. Needs `1` for per-process
  core events — and **system-wide (`-a`) uncore plus py-spy's ptrace additionally need `sudo`**.
  Without that split, the DRAM metric returns `<not supported>` and py-spy returns
  `Permission Denied`, both of which read as "this box can't do it" rather than "wrong privilege".
* **Validate the apparatus on a throwaway process first.** Doing so caught the two failures above
  before a ~16-minute model load, not after.
* **`OPENBLAS_NUM_THREADS=1`** is set by the orchestrator to stop idle BLAS spinners polluting the
  profile. Harmless at T=1 — but at T>1 it would serialize the dense BLAS path and **manufacture
  the very non-scaling signature under investigation**. Set it to `$T` for any multi-thread arm.
  The committed thread sweep never set it, so those numbers are unaffected.
* `nmi_watchdog=1` consumes a PMC; set it to 0.
* The `LOAD REPORT`'s long `MISSING` list is **benign and fully explained**: `tlut` is absent
  because 3INST is lookup-free (a computed hash, no table), `trellis_packed2`/`inv_resid_scale2`
  are stage-2 RVQ tensors that do not exist at 3 bpw, and `gate_up_proj.*` is absent because GLQ
  fuses gate+up at load while the checkpoint stores them separately (74,031 `layer_bpw` entries =
  48 x 512 x 3 + 303).

## Files

| file | what |
|---|---|
| `counters.tsv` | every number above, machine-readable, each scoped and sourced |
| `counters_*.txt` | raw `perf stat` output per metric group, perf preamble stripped |
| `attribution_by_dso.txt` / `_by_symbol.txt` | `perf report` cycle attribution |
| `attribution_pyspy_frames.txt` | Python-frame counts from 6 `py-spy dump` snapshots |

`perf.data` (12.6 MB) is deliberately not committed; it is in `.local-logs/pmu_qwen_next/`.

Scope: batch 1, decode 512, **T=1**, dtype bfloat16, `avx512fp16` tier, this checkpoint, this
box. Nothing here generalizes to batch > 1, to other thread counts, or to other models without
re-measurement.
