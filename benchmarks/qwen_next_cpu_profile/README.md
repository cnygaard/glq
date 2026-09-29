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
the step falls from **~1.41 s at T=1** (DRAM-derived; the Amdahl extrapolation independently gives
1.52 s) to 277 ms at T=96, and only the kernel is large enough to supply that drop. The kernel is
therefore ~996 ms of a T=1 step, and if it parallelizes it is on the order of **~20 ms of a 279 ms
step at T=48**. The lever recorded in
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

## Result 4 — quantizing lm_head: a 1.43x win on that layer, worth 0.6% end-to-end on CPU

`benchmarks/_lm_head_shape_probe.py` times the bare trellis matvec against the bf16 GEMV it
would replace, at lm_head's actual 248320x2560 shape. The GLQ arm is a **lower bound** (no RHT
bracket, no bf16->fp32 activation convert). Random packed data is a valid fixture and this was
checked rather than assumed: against a real Viterbi/LDLQ-quantized layer it timed within
**1.02x**, with zero subnormals and identical `absmax` — so no denormal-stall artifact.

**At 48 threads** (`lm_head_shape_probe.txt`, `lm_head_probe.tsv`):

| arm | ms | GB/s | vs bf16 |
|---|---|---|---|
| glq3 / **avx512fp16** | **3.69** | 64.5 | **1.43x FASTER** |
| glq3 / avx512 | 4.58 | 52.0 | 1.15x faster |
| **bf16 (today)** | 5.29 | **240.5** | baseline |
| fp32 | 10.73 | 237.0 | 0.49x |
| glq3 / avx2 | 14.19 | 16.8 | 0.37x SLOWER |
| glq3 / scalar | 15.00 | 15.9 | 0.35x SLOWER |

**At 1 thread the sign flips:** glq3/avx512fp16 157.12 ms vs bf16 117.62 ms = **0.75x, slower**.

**The crossover is thread count, and the mechanism is which wall each arm hits.** At 48 threads
bf16 reaches 240.5 GB/s = **78% of the ~307 GB/s theoretical peak**, so it is genuinely
DRAM-bound and cannot go faster; GLQ moves 5.3x fewer bytes and is compute-bound, so it keeps
scaling. In throughput terms GLQ goes 4.0 -> 172.3 G weights/s from 1 to 48 threads
(**42.6x, 89% parallel efficiency**) while bf16 manages only 22x. At 1 thread bf16 is nowhere
near the wall (10.8 GB/s) and GLQ's ~20-instructions-per-arithmetic-instruction decode loses.

**Two consequences worth keeping:**

1. **Traffic share is not time share.** lm_head is **32.3% of per-token bytes but only 5.29 ms
   of a ~279 ms T=48 step — about 1.9% of time** — because one large contiguous GEMV saturates
   DRAM cheaply, whereas the GLQ-quantized layers are compute-bound decode work. So a 1.43x win
   on lm_head is worth **~0.57% end-to-end on CPU**. Quantizing it is a **footprint** play
   (-0.96 GiB) and a **GPU** play (where B=1 decode is bandwidth-bound, so time share does track
   traffic share), not a CPU speed play.
2. **The sign depends on the ISA tier**, so this cannot be a single global default: avx2 and
   scalar *lose* (0.35-0.37x), avx512 and avx512fp16 win. Any default would have to be gated on
   tier and thread count, the way the dtype default is already gated on device.

This also means **~87% of a T=48 step is neither the trellis kernel (~29 ms, from 172.3 G
weights/s over this checkpoint's 5.08 G weights/token) nor lm_head (~5 ms)**. Naming that
remainder is what the T=48 arm is for.

## Result 5 — THE BOTTLENECK: 83% of cycles at 48 threads are OpenMP barrier wait

Measured on a second, identical box (2026-09-28) with the stack matched to the T=1 arm
(torch 2.13.0+cpu, transformers 5.17.0, glq from source, `avx512fp16`). Both arms passed the
footprint gate (75.31 / 75.34 GiB) and exited 0.

| T | tok/s | ms/tok | **libgomp** | glq_cpu | `[JIT]` | libtorch | python | IPC | retiring | DRAM GB/s | GHz | useful | **eff. threads** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.709* | 1410* | **0.03%** | 70.67% | 15.19% | 7.87% | 3.32% | 1.522 | 29.0% | 2.79 | 3.08 | 99.5% | **1.00** |
| 8 | 2.357 | 424.3 | **47.37%** | 34.55% | 6.52% | 6.06% | 0.96% | 1.163 | 19.8% | 12.56 | 2.02 | 49.4% | **3.95** |
| 48 | 3.412 | 293.1 | **82.89%** | 10.30% | 4.09% | 1.76% | 0.28% | 0.357 | 8.9% | 18.68 | 1.98 | 16.8% | **8.07** |

\* T=1 is derived (that arm's `RESULT` was lost to a spot reclaim); T=8 and T=48 are measured.

The symbols are explicitly barrier waits, not ambiguous runtime time:

```
61.22%  libgomp  gomp_barrier_wait_end          1.47%  libgomp  gomp_barrier_wait
17.17%  libgomp  gomp_team_barrier_wait_end     1.32%  libgomp  gomp_team_barrier_wait
10.09%  glq_cpu  zmm_matvec_impl<DecodeFp16,3>  1.07%  libgomp  gomp_team_barrier_wait_final
```

**Per-token budget at T=48 (293.1 ms measured):** barrier wait **243.0 ms**, trellis kernel
30.2 ms, oneDNN bf16 GEMMs 12.0 ms, libtorch elementwise 5.2 ms, Python+bindings 1.1 ms.

**Effective parallelism plateaus at ~8 threads' worth of useful work**, however many cores are
added — 1.00 -> 3.95 -> 8.07. That single number explains the whole sweep curve: why the knee is
at 16, why 48 -> 96 is flat, and why TTFT barely moved across a 12x thread range. Going 8 -> 48
threads (6x the cores) buys 1.45x.

### Why: too little work per parallel region, not slow barriers

GLQ contains **no `#pragma omp`** — parallelism is `at::parallel_for` over `m/32` output-row
blocks, grain size 1 (`glq_bindings_cpu.cpp:97`, `glq_moe_cpu.cpp:84`). With
`GLQ_HF_MOE_CPU_FUSED` off (the default), the per-expert Python loop issues **1,262 GLQ calls per
token** (960 expert + 302 dense), each opening its own region. An expert `gate_up` is m=1280 =
**40 blocks**; at 48 threads ~40 threads get one block each, 8 get none, and all wait for the
slowest. So the 83% is **threads idling for lack of work per region**, not synchronization
primitives being slow — the fix is work-per-region, not merely barrier count.

**Upper bound on the prize:** if that work kept its cycles and barrier cost went to zero, it
would occupy 1.98 s of the 12 s window — **~6.1x, or ~21 tok/s** against the measured 3.412.

**This also explains the recorded `GLQ_HF_MOE_CPU_FUSED` +0%.** That A/B ran at 8 threads, where
`expert_parallel = (used.size() >= at::get_num_threads())` = `(10 >= 8)` is **true** and the code
already takes the one-barrier-per-layer path (`glq_moe_cpu.cpp:197`). At 48 threads the predicate
is false, so the fused path would still open 960 regions. **The flag is not the fix**; flattening
(expert x row-block) into one region per layer *regardless of thread count* is.

### Two secondary effects worth not missing

* **AVX-512 all-core downclock: 3.08 -> ~2.0 GHz (-35%)** from T=1 to T>=8. Any per-thread
  throughput extrapolated from a single-thread measurement is 35% optimistic before parallel
  efficiency is even considered.
* **DRAM never becomes the limit**: 2.79 -> 12.56 -> 18.68 GB/s, still only **6%** of the
  ~307 GB/s peak at T=48. Both readings agree at every arm (CAS x 64 B vs perf's own metric).

## Result 6 — the fix the measurements actually pointed at: +16.8% for two lines

Wall-clock attribution (not counters — see the 48x trap above) put **13.9% of a decode step on
one line**: `final_hidden_states.index_add_(0, token_idx, h)` in `_loop_forward`
(`glq/fused_experts.py`). CPU `index_add_` dispatches to `scatter_add_`, which sorts indices
(fbgemm `radix_sort_parallel`) so parallel accumulation is safe when indices collide. At
batch-1 decode every routed expert sees exactly ONE token — so this is a 48-way parallel sort
of a single index, run ~480x per token. A lone index cannot collide, so a direct row add is
**bit-identical** and skips the sort.

**Ships ON by default**, unlike `GLQ_HF_MOE_CPU_FUSED` and `GLQ_CPU_GDN`, and deliberately so:
those two trade accuracy or memory for speed and so have to be opted into, whereas this one
produces the same bytes. `GLQ_CPU_FAST_SCATTER=0` stays as a kill switch, so a regression can
be bisected without a rebuild.

**Effect** (`fast_scatter_ab.tsv`; `--repeats 3`, ranges disjoint at both thread counts):

| threads | off | on | gain | ms/token saved |
|---|---|---|---|---|
| **48** | 3.597 (3.575-3.612) | **4.202** (4.183-4.214) | **+16.8%** | 40.0 |
| **16** | 3.192 (3.188-3.193) | **3.286** (3.284-3.292) | **+2.9%** | 9.0 |

**Mechanism, asserted rather than inferred** (`t48_scatter_on_by_dso.txt`, `_by_symbol.txt`):

| symbol | flag off | flag on |
|---|---|---|
| `fbgemm::radix_sort_parallel` | **9.54%** | **absent (0 occurrences)** |
| `gomp_barrier_wait_end` | 61.22% | 68.80% |
| `zmm_matvec_impl` | 10.09% | 11.85% |

The sort is gone from the profile entirely. The other components rise as *shares* because the
total shrank; the matvec is unchanged in absolute terms.

**Three independent checks that this is the right mechanism, not a coincidence:**

1. **Prediction vs measurement.** The profile predicted ~41 ms/token on that line; the measured
   saving at T=48 is **40.0 ms/token** — agreement to ~2%.
2. **The falsification test.** A parallel-region penalty *must* shrink with fewer threads. It
   does: 3x the threads gives 4.4x the saving. Had T=16 shown the same +16.8%, this explanation
   would have been wrong.
3. **Cross-check at a different decode length and box.** The PMU arm at decode 1300 scored 3.949
   with the flag on against 3.412 off — **+15.7%**, consistent with +16.8% at decode 128.

**Read the two A/B rows as ratios only.** The T=48 pair ran on one box and the T=16 pair on
another (spot reclaim in between). Each pair is internally controlled — same box, same session,
same load — so the ratios are sound; the absolute tok/s are not comparable across rows.

**Why counters alone could never have found this.** `radix_sort_parallel` is 9.54% of *cycles*
but 13.9% of *wall time*, and its own barrier is 8.16 of those 9.54 points — it reads as
"OpenMP overhead" in every cycle-based view. It took a wall-clock profiler, correctly placed
past three prefills, to name the Python line.

## Result 7 — the three CPU fast paths together: +26.7%, and two recorded nulls updated

`cpu_flag_matrix.tsv`. Batch 1, decode 128, repeats 3, **48 effective threads** (see the thread-cap
note below). Every comparison below has disjoint ranges.

| flags | tok/s | ms/token | weights GiB | gain |
|---|---|---|---|---|
| none | 3.586 | 278.9 | 75.3 | — |
| fast-scatter | 4.203 | 237.9 | 75.3 | +17.2% |
| fast-scatter + GDN | 4.339 | 230.5 | 75.3 | **+21.0%** |
| fast-scatter + fused-MoE | 4.292 | 233.0 | **89.2** | +19.7% |
| **all three** | **4.545** | **220.0** | **89.2** | **+26.7%** |

**`GLQ_CPU_GDN` is no longer a null, and the way it fails to be one confirms its mechanism.**
It adds **+3.2%** on top of fast-scatter but **+5.9%** on top of fast-scatter + fused-MoE — a
*fixed* serial cost is a larger fraction of a shorter step. The recorded in-situ null (2.8 tok/s
on and off) was measured at **8 threads on Emerald Rapids**, its worst case; that note's own
reasoning predicted this ("the kernel helps MORE on faster hardware, not less"). Engagement is
corroborated, not assumed: transformers' reference-fallback warning count drops 4 -> 3 in both
GDN arms.

**`GLQ_HF_MOE_CPU_FUSED` is likewise no longer +0%, but read it carefully.** It is **+19.7%**
against the *unfixed* loop — the comparison the old record made — but only **+2.1%** against the
fixed loop, because **fast-scatter is inert when the fused op runs** (the op replaces the Python
loop the fix lives in). And it costs **+13.9 GiB** (75.3 -> 89.2). On its own that is a poor
trade; it earns its place only because GDN then compounds on top of it.

TTFT also improves, 9983 -> 8937 ms.

### The thread cap that makes "96 threads" unmeasurable from the environment

**`OMP_NUM_THREADS` cannot raise torch above the physical core count.** On this 48c/2t box:

```
OMP_NUM_THREADS unset      -> torch.get_num_threads() = 48
OMP_NUM_THREADS=96         -> torch.get_num_threads() = 48   <- capped
torch.set_num_threads(96)  -> torch.get_num_threads() = 96   <- the only way up
std::thread::hardware_concurrency() : 96
```

Caught by noticing CPU utilisation topping out at 4800% (48 busy threads of 96 logical), then
confirmed from `/proc/PID/status` (`48 Rl` of 208) and a fresh interpreter. Consequences: the
arms above are all 48-thread runs despite their labels, and the thread sweep's "96" cell was a
**duplicate of its 48 cell** — see the correction in `../qwen_next_cpu_sweep/README.md`.
`run_model.py` now takes `--threads`, which calls `torch.set_num_threads()` and prints
`THREADS requested=… torch.get_num_threads()=…` so the gap cannot recur silently.

## What this run does NOT establish

* **The T=1 tok/s was never captured.** That arm's `RESULT` line had not been written when results
  were rescued and the box was reclaimed. The **~1.41 s/token** is **derived** (3.935 GB / 2.79
  GB/s), cross-checked against the Amdahl extrapolation's 1.52 s and the kernel-throughput route's
  1.77 s. T=8 and T=48 are measured; only the T=1 row is not.
* **The knee mechanism is now measured, but the *fix* is not.** That barrier wait dominates is a
  measurement; that flattening (expert x row-block) recovers it is a **hypothesis** until a patch
  is A/B'd. The ~6.1x figure is a zero-barrier-cost upper bound, not a forecast — real regions
  cannot have zero fork/join cost, and some of the work is genuinely serial.
* **Only one model, one box, one batch size.** Every number is batch 1 on a 48-core Sapphire
  Rapids at `avx512fp16`. Batch > 1 gives each region more work and should move the barrier share
  a lot; it was not measured. Nor was T=96, where the sweep's flat top lives.
* **`[JIT]` is attributed by inference.** It is assumed to be oneDNN's Xbyak-generated bf16 GEMM
  kernels (lm_head being the largest), because JIT-registered code has no symbol names. Nothing
  measured confirms which GEMM those cycles belong to.
* **The GDN recurrence is not separately visible.** It is inside `libtorch_cpu` (1.76% at T=48)
  along with every other elementwise op, so this profile cannot price it on its own — consistent
  with, but not independent confirmation of, the recorded in-situ null.
* The traffic ledger is arithmetic over `config.json`, not a measured per-tensor trace.

## Two hypotheses this work carried and then killed

Kept because both were stated confidently before being tested, and both were wrong:

* **"The flat thread curve is Python dispatch over 24,576 expert modules."** Measured at
  **3.32%** of cycles at T=1 and **0.28%** at T=48. Wrong twice over, and independently
  contradicted by the fused-MoE path measuring +0%.
* **"The non-scaling term is serial non-GLQ work."** The T=1 shares suggested `[JIT]` + libtorch +
  python + libc ≈ 400 ms against a least-squares `S` of 252.7 ms, which looked like a near-match
  at the time (under a step time that was itself wrong). The T=48 arm shows the real answer is
  none of those: they total **18.4 ms**, and 243 ms is barrier wait. **A single-thread profile
  cannot see the dominant cost of a multi-thread workload**, because at T=1 that cost is
  identically zero. That is the transferable lesson from this whole exercise.

## A resolved measurement puzzle

* An earlier reading of the shape probe on an
  **avx2** machine implied a T=1 step an order of magnitude longer, which looked like a
  contradiction. It was the ISA tier: the same probe on this box measures a **2.38x** gap between
  avx2 (374.6 ms) and avx512fp16 (157.1 ms) at lm_head's shape, far more than the ~1.5x the
  records suggested. At the top tier GLQ runs 4.05 G weights/s per thread, so 5.08 G
  weights/token / 4.05 / 0.7067 (the kernel's cycle share) = **1.77 s**. Three independent
  routes — DRAM traffic 1.41 s, Amdahl extrapolation 1.52 s, kernel throughput 1.77 s — now
  bracket the T=1 step at **1.4-1.8 s**. The denormal hypothesis was tested and falsified
  (random vs real packed data: 1.02x, zero subnormals). Separately, note that **local dev-machine
  timings were not reproducible** (the same shape gave 1.20 and 4.25 G weights/s on two runs);
  only the box numbers, whose ranges are ±0.1%, are used here.

  A fourth route — scaling T=8's measured 424.3 ms by effective-thread count and clock — gives
  1.10 s, so the honest bracket on T=1 is **~1.1-1.8 s**. Nothing in this file depends on
  pinning it down: T=8 and T=48 are both measured, and they carry every conclusion.

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
| `lm_head_shape_probe.txt` | full output of `benchmarks/_lm_head_shape_probe.py`, all four ISA tiers at 48 and 1 threads |
| `lm_head_probe.tsv` | the crossover table above, machine-readable |

`perf.data` (12.6 MB) is deliberately not committed; it is in `.local-logs/pmu_qwen_next/`.

Scope: batch 1, decode 512, **T=1**, dtype bfloat16, `avx512fp16` tier, this checkpoint, this
box. Nothing here generalizes to batch > 1, to other thread counts, or to other models without
re-measurement.
