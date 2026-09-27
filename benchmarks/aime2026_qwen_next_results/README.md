# AIME-2026 on Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4

Rescued from a spot box (18.100.53.58, RTX PRO 6000 Blackwell 95.0 GiB) on 2026-09-27.
Reproduce with `benchmarks/_aime2026_qwen_next.sh`, which pins every knob.

## ⚠️ The headline number is BUDGET-LIMITED. Do not cite it as the model's AIME-2026 score.

`aime2026_qwen_next.jsonl` — avg@8, n=30, **accuracy 0.70** — was run at `budget 16384`
against the `aime_2026` task default of **65536**, and it shows:

| | |
|---|---|
| accuracy (avg@8) | **0.70** |
| **truncated** | **119 / 240 samples (49.6%)** |
| no_answer | 31 / 240 |
| mean_gen_tokens | 11,341 (of a 16,384 budget) |
| solved_all / solved_any | 14 / 27 of 30 |
| wall clock | 3 h 43 m |
| output_tok_s | 202.9 |

Half the samples hit the cap, and a truncated chain never reaches `\boxed{}`, which is most of
the 31 no-answers. **0.70 is a floor for this checkpoint, not its score.** The gap between
`solved_any=27` and `solved_all=14` is consistent with truncation rather than capability.

The budget was cut (65536 → 32768 → 20480 → 16384) to escape a KV-capacity bind, each step
justified by a PILOT mean of 4,873 tokens — but `aime.py::_build` does `rows[:n]`, so a small-n
pilot only sees the EASIEST problems. The full set's mean came in at 11,341, 2.3x the pilot.
Do not size a budget from a small-n pilot on this task.

## Why the budget had to shrink, and what it costs

`max-num-seqs` must be **>= avg_k**: `sampling(..., n=avg_k)` makes each problem one request of
8 parallel sequences, returned only when all 8 finish, so a smaller value splits every problem
into waves and the stragglers hold the request open. Measured: 12 min/problem at 5 vs
~2.75 min/problem at 8.

But `max-num-seqs * max-model-len` must fit GPU KV or sequences get preempted — and preemption
crashes this model (see below). Capacity is CIRCULAR: raising `max-num-seqs` enlarges the
activation profile, shrinking KV. Measured on this box:

| max-num-seqs | max-model-len | KV tokens | concurrency | verdict |
|---|---|---|---|---|
| 1 (probe) | 36864 | 203,310 | 5.52x | — |
| 5 | 36864 | 214,481 | 5.82x | fits, but < avg_k -> 6 h run |
| 8 | 24576 | 187,760 | 7.64x | **oversubscribed** (needs 196,608) |
| 8 | 20480 | 170,073 | 8.30x | fits (163,840), the run below |

Always read the engine's own `GPU KV cache size` / `Maximum concurrency` at startup. Deriving
it from the util budget minus profiled usage over-predicted by 21% and would have chosen an
oversubscribed value.

## The crash these settings avoid — a vLLM bug, not GLQ

`aime2026_xid31_repro.jsonl` reproduces it with `CUDA_LAUNCH_BLOCKING=1` and
`VLLM_ENABLE_V1_MULTIPROCESSING=0` (see `benchmarks/_aime2026_xid31_repro.sh`):

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  vllm/v1/worker/gpu/model_runner.py:1599              execute_model
  vllm/v1/worker/gpu/model_states/mamba_hybrid.py:224  preprocess_state
  vllm/v1/worker/mamba_utils.py:1213                   run_fused_precopy
    precopy_mamba_align_fused_kernel[grid](...)
```

and in the kernel ring buffer (never in any application log):

```
NVRM: Xid 31, MMU Fault: ENGINE GRAPHICS, FAULT_PDE ACCESS_TYPE_VIRT_READ
```

The faulting kernel is **vLLM's own Mamba-state hybrid precopy**. No GLQ kernel, quantization
path or MoE code is on that stack. It fires when KV is oversubscribed: preemption produces a
chunked-prefill mixed batch containing a RESUMED request (`prefill_token_ids_len=26657` with
`num_computed_tokens=15680`) over a hybrid Mamba cache, with block tables full of zero entries.
Reachable by any Qwen3.8-Flash-Next serve that oversubscribes KV, quantized or not — worth
reporting upstream. Also note a recurring exactly-5.00 GiB transient allocation that appears
even without preemption; when free memory is ample the allocator reclaims and recovers, and
when it is not (1.41 GiB in the first crash) the failure precedes the illegal access.

## Files

| file | what |
|---|---|
| `aime2026_qwen_next.jsonl` | the full run. accuracy 0.70, **budget-limited**. `extra.per_item` has all 30 per-problem fractions, so a future arm can be compared item-for-item by McNemar rather than by differencing percentages |
| `aime2026_pilot3.jsonl` | n=4 avg@2 pilot, accuracy 1.0, mean_gen 4,873, truncated 0. Config validation only — NOT reportable (n=4, easiest problems) |
| `aime2026_pilot.jsonl` | the same pilot skipped by the thinking guard at mean_gen 3842 vs a 4000 floor. Kept because it shows the floor is miscalibrated for this model: 3842 tokens cannot be a no-think run, `system` was None, and the template defaults `reasoning_effort='xhigh'` |
| `aime2026_xid31_repro.jsonl` | the Xid 31 reproduction record |

## Open items

* **No bf16 baseline exists, and none is obtainable here.** `Qwen/Qwen3.8-Flash-Next` is
  335.3 GiB (131 shards) = TP=4 on 96 GiB cards; the FP8 variant is 172.8 GiB = TP=2. And
  `glq-bench` plumbs neither `tensor_parallel_size` nor `kv_transfer_config`, so neither arm
  can run on any box until that is threaded through.
* `env.glq_git_sha` is **null** in every record — the provenance cannot say which commit
  produced the number.
* `ServingMeta.kv_cache_dtype` is never populated by `load()`; it survives only inside
  `llm_kwargs`.
