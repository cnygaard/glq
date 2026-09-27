#!/usr/bin/env bash
# AIME-2026 on Qwen3.8-Flash-Next, reproducibly — one script for BOTH arms.
#
# EVERY knob is pinned explicitly, including ones that currently match a default, so a later
# default change cannot silently alter the comparison. That is not hypothetical: until #133
# `--dtype` was unreachable from the CLI and every vLLM-backed number came out bf16 whether
# or not that was intended.
#
#   --dtype bfloat16   REQUIRED, not a preference. Qwen4Exp's QSA raises NotImplementedError
#                      on fp16 (vllm/models/qwen4_exp/nvidia/qsa.py + indexer_qsa.py, 4
#                      sites). preferred_dtype already resolves bf16 for the qwen family, but
#                      pin it so both arms are provably identical here.
#   --max-num-seqs 8   MUST BE >= --avg-k, and that is not a throughput nicety.
#                      `sampling(config, budget, n=avg_k)` makes each problem ONE request with
#                      8 parallel sequences, and vLLM returns the request only when all 8
#                      finish. At max-num-seqs 5 every problem needs two waves and the 3
#                      stragglers hold the request open, so the loss is far worse than 5/8:
#                      measured 12 min/problem at 5 against ~3 min/problem at 8, i.e. a 6 h
#                      run instead of ~1.5 h. Setting it to exactly avg_k means one problem
#                      per wave with no splitting.
#
#                      Then three reasons for the rest of the numbers.
#                      (a) Hybrid-GDN: every decode slot reserves a Mamba cache block before a
#                          single request exists, so this knob is what avoids "max_num_seqs
#                          exceeds available Mamba cache blocks".
#                      (b) It must not exceed the engine's KV concurrency, or sequences get
#                          PREEMPTED and resumed -- and resume is what crashes this model (see
#                          the Xid note below). MEASURED at this exact config by starting the
#                          engine and reading what it reports:
#                            Available KV cache memory: 6.75 GiB
#                            GPU KV cache size: 203,310 tokens
#                            Maximum concurrency for 36,864 tokens per request: 5.52x
#                          5 x 36,864 = 184,320 <= 203,310, so even with every sequence at
#                          full length there is ~10% headroom and preemption cannot occur.
#                          NOT a throughput choice; raising it re-opens the crash.
#
#                          Do NOT derive this number instead of measuring it. Subtracting the
#                          profiled usage from the util budget predicted 7.14 GiB / 247,100
#                          tokens / 6.70x, which would have justified 6 -- and 6 oversubscribes
#                          the real 203,310. The engine's own startup report is the only
#                          trustworthy source, and it costs one ~2-minute engine init.
#   --gpu-mem-util     0.95, NOT the harness default 0.90. Measured: 73.29 GiB of weights on a
#                      94.97 GiB card at 0.90 (85.48 GiB) left only **2.72 GiB** of KV =
#                      94,132 tokens, i.e. "Maximum concurrency for 69,632 tokens per request:
#                      1.35x" while --max-num-seqs was 8. Eight sequences want 557,056 tokens
#                      against 94,132 available -- 6x oversubscribed, so the engine preempts
#                      and recomputes constantly. The official vLLM recipe also uses 0.95.
#
#                      WHAT KILLED THE FIRST FULL RUN IS NOT PROVEN TO BE THIS. It died at
#                      64/240 after stalling 47 minutes with zero completions, and the kernel
#                      log shows the real cause:
#
#                        NVRM: Xid (PCI:0000:2f:00): 31, pid=<EngineCore>, MMU Fault:
#                        ENGINE GRAPHICS GPC3 faulted @ 0x7f63_cb602000,
#                        FAULT_PDE ACCESS_TYPE_VIRT_READ
#
#                      Xid 31 with FAULT_PDE on a READ = a kernel read an address with no
#                      page-table mapping: an ILLEGAL ACCESS, not exhaustion. There was no
#                      host OOM (dmesg clean, 121 GiB free) and no Python traceback from
#                      EngineCore, consistent with the process being killed by the fault
#                      rather than raising. So raising gpu-mem-util does not "fix" that bug;
#                      it removes the preemption/recompute pressure that CORRELATES with it.
#
#                      LOCALISED, and it is NOT GLQ. Reproduced with CUDA_LAUNCH_BLOCKING=1
#                      and VLLM_ENABLE_V1_MULTIPROCESSING=0 (see _aime2026_xid31_repro.sh),
#                      which turned the silent death into a named error:
#
#                        RuntimeError: Triton Error [CUDA]: an illegal memory access
#                          vllm/v1/worker/gpu/model_runner.py:1599        execute_model
#                          vllm/v1/worker/gpu/model_states/mamba_hybrid.py:224 preprocess_state
#                          vllm/v1/worker/mamba_utils.py:1213             run_fused_precopy
#                            precopy_mamba_align_fused_kernel[grid](...)
#
#                      The faulting kernel is vLLM's own Mamba-state hybrid precopy. No GLQ
#                      kernel, quantization path or MoE code is on that stack. An earlier guess
#                      that blamed the MoE prefill branch above GLQ_MOE_BD_MAX_TOKENS was
#                      WRONG -- recorded here so nobody re-derives it.
#
#                      The scheduler dump at the fault shows what it choked on: a chunked
#                      prefill MIXED batch, total_num_scheduled_tokens=16381, combining a
#                      RESUMED request (prefill_token_ids_len=26657 with num_computed_tokens=
#                      15680) with a fresh 17,701-token prefill and one decode token, over a
#                      hybrid Mamba cache, with block tables full of zero entries. That is a
#                      vLLM bug in hybrid preemption/resume, reachable by any Qwen3.8-Flash-Next
#                      serve that oversubscribes KV -- quantized or not -- and worth reporting
#                      upstream with that traceback and dump.
#
#                      Which is why --max-num-seqs is set from MEASURED concurrency: if
#                      preemption never happens, resume never happens, and the vLLM bug is
#                      unreachable. This is not papering over a GLQ defect; there is none.
#   --budget 32768     was 65536 (the aime_2026 task default) and that is what made each
#                      sequence's KV ceiling unaffordable. Evidence it is ample: the pilot
#                      measured mean_gen 4,873 tokens with truncated=0, so 32768 leaves ~6.7x
#                      headroom over observed length, and it is the value CLAUDE.md specifies
#                      for AIME. Scores are NOT comparable across budgets, so both arms must
#                      use this one -- which is why it is pinned here rather than defaulted.
#   --max-model-len    32768 + 4096 for the prompt. Halving this doubles achievable
#                      concurrency, because concurrency = KV_tokens / max_model_len.
#                      At 0.95 util the KV pool is ~7.5 GiB ~= 258k tokens, so
#                      258k / 36,864 ~= 7x -- enough for --max-num-seqs 8 with only mild,
#                      normal preemption instead of thrash.
#   --task-config      Sampling belongs to the MODEL card, not the task. Qwen3.8-Flash-Next
#                      thinking mode is temp 1.0 / top_p 0.95 / top_k 20; the harness default
#                      top_k is 64, which is gemma-4 value, so top_k MUST be overridden.
#                      system:null means NO system message, which is what keeps thinking
#                      engaged. seed pinned so the drawn sample set is reproducible.
#                      min_p / presence_penalty / repetition_penalty are 0.0 / 0.0 / 1.0 on
#                      the card and those are already vLLM defaults, so they need no override
#                      -- note tasks/thinking.py could not express them anyway; it reads only
#                      temperature, top_p, top_k and seed.
#
# Thinking engagement is asserted by the harness, not trusted: tasks/thinking.py raises if
# mean generation falls below min_mean_gen, because a reasoning eval that never engaged
# reasoning still completes, still parses, and just scores like a worse model.
#
#   min_mean_gen 2000  LOWERED from the 4000 default, with evidence, not to make a run pass.
#                      A pilot (n=4, avg-k 2) was skipped at mean_gen 3842. All three ways
#                      thinking could have failed were ruled out: system was None (so the
#                      SmolLM3 template trap the error names cannot apply), the checkpoint
#                      template defaults `enable_thinking` true, and it defaults
#                      `reasoning_effort` to 'xhigh' -- the MAXIMUM. The run was already
#                      asking for the most thorough reasoning available.
#
#                      The decisive point: 3842 tokens cannot be a no-think run. A direct
#                      answer to "give the final answer as a non-negative integer in \boxed{}"
#                      is ~100-300 tokens, so 3842 is 15-35x that and extended reasoning
#                      demonstrably happened. The 4000 floor was calibrated on SmolLM3
#                      (~14-15k thinking vs ~1.8k no-think at a 32k budget); this model is far
#                      more concise per token of reasoning and those numbers do not transfer.
#
#                      2000 still leaves ~7-20x margin over a genuine no-think collapse while
#                      not failing this model. Also note `_build` does `rows[:n]`, so a small-n
#                      pilot gets the EASIEST problems -- the full n=30 reaches problems 5-15
#                      and should run considerably longer than 3842.
#
# ---------------------------------------------------------------------------------------
# GLQ arm (runs on one 96 GB card; 73.01 GiB of weights):
#
#   ./_aime2026_qwen_next.sh
#
# Pilot to validate the config before committing ~12 h (number NOT reportable at n=4):
#
#   N=4 AVGK=2 OUT=/opt/dlami/nvme/aime2026_pilot.jsonl ./_aime2026_qwen_next.sh
#
# bf16 baseline arm — BLOCKED, and not only by hardware:
#
#   MODEL=Qwen/Qwen3.8-Flash-Next QUANT=none ./_aime2026_qwen_next.sh
#
#   * 335.3 GiB across 131 shards = 3.4x a 96 GB card, so it needs ~4x96 GB, TP=4.
#   * glq-bench does NOT plumb tensor_parallel_size at all — build_llm_kwargs does not
#     accept it, load() does not pass it, no CLI flag exposes it. So this arm cannot run on
#     any box until that is threaded through (same shape as the --dtype gap, plus recording
#     it in ServingMeta/serving_command so a record states its own parallelism).
# ---------------------------------------------------------------------------------------
set -u

MODEL="${MODEL:-xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4}"
QUANT="${QUANT:-glq}"
N="${N:-30}"
AVGK="${AVGK:-8}"
OUT="${OUT:-/opt/dlami/nvme/aime2026_qwen_next.jsonl}"
GLQ_BENCH="${GLQ_BENCH:-/home/ubuntu/.glq/venv/bin/glq-bench}"

export HF_HOME="${HF_HOME:-/opt/dlami/nvme/hf_cache}"

"$GLQ_BENCH" run \
  --model "$MODEL" \
  --tasks aime_2026 \
  --quant "$QUANT" \
  --dtype bfloat16 \
  --n "$N" \
  --avg-k "$AVGK" \
  --budget 16384 \
  --max-num-seqs 8 \
  --gpu-mem-util 0.95 \
  --max-model-len 20480 \
  --task-config '{"system": null, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 0, "min_mean_gen": 2000}' \
  --out "$OUT"
echo "BENCH_EXIT=$?"
