#!/usr/bin/env bash
# Reproduce and LOCALISE the Xid 31 MMU fault that killed the first full AIME-2026 run.
#
# What happened (measured, not inferred):
#   - full run died at 64/240 after stalling 47 min with zero completions
#   - kernel ring buffer:
#       NVRM: Xid (PCI:0000:2f:00): 31, pid=<EngineCore>, name=python3.12,
#       MMU Fault: ENGINE GRAPHICS GPC3 GPCCLIENT_T1_2 faulted @ 0x7f63_cb602000,
#       Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
#   - no host OOM (dmesg clean, 121 GiB free), no CUDA OOM exception, and NO Python
#     traceback from EngineCore -- consistent with the process dying from the fault
#
# FAULT_PDE on a read = the virtual address had no page-table mapping at all. That is an
# ILLEGAL ACCESS by a kernel, not memory exhaustion, so raising gpu-mem-util does not fix it
# -- it only removes the preemption/recompute pressure that correlates with it.
#
# THIS SCRIPT DELIBERATELY USES THE FAULTING CONFIG, not the corrected one in
# _aime2026_qwen_next.sh. With gpu-mem-util 0.95 / budget 32768 the KV pool is ~7x
# oversubscribed instead of 6x under, so the fault may simply not occur -- and then
# CUDA_LAUNCH_BLOCKING would have nothing to report.
#
#   --gpu-mem-util 0.90   -> KV 2.72 GiB = 94,132 tokens
#   --max-model-len 69632 -> "Maximum concurrency ...: 1.35x" against --max-num-seqs 8
#                            i.e. 8 seqs want 557,056 tokens: constant preempt+recompute
#
# Why those env vars (vLLM troubleshooting guide):
#   CUDA_LAUNCH_BLOCKING=1          THE one that matters here. Kernel launches become
#                                   synchronous, so an MMU fault is attributed to the kernel
#                                   that caused it BY NAME instead of surfacing at some later
#                                   unrelated sync point. Costs throughput; that is the trade.
#   VLLM_LOGGING_LEVEL=DEBUG        engine-side detail around the stall and the death.
#   VLLM_ENABLE_V1_MULTIPROCESSING=0  run EngineCore IN-PROCESS. Two payoffs: any *raisable*
#                                   error lands in the client traceback that _safe_run
#                                   actually stores in the record, and runtime.py's
#                                   _capture_fd_tee (which only wraps LLM() construction,
#                                   never generation) stops being the reason output is lost.
#   PYTHONFAULTHANDLER=1            dumps a C-level stack if the process dies on a signal.
#
# NOT used, deliberately:
#   NCCL_DEBUG=TRACE                single GPU, TP=1 -- there are no NCCL collectives here.
#   VLLM_TRACE_FUNCTION=1           records every function call; enormous logs and a large
#                                   slowdown. Escalate to it only if CUDA_LAUNCH_BLOCKING
#                                   fails to localise the kernel.
#
# n=12 (96 sequences) because the fault appeared around 64-72 of 240; no need to queue 30
# problems to reach it. If it does not fault by the end, raise N.
#
#   ./_aime2026_xid31_repro.sh                # default N=12
#   N=30 ./_aime2026_xid31_repro.sh           # if 12 is not enough
set -u

MODEL="${MODEL:-xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4}"
N="${N:-12}"
OUT="${OUT:-/opt/dlami/nvme/aime2026_xid31_repro.jsonl}"
GLQ_BENCH="${GLQ_BENCH:-/home/ubuntu/.glq/venv/bin/glq-bench}"

export HF_HOME="${HF_HOME:-/opt/dlami/nvme/hf_cache}"
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONFAULTHANDLER=1

echo "=== repro config (the FAULTING one, on purpose) ==="
echo "  gpu-mem-util 0.90  max-model-len 69632  budget 65536  max-num-seqs 8  n=$N"
echo "  CUDA_LAUNCH_BLOCKING=1  VLLM_LOGGING_LEVEL=DEBUG  V1_MULTIPROCESSING=0"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

"$GLQ_BENCH" run \
  --model "$MODEL" \
  --tasks aime_2026 \
  --quant glq \
  --dtype bfloat16 \
  --n "$N" \
  --avg-k 8 \
  --budget 65536 \
  --max-num-seqs 8 \
  --gpu-mem-util 0.90 \
  --max-model-len 69632 \
  --task-config '{"system": null, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 0, "min_mean_gen": 2000}' \
  --out "$OUT"
echo "BENCH_EXIT=$?"

# The kernel ring buffer is where the Xid lands, and it is NOT in any application log.
echo "=== dmesg Xid/NVRM after the run ==="
sudo dmesg -T 2>/dev/null | grep -iE "NVRM|Xid" | tail -8 || echo "(dmesg unavailable)"
