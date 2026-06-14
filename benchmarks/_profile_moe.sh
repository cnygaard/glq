#!/bin/bash
# Phase 0: profile GLQ 26B-A4B MoE decode via vLLM's official `vllm bench latency`
# under nsys. Per the plan: VLLM_WORKER_MULTIPROC_METHOD=spawn (nsys needs spawn),
# --cuda-graph-trace=node so kernels are visible INSIDE the FULL cudagraph. Cudagraph
# capture sizes limited to <=64 (the proven config) so the >256 Python-fallback (which
# is capture-illegal) is never hit. Profiles the validated local 3bpw model at b1 + b32.
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source /home/ubuntu/venv/bin/activate

MODEL=/opt/dlami/nvme/glq_out/gemma-4-26B-A4B-it-GLQ-3bpw
CC='{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'

# DECODE-dominated: tiny prefill (16), long generation (64) so the MoE decode path
# (per-(token,expert) B=1 matvec) dominates the kernel-time breakdown, not prefill.
for B in 1 32; do
  echo "===================== nsys DECODE profile batch=$B ====================="
  nsys profile --force-overwrite=true --trace-fork-before-exec=true \
    --cuda-graph-trace=node -o /opt/dlami/nvme/glq_moe_dec_b${B} \
    vllm bench latency \
      --model "$MODEL" --quantization glq --trust-remote-code \
      --max-model-len 2048 --gpu-memory-utilization 0.90 \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --compilation-config "$CC" --async-scheduling \
      --num-iters-warmup 2 --num-iters 1 \
      --batch-size $B --input-len 16 --output-len 64 \
      --output-json /opt/dlami/nvme/glq_lat_dec_b${B}.json
  echo "--------------------- latency json b=$B ---------------------"
  cat /opt/dlami/nvme/glq_lat_dec_b${B}.json 2>/dev/null
  echo ""
  echo "--------------------- kernel time summary b=$B ---------------------"
  nsys stats --report cuda_gpu_kern_sum /opt/dlami/nvme/glq_moe_dec_b${B}.nsys-rep 2>/dev/null | head -28
done
echo "PROFILE_DEC_DONE"
