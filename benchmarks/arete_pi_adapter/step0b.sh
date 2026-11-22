#!/usr/bin/env bash
# Step 0b: one LoRA server (the lora_b2048 arm of step0.sh, same argv), then
# step0_logprobs.py: do the adapter's logprobs differ from the base model's at all?
set -u
N=/opt/dlami/nvme
OUT=$N/arete_pi_step0
M=xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw
ADAPTER=$N/hf_cache/hub/models--harness-hub--Qwen3.8-27B-pi-adapted/snapshots/7d42b119e2d4d475fa11270634b1ef98d1cd5aee
PORT=8011
export PATH=$HOME/.glq/venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.glq/venv/lib/python3.14/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$N/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /tmp || exit 1
echo "=== step0b start $(date -u +%FT%TZ)"
LOG=$OUT/lora_b2048_logprobs.log
vllm serve $M --quantization glq --dtype bfloat16 --port $PORT \
    --gpu-memory-utilization 0.92 --max-model-len 131072 --max-num-seqs 1 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --language-model-only --override-generation-config '{"temperature":1.0,"top_k":20,"top_p":0.95}' \
    --max-num-batched-tokens 2048 \
    --enable-lora --max-lora-rank 32 --max-loras 1 \
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora-modules qwen3.8-pi=$ADAPTER > $LOG 2>&1 &
PID=$!
echo "vllm PID $PID"
t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
    kill -0 $PID 2>/dev/null || { echo "vllm exited before ready"; tail -40 $LOG; exit 1; }
    sleep 5
done
echo "ready after $(( $(date +%s) - t0 )) s"
python -I $N/tools/step0_logprobs.py --base-url http://127.0.0.1:$PORT --model $M \
    --out $OUT/lora_b2048_logprobs.json
kill -INT $PID
w=0; while kill -0 $PID 2>/dev/null && [ $w -lt 120 ]; do sleep 2; w=$((w + 2)); done
echo "EXIT=0 $(date -u +%FT%TZ)"
