#!/bin/bash
# Super-VL 3 bpw on a simulated 24 GB card (gpu-mem 0.70 of the RTX PRO 4500's 31.38 GiB =
# 21.96 GiB) with a 262,144-token window: the arena shrinks to 4 GiB so the KV pool still
# holds one full-length sequence. D = run_model.py decode; E = the card's vllm serve command
# plus a ~256k-token needle prompt. Sequential, one GPU job at a time.
V=$HOME/.glq/venv
export PATH=$V/bin:$V/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=/opt/dlami/nvme/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH=/opt/dlami/nvme/tf519 PYTHONFAULTHANDLER=1
M=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw
N=/opt/dlami/nvme
echo "== D_262k run_model $(date -Is)"
GLQ_MOE_EXPERT_CACHE=4 GLQ_MOE_ARENA_MAX_TOKENS=1 python $N/bench/run_model.py --model $M \
    --batches 1 --repeats 3 --decode 256 --maxtok 256 --greedy --mm0 --dtype bfloat16 \
    --gpu-mem 0.70 --max-model-len 262144 --offload-backend uva --offload-params experts \
    --cpu-offload-gb 42 --expect red --expect-gib 8.66 --label supervl3-D_262k \
    > $N/offload_D_262k.log 2>&1 < /dev/null
echo "EXIT_D=$?"
echo "== E_serve $(date -Is)"
GLQ_MOE_EXPERT_CACHE=4 GLQ_MOE_ARENA_MAX_TOKENS=1 vllm serve $M \
    --quantization glq --dtype bfloat16 \
    --offload-backend uva --cpu-offload-gb 42 --cpu-offload-params experts \
    --gpu-memory-utilization 0.70 --max-model-len 262144 --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' --port 8000 \
    > $N/serve_262k.log 2>&1 < /dev/null &
SPID=$!
echo "serve PID $SPID"
ok=0
for i in $(seq 1 240); do
  kill -0 $SPID 2>/dev/null || { echo "serve exited early"; break; }
  curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
  sleep 5
done
echo "ready=$ok $(date -Is)"
if [ $ok = 1 ]; then python $N/bench/needle_262k.py > $N/needle_262k.log 2>&1; echo "EXIT_needle=$?"; fi
kill -INT $SPID; wait $SPID; echo "EXIT_serve=$?"
echo "END $(date -Is)"
