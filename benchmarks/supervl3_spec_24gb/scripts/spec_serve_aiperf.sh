#!/bin/bash
# One arm = vllm serve (Super-VL 3 bpw, 24 GB budget, experts on host + GLQ expert arena,
# optional draft) + AIPerf on SPEED-Bench coding, then the server is stopped by exact PID.
# Usage: spec_serve_aiperf.sh <name> <offload GiB> <arena GiB> <step tokens> <window>
#                             <request count> <osl> [<speculative-config json>]
# Arms run one at a time (one GPU job). Sampling: NVIDIA's generation_config (temperature 1.0,
# top_p 0.95), sent by AIPerf because the GLQ repo ships no generation_config.json.
name=$1 off=$2 arena=$3 st=$4 win=$5 nreq=$6 osl=$7 spec=$8
V=$HOME/.glq/venv
A=/opt/dlami/nvme/aiperf-venv
N=/opt/dlami/nvme
export PATH=$V/bin:$V/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=$N/hf_cache HF_HUB_DISABLE_TELEMETRY=1 PYTHONFAULTHANDLER=1
M=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw
SPEC_ARGS=()
[ -n "$spec" ] && SPEC_ARGS=(--speculative-config "$spec")
echo "== $name offload=$off arena=$arena step=$st window=$win n=$nreq osl=$osl spec=${spec:-none} $(date -Is)"
GLQ_MOE_EXPERT_CACHE=$arena GLQ_MOE_ARENA_MAX_TOKENS=$st GLQ_MOE_ARENA_MIN_TOKENS=$st \
HF_HUB_OFFLINE=1 PYTHONPATH=$N/tf519 vllm serve $M \
    --quantization glq --dtype bfloat16 \
    --offload-backend uva --cpu-offload-gb $off --cpu-offload-params experts \
    --gpu-memory-utilization 0.70 --max-model-len $win --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
    "${SPEC_ARGS[@]}" --port 8000 > $N/spec_serve_$name.log 2>&1 < /dev/null &
spid=$!
echo "serve PID $spid"
ok=0
for i in $(seq 1 240); do
  kill -0 $spid 2>/dev/null || { echo "serve exited early"; break; }
  curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
  sleep 5
done
echo "ready=$ok $(date -Is)"
if [ $ok = 1 ]; then
  # Tokenizer by repo id, resolved offline from the cache: aiperf 0.11.0 treats a local
  # snapshot path as a repo id and fails (HFValidationError). --custom-dataset-type needs
  # --input-file: the 70 SPEED-Bench coding rows whose text is in the HF parquet (the
  # other 10 are placeholders to be fetched from their source with specdec_bench).
  HF_HUB_OFFLINE=1 $A/bin/aiperf profile --model $M --tokenizer $M \
      --url http://localhost:8000 --endpoint-type chat --streaming \
      --input-file $N/bench/speed_bench_coding.jsonl \
      --custom-dataset-type speed-bench-coding --request-count $nreq --random-seed 42 \
      --osl $osl --extra-inputs max_completion_tokens:$osl \
      --extra-inputs temperature:1.0 --extra-inputs top_p:0.95 \
      --concurrency 1 --warmup-request-count 1 --request-timeout-seconds 3600 \
      --artifact-dir $N/aiperf/$name > $N/spec_aiperf_$name.log 2>&1 < /dev/null
  echo "EXIT_aiperf_$name=$?"
  # Optional follow-up against the same server before it stops (e.g. a long-prompt check).
  if [ -n "$POST" ]; then
    eval "$POST" > $N/spec_post_$name.log 2>&1 < /dev/null
    echo "EXIT_post_$name=$?"
  fi
fi
kill -INT $spid 2>/dev/null
for i in $(seq 1 60); do kill -0 $spid 2>/dev/null || break; sleep 2; done
kill -0 $spid 2>/dev/null && { echo "serve still up, SIGTERM $spid"; kill -TERM $spid; }
wait $spid; echo "EXIT_serve_$name=$?"
echo "END_$name $(date -Is)"
