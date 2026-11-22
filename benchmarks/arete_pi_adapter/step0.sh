#!/usr/bin/env bash
# Step 0 of the Arete pi-adapter plan: does vLLM 0.33 apply the LoRA
# harness-hub/Qwen3.8-27B-pi-adapted (rev 7d42b119) to GLQ Arete-27B 4 bpw, and at what cost?
#
# One GPU (0), one server at a time, five arms. Every arm is glq-code's Arete argv
# (role_serve_args "code" on glq 0.8.38, bf16, max-num-seqs 1) at a fixed window and util:
#
#   base_b2048        no LoRA, 2,048-token prefill step (vLLM's step on cards under 70 GiB)
#   lora_b2048        + LoRA: mechanism checks, greedy base vs adapter, a tool call, B=1 speed
#   lora_b2048_nospec + cudagraph_specialize_lora off: what the second graph set costs
#   base_b8192        no LoRA, 8,192-token step (cards of 70 GiB and up)
#   lora_b8192        + LoRA
#
# Memory comes from vLLM's own log lines. The mechanism count comes from vLLM's LoRA manager at
# DEBUG ("Successfully loaded LoRA weights for module …", one line per wrapped module when an
# adapter is activated), through a logging config that raises only that logger.
set -u
N=/opt/dlami/nvme
OUT=$N/arete_pi_step0
M=xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw
ADAPTER=$N/hf_cache/hub/models--harness-hub--Qwen3.8-27B-pi-adapted/snapshots/7d42b119e2d4d475fa11270634b1ef98d1cd5aee
PORT=8011
export PATH=$HOME/.glq/venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.glq/venv/lib/python3.14/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$N/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p $OUT
cd /tmp || exit 1

cat > $OUT/logging.json <<'EOF'
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"plain": {"format": "%(levelname)s %(asctime)s %(name)s %(message)s"}},
 "handlers": {"h": {"class": "logging.StreamHandler", "formatter": "plain", "level": "DEBUG",
                    "stream": "ext://sys.stdout"}},
 "loggers": {"vllm": {"handlers": ["h"], "level": "INFO", "propagate": false},
             "vllm.lora.model_manager": {"level": "DEBUG"}}}
EOF
export VLLM_LOGGING_CONFIG_PATH=$OUT/logging.json

echo "=== step0 start $(date -u +%FT%TZ)"
T=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
echo "instance $(curl -s -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/instance-type)"
nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import glq, vllm, torch; print("glq", glq.__version__, "vllm", vllm.__version__, "torch", torch.__version__)'
ls -la $ADAPTER/

BASE=(vllm serve $M --quantization glq --dtype bfloat16 --port $PORT
      --gpu-memory-utilization 0.92 --max-model-len 131072 --max-num-seqs 1
      --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
      --language-model-only --override-generation-config '{"temperature":1.0,"top_k":20,"top_p":0.95}')
LORA=(--enable-lora --max-lora-rank 32 --max-loras 1
      --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
      --lora-modules qwen3.8-pi=$ADAPTER)

gpu_free() {
    local used
    used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 500 ]
}

run_arm() {   # name, probe(yes|no), bench(none|base|both), extra serve args...
    local name=$1 probe=$2 bench=$3; shift 3
    local log=$OUT/$name.log
    gpu_free || { echo "ARM $name: GPU 0 not free, abort"; nvidia-smi; exit 1; }
    echo "=== ARM $name start $(date -u +%FT%TZ)"
    echo "argv: ${BASE[*]} $*" | tee $log
    "${BASE[@]}" "$@" >> $log 2>&1 &
    local pid=$!
    echo "ARM $name vllm PID $pid"
    local t0=$(date +%s)
    until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "ARM $name: vllm exited before ready"; tail -40 $log; return 1
        fi
        if [ $(( $(date +%s) - t0 )) -gt 1200 ]; then
            echo "ARM $name: not ready after 1200 s"; kill $pid; return 1
        fi
        sleep 5
    done
    echo "ARM $name ready after $(( $(date +%s) - t0 )) s"
    nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader | sed "s/^/ARM $name nvidia-smi used after ready: /"
    local mark=$(wc -l < $log)
    if [ "$probe" = yes ]; then
        python -I $N/tools/step0_probe.py --base-url http://127.0.0.1:$PORT --model $M \
            --adapter qwen3.8-pi --out $OUT/$name.probe.json
        echo "ARM $name modules with adapter weights (after the first real request):"
        tail -n +$((mark + 1)) $log | grep -c "Successfully loaded LoRA weights for module"
        echo "ARM $name modules wrapped but given no weights (after the first real request):"
        tail -n +$((mark + 1)) $log | grep -c "No LoRA weights found for module"
        tail -n +$((mark + 1)) $log | grep "Successfully loaded LoRA weights for module" \
            | sed 's/.*for module //; s/\.$//' > $OUT/$name.lora_modules.txt
    fi
    case $bench in
        base|both)
            vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model $M \
                --dataset-name random --random-input-len 1024 --random-output-len 256 \
                --num-prompts 10 --max-concurrency 1 --ignore-eos --seed 0 \
                --save-result --result-dir $OUT --result-filename $name.bench_base.json \
                > $OUT/$name.bench_base.txt 2>&1
            grep -E "Mean TTFT|Mean TPOT|Output token throughput" $OUT/$name.bench_base.txt | sed "s/^/ARM $name base: /";;
    esac
    case $bench in
        both)
            vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model $M \
                --lora-modules qwen3.8-pi \
                --dataset-name random --random-input-len 1024 --random-output-len 256 \
                --num-prompts 10 --max-concurrency 1 --ignore-eos --seed 0 \
                --save-result --result-dir $OUT --result-filename $name.bench_adapter.json \
                > $OUT/$name.bench_adapter.txt 2>&1
            grep -E "Mean TTFT|Mean TPOT|Output token throughput" $OUT/$name.bench_adapter.txt | sed "s/^/ARM $name adapter: /";;
    esac
    grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Graph capturing finished|init engine|Loaded new LoRA|LoRA" $log \
        | grep -v "Successfully loaded LoRA weights\|No LoRA weights found\|Activating LoRA\|Adding lora" \
        | cut -c1-220 | sed "s/^/ARM $name log: /" | head -30
    kill -INT $pid
    local w=0
    while kill -0 $pid 2>/dev/null && [ $w -lt 120 ]; do sleep 2; w=$((w + 2)); done
    kill -0 $pid 2>/dev/null && { echo "ARM $name: still up after SIGINT, SIGTERM"; kill $pid; sleep 10; }
    for i in $(seq 30); do gpu_free && break; sleep 2; done
    echo "=== ARM $name done $(date -u +%FT%TZ)"
}

run_arm base_b2048        no  base  --max-num-batched-tokens 2048
run_arm lora_b2048        yes both  --max-num-batched-tokens 2048 "${LORA[@]}"
run_arm lora_b2048_nospec no  none  --max-num-batched-tokens 2048 "${LORA[@]}" \
    --compilation-config '{"cudagraph_specialize_lora": false}'
run_arm base_b8192        no  none  --max-num-batched-tokens 8192
run_arm lora_b8192        no  none  --max-num-batched-tokens 8192 "${LORA[@]}"
echo "EXIT=0 $(date -u +%FT%TZ)"
