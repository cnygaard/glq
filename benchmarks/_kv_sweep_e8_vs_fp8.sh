#!/bin/bash
# KV-cache quantization sweep: GLQ E8-KV vs fp8 vs TurboQuant vs bf16.
#
# One GPU, so arms run strictly one at a time, each writing UNBUFFERED to its own
# file. Weights stay bf16 in every arm — the KV method is the only variable, as in
# vLLM's TurboQuant blog (2026-05-11), whose serving shape this reproduces:
# random 1024-in/512-out, 300 prompts, request rates 2 / 8 / inf.
#
# The bar is fp8, not TurboQuant: the blog measured fp8 at 2x capacity for
# negligible accuracy cost and rejected the 3-bit TurboQuant variants largely on
# latency and throughput. E8-KV dequantizes inside the attention kernel, so speed
# is where it is most likely to lose — which is why this gate comes before any
# accuracy eval.
#
#   REPO=/opt/dlami/nvme/glq_src OUT=/opt/dlami/nvme/kvsweep \
#     ./benchmarks/_kv_sweep_e8_vs_fp8.sh
#
# Smoke it on the small cached model first — same code path, minutes not hours:
#   MODEL=HuggingFaceTB/SmolLM2-360M-Instruct MAXLEN=4096 NPROMPTS=32 \
#     ARMS="bf16 e8_relaxed2" ./benchmarks/_kv_sweep_e8_vs_fp8.sh
set -u

REPO=${REPO:-/opt/dlami/nvme/glq_src}
VENV=${VENV:-/home/ubuntu/.glq/venv}
OUT=${OUT:-/opt/dlami/nvme/kvsweep}
HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf_cache}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
ARMS=${ARMS:-"bf16 fp8 turboquant_4bit_nc e8_relaxed2 bf16_triton fp8_triton"}

# Held FIXED across arms. Vary any of these and the arms stop being comparable —
# capacity especially is meaningless unless the pool fraction is identical.
PORT=${PORT:-8321}
GPU_UTIL=${GPU_UTIL:-0.85}
MAXLEN=${MAXLEN:-8192}
IN_LEN=${IN_LEN:-1024}
OUT_LEN=${OUT_LEN:-512}
NPROMPTS=${NPROMPTS:-300}
RATES=${RATES:-"2 8 inf"}
SEED=${SEED:-42}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

export HF_HOME PYTHONPATH="$REPO"
# sm_120 without a CUDA toolkit: FlashInfer JIT-compiles a sampler, finds no nvcc
# and ends EngineCore *after* the model has loaded. Unrelated to KV; pinned off so
# a sampler problem cannot be mistaken for an arm failing.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

BIN="$VENV/bin"
mkdir -p "$OUT"

# arm -> "serve flags|extra env (space separated KEY=VAL)"
declare -A ARM_SPEC=(
    [bf16]="|"
    [fp8]="--kv-cache-dtype fp8|"
    [turboquant_4bit_nc]="--kv-cache-dtype turboquant_4bit_nc|"
    # The six flags of E8_KV_ENV (glq/kv_compression.py). GLQ_KV_QUANT alone is
    # only the round-trip quality simulator — it saves nothing.
    [e8_relaxed2]="|GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 GLQ_KV_E8_FUSED_GATHER=1 GLQ_KV_E8_FUSED_WRITE=1"
    # E8-KV forces TRITON_ATTN. Comparing it against a FlashAttention baseline
    # would confound the KV method with the attention backend, so these two give
    # a like-for-like floor; the plain bf16/fp8 arms give the realistic default.
    [bf16_triton]="|VLLM_ATTENTION_BACKEND=TRITON_ATTN"
    [fp8_triton]="--kv-cache-dtype fp8|VLLM_ATTENTION_BACKEND=TRITON_ATTN"
)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

wait_gpu_free() {  # $1 = seconds to wait for the card to drain
    local deadline=$((SECONDS + ${1:-180})) used
    while [ $SECONDS -lt $deadline ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "$used" ] && [ "$used" -lt 1000 ] && { log "  GPU free (${used} MiB)"; return 0; }
        sleep 5
    done
    log "  WARNING: GPU still holding ${used:-?} MiB — the next arm's capacity will be understated"
    return 1
}

wait_ready() {  # $1 = server pid, $2 = log file
    local pid=$1 logf=$2 deadline=$((SECONDS + READY_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        # Liveness first: a dead server means failing now beats waiting out the
        # timeout, and the log already holds the reason.
        kill -0 "$pid" 2>/dev/null || { log "  server pid $pid exited"; return 1; }
        curl -sf -o /dev/null "http://127.0.0.1:${PORT}/v1/models" && return 0
        sleep 5
    done
    log "  server not ready within ${READY_TIMEOUT}s"
    return 1
}

run_arm() {
    local arm=$1 spec flags envs serve_log pid rc cap
    spec=${ARM_SPEC[$arm]:-}
    if [ -z "$spec" ]; then log "ARM $arm UNKNOWN — skipped"; return 1; fi
    if [ -f "$OUT/$arm.done" ]; then log "ARM $arm already done — skipped"; return 0; fi

    flags=${spec%%|*}; envs=${spec#*|}
    serve_log="$OUT/$arm.server.log"
    log "=== ARM $arm ==="
    log "  flags: ${flags:-none}   env: ${envs:-none}"

    # setsid: the server gets its own process group, so teardown kills the whole
    # tree by group rather than by pattern (a pattern match would also catch the
    # ssh command running this script).
    # shellcheck disable=SC2086
    setsid nohup env $envs stdbuf -oL -eL "$BIN/vllm" serve "$MODEL" \
        --port "$PORT" --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$GPU_UTIL" --seed "$SEED" $flags \
        > "$serve_log" 2>&1 &
    pid=$!
    log "  server pid $pid -> $serve_log"

    if ! wait_ready "$pid" "$serve_log"; then
        log "  ARM $arm FAILED to start; see $serve_log"
        kill -- -"$pid" 2>/dev/null; wait_gpu_free 120
        echo "$arm start-failed" >> "$OUT/failures.txt"
        return 1
    fi

    cap=$(grep -oE "KV cache size: [0-9,]+ tokens" "$serve_log" | head -1)
    log "  ready. ${cap:-KV cache size not found in log}"
    echo "${cap:-unknown}" > "$OUT/$arm.capacity.txt"

    # Warm the server so the first measured rate does not pay for lazy init.
    stdbuf -oL -eL "$BIN/vllm" bench serve --base-url "http://127.0.0.1:${PORT}" \
        --model "$MODEL" --dataset-name random --random-input-len "$IN_LEN" \
        --random-output-len "$OUT_LEN" --num-prompts 5 --seed "$SEED" \
        > "$OUT/$arm.warmup.log" 2>&1
    log "  warmup rc=$?"

    for rate in $RATES; do
        log "  bench rate=$rate"
        stdbuf -oL -eL "$BIN/vllm" bench serve \
            --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" \
            --dataset-name random --random-input-len "$IN_LEN" \
            --random-output-len "$OUT_LEN" --num-prompts "$NPROMPTS" \
            --request-rate "$rate" --seed "$SEED" \
            --save-result --result-dir "$OUT" \
            --result-filename "$arm.rate$rate.json" \
            > "$OUT/$arm.rate$rate.log" 2>&1
        log "    rc=$?"
    done

    # Signal the `vllm serve` leader alone and let it drain. Signalling the whole
    # group at once kills EngineCore and the API server simultaneously, and the
    # API server's output handler then raises EngineDeadError into the log — a
    # teardown artifact that reads exactly like a real crash, in every arm.
    log "  stopping server (pid $pid)"
    kill -TERM "$pid" 2>/dev/null
    local drain=$((SECONDS + 60))
    while kill -0 "$pid" 2>/dev/null && [ $SECONDS -lt $drain ]; do sleep 2; done
    # Anything still alive after the grace period: take the group, including
    # EngineCore workers that outlived their parent.
    kill -9 -- -"$pid" 2>/dev/null
    wait_gpu_free 300
    touch "$OUT/$arm.done"
    log "=== ARM $arm DONE ==="
}

log "sweep: model=$MODEL util=$GPU_UTIL maxlen=$MAXLEN in/out=$IN_LEN/$OUT_LEN prompts=$NPROMPTS rates='$RATES'"
log "out: $OUT"

# Prefetch once: a 60 GB pull inside arm 1 would inflate its load time and repeat.
log "prefetching $MODEL"
stdbuf -oL -eL "$BIN/hf" download "$MODEL" > "$OUT/prefetch.log" 2>&1
log "prefetch rc=$?"

wait_gpu_free 120
for arm in $ARMS; do run_arm "$arm"; done

log "=== SUMMARY ==="
"$BIN/python" "$(dirname "$0")/_kv_sweep_summary.py" --out "$OUT" --arms "$ARMS" \
    --rates "$RATES" 2>&1 | tee "$OUT/summary.txt"
log "KV_SWEEP_ALL_DONE"
