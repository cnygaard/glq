#!/bin/bash
# KV-quantization ACCURACY sweep: long-context retrieval (MRCR) + reasoning (AIME).
#
# The companion to _kv_sweep_e8_vs_fp8.sh, which measures capacity and speed. This
# one measures whether the context a bigger cache buys is actually usable: MRCR
# puts the answer tens or hundreds of thousands of tokens back, so KV quantization
# error shows up as retrieval failure at exactly the lengths that matter. It is
# where vLLM's TurboQuant blog found its 3-bit variants failing while short-context
# reasoning still looked fine — so AIME alone would not have caught it.
#
# Arms run one at a time (single GPU), each unbuffered to its own file. Unlike the
# serving sweep this drives glq-bench, which builds an in-process vLLM engine, so
# the KV configuration is applied as ENV around the whole run.
#
#   REPO=/opt/dlami/nvme/glq_src OUT=/opt/dlami/nvme/kvacc \
#     ./benchmarks/_kv_accuracy_sweep.sh
#
# Smoke on short buckets first — proves dataset, scoring and wiring in minutes:
#   MODEL=HuggingFaceTB/SmolLM2-360M-Instruct MAXLEN=16384 PER_BUCKET=2 \
#     TASKS=mrcr ARMS="bf16 e8_relaxed2" ./benchmarks/_kv_accuracy_sweep.sh
set -u

REPO=${REPO:-/opt/dlami/nvme/glq_src}
VENV=${VENV:-/home/ubuntu/.glq/venv}
OUT=${OUT:-/opt/dlami/nvme/kvacc}
HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf_cache}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
ARMS=${ARMS:-"bf16 fp8 turboquant_4bit_nc e8_relaxed2"}
TASKS=${TASKS:-"mrcr"}

# Held fixed across arms. MAXLEN is the lever that makes this a KV test at all:
# at 131072 the pool is the binding constraint, which is the regime the capacity
# claim is about. bf16 may simply fail to start here — that is a result, not a bug.
MAXLEN=${MAXLEN:-131072}
GPU_UTIL=${GPU_UTIL:-0.85}
PER_BUCKET=${PER_BUCKET:-8}
NEEDLES=${NEEDLES:-8}
AVG_K=${AVG_K:-4}

export HF_HOME PYTHONPATH="$REPO"
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

BIN="$VENV/bin"
mkdir -p "$OUT"

# vLLM's KV dtype is an ENGINE argument with no env var behind it, so fp8 and
# turboquant go through glq-bench's --kv-cache-dtype. Selecting them via the
# environment would run every arm as bf16 and label it fp8. GLQ's E8 path is the
# opposite: it is env-driven by design and takes no engine flag.
declare -A ARM_KVDTYPE=(
    [bf16]=""
    [fp8]="fp8"
    [turboquant_4bit_nc]="turboquant_4bit_nc"
    [e8_relaxed2]=""
)
declare -A ARM_ENV=(
    [bf16]=""
    [fp8]=""
    [turboquant_4bit_nc]=""
    [e8_relaxed2]="GLQ_KV_QUANT=e8_relaxed:2 GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 GLQ_KV_E8_FUSED_GATHER=1 GLQ_KV_E8_FUSED_WRITE=1"
)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

wait_gpu_free() {
    local deadline=$((SECONDS + ${1:-300})) used
    while [ $SECONDS -lt $deadline ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "$used" ] && [ "$used" -lt 1000 ] && { log "  GPU free (${used} MiB)"; return 0; }
        sleep 5
    done
    log "  WARNING: GPU still holding ${used:-?} MiB"
    return 1
}

run_arm() {
    local arm=$1 envs rc
    envs=${ARM_ENV[$arm]+${ARM_ENV[$arm]}}
    if [ -z "${ARM_ENV[$arm]+x}" ]; then log "ARM $arm UNKNOWN — skipped"; return 1; fi
    if [ -f "$OUT/$arm.done" ]; then log "ARM $arm already done — skipped"; return 0; fi

    log "=== ARM $arm ==="
    log "  env: ${envs:-none}"
    # Per-task config: MRCR wants the bucket size and needle count; AIME wants
    # avg@k, because a single pass at n=30 has variance that swamps any KV effect.
    local cfg="{\"mrcr\":{\"per_bucket\":$PER_BUCKET,\"needles\":$NEEDLES,\"max_ctx\":$MAXLEN},\"aime_2026\":{\"avg_k\":$AVG_K}}"

    local kvd=${ARM_KVDTYPE[$arm]:-} kvflag=()
    [ -n "$kvd" ] && kvflag=(--kv-cache-dtype "$kvd")
    log "  kv-cache-dtype: ${kvd:-engine default}"

    # shellcheck disable=SC2086
    env $envs stdbuf -oL -eL "$BIN/glq-bench" run \
        --model "$MODEL" --tasks "$TASKS" \
        --max-model-len "$MAXLEN" --gpu-mem-util "$GPU_UTIL" \
        "${kvflag[@]}" \
        --task-config "$cfg" \
        --out "$OUT/$arm.jsonl" \
        > "$OUT/$arm.log" 2>&1
    rc=$?
    log "  rc=$rc"
    if [ $rc -ne 0 ]; then
        # An arm that cannot serve MAXLEN at all is the capacity claim showing up as
        # a hard failure — record it and keep the other arms' numbers.
        echo "$arm rc=$rc" >> "$OUT/failures.txt"
        grep -iE "max seq len|KV cache|out of memory|ValueError|RuntimeError" "$OUT/$arm.log" \
            | tail -3 | sed 's/^/    /'
    fi
    wait_gpu_free 300
    touch "$OUT/$arm.done"
    log "=== ARM $arm DONE ==="
}

log "accuracy sweep: model=$MODEL tasks=$TASKS maxlen=$MAXLEN util=$GPU_UTIL"
log "  mrcr: needles=$NEEDLES per_bucket=$PER_BUCKET   aime: avg_k=$AVG_K"
log "out: $OUT"

wait_gpu_free 120
for arm in $ARMS; do run_arm "$arm"; done

log "=== SUMMARY ==="
"$BIN/python" - "$OUT" "$ARMS" <<'PY' 2>&1 | tee "$OUT/summary.txt"
import json, os, sys
out, arms = sys.argv[1], sys.argv[2].split()
print(f"\n{'arm':22s} {'task':16s} {'metric':8s} {'value':>8s}  by bucket / detail")
for arm in arms:
    path = os.path.join(out, f"{arm}.jsonl")
    if not os.path.exists(path):
        print(f"{arm:22s} {'— did not run':16s}")
        continue
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        r = rec.get("result", rec)
        extra = r.get("extra", {}) or {}
        detail = extra.get("by_bucket") or {
            k: extra[k] for k in ("avg_k", "solved_any", "prefix_misses") if k in extra}
        print(f"{arm:22s} {r.get('task',''):16s} {r.get('metric',''):8s} "
              f"{r.get('value', float('nan')):8.4f}  {detail}")
fails = os.path.join(out, "failures.txt")
if os.path.exists(fails):
    print("\nArms that failed (a hard failure at this context length IS the capacity result):")
    print(open(fails).read().rstrip())
PY
log "KV_ACCURACY_SWEEP_ALL_DONE"
