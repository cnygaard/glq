#!/usr/bin/env bash
# Phase 5.2: sweep mmlu_pro 1% on Gemma-4-E4B-it across the E8 KV
# variants. Each invocation reloads vLLM (the engine reads
# GLQ_KV_QUANT once at plugin load) so the cost is ~80s × 4 = 5min
# overhead. Per-variant inference time at N=120 with vLLM batched
# decode is ~1-3min, so the whole sweep finishes in ~15-25 min.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${N:-120}"
MODEL="${MODEL:-unsloth/gemma-4-E4B-it}"
OUT_DIR="${OUT_DIR:-/tmp/logs}"
mkdir -p "$OUT_DIR"

run_one() {
    local spec="$1"
    local label tag
    if [[ -z "$spec" ]]; then
        label="baseline-fp16"
        tag="baseline"
        unset GLQ_KV_QUANT
    else
        label="$spec"
        tag="${spec//:/_}"
        export GLQ_KV_QUANT="$spec"
    fi
    local out="${OUT_DIR}/mmlu_vllm_${tag}.json"
    echo "=== $label  →  $out ==="
    python "$SCRIPT_DIR/kv_compress_mmlu_vllm.py" \
        --model "$MODEL" --n "$N" --out "$out" --label "$label" 2>&1 \
        | tee "${OUT_DIR}/mmlu_vllm_${tag}.log" \
        | grep -E 'accuracy:|parse rate:|wall time:|output tok:|load:'
    echo
}

run_one ""
run_one "e8_relaxed:1"
run_one "e8_relaxed:2"
run_one "e8_relaxed:3"

echo "=== summary ==="
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("/tmp/logs/mmlu_vllm_*.json")):
    if "summary" in p: continue
    d = json.load(open(p))
    rows.append((d["label"], d["accuracy"], d["parse_rate"],
                 d["output_tok_per_sec"], d["elapsed_sec"]))
print(f"{'variant':<24} {'acc':>6} {'parse':>6} {'tok/s':>8} {'time':>8}")
for label, acc, parse, tps, dt in rows:
    print(f"{label:<24} {acc:>6.3f} {parse:>6.3f} {tps:>8.0f} {dt:>7.1f}s")
PY
