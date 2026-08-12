#!/bin/bash
# AIME-2026 avg@8 matrix for SmolLM3-3B: bf16 / GLQ-3inst-4bpw / NVFP4 / GLQ-3inst-2bpw.
#
# Runs ON the box. Arms are selected by argument so a reclaimed box can be resumed with
# only the arms that did not finish:  ./_aime2026_smollm3.sh nvfp4 glq2
#
# Sampling comes from SmolLM3's model card (temp 0.6, no top_k) and **no system message** —
# per the bench-quality skill, any system message deletes SmolLM3's reasoning preamble and
# the run silently scores like a no-think one. glq-bench's aime task enforces a mean_gen
# floor of 4000 that RAISES, so a broken-thinking arm is recorded as skipped, never as a
# quality number. A real thinking run sits at ~14-15k mean tokens.
#
# ARM ORDER IS BY INFORMATION VALUE, not by bpw — spot boxes keep dying mid-matrix, so the
# arm that most changes a conclusion must bank first. AIME at n=30 has a ~4.5 pt SE and
# cannot separate adjacent bpw rungs (true separation ~1-2 pt); wikitext2_ppl did that in
# 5 min/arm. What AIME *can* see is a large gap, so the arms worth its GPU time are the ones
# PPL puts far apart: nvfp4 (+9.5% PPL) and glq2 (+28%). glq3 (+5.1%) earns its slot only
# because it sits BELOW nvfp4 on PPL despite carrying one less bit — if that ordering holds
# on a reasoning task it is the strongest quality-per-bit evidence in the matrix.
set -uo pipefail
cd /opt/dlami/nvme/glq_src
export HF_HOME=/opt/dlami/nvme/hf_cache
export HF_TOKEN=$(cat /opt/dlami/nvme/.hftok)   # 2bpw repo is private; never on a cmdline
export PYTHONUNBUFFERED=1
OUT=/opt/dlami/nvme

declare -A MODELS=(
  [bf16]=HuggingFaceTB/SmolLM3-3B
  [glq4]=xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel
  [nvfp4]=Firworks/SmolLM3-3B-nvfp4
  [glq3]=xv0y5ncu/SmolLM3-3B-trellis-3inst-3bpw
  [glq2]=xv0y5ncu/SmolLM3-3B-trellis-3inst-2bpw
)

arm () {
  label=$1; model=${MODELS[$label]}
  echo "=== [$label] $(date -u +%H:%M:%S) $model ==="
  ( /home/ubuntu/venv/bin/glq-bench run --model "$model" --tasks aime_2026 \
      --avg-k 8 --budget 32768 \
      --task-config '{"system": null, "temperature": 0.6, "top_k": null}' \
      --out $OUT/aime_${label}.jsonl ) > $OUT/raw_aime_${label}.log 2>&1
  echo "    exit=$?  (raw: $OUT/raw_aime_${label}.log)"
  grep -aE "aime_2026 done|mean_gen|Traceback|Error|skipped" $OUT/raw_aime_${label}.log | tail -4
  # Echo the whole record to stdout so provenance survives the box. Three spot boxes have
  # now been reclaimed mid-matrix; the first took the only copy of its records with it.
  [ -f "$OUT/aime_${label}.jsonl" ] && echo "### AREC ${label}: $(cat $OUT/aime_${label}.jsonl)"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null; done
  sleep 5   # let vLLM's GPU memory actually free before the next arm loads
}

for a in "$@"; do
  [ -n "${MODELS[$a]:-}" ] || { echo "=== unknown arm '$a' — skipping ==="; continue; }
  arm "$a"
done
echo "AIME MATRIX DONE $(date -u +%H:%M:%S)"
