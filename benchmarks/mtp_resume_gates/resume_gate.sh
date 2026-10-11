#!/bin/bash
# Item 4: prove resume + --streaming reproduces an uninterrupted run byte-for-byte.
set -u
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
PY=/home/ubuntu/.glq/venv/bin/python
OUT=/opt/dlami/nvme/smol_resume
rm -rf "$OUT"
ARGS="--model HuggingFaceTB/SmolLM3-3B --output $OUT --bpw 3 --nsamples 8 --streaming --resume"

echo "=== PHASE 1: run, then interrupt it part-way (stands in for a spot reclaim) ==="
timeout --signal=KILL 200 stdbuf -oL -eL $PY -m glq.quantize_model $ARGS
echo "PHASE1_EXIT=$?  (137 = stopped on purpose)"
echo "--- shards checkpointed ---"
ls "$OUT/_resume"/layer_*.safetensors 2>/dev/null | wc -l

echo
echo "=== PHASE 2: restart the identical command; it must replay, not redo ==="
stdbuf -oL -eL $PY -m glq.quantize_model $ARGS
echo "PHASE2_EXIT=$?"
echo "--- layers it replayed rather than quantized ---"
grep -c "replayed from checkpoint" /opt/dlami/nvme/resume_gate.log 2>/dev/null || true

echo
echo "=== GATE: resumed checkpoint vs the uninterrupted reference ==="
$PY /opt/dlami/nvme/glq_src/benchmarks/_ckpt_bytecmp.py /opt/dlami/nvme/smol_ref "$OUT"
echo "BYTECMP_EXIT=$?"
