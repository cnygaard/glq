#!/bin/bash
# S0 gate — SmolLM3-3B, 2 bpw: QTIP-trellis (HYB) vs e8p, BOTH through the production
# path (glq.quantize_model --codebook ... → compressed safetensors → from_pretrained).
#
# This is the shippable-format gate: it proves the stored trellis checkpoint loads and
# decodes correctly at scale AND still beats e8p. Reference (scratch, bf16-dequant,
# independent-per-layer): bf16 9.12 / e8p 13.79 / trellis-hyb 11.94.
# One GPU job at a time; run under nohup with a tee'd log.
set -u
cd /opt/dlami/nvme/bench
export HF_HOME=/opt/dlami/nvme/hf_cache
export HF_TOKEN=$(cat /opt/dlami/nvme/.hftok)
export CUDA_HOME=/usr/local/cuda
PY=/home/ubuntu/venv/bin/python
Q=/opt/dlami/nvme/qtip
M=HuggingFaceTB/SmolLM3-3B

echo "=== [1/3] quantize trellis-hyb 2bpw (pure-torch Viterbi — the long pole) ==="
date
GLQ_TRELLIS_VARIANT=hyb $PY -u -m glq.quantize_model --model "$M" \
  --output "$Q/SmolLM3-3B-trellis-store" --bpw 2 --codebook trellis --nsamples 128 --seqlen 2048
echo "trellis quant exit=$?"
date

echo "=== [2/3] quantize e8p 2bpw (same driver — the A/B baseline) ==="
$PY -u -m glq.quantize_model --model "$M" \
  --output "$Q/SmolLM3-3B-e8p-store" --bpw 2 --codebook e8p --nsamples 128 --seqlen 2048
echo "e8p quant exit=$?"
date

echo "=== [3/3] footprint + PPL (bf16 / trellis / e8p) ==="
du -sh "$Q/SmolLM3-3B-trellis-store" "$Q/SmolLM3-3B-e8p-store"
for CK in "$M" "$Q/SmolLM3-3B-trellis-store" "$Q/SmolLM3-3B-e8p-store"; do
  $PY -u _ppl_checkpoint.py --model "$CK" 2>&1 | grep -E "^PPL"
done
echo "=== S0 GATE DONE ==="
date
