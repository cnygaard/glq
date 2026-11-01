#!/bin/bash
# gemma-4 served text-only: `--language-model-only` sets every modality limit to 0, and vLLM
# 0.32 then skips the towers gemma4_mm marks with `_mark_tower_model` (E4B: 0.58 GiB audio +
# 0.32 GiB vision). glq-code does not pass the flag for gemma-4 yet, so it is appended here.
# Each pair's twin without the flag is in queues.sh, so "Model loading took" differs by the
# towers alone. Run after queue 0, on the same GPU.
#   queue_textonly.sh GPU PORT
N=/opt/dlami/nvme
R=$N/tier_run.sh
G=$1 P=$2
E4B=xv0y5ncu/gemma-4-E4B-it-GLQ-trellis-3inst-4bpw
G26_4=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw
export TIER_VLLM_ARGS="--language-model-only"
$R x8_e4b_lmo      $G 8  16 $P $E4B --force
$R x24_g26b4_lmo   $G 24 64 $P $G26_4
echo "QUEUE_END textonly $(date -Is)"
