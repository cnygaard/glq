#!/bin/bash
# The gemma-4 multimodal rows again: in run 2 an explicit 2048-token batch stopped every one
# of them at vLLM's "Chunked MM input disabled" check, which a real card of the tier never
# hits (vLLM raises its own default to the 2496-token item). `tier_launch.MM_ITEM_FLOOR` now
# applies that rule. Plus E4B text-only at glq-chat's 8192 floor on an 8 GiB card.
#   queue_gemma.sh GPU PORT
N=/opt/dlami/nvme
R=$N/tier_run.sh
G=$1 P=$2
E4B=xv0y5ncu/gemma-4-E4B-it-GLQ-trellis-3inst-4bpw
G26_3=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw
G26_4=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw
G31M=xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8
G31T=xv0y5ncu/gemma-4-31B-it-GLQ-trellis-3inst-4bpw
$R t24_g26b4       $G 24 64 $P $G26_4
$R t16_g26b3       $G 16 32 $P $G26_3 --force
$R t16_g26b4       $G 16 32 $P $G26_4 --force
$R t12_g26b3       $G 12 16 $P $G26_3 --force
$R t32_g31mix      $G 32 64 $P $G31M
$R t48_g31t4       $G 48 64 $P $G31T
TIER_VLLM_ARGS="--language-model-only" $R x8_e4b_lmo_8k $G 8 16 $P $E4B --force --max-model-len 8192
echo "QUEUE_END gemma $(date -Is)"
