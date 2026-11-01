#!/bin/bash
# The tier matrix as two queues, one per GPU, each run in order. Rows marked --force were
# refused by the planner's margins in plan-only runs; they run anyway, so each either serves
# (the margin was over-cautious) or fails with vLLM's own KV figure, which still gives the
# real overhead as budget - loaded - KV.
#   queues.sh 0    # GPU 0: small cards
#   queues.sh 1    # GPU 1: Super-VL
N=/opt/dlami/nvme
R=$N/tier_run.sh
SV3=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw
SV4=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw
L3=xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-3bpw
L4=xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw
SMOL=xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel
E4B=xv0y5ncu/gemma-4-E4B-it-GLQ-trellis-3inst-4bpw
G26_3=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw
G26_4=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw
G31M=xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8
G31T=xv0y5ncu/gemma-4-31B-it-GLQ-trellis-3inst-4bpw
if [ "$1" = 0 ]; then
  P=8000
  $R t8_smol3        0 8  16 $P $SMOL --force
  $R t8_e4b          0 8  16 $P $E4B --force
  $R t8_l3_r32       0 8  32 $P $L3 --force
  $R t8_l3_r16       0 8  16 $P $L3 --force
  $R t12_e4b         0 12 16 $P $E4B --force
  $R t12_g26b3       0 12 16 $P $G26_3 --force
  $R t12_l4_r32      0 12 32 $P $L4
  $R t16_g26b3       0 16 32 $P $G26_3 --force
  $R t16_g26b4       0 16 32 $P $G26_4 --force
  $R t16_l4          0 16 32 $P $L4
  $R t24_g26b4       0 24 64 $P $G26_4
  $R t24_l4          0 24 64 $P $L4
  $R t32_g31mix      0 32 64 $P $G31M
  $R t48_g31t4       0 48 64 $P $G31T
else
  P=8001
  $R t16_sv3         1 16 64 $P $SV3 --force
  $R t24_sv3_auto64  1 24 64 $P $SV3 --dspark
  $R t24_sv3_auto128 1 24 128 $P $SV3 --dspark
  $R t24_sv3_pin3    1 24 64 $P $SV3 --dspark 3 --max-model-len 262144
  $R t32_sv4_r64     1 32 64 $P $SV4
  $R t32_sv4_auto128 1 32 128 $P $SV4 --dspark
  $R t48_sv4_auto64  1 48 64 $P $SV4 --dspark
  $R t48_sv4_auto128 1 48 128 $P $SV4 --dspark
  $R t96_sv3_auto    1 95.6 128 $P $SV3 --dspark
  $R t96_sv4_auto    1 95.6 128 $P $SV4 --dspark
  $R t96_sv4_1m      1 95.6 128 $P $SV4 --dspark --max-model-len 1048576
fi
echo "QUEUE_END $1 $(date -Is)"
