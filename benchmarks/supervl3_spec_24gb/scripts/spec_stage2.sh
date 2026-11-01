#!/bin/bash
# Stage 2: depth screening on the real workload (SPEED-Bench coding, 5 prompts, osl 1024),
# DSpark only (MTP failed in Stage 1 on a GLQ bug). Window 16,384, 24 GB budget. Arena sized
# so each verify step (1 + k tokens x top_k 22) fits: k=1 4 GiB, k=2 5.5, k=4 9; baseline 8.
S=/opt/dlami/nvme/bench/spec_serve_aiperf.sh
DS=RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark
d() { echo '{"method": "dspark", "model": "'$DS'", "num_speculative_tokens": '$1"$2"'}'; }
bash $S S2_base 42 8   1 16384 5 1024
bash $S S2_ds1  42 4   2 16384 5 1024 "$(d 1)"
bash $S S2_ds2  42 5.5 3 16384 5 1024 "$(d 2)"
bash $S S2_ds4  42 9   5 16384 5 1024 "$(d 4)"
bash $S S2_ds4av 42 9  5 16384 5 1024 "$(d 4 ', "enable_adaptive_verification": true')"
echo "END_stage2 $(date -Is)"
