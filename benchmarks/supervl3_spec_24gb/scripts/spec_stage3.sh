#!/bin/bash
# Stage 3: final AIPerf runs, SPEED-Bench coding, 10 requests, osl 4096, concurrency 1.
# Same prompts (seed 42), sampling 1.0 / 0.95, 21.96 GiB cap, experts on host + GLQ arena.
# Args to spec_serve_aiperf.sh: name offload arena step window nreq osl [spec json]
# A0 = no draft, A2 = best DSpark depth from Stage 2 (K), A3 = DSpark k=1 at the full
# 1,048,576 window with a 3.5 GiB arena (45 slots/layer -> cap 2 = one k=1 verify step).
S=/opt/dlami/nvme/bench/spec_serve_aiperf.sh
DS=RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark
K=${K:?set K to the best DSpark depth from Stage 2}
ARENA=${ARENA:?set ARENA (GiB) for depth K}
d() { echo '{"method": "dspark", "model": "'$DS'", "num_speculative_tokens": '$1'}'; }
bash $S A0_base 42 8 1 16384 10 4096
bash $S A2_ds$K 42 $ARENA $((K + 1)) 16384 10 4096 "$(d $K)"
bash $S A3_ds1_1m 42 3.5 2 1048576 10 4096 "$(d 1)"
echo "STAGE3_END $(date -Is)"
