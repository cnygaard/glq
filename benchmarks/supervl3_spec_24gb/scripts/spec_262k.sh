#!/bin/bash
# The launcher target: a 262,144-token window with DSpark on the 24 GB budget (21.96 GiB).
# T3 = k=3 (arena 7 GiB, 90 slots >= 4 x 22), then T2 = k=2 (5.15 GiB, 66 slots >= 3 x 22).
# Each arm: AIPerf coding, 10 requests, osl 4096. T3 also runs a ~250k needle and the
# multi-turn prefix-cache probe against the same server before it stops.
N=/opt/dlami/nvme
S=$N/bench/spec_serve_aiperf.sh
DS=RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark
d() { echo '{"method": "dspark", "model": "'$DS'", "num_speculative_tokens": '$1'}'; }
PY="PYTHONPATH=$N/tf519 HF_HUB_OFFLINE=1 HF_HOME=$N/hf_cache $HOME/.glq/venv/bin/python"

POST="$PY $N/bench/prefix_turns.py 20000; $PY $N/bench/needle_spec.py 250000" \
  bash $S T3_ds3_262k 42 7 4 262144 10 4096 "$(d 3)"
POST="" bash $S T2_ds2_262k 42 5.15 3 262144 10 4096 "$(d 2)"
echo "SPEC262K_END $(date -Is)"
