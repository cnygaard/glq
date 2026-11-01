#!/bin/bash
# Speculative decoding at a 524,288-token window on the 24 GB budget (21.96 GiB).
# vLLM estimated 740,544 tokens as the most that fits with DSpark k=1 and a 3.47 GiB arena,
# so k=1 should start; k=2 needs a 3-token arena (66 slots/layer = 5.1 GiB) and is tight.
# Each arm: AIPerf coding, 10 requests, osl 4096. The first arm that starts also gets one
# ~500k-token needle prompt (needle_spec.py) with the draft running.
N=/opt/dlami/nvme
S=$N/bench/spec_serve_aiperf.sh
DS=RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark
d() { echo '{"method": "dspark", "model": "'$DS'", "num_speculative_tokens": '$1'}'; }
NEEDLE="PYTHONPATH=$N/tf519 HF_HUB_OFFLINE=1 HF_HOME=$N/hf_cache $HOME/.glq/venv/bin/python $N/bench/needle_spec.py 500000"

POST="$NEEDLE" bash $S L2_ds2_512k 42 5.15 3 524288 10 4096 "$(d 2)"
if [ -f $N/aiperf/L2_ds2_512k/profile_export_aiperf.json ]; then P=""; else P="$NEEDLE"; fi
POST="$P" bash $S L1_ds1_512k 42 3.5 2 524288 10 4096 "$(d 1)"
echo "SPEC512K_END $(date -Is)"
