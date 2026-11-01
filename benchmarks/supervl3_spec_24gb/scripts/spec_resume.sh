#!/bin/bash
# Resume on the replacement box. Waits for setup_box.sh (PID as $1), re-screens DSpark k=2/3/4
# on coding (5 requests, osl 1024), so the depths are compared on one box, keeps k=2 unless a
# deeper draft beats it by more than 3% (the per-request spread at n=5), then runs Stage 3.
N=/opt/dlami/nvme
S=$N/bench/spec_serve_aiperf.sh
DS=RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark
d() { echo '{"method": "dspark", "model": "'$DS'", "num_speculative_tokens": '$1'}'; }
# Arena per depth: (1 + k) x 22 experts per layer must fit, 1.976 MiB/slot over 40 layers.
declare -A ARENA=([2]=5.5 [3]=7 [4]=9)

while kill -0 "$1" 2>/dev/null; do sleep 30; done
if ! grep -q '^EXIT_dl=0' $N/setup_box.log || [ ! -s $N/bench/speed_bench_coding.jsonl ]; then
  echo "setup incomplete, nothing run $(date -Is)"; exit 1
fi
echo "setup done $(date -Is)"

for k in 2 3 4; do
  bash $S S2b_ds$k 42 ${ARENA[$k]} $((k + 1)) 16384 5 1024 "$(d $k)"
done

K=$($N/aiperf-venv/bin/python -I - <<'EOF'
import json, sys
rate = {}
for k in (2, 3, 4):
    try:
        j = json.load(open(f"/opt/dlami/nvme/aiperf/S2b_ds{k}/profile_export_aiperf.json"))
        rate[k] = j["output_token_throughput_per_user"]["avg"]
    except (OSError, KeyError, ValueError):
        pass
print("SCREEN per-request output tok/s:", rate, file=sys.stderr)
if not rate:
    sys.exit(1)
top = max(rate, key=rate.get)
print(top if 2 not in rate or rate[top] > rate[2] * 1.03 else 2)
EOF
) || { echo "screening produced no results $(date -Is)"; exit 1; }
echo "PICK K=$K ARENA=${ARENA[$K]} $(date -Is)"

K=$K ARENA=${ARENA[$K]} bash $N/bench/spec_stage3.sh
echo "RESUME_END $(date -Is)"
