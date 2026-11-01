#!/bin/bash
# Stage 1: can each draft route load and draft on Super-VL 3 bpw under expert offload + arena,
# on the 24 GB budget (gpu-mem 0.70 = 21.96 GiB), window 16,384? run_model.py asserts drafted>0
# and graphs captured. No --expect-gib: whether the draft's experts get offloaded is one of
# the questions, so the footprint is read from the log rather than asserted.
V=$HOME/.glq/venv
export PATH=$V/bin:$V/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=/opt/dlami/nvme/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH=/opt/dlami/nvme/tf519 PYTHONFAULTHANDLER=1
M=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw
DS=RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark
N=/opt/dlami/nvme
COMMON="--model $M --batches 1 --repeats 3 --decode 256 --maxtok 256 --greedy --mm0 --dtype bfloat16 --gpu-mem 0.70 --max-model-len 16384 --offload-backend uva --offload-params experts --expect red"
run() {  # name, offload GiB, arena GiB, step tokens, spec json
  local name=$1 off=$2 arena=$3 st=$4 spec=$5
  echo "== $name offload=$off arena=$arena step=$st spec=$spec $(date -Is)"
  GLQ_MOE_EXPERT_CACHE=$arena GLQ_MOE_ARENA_MAX_TOKENS=$st GLQ_MOE_ARENA_MIN_TOKENS=$st \
    python $N/bench/run_model.py $COMMON --cpu-offload-gb $off --speculative-config "$spec" \
    --label supervl3-$name > $N/spec_$name.log 2>&1 < /dev/null
  echo "EXIT_$name=$?"
}
run F1_mtp1 48 4 2 '{"method": "mtp", "num_speculative_tokens": 1}'
run F2_dspark4 42 9 5 '{"method": "dspark", "model": "'$DS'", "num_speculative_tokens": 4}'
echo "END $(date -Is)"
