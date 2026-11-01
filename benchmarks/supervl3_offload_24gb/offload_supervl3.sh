#!/bin/bash
# Super-VL 3 bpw expert offload on a simulated 24 GB card (RTX PRO 4500, gpu-mem 0.70).
# Arms run one after another (one GPU job at a time).
V=$HOME/.glq/venv
export PATH=$V/bin:$V/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=/opt/dlami/nvme/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH=/opt/dlami/nvme/tf519 PYTHONFAULTHANDLER=1
M=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw
COMMON="--model $M --batches 1 --repeats 3 --decode 256 --maxtok 256 --greedy --mm0 --dtype bfloat16 --gpu-mem 0.70 --max-model-len 8192 --offload-backend uva --offload-params experts --expect red"
run() {  # name, offload GiB, expect GiB, extra env...
  local name=$1 off=$2 exp=$3; shift 3
  echo "== $name  offload=$off  env=$*  $(date -Is)"
  env "$@" python /opt/dlami/nvme/bench/run_model.py $COMMON --cpu-offload-gb $off --expect-gib $exp \
      --label supervl3-$name > /opt/dlami/nvme/offload_$name.log 2>&1 < /dev/null
  echo "EXIT_$name=$?"
}
python -c "import glq.inference_kernel as ik; print(\"cuda ext\", ik._try_load_cuda_ext())" 2>&1 | tail -1
#run A_off   42    4.58
run B_cache 42    12.53 GLQ_MOE_EXPERT_CACHE=8 GLQ_MOE_ARENA_MAX_TOKENS=1
#run C_isovram 31.7 12.5
echo "END $(date -Is)"
