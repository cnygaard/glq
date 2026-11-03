#!/bin/bash
# Kolibri-1 GLQ 3 bpw, B=1, the glq-code --dspark plan for this box (RTX PRO 4500, 30 GiB RAM):
# 18 GiB of experts on host (UVA), a 4 GiB arena, util 0.902, fp16, 262,144 window.
# Arm A without a drafter, arm B with intuit/Kolibri-1-dspark at 4 draft tokens. Same flags.
N=/opt/dlami/nvme
export PATH=$HOME/.glq/venv/bin:$PATH HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfread
export PYTHONUNBUFFERED=1 GLQ_MOE_EXPERT_CACHE=4 GLQ_MOE_ARENA_MAX_TOKENS=5
COMMON=(--model xv0y5ncu/Kolibri-1-GLQ-trellis-3inst-3bpw --batches 1 --repeats 3
        --dtype float16 --gpu-mem 0.902 --max-model-len 262144
        --offload-backend uva --cpu-offload-gb 18 --offload-params experts)
cd $N
python $N/glq_kolibri_dspark/run_model.py "${COMMON[@]}" --expect-gib 14.7 \
    > $N/kolibri_dspark_armA_nodraft.log 2>&1
echo "ARM_A_EXIT=$?"
python $N/glq_kolibri_dspark/run_model.py "${COMMON[@]}" --expect-gib 15.47 \
    --speculative-config '{"method": "dspark", "model": "intuit/Kolibri-1-dspark", "num_speculative_tokens": 4}' \
    > $N/kolibri_dspark_armB_dspark4.log 2>&1
echo "ARM_B_EXIT=$?"
