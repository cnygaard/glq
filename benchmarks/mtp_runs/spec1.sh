#!/bin/bash
# num_speculative_tokens=1 vs 3, at 40 GiB offload and unoffloaded.
#
# The question: does a SHORTER draft window survive offload better? Two forces pull opposite
# ways, and only measurement separates them.
#   - fewer token positions per verify step. If UVA charges per byte actually ACCESSED, a
#     2-position step routes to fewer distinct experts than a 4-position step, so the step is
#     cheaper and the penalty smaller.
#   - fewer tokens per step to show for it. num_spec=1 can accept at most 1 draft token, so
#     ~1.4-1.7 tokens/step against 2.204 for num_spec=3.
#
# The unoffloaded arm is the reference: without it, a change at 40 GiB cannot be attributed to
# the offload rather than to the smaller window.
#
# Already measured at 40 GiB: no spec 20.109, num_spec=3 21.554 (1.07x).
# Unoffloaded: no spec 32.330, num_spec=3 54.672 (1.69x).
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
export PATH=/home/ubuntu/.glq/venv/bin:$PATH
PY=/home/ubuntu/.glq/venv/bin/python
HARNESS=/opt/dlami/nvme/glq_src/benchmarks/run_model.py
NEW=/opt/dlami/nvme/fn_mtp3
SPEC1='{"method":"mtp","num_speculative_tokens":1}'

run () {
    local label="$1"; shift
    echo ""; echo "############ $label ############"
    "$PY" "$HARNESS" --model "$NEW" --runtime vllm \
        --batches 1 --decode 256 --repeats 3 \
        --max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16 \
        --label "$label" "$@"
    echo "EXIT_${label}=$?"
    sleep 10
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

run off40_spec1 --cpu-offload-gb 40 --offload-params experts --offload-backend uva \
    --speculative-config "$SPEC1"
run noff_spec1 --speculative-config "$SPEC1"

echo ""; echo "SPEC1_DONE"
