#!/bin/bash
# num_speculative_tokens=2 at 40 GiB offload — fills the curve between 1 and 3.
#
# Prediction, from the per-position step cost and per-position acceptance already measured:
#   3 positions/step -> step ~= 49.7 + 2*18 = 86 ms
#   accepted ~= 0.64 (pos0) + ~0.35 (pos1) -> ~1.99 tokens/step
#   -> ~23 tok/s, just below num_spec=1 (23.952) and above num_spec=3 (21.554)
# Stated before running so the result tests the cost model rather than merely adding a point.
#
# Measured so far at 40 GiB: no spec 20.109 | spec1 23.952 (1.19x) | spec3 21.554 (1.07x)
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
export PATH=/home/ubuntu/.glq/venv/bin:$PATH

echo ""; echo "############ off40_spec2 ############"
/home/ubuntu/.glq/venv/bin/python /opt/dlami/nvme/glq_src/benchmarks/run_model.py \
    --model /opt/dlami/nvme/fn_mtp3 --runtime vllm \
    --batches 1 --decode 256 --repeats 3 \
    --max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16 \
    --label off40_spec2 \
    --cpu-offload-gb 40 --offload-params experts --offload-backend uva \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
echo "EXIT_off40_spec2=$?"
echo ""; echo "SPEC2_DONE"
