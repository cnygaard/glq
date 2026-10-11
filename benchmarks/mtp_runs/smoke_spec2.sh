#!/bin/bash
# Re-run only the SPECULATIVE arms: NEW_spec now that glq_vllm translates vLLM's renumbered
# MTP prefix, and PUB_spec now that stats are enabled so the acceptance rate is readable.
# The no-spec controls from the previous pass stand (48.35 GiB / 32.33 tok/s, both arms).
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
export PATH=/home/ubuntu/.glq/venv/bin:$PATH
PY=/home/ubuntu/.glq/venv/bin/python
HARNESS=/opt/dlami/nvme/glq_src/benchmarks/run_model.py
SPEC='{"method":"mtp","num_speculative_tokens":3}'
PUB=$(ls -d /opt/dlami/nvme/hf_cache/hub/models--xv0y5ncu--Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4/snapshots/*/ | head -1)
NEW=/opt/dlami/nvme/fn_mtp3

run () {
    local label="$1"; local model="$2"; shift 2
    echo ""; echo "############ $label ############"
    "$PY" "$HARNESS" --model "$model" --runtime vllm \
        --batches 1 --decode 256 --repeats 3 \
        --max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16 \
        --label "$label" "$@"
    echo "EXIT_${label}=$?"
    sleep 10
}

run NEW_spec "$NEW" --speculative-config "$SPEC"
run PUB_spec "$PUB" --speculative-config "$SPEC"
echo ""; echo "SPEC_RERUN_DONE"
