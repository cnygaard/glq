#!/bin/bash
# Does MTP speculation still pay when decode is PCIe-bound?
#
# Hypothesis: speculation verifies k+1 tokens in ONE forward pass, so it amortizes the
# per-step weight streaming across every accepted token — which predicts the speculation
# RATIO grows with offload. Counter-pressure: the draft head's own weights add offload
# volume, and `--offload-params experts` matches `mtp.layers.48.mlp.experts` too, so the
# draft streams as well.
#
# The no-offload row is already measured (32.330 / 54.672 tok/s, 48.35 / 49.41 GiB).
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
export PATH=/home/ubuntu/.glq/venv/bin:$PATH
PY=/home/ubuntu/.glq/venv/bin/python
HARNESS=/opt/dlami/nvme/glq_src/benchmarks/run_model.py
SPEC='{"method":"mtp","num_speculative_tokens":3}'
NEW=/opt/dlami/nvme/fn_mtp3

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

# The experts are 42.19 GiB of the 48.35 GiB resident, so 20 is about half of them and
# 40 is nearly all. uva named explicitly: cpu_offload_gb alone selects it implicitly, and
# a sweep that does not name the backend is not reproducible.
for OFF in 20 40; do
    run "off${OFF}_nospec" --cpu-offload-gb $OFF --offload-params experts --offload-backend uva
    run "off${OFF}_spec"   --cpu-offload-gb $OFF --offload-params experts --offload-backend uva \
        --speculative-config "$SPEC"
done

echo ""; echo "OFFLOAD_MATRIX_DONE"
