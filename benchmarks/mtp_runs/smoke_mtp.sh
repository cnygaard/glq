#!/bin/bash
# MTP speculative smoke + decode bench.
#
# Order: the NEW checkpoint (quantized 3 bpw draft head) first, then the published one
# (bf16 draft head), each with and without --speculative-config. The no-spec arms are the
# control: without them a tok/s difference cannot be attributed to speculation.
#
# One GPU job at a time — these run strictly in sequence.
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
# FlashInfer's sampling JIT shells out to `ninja` BY NAME, so a vllm launched without the
# venv's bin dir on PATH dies AFTER loading the model. That cost a full run on a previous box.
export PATH=/home/ubuntu/.glq/venv/bin:$PATH

PY=/home/ubuntu/.glq/venv/bin/python
HARNESS=/opt/dlami/nvme/glq_src/benchmarks/run_model.py
SPEC='{"method":"mtp","num_speculative_tokens":3}'
PUB=$(ls -d /opt/dlami/nvme/hf_cache/hub/models--xv0y5ncu--Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4/snapshots/*/ | head -1)
NEW=/opt/dlami/nvme/fn_mtp3

echo "published checkpoint: $PUB"
echo "new checkpoint:       $NEW"

run () {   # $1 = label, $2 = model dir, rest = extra args
    local label="$1"; local model="$2"; shift 2
    echo ""
    echo "############################################################"
    echo "### $label"
    echo "############################################################"
    "$PY" "$HARNESS" --model "$model" --runtime vllm \
        --batches 1 --decode 256 --repeats 3 \
        --max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16 \
        --label "$label" "$@"
    echo "EXIT_${label}=$?"
    # vLLM holds the card until the process is gone; give the allocator a moment so the
    # next arm does not start against a partially released pool.
    sleep 10
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

run NEW_spec   "$NEW" --speculative-config "$SPEC"
run NEW_nospec "$NEW"
run PUB_spec   "$PUB" --speculative-config "$SPEC"
run PUB_nospec "$PUB"

echo ""
echo "ALL_SMOKES_DONE"
# NOTE on --dtype bfloat16: it is REQUIRED here, not a preference. GLQ defaults to fp16 on
# CUDA (measured +12.8% free), but Qwen4Exp's 12 sparse-attention layers raise
# NotImplementedError("Qwen4Exp QSA currently requires BF16"). Without it every arm dies at
# engine start, after the weights have loaded.
