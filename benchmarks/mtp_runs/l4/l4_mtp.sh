#!/bin/bash
# MTP speculation on an L4 — the card the whole exercise was aimed at.
#
# The L4 has 22.5 GiB, so heavy offload is MANDATORY: the checkpoint is 48.35 GiB resident
# unoffloaded and 28.34 at 20 GiB offload, neither of which fits. 40 GiB offload leaves
# 9.09 GiB of weights (measured on a Blackwell), which does.
#
# What makes this more than a repeat: the L4 is PCIe 4 against the g7e's PCIe 5, and the
# per-GiB offload cost is NOT portable across PCIe generations (only ms/token/GiB is). The
# per-position step slope should therefore be STEEPER here, which predicts num_spec=1 wins
# by more than it did on the Blackwell — and possibly that speculation barely pays at all.
#
# Reference points: Blackwell at 40 GiB offload -> nospec 20.109, spec1 23.952, spec3 21.554.
# Earlier L4 baseline for the PUBLISHED checkpoint: 6.271 tok/s at 6.54 GiB weights.
set -u
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
export PATH=/home/ubuntu/.glq/venv/bin:$PATH
PY=/home/ubuntu/.glq/venv/bin/python
HARNESS=/opt/dlami/nvme/glq_src/benchmarks/run_model.py

echo "### waiting for install"
for i in $(seq 1 120); do pgrep -f "install.sh" >/dev/null || break; sleep 15; done

echo "### environment"
$PY -c "import glq,torch,vllm,transformers as t;print(' glq',glq.__version__,'torch',torch.__version__,'vllm',vllm.__version__,'tf',t.__version__)"
$PY -c "import glq.quantize_model as q;print(' branch code:', hasattr(q,'quantize_mtp_head'))"
$PY -c "import glq_vllm.config as c;print(' moe fix:', hasattr(c.GLQvLLMConfig,'_moe_is_quantized'), '| mtp renumber:', hasattr(c.GLQvLLMConfig,'_mtp_forms'))"
# A torch-ABI-mismatched flash_attn raises SIGBUS at model load from inside a
# suppress(ModuleNotFoundError), so it looks like a GLQ bug. Remove it before serving.
$PY -m pip uninstall -y flash_attn flash-attn 2>/dev/null | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "### download"
$PY /opt/dlami/nvme/dl_l4.py
CKPT=$(ls -d /opt/dlami/nvme/hf_cache/hub/models--xv0y5ncu--Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp/snapshots/*/ | head -1)
echo "checkpoint: $CKPT"

run () {
    local label="$1"; shift
    echo ""; echo "############ $label ############"
    "$PY" "$HARNESS" --model "$CKPT" --runtime vllm \
        --batches 1 --decode 256 --repeats 3 \
        --max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16 \
        --label "$label" "$@"
    echo "EXIT_${label}=$?"
    sleep 10
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

OFF="--cpu-offload-gb 40 --offload-params experts --offload-backend uva"
run l4_off40_nospec $OFF
run l4_off40_spec1  $OFF --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
run l4_off40_spec3  $OFF --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

echo ""; echo "L4_MTP_DONE"
