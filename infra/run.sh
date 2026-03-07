#!/bin/bash
# Setup fresh GPU VM, install deps, sync code, run benchmark.
# Usage: ./run.sh <IP> [MODEL] [METHODS...]
# Example: ./run.sh 16.171.154.8 HuggingFaceTB/SmolLM-360M baseline glq quip_gptq
set -euo pipefail

IP="${1:?Usage: $0 <IP> [MODEL] [METHODS...]}"
MODEL="${2:-HuggingFaceTB/SmolLM-360M}"
shift 2 || true
METHODS="${*:-baseline glq quip_gptq}"

KEY="$(dirname "$0")/gpu-key.pem"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY ubuntu@$IP"
RSYNC="rsync -aW --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='*.egg-info' --exclude='node_modules' -e 'ssh -o StrictHostKeyChecking=no -i $KEY'"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Connecting to $IP ==="
for i in $(seq 1 30); do
    if $SSH "echo ok" 2>/dev/null; then
        echo "Connected."
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 10
done

echo "=== Setting up Python venv ==="
$SSH "python3 -m venv ~/venv 2>/dev/null || true"

echo "=== Installing PyTorch ==="
$SSH "source ~/venv/bin/activate && \
    pip install --quiet pip --upgrade && \
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu130"

echo "=== Installing llm-compressor + deps ==="
$SSH "source ~/venv/bin/activate && \
    pip install --quiet 'llmcompressor>=0.5' transformers datasets accelerate sentencepiece protobuf triton fast-hadamard-transform"

echo "=== Syncing code ==="
eval $RSYNC "$PROJ/glq/" "ubuntu@$IP:~/golay-leech-quant/glq/"
eval $RSYNC "$PROJ/compare_methods.py" "$PROJ/pyproject.toml" "ubuntu@$IP:~/golay-leech-quant/"
eval $RSYNC "$PROJ/llm-compressor/src/llmcompressor/modifiers/glq/" "ubuntu@$IP:~/golay-leech-quant/glq_modifier/"

echo "=== Installing GLQ package ==="
$SSH "source ~/venv/bin/activate && cd ~/golay-leech-quant && pip install --quiet -e ."

echo "=== Installing GLQ modifier into llm-compressor ==="
LLMC_MODS=\$($SSH "source ~/venv/bin/activate && python3 -c 'import llmcompressor.modifiers as m; import os; print(os.path.dirname(m.__file__))'")
$SSH "cp -r ~/golay-leech-quant/glq_modifier $LLMC_MODS/glq"

echo "=== Verifying imports ==="
$SSH "source ~/venv/bin/activate && python3 -c '
import torch; print(f\"torch {torch.__version__}, cuda={torch.cuda.is_available()}\")
from llmcompressor.modifiers.glq import GLQModifier; print(\"GLQModifier OK\")
from llmcompressor.modifiers.quantization import GPTQModifier; print(\"GPTQModifier OK\")
from llmcompressor.modifiers.transform import QuIPModifier; print(\"QuIPModifier OK\")
'"

echo "=== Running benchmark ==="
echo "  Model: $MODEL"
echo "  Methods: $METHODS"
$SSH "source ~/venv/bin/activate && cd ~/golay-leech-quant && \
    GLQ_MODEL=$MODEL python3 -u compare_methods.py $METHODS 2>&1" | tee "$(dirname "$0")/compare_output.log"

echo ""
echo "=== Done. Output saved to infra/compare_output.log ==="
