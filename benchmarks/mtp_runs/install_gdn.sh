#!/bin/bash
# GDN fast-path kernels for the calibration forward. GPU box only: these bind at import
# time with no device check, so on a CPU-only box they break the model outright.
V=/home/ubuntu/.glq/venv/bin
export CUDA_HOME=/usr/local/cuda-13.2
export MAX_JOBS=8
echo "=== flash-linear-attention (Triton, no compile) ==="
$V/pip install -q flash-linear-attention 2>&1 | tail -5
echo "FLA_EXIT=$?"
echo "=== causal-conv1d (CUDA, compiles) ==="
$V/pip install -q causal-conv1d 2>&1 | tail -15
echo "CCONV_EXIT=$?"
echo "=== verify, and confirm torch was not disturbed ==="
$V/python -c "
import torch; print(\"torch\", torch.__version__)
try:
    import fla; print(\"fla OK\")
except Exception as e: print(\"fla FAILED:\", type(e).__name__, e)
try:
    import causal_conv1d; print(\"causal_conv1d OK\")
except Exception as e: print(\"causal_conv1d FAILED:\", type(e).__name__, e)
"
echo "INSTALL_GDN_DONE"
