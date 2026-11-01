#!/bin/bash
# Stage 0 (no GPU): AIPerf 0.11.0 in its own venv (never ~/.glq/venv), its CLI help, and the
# RedHatAI DSpark drafter for Super-VL in the shared HF cache.
N=/opt/dlami/nvme
set -x
python3.12 -m venv $N/aiperf-venv
$N/aiperf-venv/bin/pip install -q --upgrade pip
$N/aiperf-venv/bin/pip install -q "aiperf==0.11.0"
echo "EXIT_pip=$?"
$N/aiperf-venv/bin/aiperf --version
$N/aiperf-venv/bin/aiperf profile --help > $N/bench/aiperf_profile_help.txt 2>&1
$N/aiperf-venv/bin/aiperf speed-bench-report --help > $N/bench/aiperf_report_help.txt 2>&1
$N/aiperf-venv/bin/pip list 2>/dev/null | command grep -iE "^(transformers|tokenizers|datasets|huggingface)" 
# The token as a path: under `set -x`, `HF_TOKEN=$(cat ...)` prints it into the log.
HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfreadtok $N/aiperf-venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark'))
print(snapshot_download('nvidia/SPEED-Bench', repo_type='dataset', allow_patterns=['qualitative/*', 'README.md']))
"
echo "EXIT_dl=$?"
echo "END $(date -Is)"
