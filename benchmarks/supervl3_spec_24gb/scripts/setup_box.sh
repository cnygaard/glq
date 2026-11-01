#!/bin/bash
# Fresh-box setup for the Super-VL 3 bpw speculative-decoding AIPerf runs. Needs, copied in
# beforehand: $N/bench/{install.sh,make_speed_bench_coding.py} and the read token in
# $N/.hfreadtok (mode 600). Everything lands on the instance NVMe, which a stop wipes.
N=/opt/dlami/nvme
M=xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw
set -x
mkdir -p $N/hf_cache $N/aiperf
# 1. glq + vLLM through the installer, at the version the earlier runs used.
bash $N/bench/install.sh --yes --components core,vllm --glq-version 0.8.33
echo "EXIT_install=$?"
# 2. The installer pins transformers 5.17.0; Super-VL's native config class needs >= 5.18.
#    An overlay keeps ~/.glq/venv exactly as the installer left it.
~/.glq/venv/bin/pip install -q --no-deps --target $N/tf519 'transformers==5.19.0'
echo "EXIT_tf519=$?"
PYTHONPATH=$N/tf519 ~/.glq/venv/bin/python -c "
import transformers, vllm, glq, torch
print('VERSIONS tf', transformers.__version__, 'vllm', vllm.__version__, 'glq', glq.__version__, 'torch', torch.__version__)
import glq.inference_kernel as ik
print('EXT_OK', ik._try_load_cuda_ext())
"
# 3. AIPerf 0.11.0 in its own venv (never ~/.glq/venv).
python3.12 -m venv $N/aiperf-venv
$N/aiperf-venv/bin/pip install -q --upgrade pip
$N/aiperf-venv/bin/pip install -q "aiperf==0.11.0"
echo "EXIT_aiperf=$?"
# 4. Model, DSpark drafter and SPEED-Bench into the shared cache, with the read-only token.
#    Passed as a path: under `set -x`, `HF_TOKEN=$(cat ...)` prints the token into the log.
HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfreadtok $N/aiperf-venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$M'))
print(snapshot_download('RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark'))
print(snapshot_download('nvidia/SPEED-Bench', repo_type='dataset', allow_patterns=['qualitative/*', 'README.md']))
"
echo "EXIT_dl=$?"
# 5. The 70 SPEED-Bench coding prompts as aiperf JSONL.
P=$(ls $N/hf_cache/hub/datasets--nvidia--SPEED-Bench/snapshots/*/qualitative/test-00000-of-00001.parquet)
$N/aiperf-venv/bin/python -I $N/bench/make_speed_bench_coding.py "$P" $N/bench/speed_bench_coding.jsonl
echo "END $(date -Is)"
