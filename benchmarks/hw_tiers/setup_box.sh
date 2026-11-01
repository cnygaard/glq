#!/bin/bash
# Fresh-box setup for the hardware-tier sweep (2x RTX PRO 6000, every tier simulated by a
# utilization cap). Needs, copied in beforehand: $N/bench/install.sh and the read-only token in
# $N/.hfread (mode 600). Everything lands on the instance NVMe, which a stop wipes.
N=/opt/dlami/nvme
set -x
mkdir -p $N/hf_cache $N/tiers
# 1. glq + vLLM + chat UI + pi + bench deps through the installer, at the release the branch
#    overlays (0.8.33). The overlay at $N/glq_dspark is rsynced separately.
bash $N/bench/install.sh --yes --no-start --components core,vllm,chat,picode,bench --glq-version 0.8.33
echo "EXIT_install=$?"
# 2. The installer pins transformers 5.17.0; Super-VL's native config class needs >= 5.18.
~/.glq/venv/bin/pip install -q --no-deps --target $N/tf519 'transformers==5.19.0'
echo "EXIT_tf519=$?"
PYTHONPATH=$N/tf519 ~/.glq/venv/bin/python -c "
import transformers, vllm, glq, torch
print('VERSIONS tf', transformers.__version__, 'vllm', vllm.__version__, 'glq', glq.__version__, 'torch', torch.__version__)
import glq.inference_kernel as ik
print('EXT_OK', ik._try_load_cuda_ext())
"
# 3. What the old box lacked: precompiled FlashInfer kernels (their absence cost ~75 s of JIT
#    per start there).
~/.glq/venv/bin/pip list 2>/dev/null | grep -i flashinfer
# 4. The RAM tiers are emulated with a cgroup memory cap.
systemd-run --user --scope -q -p MemoryMax=1G true && echo "SCOPE_OK user" \
  || { sudo -n systemd-run --scope -q -p MemoryMax=1G true && echo "SCOPE_OK sudo"; }
# 5. Every checkpoint in the matrix, and the DSpark drafter, with the read-only token. Passed
#    as a path: under `set -x`, `HF_TOKEN=$(cat ...)` prints the token into the log.
HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfread HF_HUB_ENABLE_HF_TRANSFER=0 ~/.glq/venv/bin/python -c "
from huggingface_hub import snapshot_download
for r in ['xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel',
          'xv0y5ncu/gemma-4-E4B-it-GLQ-trellis-3inst-4bpw',
          'xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw',
          'xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw',
          'xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8',
          'xv0y5ncu/gemma-4-31B-it-GLQ-trellis-3inst-4bpw',
          'xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-3bpw',
          'xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw',
          'RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark',
          'xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw',
          'xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw']:
    print('DL', r, snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model']), flush=True)
"
echo "EXIT_dl=$?"
du -sh $N/hf_cache
echo "END $(date -Is)"
