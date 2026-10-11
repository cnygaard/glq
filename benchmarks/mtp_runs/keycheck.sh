#!/bin/bash
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
exec stdbuf -oL -eL /home/ubuntu/.glq/venv/bin/python -m glq.quantize_model \
  --model Qwen/Qwen3.8-Flash-Next --output /opt/dlami/nvme/fn_dryrun \
  --bpw 3 --nsamples 128 --streaming --resume --mtp-bpw 3
