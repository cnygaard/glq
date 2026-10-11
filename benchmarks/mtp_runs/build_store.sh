#!/bin/bash
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
C=$(ls -d /opt/dlami/nvme/hf_cache/hub/models--xv0y5ncu--Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4/snapshots/*/ | head -1)
echo "source checkpoint: $C"
exec stdbuf -oL -eL /home/ubuntu/.glq/venv/bin/python -m glq.resume \
  --checkpoint "$C" --store-dir /opt/dlami/nvme/fn_mtp3/_resume --mtp-bpw 3
