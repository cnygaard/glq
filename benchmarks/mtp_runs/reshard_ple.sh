#!/bin/bash
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
C=$(ls -d /opt/dlami/nvme/hf_cache/hub/models--xv0y5ncu--Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4/snapshots/*/ | head -1)
# Only layer 1 changes: it owns the PLE table, whose rht_blocks leaf the old whitelist
# dropped. Every other layer own only recognised leaves, so their shards are already right.
exec /home/ubuntu/.glq/venv/bin/python -m glq.resume --checkpoint "$C" \
  --store-dir /opt/dlami/nvme/fn_mtp3/_resume --mtp-bpw 3 --layers 1
