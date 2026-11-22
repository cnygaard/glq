#!/usr/bin/env bash
# Step 0 prep for the pi adapter on Arete: fetch the adapter at its pinned revision and the
# Arete 4 bpw checkpoint into the NVMe HF cache. Network only, no GPU.
set -u
N=/opt/dlami/nvme
export PATH=$HOME/.glq/venv/bin:$PATH HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfread
echo "=== download start $(date -u +%FT%TZ)"
python - <<'EOF'
from huggingface_hub import snapshot_download
print("ADAPTER", snapshot_download("harness-hub/Qwen3.8-27B-pi-adapted",
                                  revision="7d42b119e2d4d475fa11270634b1ef98d1cd5aee"))
print("ARETE", snapshot_download("xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw"))
EOF
echo "EXIT=$? $(date -u +%FT%TZ)"
