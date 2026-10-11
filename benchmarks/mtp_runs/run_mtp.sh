#!/bin/bash
# Item 5, attempt 2: replay the published Flash-Next decoder from the rebuilt store and
# quantize the MTP draft head at 3 bpw.
#
# GLQ_PLE_BPW=4 is REQUIRED, not a tuning choice. The published checkpoint is ple_bpw 4
# (its repo is named -ple4) while the descriptor defaults to 3, and the resumed PLE
# artifacts are 4 bpw. ple_reuse_decision() refuses the mismatch rather than shipping
# 4 bpw weights declared as 3, which vLLM would size its buffers from.
cd /opt/dlami/nvme
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTHONPATH=/opt/dlami/nvme/glq_src
export PYTHONUNBUFFERED=1
export GLQ_PLE_BPW=4

# Wait for the GDN kernel compile so the calibration forward gets the fast path rather
# than the reference implementation. Capped: the run is correct either way, just slower.
for i in $(seq 1 60); do
  pgrep -f "[i]nstall_gdn.sh" >/dev/null || break
  echo "waiting for the GDN kernel install ($i/60) ..."
  sleep 30
done
/home/ubuntu/.glq/venv/bin/python -c "
import importlib
for m in (\"fla\", \"causal_conv1d\"):
    try:
        importlib.import_module(m); print(f\"  {m}: available\")
    except Exception as e:
        print(f\"  {m}: NOT available ({type(e).__name__}) - reference path will be used\")
"
exec stdbuf -oL -eL /home/ubuntu/.glq/venv/bin/python -m glq.quantize_model \
  --model Qwen/Qwen3.8-Flash-Next \
  --output /opt/dlami/nvme/fn_mtp3 \
  --bpw 3 --nsamples 128 --seqlen 2048 \
  --streaming --resume \
  --resume-bucket glq-resume-eu-north-1-2859f54e \
  --resume-prefix flashnext-mtp3/95efd67688d022da70c7675a836db31d \
  --mtp-bpw 3
