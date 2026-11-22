#!/bin/bash
# glq-code with the pi adapter and RedHatAI's DSpark drafter, the window left to the planner.
# Before this branch the plan for 262,144 tokens missed the drafter's own KV and vLLM refused it
# at startup (21.39 GiB needed); with it, the plan should hold that window and serve.
#   kv_ds3       --pi-adapter --draft-model D --dspark 3   (pinned depth)
#   kv_auto      --pi-adapter --draft-model D              (auto depth, up to 4)
#   kv_base_ds3  --draft-model D --dspark 3, no adapter    (the control)
# glq comes from $N/glq_overlay_kv: the installed 0.8.38 package with this branch's changed
# files ($N/glq_kv_files) on top. One GPU job at a time; each row ends before the next starts.
set -u
N=/opt/dlami/nvme
OUT=$N/arete_pi_step5
M=xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw
D=RedHatAI/Qwen3.8-27B-speculator.dspark
PY=/home/ubuntu/.glq/venv/bin/python
SITE=/home/ubuntu/.glq/venv/lib/python3.14/site-packages
echo "=== dspark_kv start $(date -Is)"
rm -rf $N/glq_overlay_kv && mkdir -p $N/glq_overlay_kv && cp -a $SITE/glq $N/glq_overlay_kv/
cp $N/glq_kv_files/*.py $N/glq_overlay_kv/glq/
cp $N/glq_kv_files/installer/*.py $N/glq_overlay_kv/glq/installer/
find $N/glq_overlay_kv -name __pycache__ -prune -exec rm -rf {} +
grep -q "def draft_kv_bytes_per_token" $N/glq_overlay_kv/glq/supervisor.py || { echo "OVERLAY_STALE"; exit 1; }
(cd $N/glq_overlay_kv/glq && sha256sum chat.py code.py installer/configure.py supervisor.py tooling.py)

row() {   # row TAG PROBE_MODEL flags...
    local tag=$1 probe=$2; shift 2
    OVERLAY=$N/glq_overlay_kv PROBE_MODEL=$probe bash $OUT/step5_run.sh $tag - - $M "$@"
    $PY -I $OUT/step5_check.py $OUT $tag
    grep -E "Available KV cache memory|GPU KV cache size|Model loading took" $OUT/$tag.vllm.log | cut -c1-300
    grep -E "SpecDecoding metrics" $OUT/$tag.vllm.log | tail -3 | cut -c1-400
}
row kv_ds3 qwen3.8-pi --pi-adapter --draft-model $D --dspark 3
row kv_auto qwen3.8-pi --pi-adapter --draft-model $D
row kv_base_ds3 $M --draft-model $D --dspark 3
echo "=== dspark_kv end $(date -Is)"
