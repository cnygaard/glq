#!/bin/bash
# One Step 5 row: glq-code --pi-adapter on GPU 0, either on the real card (VRAM_GIB=-) or planned
# for a smaller card and host (PR #202's emulation), with pi reading hello.py.
#   step5_run.sh TAG VRAM_GIB|- RAM_GIB|- MODEL [flags...]
# Adapted from benchmarks/arete_24gb/tier24_run.sh. Differences:
#   * glq comes from $N/glq_overlay (the installed 0.8.38 package with the branch's files on top),
#     through PYTHONPATH; vLLM's engine processes inherit it.
#   * stdout and stderr go to separate files, so the "63" check reads pi's stdout and cannot be
#     satisfied by a supervisor line such as "ready in 63 s" (stderr).
#   * the probe asks the adapter's served name (TIER_PROBE_MODEL).
#   * RAM_GIB=- runs without the cgroup cap.
# Each row gets its own HOME, so ~/.glq/vllm.log, pi's models.json and pi's sessions are per row
# and the box's own ~/.pi is not touched.
set -u
N=/opt/dlami/nvme
OUT=$N/arete_pi_step5
TAG=$1 VRAM=$2 RAM=$3 MODEL=$4; shift 4
H=$OUT/home_$TAG
rm -rf "$H"; mkdir -p "$H/.pi/agent" "$H/.glq" "$H/work" "$H/.cache" "$N/tiers/shared_flashinfer"
ln -s /home/ubuntu/.nvm "$H/.nvm"
ln -s "$N/tiers/shared_flashinfer" "$H/.cache/flashinfer"
# CACHE_FROM=<tag>: start with a copy of that row's vLLM compile cache and Triton cache, so the
# row measures a second launch (warm) rather than a first one on a fresh HOME (cold).
if [ -n "${CACHE_FROM:-}" ]; then
    cp -a "$OUT/home_$CACHE_FROM/.cache/vllm" "$H/.cache/" && cp -a "$OUT/home_$CACHE_FROM/.triton" "$H/"
fi
printf "RETRY_LIMIT = 7\n\n\ndef main():\n    print(sum(range(RETRY_LIMIT)) * 3)\n\n\nif __name__ == \"__main__\":\n    main()\n" > "$H/work/hello.py"
LOG=$OUT/$TAG.log
echo "RUN_START $TAG vram=$VRAM ram=$RAM model=$MODEL flags: $* plan_only=${TIER_PLAN_ONLY:-0} $(date -Is)" | tee "$LOG"
TAIL=(-- -p "Read hello.py in the current directory with your read tool, then tell me exactly what it prints when run.")
if [ "$RAM" = - ]; then CAP=(); else CAP=(sudo -n systemd-run --scope -q -p MemoryMax=${RAM}G -p MemorySwapMax=0); fi
SECONDS=0
cd "$H/work"
"${CAP[@]}" sudo -n -u ubuntu env HOME="$H" CUDA_VISIBLE_DEVICES=0 TIER_CMD=code TIER_VRAM_GIB=$VRAM \
      TIER_RAM_GIB=$([ "$RAM" = - ] && echo "" || echo "$RAM") TIER_PLAN_ONLY=${TIER_PLAN_ONLY:-0} \
      TIER_PROBE_MODEL=${PROBE_MODEL:-qwen3.8-pi} PYTHONPATH=$N/glq_overlay \
      HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfread PYTHONUNBUFFERED=1 \
      PATH=/home/ubuntu/.glq/venv/bin:$PATH \
  /home/ubuntu/.glq/venv/bin/python $OUT/step5_launch.py --model "$MODEL" "$@" "${TAIL[@]}" \
  < /dev/null > >(tee "$OUT/$TAG.stdout" >> "$LOG") 2> >(tee "$OUT/$TAG.stderr" >> "$LOG")
RC=$?
wait
echo "EXIT=$RC wall=${SECONDS}s" | tee -a "$LOG"
cp "$H/.glq/vllm.log" "$OUT/$TAG.vllm.log" 2>/dev/null
mkdir -p "$OUT/${TAG}_pi_sessions" && cp -r "$H/.pi/agent/sessions/." "$OUT/${TAG}_pi_sessions/" 2>/dev/null
cp "$H/.pi/agent/models.json" "$OUT/$TAG.models.json" 2>/dev/null
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/$TAG.gpu_after.txt"
echo "RUN_END $TAG $(date -Is)" | tee -a "$LOG"
