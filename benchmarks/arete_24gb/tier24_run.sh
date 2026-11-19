#!/bin/bash
# One emulated-tier row: glq-code or glq-chat planned for a TIER_VRAM card and a TIER_RAM host,
# on GPU 0 of this box, inside a cgroup that caps the whole process tree at TIER_RAM.
#   tier24_run.sh TAG code|chat VRAM_GIB RAM_GIB MODEL [flags...]
# Adapted from benchmarks/hw_tiers/tier_run.sh for an installed glq 0.8.37 (no PYTHONPATH tree).
# Each row gets its own HOME, so ~/.glq/vllm.log and pi's models.json are per row. A file
# $N/tiers/config_<TAG>.json, if present, becomes that HOME's ~/.glq/config.json.
set -u
N=/opt/dlami/nvme
TAG=$1 CMD=$2 VRAM=$3 RAM=$4 MODEL=$5; shift 5
H=$N/tiers/home_$TAG
rm -rf "$H"; mkdir -p "$H/.pi/agent" "$H/.glq" "$H/work" "$H/.cache" "$N/tiers/shared_flashinfer"
ln -s /home/ubuntu/.nvm "$H/.nvm"
ln -s "$N/tiers/shared_flashinfer" "$H/.cache/flashinfer"
[ -f "$N/tiers/config_$TAG.json" ] && cp "$N/tiers/config_$TAG.json" "$H/.glq/config.json"
printf "RETRY_LIMIT = 7\n\n\ndef main():\n    print(sum(range(RETRY_LIMIT)) * 3)\n\n\nif __name__ == \"__main__\":\n    main()\n" > "$H/work/hello.py"
LOG=$N/tiers/$TAG.log
echo "RUN_START $TAG cmd=$CMD vram=$VRAM ram=$RAM model=$MODEL flags: $* plan_only=${TIER_PLAN_ONLY:-0} $(date -Is)" | tee "$LOG"
if [ "$MODEL" = "-" ]; then MARGS=(); else MARGS=(--model "$MODEL"); fi
if [ "$CMD" = code ]; then
    TAIL=(-- -p "Read hello.py in the current directory with your read tool, then tell me exactly what it prints when run.")
else
    TAIL=(--no-browser --no-share)
fi
SECONDS=0
cd "$H/work"
sudo -n systemd-run --scope -q -p MemoryMax=${RAM}G -p MemorySwapMax=0 \
  sudo -n -u ubuntu env HOME="$H" CUDA_VISIBLE_DEVICES=0 TIER_CMD=$CMD TIER_VRAM_GIB=$VRAM \
      TIER_RAM_GIB=$RAM TIER_PLAN_ONLY=${TIER_PLAN_ONLY:-0} \
      HF_HOME=$N/hf_cache HF_TOKEN_PATH=$N/.hfread PYTHONUNBUFFERED=1 \
      PATH=/home/ubuntu/.glq/venv/bin:$PATH \
  /home/ubuntu/.glq/venv/bin/python $N/tiers/tier24_launch.py "${MARGS[@]}" "$@" "${TAIL[@]}" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "EXIT=$RC wall=${SECONDS}s" | tee -a "$LOG"
cp "$H/.glq/vllm.log" "$N/tiers/$TAG.vllm.log" 2>/dev/null
echo "RUN_END $TAG $(date -Is)" | tee -a "$LOG"
