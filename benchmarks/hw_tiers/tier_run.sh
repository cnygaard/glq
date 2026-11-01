#!/bin/bash
# One tier row: glq-code planned for a TIER_VRAM card and a TIER_RAM host, on one GPU of this
# box, inside a cgroup that holds the whole process tree (glq-code, vLLM, pi) to TIER_RAM.
#   tier_run.sh TAG GPU VRAM_GIB RAM_GIB PORT MODEL [glq-code flags...]
# Each row gets its own HOME, so two rows on two GPUs never share ~/.glq/vllm.log, pi's
# models.json or a compile cache.
set -u
N=/opt/dlami/nvme
TAG=$1 GPU=$2 VRAM=$3 RAM=$4 PORT=$5 MODEL=$6; shift 6
H=$N/tiers/home_$TAG
rm -rf "$H"; mkdir -p "$H/.pi/agent" "$H/work" "$H/.cache" "$N/tiers/shared_flashinfer"
ln -s /home/ubuntu/.nvm "$H/.nvm"
# One FlashInfer JIT cache for every row. Per-row, each row rebuilt the same kernels cold, and
# that parallel nvcc build (~0.5 GiB per cudafe++) inside a 64 GiB cap with 38 GiB pinned was
# what the cgroup killed (t32_sv4_r64, run 1). A user's box builds them once.
ln -s "$N/tiers/shared_flashinfer" "$H/.cache/flashinfer"
printf "RETRY_LIMIT = 7\n\n\ndef main():\n    print(sum(range(RETRY_LIMIT)) * 3)\n\n\nif __name__ == \"__main__\":\n    main()\n" > "$H/work/hello.py"
LOG=$N/tiers/$TAG.log
echo "RUN_START $TAG gpu=$GPU vram=$VRAM ram=$RAM model=$MODEL flags: $* $(date -Is)" | tee "$LOG"
SECONDS=0
cd "$H/work"
# A system scope, not `--user`: without lingering, the user manager (and every scope under it)
# stops when the ssh session that launched this ends.
sudo -n systemd-run --scope -q -p MemoryMax=${RAM}G -p MemorySwapMax=0 \
  sudo -n -u ubuntu env HOME="$H" CUDA_VISIBLE_DEVICES=$GPU TIER_VRAM_GIB=$VRAM TIER_RAM_GIB=$RAM \
      TIER_VLLM_ARGS="${TIER_VLLM_ARGS:-}" \
      PYTHONPATH=$N/glq_dspark:$N/tf519 HF_HOME=$N/hf_cache PYTHONUNBUFFERED=1 \
      PATH=/home/ubuntu/.glq/venv/bin:$PATH \
  /home/ubuntu/.glq/venv/bin/python $N/glq_dspark/benchmarks/tier_launch.py \
      --model "$MODEL" --base-url "http://127.0.0.1:$PORT/v1" "$@" \
      -- -p "Read hello.py in the current directory with your read tool, then tell me exactly what it prints when run." \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "EXIT=$RC wall=${SECONDS}s" | tee -a "$LOG"
cp "$H/.glq/vllm.log" "$N/tiers/$TAG.vllm.log" 2>/dev/null
echo "RUN_END $TAG $(date -Is)" | tee -a "$LOG"
