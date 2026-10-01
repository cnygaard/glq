#!/usr/bin/env bash
# Weight-offload frontier sweep: how much VRAM can a model give up, and what does it cost?
#
# The question this answers: GLQ's goal is "fit larger LLMs on smaller GPUs", so the useful
# output is a frontier -- resident GiB against tok/s -- not a yes/no. vLLM has TWO offload
# backends and they fail differently, so both are swept:
#
#   uva      p.data becomes a CUDA *view* of pinned host memory; the GPU reads over PCIe on
#            demand. Cost is per byte ACTUALLY ACCESSED, so it is nearly free for a sparse
#            lookup (0.8.23's PLE table) and bandwidth-bound for dense weights, which are
#            read in full every token at B=1.
#   prefetch params live in pinned host storage, a GPU buffer is swapped in around forward,
#            and copies are issued `prefetch_step` groups ahead -- so transfer can HIDE
#            behind compute. The free zone is roughly
#            offloaded_bytes / PCIe_BW < 1 / tok_s_resident.
#
# NVFP4 goes first and GLQ second, deliberately: NVFP4 is quantization vLLM supports natively,
# so it measures the MECHANISM with no GLQ compatibility bug in the path. If NVFP4 cannot use
# offload, GLQ cannot either and no GLQ work is justified.
#
# `experts`-filtered arms are the interesting ones for MoE: only routed experts are read per
# token (26B-A4B touches ~4B of 26B), so offloading just the experts is the one case with the
# PLE's sparsity. vLLM supports it as a config -- no custom code.
#
# Usage:
#   MODEL=nvidia/Gemma-4-26B-A4B-NVFP4 QUANT=none bash benchmarks/_offload_sweep.sh
#   MODEL=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw QUANT=glq bash benchmarks/_offload_sweep.sh
#
# One GPU job at a time, sequentially, per CLAUDE.md. Results land in $OUT as TSV plus one
# full log per arm.
set -u

MODEL=${MODEL:?set MODEL}
QUANT=${QUANT:-none}
PY=${PY:-/home/ubuntu/.glq/venv/bin/python}
OUT=${OUT:-/opt/dlami/nvme/offload_sweep}
BATCHES=${BATCHES:-1}
DECODE=${DECODE:-64}
GPU_MEM=${GPU_MEM:-0.90}
# Multimodal checkpoints (Gemma-4 26B-A4B carries a vision encoder) refuse to start when the
# encoder's max_tokens_per_mm_item exceeds max_num_batched_tokens, which is derived from
# max_model_len: at 2048 every arm died with
#   "Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than
#    max_num_batched_tokens (2048)"
# before offload was even reached. 4096 clears it, and --mm0 zeroes all three
# limit_mm_per_prompt keys for a text-only serve (run_model.py's needs_mm0() auto-detect does
# not fire for nvidia's repacked NVFP4 arch name, so pass it explicitly rather than relying
# on detection).
MAXLEN=${MAXLEN:-4096}
TAG=$(echo "$MODEL" | tr '/' '_')

mkdir -p "$OUT"
TSV="$OUT/${TAG}_${QUANT}.tsv"
printf 'arm\tbackend\tbudget_gb\tfilter\tgroup\tresident_gib\tttft_ms\ttokps_b1\tstatus\n' > "$TSV"

# arm-name | extra run_model.py flags
ARMS=(
  "resident|"
  "uva_4|--offload-backend uva --cpu-offload-gb 4"
  "uva_8|--offload-backend uva --cpu-offload-gb 8"
  "uva_12|--offload-backend uva --cpu-offload-gb 12"
  "uva_experts_12|--offload-backend uva --cpu-offload-gb 12 --offload-params experts"
  "prefetch_g4|--offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1"
  "prefetch_g2|--offload-backend prefetch --offload-group-size 2 --offload-num-in-group 1"
  "prefetch_experts_g4|--offload-backend prefetch --offload-group-size 4 --offload-params experts"
)

echo "=== offload sweep: $MODEL (quant=$QUANT) ==="
echo "    results -> $TSV"

for entry in "${ARMS[@]}"; do
    arm=${entry%%|*}
    flags=${entry#*|}
    log="$OUT/${TAG}_${QUANT}_${arm}.log"
    echo "--- arm $arm : $flags"

    # shellcheck disable=SC2086
    HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf_cache} PYTHONFAULTHANDLER=1 \
      stdbuf -oL -eL "$PY" benchmarks/run_model.py \
        --model "$MODEL" --runtime vllm --quant "$QUANT" \
        --batches "$BATCHES" --decode "$DECODE" --gpu-mem "$GPU_MEM" \
        --max-model-len "$MAXLEN" --mm0 --label "offload_$arm" $flags > "$log" 2>&1
    rc=$?

    # "Model loading took X GiB" is the ONLY trustworthy resident figure: nvidia-smi measures
    # the KV pool too, and a parent-process max_memory_allocated() reads 0 because weights
    # load in the EngineCore subprocess.
    resident=$(/usr/bin/grep -aoE "Model loading took [0-9.]+ GiB" "$log" | tail -1 | awk '{print $4}')
    ttft=$(/usr/bin/grep -aoE "ttft_ms=[0-9.]+" "$log" | tail -1 | cut -d= -f2)
    tokps=$(/usr/bin/grep -aoE "per_seq_tokps=[0-9.]+" "$log" | tail -1 | cut -d= -f2)
    # The mechanism, not just the outcome: a run that silently stayed resident still generates
    # correct tokens at a normal-looking rate, so record whether vLLM said it offloaded.
    offl=$(/usr/bin/grep -aoE "Total CPU offloaded parameters: [0-9.]+ ?[A-Za-z]*" "$log" | tail -1)
    status="ok"; [ "$rc" -ne 0 ] && status="FAILED(rc=$rc)"
    [ -z "${resident:-}" ] && status="$status;no-footprint"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$arm" "$(echo "$flags" | /usr/bin/grep -oE 'backend [a-z]+' | awk '{print $2}')" \
      "$(echo "$flags" | /usr/bin/grep -oE 'offload-gb [0-9]+' | awk '{print $2}')" \
      "$(echo "$flags" | /usr/bin/grep -oE 'offload-params [a-z,]+' | awk '{print $2}')" \
      "$(echo "$flags" | /usr/bin/grep -oE 'group-size [0-9]+' | awk '{print $2}')" \
      "${resident:--}" "${ttft:--}" "${tokps:--}" "$status" >> "$TSV"

    echo "    resident=${resident:--} GiB  ttft=${ttft:--} ms  tok/s=${tokps:--}  [$status]"
    echo "    vllm said: ${offl:-<no offload line>}"
done

echo "=== done ==="
column -t -s $'\t' "$TSV"
