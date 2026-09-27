#!/usr/bin/env bash
# Qwen-Next CPU decode, thread sweep.
#
# --expect-gib 75 is the QUANTIZATION GATE: on a cpu device_map, run_model uses process RSS as
# the footprint, so a dense bf16 load (157+ GiB) fails loudly instead of quietly benchmarking
# the wrong model. 377 GB of RAM here would happily hold a dense load, which is exactly why
# the gate is needed rather than optional.
#
# HF_TOKEN is read from a mode-0600 file, never passed as an argument or exported by the
# caller: on a command line it lands in shell history and in `ps` for every user on the box.
# Authenticated pulls get higher rate limits and materially faster downloads.
set -u
T="${T:?set T}"
M=xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4
export HF_HOME=$HOME/hf_cache
export OMP_NUM_THREADS="$T"
[ -r "$HOME/.hftok" ] && export HF_TOKEN="$(cat "$HOME/.hftok")"
echo "### OMP_NUM_THREADS=$T  token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo MISSING)  $(date -u +%H:%M:%SZ)"
$HOME/.glq/venv/bin/python $HOME/harness/run_model.py \
  --model "$M" --runtime hf --device-map cpu \
  --batches 1 --decode 32 --repeats 3 --warmup 8 \
  --expect-gib 75 --label "cpu_t$T"
echo "EXIT=$? T=$T"
