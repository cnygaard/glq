#!/usr/bin/env bash
# Hardware-counter profile of one GLQ CPU decode window. Results + findings:
# benchmarks/qwen_next_cpu_profile/README.md
#
# Region scoping without a code change: run_model.py prints FOOTPRINT immediately after load and
# before _two_point, so that line is an exact phase marker. _two_point then runs
# warmup(N) -> generate(1) -> generate(DECODE); the last call is the steady-state window, so we
# settle past the first two and fire one metric group at a time into the long one. Multiplexing
# degrades metric accuracy, hence sequential windows rather than one big -e list -- and one model
# load serves every window, because a load costs ~143 s and 75 GiB.
#
# Privilege split, established by dry-run against a throwaway process rather than discovered
# mid-run: per-PID core events work at perf_event_paranoid=1, but system-wide (-a) uncore AND
# py-spy's ptrace both additionally need sudo. Without the split the DRAM metric returns
# "<not supported>" and py-spy returns "Permission Denied" -- both of which read as "this box
# can't do it" rather than "wrong privilege". Uncore PMUs are per-socket and CANNOT be attributed
# to a PID, so the -a windows are only meaningful because the box is otherwise idle.
#
# Prerequisites on the box:
#   sudo sysctl -w kernel.perf_event_paranoid=1 kernel.nmi_watchdog=0   # 4 blocks even task-clock
#   <venv>/bin/pip install py-spy                                       # Python frames perf can't see
set -u
T="${T:?set T}"; LABEL="${LABEL:?set LABEL}"; DECODE="${DECODE:-512}"
SETTLE="${SETTLE:-45}"; FULL="${FULL:-1}"
M="${MODEL:-xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4}"
BASE=/opt/dlami/nvme; [ -w "$BASE" ] 2>/dev/null || BASE=$HOME/results
OUT=$BASE/pmu_$LABEL; mkdir -p "$OUT"
PY=${PY:-$HOME/.glq/venv/bin/python}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export OMP_NUM_THREADS="$T"

# Match BLAS to the arm, do NOT pin it to 1. Pinning it is harmless at T=1 but at T>1 it
# serializes the dense oneDNN/BLAS path -- which is precisely the non-scaling signature this
# script exists to investigate, so it would manufacture its own answer. (The committed thread
# sweep never set it, so those numbers are unaffected by the earlier version of this script.)
export OPENBLAS_NUM_THREADS="$T"

[ -r "$HOME/.hftok" ] && export HF_TOKEN="$(cat "$HOME/.hftok")"
LOG=$OUT/run.log

echo "=== pmu_profile LABEL=$LABEL T=$T DECODE=$DECODE OUT=$OUT $(date -u +%FT%TZ)"
echo "extra env: GLQ_CPU_GDN=${GLQ_CPU_GDN:-unset} GLQ_HF_MOE_CPU_FUSED=${GLQ_HF_MOE_CPU_FUSED:-unset}"

# ---- mechanism assertion: did OMP_NUM_THREADS actually reach torch, and which ISA tier? -------
# Not cosmetic: if torch's intra-op pool ignored the env var, the whole thread axis is fiction.
$PY - > "$OUT/mechanism.txt" 2>&1 <<'PYEOF'
import os, torch
from glq.inference_kernel_cpu import _try_load_cpu_ext, cpu_ext_status
_try_load_cpu_ext()
print("OMP_NUM_THREADS   =", os.environ.get("OMP_NUM_THREADS"))
print("torch.num_threads =", torch.get_num_threads())
print("torch.interop     =", torch.get_num_interop_threads())
print("glq cpu ext       =", cpu_ext_status())
print("GLQ_CPU_GDN       =", os.environ.get("GLQ_CPU_GDN", "unset"))
print("GLQ_HF_MOE_CPU_FUSED =", os.environ.get("GLQ_HF_MOE_CPU_FUSED", "unset"))
PYEOF
cat "$OUT/mechanism.txt"

{ nproc; free -g; swapon --show; } > "$OUT/env.txt" 2>&1

$PY "${RUN_MODEL:-$HOME/harness/run_model.py}" --model "$M" --runtime hf --device-map cpu \
  --batches 1 --decode "$DECODE" --repeats 1 --warmup 8 \
  --expect-gib "${EXPECT_GIB:-75}" --label "$LABEL" > "$LOG" 2>&1 &
PID=$!
echo "RUN_PID=$PID   (log $LOG)"

# ---- wait for load to finish, identified by the FOOTPRINT marker ----
for _ in $(seq 1 900); do
  grep -q '^FOOTPRINT' "$LOG" 2>/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "ENDED_BEFORE_DECODE"; tail -30 "$LOG"; exit 1; }
  sleep 1
done
grep '^FOOTPRINT' "$LOG" || { echo "NO_FOOTPRINT_TIMEOUT"; tail -20 "$LOG"; }
echo "--- settling ${SETTLE}s past warmup + the 1-token call ---"
sleep "$SETTLE"

alive(){ kill -0 $PID 2>/dev/null; }
win(){ # win <name> <command...>
  local n="$1"; shift
  if ! alive; then echo "SKIPPED $n (decode already finished)" | tee -a "$OUT/windows.txt"; return; fi
  echo "--- window $n @ $(date -u +%T)Z" | tee -a "$OUT/windows.txt"
  "$@" > "$OUT/$n.txt" 2>&1
}

win topdownL1 perf stat -p $PID -M TopdownL1 -- sleep 12
win topdownL2 perf stat -p $PID -M TopdownL2 -- sleep 12
win ipc       perf stat -p $PID -e cycles,instructions,branches,branch-misses -- sleep 12

if [ "$FULL" = "1" ]; then
  # Two independent DRAM readings. They must agree before either is quotable; at T=1 they came
  # in at 2.79 GB/s (CAS x 64 B) and 2.88 GB/s (perf's own metric).
  win dram_metric sudo perf stat -a -M tma_info_system_dram_bw_use -- sleep 12
  win dram_cas    sudo perf stat -a -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ -- sleep 12
  # FP events only -- these do NOT count integer SIMD, so the true vector share is higher.
  win fp_width    perf stat -p $PID \
      -e fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.scalar_single -- sleep 12
  win tma_l3mem   perf stat -p $PID -M TmaL3mem -- sleep 12
fi

# ---- native symbol attribution: which .so, which function -------------------------------------
# TMA alone cannot tell "efficient" from "efficiently doing useless work" -- an interpreter and a
# spinning barrier both RETIRE instructions. It is only interpretable next to this.
if alive; then
  echo "--- perf record (lbr, 30s) @ $(date -u +%T)Z"
  perf record -p $PID --call-graph=lbr -F 999 -o "$OUT/perf.data" -- sleep 30 > "$OUT/record.txt" 2>&1
  perf report -i "$OUT/perf.data" --stdio --no-children -F overhead,dso     > "$OUT/by_dso.txt" 2>&1
  perf report -i "$OUT/perf.data" --stdio --no-children -F overhead,dso,sym > "$OUT/by_sym.txt" 2>&1
fi

# ---- Python-frame attribution: the only view that prices the per-expert loop directly ---------
if alive && [ "$FULL" = "1" ]; then
  echo "--- py-spy (40s) @ $(date -u +%T)Z"
  sudo env "PATH=$PATH" "$(dirname "$PY")/py-spy" record -p $PID -d 40 -r 100 \
      -f speedscope -o "$OUT/pyspy.speedscope.json" > "$OUT/pyspy_record.txt" 2>&1 \
    || echo "pyspy record failed, see $OUT/pyspy_record.txt"
  for _ in 1 2 3 4 5 6; do
    alive && sudo env "PATH=$PATH" "$(dirname "$PY")/py-spy" dump --pid $PID >> "$OUT/pyspy_dumps.txt" 2>&1
  done
fi

# Waiting matters: without the RESULT line the arm has counters but no tok/s to anchor them to,
# which is exactly what a mid-run rescue cost the first T=1 arm.
echo "--- waiting for the run to finish so RESULT is recorded ---"
wait $PID; echo "RUN_EXIT=$?" | tee -a "$OUT/windows.txt"
grep -E '^(FOOTPRINT|RESULT|PEAK_RSS_GIB|DEGRADED|FAIL)' "$LOG" | tee "$OUT/result.txt"
echo "=== pmu_profile DONE $LABEL $(date -u +%FT%TZ)"
