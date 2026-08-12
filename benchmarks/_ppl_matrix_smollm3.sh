#!/bin/bash
# WikiText-2 PPL matrix for SmolLM3-3B: bf16 vs GLQ-3inst-3bpw vs GLQ-3inst-4bpw vs NVFP4.
#
# Why PPL and not another speed sweep: the L4 speed matrix (2026-08-06) showed NVFP4 via
# Marlin beating GLQ-4bpw at every batch size, leaving GLQ only a ~28% footprint edge
# (1.81 vs 2.51 GiB). GLQ's actual claim is quality-per-bit, and that is unmeasured.
# wikitext2_ppl resolves adjacent-bpw differences (~0.1 ppl) that no MMLU-Pro/AIME sample
# size can, in ~5 min per arm.
#
# Usage:  ./_ppl_matrix_smollm3.sh <box-ip> [ssh-key]
#
# Runs ON the box; expects glq_src synced and the venv at /home/ubuntu/venv.
set -uo pipefail
BOX="${1:?usage: $0 <box-ip> [ssh-key]}"
KEY="${2:-infra/gpu-key.pem}"
SSH=(ssh -o ConnectTimeout=20 -i "$KEY" "ubuntu@$BOX")

"${SSH[@]}" 'bash -s' <<'REMOTE'
set -uo pipefail
cd /opt/dlami/nvme/glq_src
export HF_HOME=/opt/dlami/nvme/hf_cache
export HF_TOKEN=$(cat /opt/dlami/nvme/.hftok)
# Python block-buffers stdout at ~8 KB when it is a pipe, so a 35-min quantize logs
# NOTHING for minutes and then dumps in bursts — indistinguishable from a hang, which is
# the wrong signal when spot boxes vanish mid-run. `stdbuf -oL` does not help: it sits
# below Python's own buffering layer. quantize_model.main() now line-buffers itself; this
# covers every other child process too (glq-bench, its vLLM subprocesses).
export PYTHONUNBUFFERED=1
OUT=/opt/dlami/nvme

# --- Step 0: neither the 3bpw nor the 2bpw checkpoint exists on HF -------------------
# The org has 3inst at 4/5/6 bpw only. Substituting SmolLM3-3B-GLQ-3.5bpw would be a
# different codec (e8-shell, not trellis), so quantize both. Integer bpw only for trellis;
# nsamples 128 per CLAUDE.md. 3INST variant is required for new quants.
# ~30-40 min each for a 3B, so this is the long pole of the job.
quant () {
  bpw=$1; dst=$OUT/SmolLM3-3B-trellis-3inst-${bpw}bpw
  if [ -f "$dst/config.json" ]; then
    echo "=== [0] ${bpw}bpw checkpoint already present, skipping quantize ==="
    return 0
  fi
  echo "=== [0] quantize SmolLM3-3B trellis-3inst-${bpw}bpw $(date -u +%H:%M:%S) ==="
  # Must be the console script (pyproject [project.scripts] glq-quantize), NOT
  # `python glq/quantize_model.py` — the module uses relative imports (`from .codebook
  # import ...`) so running it as a file dies instantly with
  # "attempted relative import with no known parent package".
  #
  # HOST RAM: this does NOT pass --streaming, so the whole model is resident. On a 32 GiB
  # box (g5.2xlarge) the 3bpw run was OOM-killed — exit 137, "Killed", which reads like a
  # spot reclaim but is the kernel. Either give it a roomy box or add --streaming (which
  # loads one layer at a time, quantize_model.py:1262). Deliberately NOT adding --streaming
  # here: it changes the quantize path, and the 4bpw reference checkpoint we are comparing
  # against was produced without it.
  GLQ_TRELLIS_VARIANT=3inst stdbuf -oL -eL /home/ubuntu/venv/bin/glq-quantize \
      --model HuggingFaceTB/SmolLM3-3B --codebook trellis --bpw "$bpw" --nsamples 128 \
      --output "$dst" > $OUT/ppl_quant${bpw}bpw.log 2>&1
  echo "    quantize exit=$? (log: $OUT/ppl_quant${bpw}bpw.log)"
}
Q3=$OUT/SmolLM3-3B-trellis-3inst-3bpw
Q2=$OUT/SmolLM3-3B-trellis-3inst-2bpw

# --- PPL arms -----------------------------------------------------------------------
# wikitext2_ppl is kind="hf": it loads the model itself, NOT through vLLM. The NVFP4 arm
# therefore depends on transformers+compressed-tensors being able to load an
# nvfp4-pack-quantized checkpoint on sm_89 — untested, because the box died before the
# check ran. It is expected to be the fragile arm; `|| true` keeps one failure from
# killing the other three, and the exit code is printed so a skip is never silent.
arm () {
  label=$1; model=$2
  echo "=== [$label] $(date -u +%H:%M:%S)  model=$model ==="
  ( /home/ubuntu/venv/bin/glq-bench run --model "$model" --tasks wikitext2_ppl \
      --out $OUT/ppl_${label}.jsonl ) > $OUT/raw_ppl_${label}.log 2>&1
  echo "    exit=$?  (raw: $OUT/raw_ppl_${label}.log)"
  grep -aE "wikitext2_ppl done|perplexity|Traceback|Error|not supported" \
      $OUT/raw_ppl_${label}.log | tail -4
  # Echo the whole record to stdout, which the LOCAL driver log captures. Two spot boxes
  # have now been reclaimed mid-run; the first took the only copy of the bf16/glq4 records
  # with it. Printing here means provenance survives even if the box vanishes before any
  # scp, at the cost of a few hundred bytes of log.
  [ -f "$OUT/ppl_${label}.jsonl" ] && \
    echo "### RECORD ${label}: $(cat $OUT/ppl_${label}.jsonl)"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null; done
}

# ORDER MATTERS. Four spot boxes have now been lost mid-run, one after ~4 minutes. Running
# the two ~35-min quantizes first meant a short-lived box produced NOTHING. The three arms
# that need no quantize are ~5 min each off existing HF checkpoints, so they go first and
# bank real results inside the first quarter hour; each quantize is then immediately
# followed by its own PPL arm, so even a mid-run death leaves the earlier arms measured.
arm bf16   HuggingFaceTB/SmolLM3-3B
arm glq4   xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel
arm nvfp4  Firworks/SmolLM3-3B-nvfp4

quant 3
[ -f "$Q3/config.json" ] && arm glq3 "$Q3" || echo "=== [glq3] SKIPPED: quantize failed ==="
quant 2
[ -f "$Q2/config.json" ] && arm glq2 "$Q2" || echo "=== [glq2] SKIPPED: quantize failed ==="

# --- per-layer breakdown -------------------------------------------------------------
# The average SQNR hides where a low bpw actually does its damage. layer_metrics.json is
# written by quantize_model.py for every run; this ranks the worst matrices and rolls up
# by matrix type, which is what decides whether mixed-bpw is worth trying at 2 bpw.
for bpw in 2 3; do
  d=$OUT/SmolLM3-3B-trellis-3inst-${bpw}bpw
  [ -f "$d/layer_metrics.json" ] || { echo "=== [layers ${bpw}bpw] no layer_metrics.json ==="; continue; }
  echo "=== [layers] ${bpw}bpw per-layer quality ==="
  /home/ubuntu/venv/bin/python - "$d/layer_metrics.json" <<'PY'
import json, sys, re, statistics as st
m = json.load(open(sys.argv[1]))
rows = [(k, v.get('sqnr'), v.get('proxy_loss')) for k, v in m.items() if v.get('sqnr') is not None]
rows.sort(key=lambda r: r[1])
print(f"  {len(rows)} matrices | SQNR min {rows[0][1]:.1f} / median "
      f"{st.median(r[1] for r in rows):.1f} / max {rows[-1][1]:.1f} dB")
print("  --- 10 worst by SQNR ---")
for k, s, p in rows[:10]:
    print(f"    {k:<52} {s:6.1f} dB  proxy={p:.4g}")
byt = {}
for k, s, p in rows:
    t = re.sub(r'.*\.(\w+)_proj$', r'\1', k) if k.endswith('_proj') else k.split('.')[-1]
    byt.setdefault(t, []).append(s)
print("  --- by matrix type (mean SQNR) ---")
for t, v in sorted(byt.items(), key=lambda kv: st.mean(kv[1])):
    print(f"    {t:<14} n={len(v):3d}  mean={st.mean(v):6.1f} dB  min={min(v):6.1f}")
PY
done

echo "=== SUMMARY $(date -u +%H:%M:%S) ==="
for f in $OUT/ppl_*.jsonl; do
  [ -e "$f" ] || continue
  /home/ubuntu/venv/bin/python - "$f" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).read().strip().split("\n")[0])
b = r["benchmark"]
print(f"  {sys.argv[1].split('/')[-1]:<22} {b['metric']}={b['value']}  "
      f"gpu={(r.get('hardware') or {}).get('gpu_model')}")
PY
done
echo "PPL MATRIX DONE $(date -u +%H:%M:%S)"
REMOTE
