#!/bin/bash
# Stage 3 #203 gate part 1: e2e oracle parity (eager). Runs the Python-loop
# oracle (fallback) then the grouped-GEMM path (grouped) on the on-disk 3bpw 26B,
# greedy/deterministic, and diffs the 48 output token IDs per prompt. grouped MUST
# match fallback token-for-token -> validates the whole grouped chain (#199-#202).
cd /opt/dlami/nvme
source /home/ubuntu/venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf_cache
export GLQ_PARITY_MODEL=/opt/dlami/nvme/glq_out/gemma-4-26B-A4B-it-GLQ-3bpw
export GLQ_PARITY_EAGER=1

echo "===== run fallback (oracle) ====="
python -u moe_parity.py fallback > parity_fallback.log 2>&1; echo "fallback rc=$?"
echo "===== run grouped ====="
python -u moe_parity.py grouped > parity_grouped.log 2>&1; echo "grouped rc=$?"

echo "===== PARITY DIFF (grouped vs fallback) ====="
python - <<'PY'
import re
def parse(fn):
    d = {}
    for line in open(fn):
        m = re.match(r"PARITY\[\w+\]\[(\d+)\]: (\[.*\])", line)
        if m:
            d[int(m.group(1))] = m.group(2)
    return d
fb, gp = parse("parity_fallback.log"), parse("parity_grouped.log")
ok = bool(fb) and bool(gp)
for i in sorted(fb):
    match = fb.get(i) == gp.get(i)
    print(f"  prompt {i}: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        ok = False
        print("    fallback:", fb.get(i))
        print("    grouped :", gp.get(i))
print("PARITY_RESULT:", "ALL MATCH" if ok else "MISMATCH/EMPTY")
PY
echo "===== throughput (eager) ====="
grep -h "RESULT\[" parity_fallback.log parity_grouped.log
echo "===== coherence (grouped TEXT) ====="
grep -h "TEXT\[grouped\]" parity_grouped.log | head -3
echo "PARITY_DRIVER_DONE"
