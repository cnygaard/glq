#!/bin/bash
# Gemma-4-31B is DENSE (enable_moe_block:False) -> the grouped MoE path is inert
# (its gate needs MoE block-diag meta dense layers lack). This run:
#   1. confirms the grouped op + seam don't regress LARGE dense serving (coherence),
#   2. captures a current 31B mix-5bpw b1/b32 throughput number on the Blackwell
#      (sm_120) -- relevant to the open 0.52 tok/s concern (task #45),
#   3. re-confirms grouped==fused (flag inert) modulo the known benign cross-launch
#      greedy-tie divergence on the single open-ended prompt (prompt 3, "ocean").
cd /opt/dlami/nvme
source /home/ubuntu/venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf_cache
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null)
export GLQ_PARITY_MODEL=xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8
export GLQ_PARITY_EAGER=0   # FULL cudagraph (capture sizes <=64)

python -u moe_parity.py fused   > b31_fused.log 2>&1;   echo "fused rc=$?"
python -u moe_parity.py grouped > b31_grouped.log 2>&1; echo "grouped rc=$?"

echo "===== coherence (grouped TEXT, first 4 prompts) ====="
grep -h "TEXT\[grouped\]" b31_grouped.log | head -4
echo "===== capture clean? ====="
grep -iE "Capturing CUDA graphs|illegal|Cannot copy|RuntimeError|Engine core" b31_grouped.log | tail -3
echo "===== per-prompt fused-vs-grouped (prompt 3 may differ benignly: cross-launch tie) ====="
python - <<'PY'
import re
def parse(fn):
    d = {}
    for l in open(fn):
        m = re.match(r"PARITY\[\w+\]\[(\d+)\]: (\[.*\])", l)
        if m: d[int(m.group(1))] = m.group(2)
    return d
fu, gp = parse("b31_fused.log"), parse("b31_grouped.log")
for i in sorted(set(fu) | set(gp)):
    print(f"  prompt {i}:", "MATCH" if fu.get(i) == gp.get(i) else "DIFFER")
factual = [i for i in (0,1,2,4) if fu.get(i) != gp.get(i)]
print("FACTUAL_PROMPTS_ALL_MATCH:", not factual, "(differing factual:", factual, ")")
PY
echo "===== throughput (cudagraph) ====="
grep -h "RESULT\[" b31_fused.log b31_grouped.log
echo "B31_DONE"
