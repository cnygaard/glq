#!/bin/bash
# Regression check: E4B is the DENSE Gemma-4 (no MoE), so the grouped MoE path is
# inert (its gate needs MoE block-diag meta dense layers don't have). This confirms
# the grouped op registration + dispatch seam don't break dense-model serving:
# grouped vs fused must be token-identical + coherent, with unchanged throughput.
cd /opt/dlami/nvme
source /home/ubuntu/venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf_cache
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null)
export GLQ_PARITY_MODEL=xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw
export GLQ_PARITY_EAGER=0   # cudagraph

python -u moe_parity.py fused   > e4b_fused.log 2>&1;   echo "fused rc=$?"
python -u moe_parity.py grouped > e4b_grouped.log 2>&1; echo "grouped rc=$?"

echo "===== inert? (grouped == fused token IDs — dense model, flag does nothing) ====="
python - <<'PY'
import re
def parse(fn):
    d = {}
    for l in open(fn):
        m = re.match(r"PARITY\[\w+\]\[(\d+)\]: (\[.*\])", l)
        if m:
            d[int(m.group(1))] = m.group(2)
    return d
fu, gp = parse("e4b_fused.log"), parse("e4b_grouped.log")
ok = bool(fu) and all(fu[i] == gp.get(i) for i in fu)
print("E4B_FLAG_INERT:", "IDENTICAL (no regression)" if ok else "DIFFER/EMPTY")
PY
grep -h "TEXT\[grouped\]" e4b_grouped.log | head -3
grep -h "RESULT\[" e4b_fused.log e4b_grouped.log
echo "E4B_DONE"
