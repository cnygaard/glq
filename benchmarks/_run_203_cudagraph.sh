#!/bin/bash
# Stage 3 #203 gate part 2: FULL-cudagraph capture + the real payoff measurement
# (grouped vs the C++ block-diag fused path, both cudagraph) + determinism.
cd /opt/dlami/nvme
source /home/ubuntu/venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf_cache
export GLQ_PARITY_MODEL=/opt/dlami/nvme/glq_out/gemma-4-26B-A4B-it-GLQ-3bpw
export GLQ_PARITY_EAGER=0   # FULL cudagraph (capture sizes <=64)

echo "===== fused (block-diag) cudagraph ====="
python -u moe_parity.py fused > cg_fused.log 2>&1; echo "fused rc=$?"
echo "===== grouped cudagraph (run 1) ====="
python -u moe_parity.py grouped > cg_grouped1.log 2>&1; echo "grouped1 rc=$?"
echo "===== grouped cudagraph (run 2, determinism) ====="
python -u moe_parity.py grouped > cg_grouped2.log 2>&1; echo "grouped2 rc=$?"

echo "===== capture clean (grouped)? ====="
grep -iE "Capturing CUDA graphs|illegal|Cannot copy|RuntimeError|Engine core" cg_grouped1.log | tail -4

echo "===== determinism (grouped run1 vs run2) ====="
python - <<'PY'
import re
def parse(fn):
    d = {}
    for l in open(fn):
        m = re.match(r"PARITY\[\w+\]\[(\d+)\]: (\[.*\])", l)
        if m:
            d[int(m.group(1))] = m.group(2)
    return d
a, b = parse("cg_grouped1.log"), parse("cg_grouped2.log")
det = bool(a) and all(a[i] == b.get(i) for i in a)
print("DETERMINISM:", "BIT-EXACT across runs" if det else "NONDETERMINISTIC")
PY

echo "===== b32 throughput (cudagraph): grouped vs block-diag ====="
grep -h "RESULT\[" cg_fused.log cg_grouped1.log
echo "CUDAGRAPH_DRIVER_DONE"
