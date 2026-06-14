#!/bin/bash
# Ship-gate validation for grouped-as-default on L4 24GB (sm_89), published 4bpw
# 26B-A4B. Confirms the SHIPPED default (mixed: b1 block-diag matvec + num_tokens>=2
# grouped-GEMM) serves correctly + captures clean + is deterministic + wins at
# batch -- the co-resident mix that was never run before the spot box died.
# 24GB can't fit the full [1..64] capture with the ~15GB model -> capture [1,2,4,8].
cd /opt/dlami/nvme
source /home/ubuntu/venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf_cache
export GLQ_PARITY_MODEL=xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw
export GLQ_PARITY_CAPTURE=1,2,4,8
export GLQ_PARITY_BATCHES=1,8

# 1) oracle (eager Python per-expert loop) -- the correctness reference
export GLQ_PARITY_EAGER=1
python -u moe_parity.py fallback  > l4_oracle.log    2>&1; echo "oracle rc=$?"

# 2-4) shipped default (mixed) x2 for determinism + blockdiag A/B -- all FULL cudagraph
export GLQ_PARITY_EAGER=0
python -u moe_parity.py fused     > l4_default1.log  2>&1; echo "default1 rc=$?"
python -u moe_parity.py fused     > l4_default2.log  2>&1; echo "default2 rc=$?"
python -u moe_parity.py blockdiag > l4_blockdiag.log 2>&1; echo "blockdiag rc=$?"

PARSE='
import re,sys
def parse(fn):
    d={}
    for l in open(fn):
        m=re.match(r"PARITY\[\w+\]\[(\d+)\]: (\[.*\])", l)
        if m: d[int(m.group(1))]=m.group(2)
    return d
'
echo "===== GATE1 correctness: default(cudagraph) factual tokens vs eager oracle ====="
python - <<PY
$PARSE
o,df=parse("l4_oracle.log"),parse("l4_default1.log")
factual=[i for i in (0,1,2,4) if o.get(i)!=df.get(i)]
print("  factual prompts:", "ALL MATCH oracle" if (o and not factual) else f"DIFFER {factual}")
print("  prompt3 open-ended:", "match" if o.get(3)==df.get(3) else "benign cross-impl diverge")
PY
echo "===== GATE2 cudagraph capture clean (default)? ====="
grep -iE "Capturing CUDA graph|illegal|Cannot copy|RuntimeError|Engine core|out of memory|CUDA error" l4_default1.log | tail -4 || true
echo "  coherence (default):"; grep -h "TEXT\[fused\]" l4_default1.log | head -4
echo "===== GATE3 determinism (default run1 vs run2) ====="
python - <<PY
$PARSE
a,b=parse("l4_default1.log"),parse("l4_default2.log")
print("  ", "BIT-EXACT run-to-run" if (a and all(a[i]==b.get(i) for i in a)) else "NONDETERMINISTIC")
PY
echo "===== GATE4 throughput: default(grouped b>=2) vs blockdiag (b1 same, b8 differs) ====="
grep -h "RESULT\[" l4_default1.log l4_blockdiag.log
echo "L4_SHIP_VALIDATE_DONE"
