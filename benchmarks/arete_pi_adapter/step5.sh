#!/bin/bash
# Step 5 of the Arete pi-adapter plan, one GPU job at a time on GPU 0:
#   overlay  the installed glq 0.8.38 package, with the branch's three changed files on top
#            (the branch differs from v0.8.38 only in glq/code.py, glq/tooling.py and
#            glq/installer/configure.py), sha256-checked against the local worktree
#   full       glq-code --pi-adapter on the real card (RTX PRO 6000, 95.6 GiB)
#   t24_refuse planned for a 23.99 GiB card / 32 GiB host: the fit check must refuse, no GPU
#   t24_force  the same with --force: does vLLM hold the window the plan would have used?
set -u
N=/opt/dlami/nvme
OUT=$N/arete_pi_step5
M=xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw
PY=/home/ubuntu/.glq/venv/bin/python
echo "=== step5 start $(date -Is)"

SP=$(env -u PYTHONPATH $PY -c "import glq, os; print(os.path.dirname(glq.__file__))")
echo "installed glq: $SP"
rm -rf $N/glq_overlay && mkdir -p $N/glq_overlay && cp -a "$SP" $N/glq_overlay/glq
for f in glq/code.py glq/tooling.py glq/installer/configure.py; do
    cp "$OUT/branch/$f" "$N/glq_overlay/$f"
done
find $N/glq_overlay -name __pycache__ -prune -exec rm -rf {} +
(cd $N/glq_overlay && sha256sum -c "$OUT/branch/SHA256SUMS") || { echo "OVERLAY_SHA_FAIL"; exit 1; }
PYTHONPATH=$N/glq_overlay $PY -c "import glq, glq.tooling as t; print('overlay', glq.__version__, glq.__file__); print(t.pi_adapter('$M'))" || exit 1

bash $OUT/step5_run.sh full - - $M --pi-adapter
$PY -I $OUT/step5_check.py $OUT full

bash $OUT/step5_run.sh t24_refuse 23.99 32 $M --pi-adapter
if grep -q "^EXIT=2 " $OUT/t24_refuse.log && [ ! -s $OUT/t24_refuse.vllm.log ]; then
    echo "GATE t24_refuse refused_before_launch PASS"
else
    echo "GATE t24_refuse refused_before_launch FAIL"
fi
grep -v '^TIER ' $OUT/t24_refuse.stderr | head -20

bash $OUT/step5_run.sh t24_force 23.99 32 $M --pi-adapter --force
$PY -I $OUT/step5_check.py $OUT t24_force

echo "=== step5 end $(date -Is)"
