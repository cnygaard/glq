#!/bin/bash
# Arete-27B 4 bpw: real glq-code / glq-chat runs on an emulated 24 GiB card (24,564 MiB), one
# at a time. A row the fit check refuses (EXIT=2) is repeated with --force to see whether vLLM
# serves it anyway. Then the real 32 GB card as a control.
cd /opt/dlami/nvme/tiers
M=xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw
for cmd in code chat; do
    bash tier24_run.sh r24_$cmd $cmd 23.99 32 $M
    if grep -q "^EXIT=2 " r24_$cmd.log; then bash tier24_run.sh r24_${cmd}_force $cmd 23.99 32 $M --force; fi
done
bash tier24_run.sh r32_code code 31.86 32 $M
echo QUEUE_DONE $(date -Is)
