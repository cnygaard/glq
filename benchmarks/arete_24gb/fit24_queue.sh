#!/bin/bash
# fix/fit-arete-24gb on the box: the same glq-code / glq-chat rows as run_2026-11-19, now with the
# family overhead. No --force anywhere: a row that refuses is a result. One GPU job at a time.
cd /opt/dlami/nvme/tiers
A=xv0y5ncu/Arete-27B-GLQ-trellis-3inst
bash tier24_run.sh f24_code code 23.99 32 ${A}-4bpw
bash tier24_run.sh f24_chat chat 23.99 32 ${A}-4bpw
bash tier24_run.sh f22_code code 22.49 32 ${A}-4bpw
until grep -q "^DONE_5\|^FAIL_5" ../arete56_download.log; do sleep 30; done
bash tier24_run.sh f32_code5 code 31.86 32 ${A}-5bpw
bash tier24_run.sh f32_chat5 chat 31.86 32 ${A}-5bpw
until grep -q "^DONE_6\|^FAIL_6" ../arete56_download.log; do sleep 30; done
bash tier24_run.sh f32_code6 code 31.86 32 ${A}-6bpw
bash tier24_run.sh f32_chat6 chat 31.86 32 ${A}-6bpw
echo QUEUE_DONE $(date -Is)
