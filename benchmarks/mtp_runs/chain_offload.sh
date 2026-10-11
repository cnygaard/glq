#!/bin/bash
# Wait out the in-flight GPU job, then run the offload matrix. One GPU job at a time.
for i in $(seq 1 80); do
  pgrep -f "[s]moke_spec2.sh" >/dev/null || break
  sleep 15
done
exec /opt/dlami/nvme/offload_spec.sh
