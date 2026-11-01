#!/bin/bash
# glq-code end to end from the branch overlay: $1 = run label, rest = glq-code flags.
set -u
N=/opt/dlami/nvme
TAG=$1; shift
export PYTHONPATH=$N/glq_dspark:$N/tf519 HF_HOME=$N/hf_cache
export NVM_DIR=$HOME/.nvm; . $NVM_DIR/nvm.sh >/dev/null
mkdir -p $N/codetest && cd $N/codetest
printf "RETRY_LIMIT = 7\n\n\ndef main():\n    print(sum(range(RETRY_LIMIT)) * 3)\n\n\nif __name__ == \"__main__\":\n    main()\n" > hello.py
# Speculation and prefix-cache counters while the server is up; the last scrape is kept.
( until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
  while curl -sf localhost:8000/metrics -o $N/metrics_$TAG.prom.tmp; do
    mv $N/metrics_$TAG.prom.tmp $N/metrics_$TAG.prom; sleep 10; done ) &
W=$!
echo "RUN_START $TAG $(date -Is) flags: $*"
SECONDS=0
~/.glq/venv/bin/glq-code --model xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw "$@" \
  -- -p "Read hello.py in the current directory with your read tool, then tell me exactly what it prints when run."
echo "EXIT=$? wall=${SECONDS}s"
kill $W 2>/dev/null
cp ~/.glq/vllm.log $N/vllm_$TAG.log
echo "RUN_END $TAG $(date -Is)"
