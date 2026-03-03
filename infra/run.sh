#!/bin/bash
# Run from infra/ after `tofu apply`.
# Uploads code, installs deps, runs GLQ E2E test via GPTQModel.
set -euo pipefail

IP=$(tofu output -raw instance_ip)
KEY="gpu-key.pem"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY ubuntu@$IP"
RSYNC="rsync -aW --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='*.egg-info' -e 'ssh -o StrictHostKeyChecking=no -i $KEY'"

echo "=== Waiting for instance setup ==="
for i in $(seq 1 60); do
    if $SSH "test -f /tmp/setup-done" 2>/dev/null; then
        echo "Instance ready."
        break
    fi
    echo "  Waiting... ($i/60)"
    sleep 10
done

echo "=== Syncing code ==="
eval $RSYNC ../glq/ ubuntu@$IP:~/glq/
eval $RSYNC ../GPTQModel/ ubuntu@$IP:~/GPTQModel/
eval $RSYNC ../test_glq_e2e.py ../pyproject.toml ubuntu@$IP:~/

echo "=== Installing packages ==="
$SSH "source .venv/bin/activate && \
    cd ~ && pip install -e . --no-deps -q && \
    cd ~/GPTQModel && pip install -r requirements.txt -q 2>&1 | tail -5 && \
    pip install qwen-vl-utils -q 2>/dev/null || true"

echo "=== Running GLQ E2E test ==="
$SSH "source .venv/bin/activate && cd ~ && \
    PYTHONPATH=GPTQModel:. python -u test_glq_e2e.py --ppl 2>&1" | tee ../test_glq_e2e.out

echo ""
echo "=== Done. Results saved to ../test_glq_e2e.out ==="
echo "=== To destroy: tofu destroy ==="
