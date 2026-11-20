#!/usr/bin/env bash
# The branch's glq-code, end to end up to the wire: it writes pi's entry, starts pi, and pi
# sends each level's chat_template_kwargs, captured by a stub server. --no-serve and a
# separate HOME: no GPU, no vLLM, and the user's ~/.pi and ~/.glq are never read or written.
set -u
N=/opt/dlami/nvme
P=$N/pi_probe
SRC=$N/glq_thinking_src
PORT=18999
M=xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw
E2E_HOME=$P/e2e_home
NODE_BIN=/home/ubuntu/.nvm/versions/node/v26.13.1/bin
PY=/home/ubuntu/.glq/venv/bin/python

rm -rf "$E2E_HOME"; mkdir -p "$E2E_HOME"
CAP=$P/capture_e2e.jsonl; : > "$CAP"
python3 -I "$P/capture_server.py" $PORT "$CAP" & SRV=$!
echo "stub server PID $SRV"

run() {   # env for one glq-code / pi run under the separate HOME
  HOME=$E2E_HOME PATH=$NODE_BIN:/usr/bin:/bin PYTHONPATH=$SRC \
  HF_HOME=$N/hf_cache HF_HUB_OFFLINE=1 "$@" < /dev/null
}
run $PY -c "import glq, glq.tooling as t; print('glq from', glq.__file__); print('has pi_model_fields', hasattr(t, 'pi_model_fields'))"

last_body() {   # the chat_template_kwargs of the newest captured request since line $1
  python3 -I -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2]):]
b = rows[-1]['body'] if rows else None
print(json.dumps({k: b[k] for k in ('chat_template_kwargs', 'reasoning_effort') if k in b}, sort_keys=True) if b is not None else 'NO REQUEST')
" "$CAP" "$1"
}

for lvl in default off low medium xhigh; do
  pi_flags=(--no-session -p hi); [ "$lvl" = default ] || pi_flags=(--thinking "$lvl" "${pi_flags[@]}")
  n0=$(wc -l < "$CAP")
  run timeout 120 $PY -m glq.code --model $M --no-serve --base-url http://127.0.0.1:$PORT/v1 \
      --max-model-len 65536 -- "${pi_flags[@]}" > "$P/e2e_$lvl.out" 2>&1
  echo "E2E level=$lvl rc=$? sent=$(last_body "$n0")"
done
echo "--- the entry glq-code wrote ($E2E_HOME/.pi/agent/models.json)"
python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d['providers']['glq']['models'], indent=1))
" "$E2E_HOME/.pi/agent/models.json"
echo "--- pi's own view of it"
HOME=$E2E_HOME PATH=$NODE_BIN:/usr/bin:/bin timeout 60 $NODE_BIN/pi --list-models Arete < /dev/null 2>&1 | head -3

echo "--- other pi versions, reading the entry glq-code wrote"
for v in 0.79.9 0.79.0 0.74.0 mz0731; do
  for lvl in default off low medium xhigh; do
    f=(); [ "$lvl" = default ] || f=(--thinking "$lvl")
    n0=$(wc -l < "$CAP")
    PATH=$NODE_BIN:$PATH PI_CODING_AGENT_DIR=$E2E_HOME/.pi/agent timeout 60 "$P/pi_$v/node_modules/.bin/pi" \
      --provider glq --model $M --no-session "${f[@]}" -p hi < /dev/null > "$P/old2_${v}_$lvl.out" 2>&1
    echo "OLD pi=$v level=$lvl rc=$? sent=$(last_body "$n0")"
  done
done
kill $SRV
