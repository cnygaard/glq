#!/usr/bin/env bash
# What does pi put on the wire for each thinking level, with the glq entry carrying
# Qwen3.8 chat-template thinking? Stub server, no model, no GPU; a private agent dir
# (PI_CODING_AGENT_DIR) and session dir, so the user's own pi config is untouched.
set -u
N=/opt/dlami/nvme
P=$N/pi_probe
PORT=18999
source ~/.nvm/nvm.sh > /dev/null
for v in 0.74.0 0.79.0; do                       # schema only, older versions
  pre=$P/pi_$v
  [ -x "$pre/node_modules/.bin/pi" ] || npm install --silent --prefix "$pre" --ignore-scripts \
    "@earendil-works/pi-coding-agent@$v" > "$P/npm_$v.log" 2>&1
  agent=$P/agent_${v}_thinking; rm -rf "$agent"; mkdir -p "$agent"
  cp "$P/models_thinking.json" "$agent/models.json"
  out=$(PI_CODING_AGENT_DIR=$agent timeout 60 "$pre/node_modules/.bin/pi" --list-models 2>&1)
  echo "VERSION $v glq_listed=$(printf '%s\n' "$out" | grep -c Arete-27B)" \
       "decoy_listed=$(printf '%s\n' "$out" | grep -c decoy-model)" \
       "err=$(printf '%s\n' "$out" | grep -i -m1 -E 'invalid|error|schema' | cut -c1-200)"
done

CAP=$P/capture.jsonl; rm -f "$CAP"
python3 -I "$P/capture_server.py" $PORT "$CAP" & SRV=$!
echo "stub server PID $SRV"
agent=$P/agent_send; rm -rf "$agent"; mkdir -p "$agent"
sed "s#http://127.0.0.1:8000/v1#http://127.0.0.1:$PORT/v1#" "$P/models_thinking.json" > "$agent/models.json"
for pi_v in 1.6.0 0.79.9; do
  for lvl in default off minimal low medium high xhigh max; do
    flag=(); [ "$lvl" = default ] || flag=(--thinking "$lvl")
    n0=$(wc -l < "$CAP" 2>/dev/null || echo 0)
    PI_CODING_AGENT_DIR=$agent timeout 60 "$P/pi_$pi_v/node_modules/.bin/pi" \
      --provider glq --model xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw --no-session \
      "${flag[@]}" -p "hi" > "$P/send_${pi_v}_$lvl.out" 2>&1
    rc=$?
    python3 -I - "$CAP" "$n0" "$pi_v" "$lvl" "$rc" <<'EOF'
import json, sys
cap, n0, v, lvl, rc = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(cap)][n0:] if n0 >= 0 else []
if not rows:
    print(f"SEND pi={v} level={lvl} rc={rc} NO REQUEST"); sys.exit()
b = rows[-1]["body"]
keys = {k: b[k] for k in ("chat_template_kwargs", "reasoning_effort", "enable_thinking",
                          "reasoning", "thinking") if k in b}
print(f"SEND pi={v} level={lvl} rc={rc} {json.dumps(keys, sort_keys=True)}")
EOF
  done
done
kill $SRV
