#!/usr/bin/env bash
# Which pi versions accept the glq provider entry with Qwen3.8 chat-template thinking?
# pi drops the WHOLE models.json on a schema error, so the probe file also carries a
# decoy provider: if it vanishes from --list-models, the user's own providers would too.
# CPU only; installs each old pi into its own prefix, never globally.
set -u
N=/opt/dlami/nvme
P=$N/pi_probe
source ~/.nvm/nvm.sh > /dev/null
for v in 0.79.8 0.79.9 1.6.0; do
  pre=$P/pi_$v
  if [ ! -x "$pre/node_modules/.bin/pi" ]; then
    npm install --silent --prefix "$pre" --ignore-scripts "@earendil-works/pi-coding-agent@$v" \
      > "$P/npm_$v.log" 2>&1 || { echo "VERSION $v npm install failed"; continue; }
  fi
  for case in thinking plain; do
    agent=$P/agent_${v}_$case
    rm -rf "$agent" && mkdir -p "$agent"
    cp "$P/models_$case.json" "$agent/models.json"
    out=$(PI_CODING_AGENT_DIR=$agent timeout 60 "$pre/node_modules/.bin/pi" --list-models 2>&1)
    glq=$(printf '%s\n' "$out" | grep -c 'Arete-27B')
    decoy=$(printf '%s\n' "$out" | grep -c 'decoy-model')
    err=$(printf '%s\n' "$out" | grep -i -m1 -E 'invalid|error|schema' | cut -c1-200)
    echo "VERSION $v case=$case glq_listed=$glq decoy_listed=$decoy err=${err:-none}"
    printf '%s\n' "$out" | grep -E 'Arete-27B|decoy-model|^provider|thinking' | head -4 | sed 's/^/    /'
  done
done
