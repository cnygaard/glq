"""Gates for one Step 5 row, from the files step5_run.sh leaves in $OUT.

    python -I step5_check.py /opt/dlami/nvme/arete_pi_step5 TAG

Prints one `GATE <tag> <check> PASS|FAIL <detail>` line per check and `NOTE` lines with the
memory figures vLLM logged, then `GATE_RESULT <tag> PASS|FAIL`. Exit 1 on any FAIL.
"""
import glob
import json
import os
import re
import sys

out, tag = sys.argv[1], sys.argv[2]
p = lambda ext: os.path.join(out, f"{tag}{ext}")          # noqa: E731
read = lambda f: open(f, errors="replace").read() if os.path.exists(f) else ""  # noqa: E731
log, stdout, vlog = read(p(".log")), read(p(".stdout")), read(p(".vllm.log"))
fails = 0


def gate(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"GATE {tag} {name} {'PASS' if ok else 'FAIL'} {detail}".rstrip())


tier = {}
for line in stdout.splitlines():
    if line.startswith("TIER "):
        d = json.loads(line[5:])
        tier[d["phase"]] = d

m = re.search(r"^EXIT=(\d+)", log, re.M)
gate("exit_0", m is not None and m.group(1) == "0", f"EXIT={m.group(1) if m else None}")

pi_out = "\n".join(l for l in stdout.splitlines() if not l.startswith("TIER "))
gate("answer_63", re.search(r"\b63\b", pi_out) is not None, repr(pi_out.strip()[-200:]))

loaded = re.findall(r"Loaded new LoRA adapter: name '([^']+)'", vlog)
gate("adapter_loaded", loaded == ["qwen3.8-pi"], f"{loaded}")

args = re.search(r"non-default args: (\{.*)", vlog)
args = args.group(1) if args else ""
for flag in ("'enable_lora': True", "'max_lora_rank': 32",
             "'enable_thinking': False", "'tool_call_parser': 'qwen3_xml'"):
    gate(f"arg[{flag}]", flag in args)
# vLLM's "non-default args" line leaves out a value equal to its default, and max_loras
# defaults to 1, so this one is read from the argv glq-code handed to vLLM.
argv = (tier.get("argv") or {}).get("argv") or []
gate("argv[--max-loras 1]", "--max-loras" in argv
     and argv[argv.index("--max-loras") + 1] == "1")
targets = re.search(r"'lora_target_modules': \[([^\]]*)\]", args)
gate("lora_targets", targets is not None and len(targets.group(1).split(",")) == 7,
     targets.group(1) if targets else "missing")

plan = tier.get("plan", {})
served = re.search(r"'max_model_len': (\d+)", args)
gate("planned_window_served",
     served is not None and plan.get("window") == int(served.group(1)),
     f"plan {plan.get('window')} served {served.group(1) if served else None}")

models = json.loads(read(p(".models.json")) or "{}")
entries = [m for prov in models.get("providers", {}).values() for m in prov.get("models", [])]
pi_entry = [m for m in entries if m.get("id") == "qwen3.8-pi"]
gate("pi_entry", len(pi_entry) == 1 and not any(k in pi_entry[0] for k in
     ("reasoning", "thinkingLevelMap", "compat")), json.dumps(pi_entry)[:300])

sessions = glob.glob(os.path.join(out, f"{tag}_pi_sessions", "**", "*.jsonl"), recursive=True)
models_used, thinking_chars, assistant_msgs, tool_calls = set(), 0, 0, 0
for s in sessions:
    for line in open(s, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "thinking_level_change":
            print(f"NOTE {tag} pi thinking_level_change {json.dumps(e)[:160]}")
        msg = e.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        assistant_msgs += 1
        if msg.get("model"):
            models_used.add(msg["model"])
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "thinking":
                thinking_chars += len(c.get("thinking") or "")
            if isinstance(c, dict) and c.get("type") == "toolCall":
                tool_calls += 1
gate("pi_session_model", models_used == {"qwen3.8-pi"},
     f"{sorted(models_used)} over {assistant_msgs} assistant messages in {len(sessions)} session(s)")
gate("pi_tool_call", tool_calls >= 1, f"{tool_calls} tool call(s)")
gate("thinking_off_pi", thinking_chars == 0, f"{thinking_chars} thinking chars")

probe = tier.get("probe", {})
ans = probe.get("answer") or {}
gate("probe_answer", "paris" in (ans.get("content") or "").lower(),
     repr((ans.get("content") or probe.get("error") or "")[:120]))
gate("thinking_off_probe", ans.get("reasoning_chars") == 0, f"{ans.get('reasoning_chars')}")

gpu = read(p(".gpu_after.txt")).strip()
gate("gpu_free", gpu == "", gpu or "no compute apps")

for pat in (r"Model loading took [\d.]+ GiB", r"Available KV cache memory: [\d.]+ GiB",
            r"GPU KV cache size: [\d,]+ tokens", r"Maximum concurrency for [\d,]+ tokens per "
            r"request: [\d.]+x", r"Graph capturing finished in [\d.]+ secs, took [-\d.]+ GiB",
            r"init engine \(profile, create kv cache, warmup model\) took [\d.]+ s"):
    for hit in re.findall(pat, vlog)[:1]:
        print(f"NOTE {tag} {hit}")
for r in probe.get("runs") or []:
    print(f"NOTE {tag} probe ttft_s {r.get('ttft_s')} decode_tps {r.get('decode_tps')} "
          f"tokens {r.get('tokens')}")
print(f"NOTE {tag} plan {json.dumps(plan)}")
print(f"NOTE {tag} ready {json.dumps(tier.get('ready'))}")
print(f"GATE_RESULT {tag} {'PASS' if not fails else 'FAIL'}")
sys.exit(1 if fails else 0)
