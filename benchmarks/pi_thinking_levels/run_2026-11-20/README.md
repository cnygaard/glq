# pi reasoning levels for Arete-27B in glq-code (2026-11-20)

## The problem

In glq-code on `xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw`, `/thinking` showed
"Thinking level: off" and offered no levels.

glq-code's entry in `~/.pi/agent/models.json` carried only `id`, `contextWindow` and
`maxTokens`. pi therefore treated the model as one that cannot think. The server went on
thinking at the chat template's default, `xhigh`.

## The fix

glq-code adds the family's pi fields to that entry: `reasoning`, `thinkingLevelMap`, and
`compat.chatTemplateKwargs` with `thinkingFormat: "chat-template"`. They live in
`glq/tooling.py`, `_QWEN38_27B.pi_thinking`, which covers Arete-27B and Qwen3.8-27B.

## Setup

- Box: RTX PRO 4500 (32 GB).
- vLLM 0.33.0 serving Arete 4 bpw: the user's own glq-code session, glq 0.8.38. It was not
  restarted.
- pi 1.6.0 installed globally. Older versions were installed into separate prefixes under
  `/opt/dlami/nvme/pi_probe`.
- No GPU job was started. vLLM saw `/tokenize` calls and ten 1-token chat completions.

## Results

| check | script | log | result |
|---|---|---|---|
| schema | `pi_schema_probe.sh`, `pi_send_probe.sh` (top), `old_pi_wire_offspelled.txt` (line 1) | `probe.txt`, `send_probe.txt` | pi 0.73.1, 0.74.0, 0.79.0, 0.79.8, 0.79.9 and 1.6.0 all list the entry and keep a decoy provider. This matters because pi drops the whole `models.json` when one entry fails its schema check (`dist/core/model-config.js`). |
| wire | `pi_send_probe.sh` + `capture_server.py` | `send_probe.txt` | pi 0.79.9 and 1.6.0 send `chat_template_kwargs`, with `reasoning_effort` set to the map value and omitted when off. They clamp unmapped levels: minimal to low, high to xhigh, and max to xhigh (1.6.0) or medium (0.79.9). |
| older pi | `old_pi_wire_offspelled.txt` | same | pi 0.73.1, 0.74.0 and 0.79.0 have no chat-template format. They send a top-level `reasoning_effort`, which was `"off"` at the off level with the first map. |
| render | `render_levels.py` | `render_levels.txt` | Through `/tokenize`: xhigh and no kwargs are identical (60 tokens). low gives the low instruction, medium no instruction, off the closed think block. An unaccepted value gets HTTP 400. |
| real chat path | `chat_path_levels.py` | `chat_path_levels.txt` | `usage.prompt_tokens`: xhigh 60, low 48, medium 22, off 24, for both styles of request. A top-level `"off"` is HTTP 400 (vLLM accepts only none/minimal/low/medium/high/xhigh/max), and `"none"` gives 24. So the map's off value became `"none"`. |
| end to end | `glq_code_e2e.sh` | `glq_code_e2e_v2.txt` | The branch's glq-code with `--no-serve` against the stub, under a separate HOME. It writes the entry, and pi lists it with thinking `yes`. Every level from 1.6.0 (through glq-code), 0.79.9, 0.79.0, 0.74.0 and 0.73.1 sends only values with a 200 row in the real-chat-path check. |

`models_thinking.json` is the probe file from before the `"none"` change, so it still maps
off to `"off"`. The end-to-end run used the entry glq-code itself wrote.

The scripts write `.log` files on the box. They are kept here as `.txt`, byte for byte,
because the repo ignores `*.log`.

## Not measured

How long Arete actually reasons at each level. The template adds an instruction for xhigh
and low and none for medium, and Arete's AIME numbers were measured at xhigh.
