# Chat with a GLQ model

GLQ checkpoints serve over vLLM's OpenAI-compatible API, so any OpenAI-compatible
client works. Two are set up here.

## Gradio (default — same venv, no Node)

```bash
pip install glq vllm 'glq[chat]'
vllm serve xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel --quantization glq   # terminal 1
python examples/chat/app.py                                                   # terminal 2
```

Then open http://localhost:7860. The dropdown is populated from the server's
`/v1/models`, falling back to `~/.glq/config.json` if the server isn't up yet.

`--base-url` points it at a server on another host or port.

## Open WebUI (optional)

Nicer UI, but **install it in its own virtualenv** — never alongside glq:

```bash
python3 -m venv ~/.glq/venv-webui
~/.glq/venv-webui/bin/pip install open-webui
OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=glq \
  ~/.glq/venv-webui/bin/open-webui serve --port 8080
```

It pins 119 dependencies exactly, among them `transformers==5.5.4`, while GLQ needs
transformers ≥ 5.13.1 to serve gemma-4 — installing it into the glq venv silently
downgrades transformers and breaks GLQ. It also requires Python ≥ 3.11, < 3.13 where
glq supports 3.10. Note its licence is the "Open WebUI License", not an OSI-standard one.

`install.sh --chat=openwebui` does the above for you, into that separate venv.

## Other clients

`../opencode/` and `../pi/` hold configs for the opencode and pi coding agents against
the same endpoint.
