# Use a GLQ model with opencode

GLQ models serve over vLLM's OpenAI-compatible API, so opencode talks to them as
an OpenAI-compatible provider.

1. Serve the model (add `--trust-remote-code` for models that need it):

   ```bash
   pip install glq vllm
   vllm serve xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw --port 8000
   ```

2. Put `opencode.json` at `~/.config/opencode/opencode.json` (edit the `models`
   map to list the repos you serve), then select the `glq` provider in opencode.

The `apiKey` is a placeholder — vLLM does not check it, but the field must be set.
