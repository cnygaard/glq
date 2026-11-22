# The pi adapter on GLQ Arete-27B (`glq-code --pi-adapter`)

[harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted)
(revision `7d42b119`) is a PEFT LoRA:
- rank 32, alpha 64;
- targets `q/k/v/o_proj` on the 16 full-attention layers and `gate/up/down_proj` on all 64 MLPs;
- trained with RL on Qwen3.8-27B's bf16 weights to drive pi, with thinking off.

These runs check that vLLM applies it to the GLQ Arete-27B 4 bpw checkpoint unchanged, what it
costs, and that `glq-code --pi-adapter` works end to end. Whether it helps Arete is **not** measured
here.

Setup for every run:
- one RTX PRO 6000 Blackwell Server Edition (96 GB), on a g7e;
- vLLM 0.33.0, torch 2.14.0, glq 0.8.38 with this branch's three changed files on top;
- `xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw`, bf16, one request at a time;
- 2026-11-22.

Raw logs are not in the repo.

## Step 0: does vLLM apply the LoRA to GLQ layers? (`step0.sh`, `step0b.sh`)

Hand-run `vllm serve` with glq-code's Arete arguments, a 131,072-token window, util 0.92 and
`--max-num-seqs 1`. Arms were run with and without `--enable-lora …`, at vLLM's 2,048-token
prefill step (cards under 70 GiB) and at 8,192.

- **The mechanism engaged.**
  - The served adapter is activated at startup ("Loaded new LoRA adapter").
  - vLLM's LoRA manager logged weights for 160 modules: 64 `gate_up_proj`, 64 `down_proj`,
    16 `qkv_proj` and 16 `o_proj`.
- **The adapter changes the output.** `step0_logprobs.py` sent two prompts at temperature 0,
  each twice to the base model and once to the adapter.
  - The base model repeated itself exactly: the largest logprob difference was 0.000.
  - The adapter moved logprobs by up to 0.18 and 0.28.
  - The greedy text was the same over those ~95 tokens, and both made the same `read` tool
    call.
- **Memory.** "Available KV cache memory" fell from 68.22 to 67.86 GiB at the 2,048-token step,
  and from 67.98 to 67.63 GiB at 8,192.
  - That is 0.36 and 0.35 GiB: about 0.31 GiB in loaded weights and 0.05 GiB in graphs.
  - Turning `cudagraph_specialize_lora` off made no measurable difference.
- **Speed.** `vllm bench serve`, 10 random prompts of 1,024 tokens in and 256 out, concurrency 1:

  | server | model asked | decode | mean TTFT |
  |---|---|--:|--:|
  | no LoRA | base | 38.81 tok/s | 405.5 ms |
  | `--enable-lora` | base | 38.76 tok/s | 407.8 ms |
  | `--enable-lora` | adapter | 33.20 tok/s | 407.4 ms |

## Step 5: `glq-code --pi-adapter` end to end (`step5.sh`)

`step5_run.sh` runs glq-code with a fresh HOME per row, so `vllm.log`, pi's `models.json` and
pi's sessions belong to that row. The 24 GB rows emulate a 23.99 GiB card and a 32 GiB host on
the same GPU, the method from PR #202 (`benchmarks/arete_24gb/`). pi is asked to read `hello.py`
and say what it prints. `step5_check.py` gates each row.

| row | window | KV cache | KV tokens | decode | start to ready | result |
|---|--:|--:|--:|--:|--:|---|
| `full`: 96 GB, adapter, first launch | 262,144 | 22.97 GiB | 364,581 | 32.77 tok/s | 326 s | all gates pass |
| `full_warm`: same, compile caches warm | 262,144 | 23.00 GiB | 364,581 | 32.78 tok/s | 186 s | all gates pass |
| `base_cold`: 96 GB, no adapter, first launch | 262,144 | 23.31 GiB | 370,227 | 38.59 tok/s | 298 s | control |
| `t24_refuse`: 24 GB, adapter | — | — | — | — | — | refused before launch, SHORT by 0.30 GiB |
| `t24_force`: 24 GB, adapter, `--force` | 16,384 | 2.37 GiB | 25,988 | 32.82 tok/s | 320 s | all gates pass |

Decode is the launcher's probe: 256 tokens at temperature 0, three repeats after a warm-up,
which agreed within 0.01 tok/s in every row.

The gates on the adapter rows:
- pi answered 63 through one `read` tool call;
- vLLM loaded `qwen3.8-pi` with the 7 target modules and rank 32;
- pi's session used model `qwen3.8-pi`, its thinking level was "off", and it had 0 thinking
  characters;
- the probe's answer had no reasoning;
- the planned window was served;
- no GPU process was left afterwards.

Notes:

- **24 GB.** The refusal comes from Arete's family `runtime_overhead`. That 3.25 GiB is the
  5 bpw worst case plus a margin; 4 bpw measured 2.48–2.62 GiB in #202.
  - Counted the planner's way, the forced run used about 2.72 GiB: a 22.07 GiB budget, less
    16.98 GiB of weights, less 2.37 GiB of KV.
  - The planner reserved 3.65 GiB: 3.25 plus the adapter's 0.40.
- **Graph capture.** About 100 s goes to PIECEWISE CUDA graph capture in every 96 GB row,
  whether the adapter is on or off and whether the caches are warm. Step 0's hand-run server,
  with a 131,072-token window and util 0.92, captured in 6–8 s. This comes from glq-code's
  serve setup, not from the adapter, and its cause is not identified.
- **First request.** vLLM warns that the LoRA shrink kernel is JIT-compiled on the first real
  request. The warm-up request's TTFT was 0.175 s, against about 0.105 s after it.
- **Deprecation warning.** vLLM 0.33 logs "deprecated support for supporting different
  tokenizers for different LoRAs" on the first LoRA request whenever the server has a
  tokenizer (`InputProcessor._validate_lora`), so it cannot be avoided from glq. glq-code
  downloads only `adapter_config.json` and `adapter_model.safetensors`, and the base
  tokenizer is used.
  - Its vocabulary, merges and added tokens match the adapter's.
  - The pre-tokenizer regex differs (`[\p{L}\p{M}]+` against `\p{L}+`), which matters only for
    decomposed accents.
