# GLQ — fit larger LLMs on smaller GPUs

**E8-lattice post-training quantization** for LLM weights: **2–8 bits/weight**,
served on **vLLM · HuggingFace Transformers**, with **deterministic**
fused CUDA kernels and opt-in **KV-cache compression (up to ~4×)**. Validated
from 24 GB 3090-class GPUs (A10G, sm_86) to a 96 GB RTX PRO 6000 Blackwell
(sm_120).

GLQ encodes each group of 8 weights as a 16-bit index into a 65,536-entry E8
lattice codebook; a Randomized Hadamard Transform makes the weights incoherent
so Euclidean nearest-neighbour rounding is near-optimal under the Hessian-weighted
proxy loss, and a fused CUDA kernel matmuls **directly against the compressed
indices** — on the GPU serving path the dense weight is never materialized, so
GPU memory drops with the compression ratio (CPU inference and a few
architecture fallbacks dequantize instead).

**What you get**
- **NEW in v0.7 — trellis (TCQ) codebook**: single-stream decode at **bf16 speed**
  (176 vs 180 tok/s, SmolLM3-3B 4 bpw vs bf16, RTX PRO 6000 / vLLM) in a **third of the
  memory**, and good quality at 2–3 bpw. See
  [Trellis codebook](#trellis-codebook---codebook-trellis--qtip-derived-tcq).
- **2–8 bpw**, **no group-size constraint**, optional **per-layer mixed precision**.
- **Serve anywhere** — a vLLM plugin (weight + MoE + embedding + KV cache) and an
  HF Transformers integration. `pip install glq`, load, run.
- **Small footprint** — smallest of the ~4-bit quantizers we measured (vs AWQ /
  NVFP4 on a 26B); a 31B fits ≈16.5 GiB at 5 bpw where bf16 needs ≈58 GiB, with
  quality within noise of bf16 on our paired reasoning evals.
- **Deterministic kernels** — bit-identical logits across runs (reproducible
  lm-eval scoring / on-policy RL rollouts).
- **Long context** — opt-in E8 KV-cache compression (up to ~4× smaller KV → more
  context in the same VRAM).

**Pick your path** → [run a model](#run-a-pre-quantized-model) ·
[fit a bigger model on your card](#available-pre-quantized-checkpoints) ·
[quantize your own](#quantize-your-own-model) ·
[how GLQ compares](#how-glq-compares) · serve with
[vLLM](#docker-image-nvidia-gpu) ·
[how it works](#how-it-works)

## Quickstart

### Run a pre-quantized model

```bash
pip install glq         # requires PyTorch ≥ 2.0
```

Python ≥ 3.10. Triton ships with PyTorch on CUDA and is used
automatically. The CUDA C extension JIT-builds on first run
(~30 s); CPU falls back to dequantize-then-matmul.

```python
import glq.hf_integration  # registers GLQ with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "xv0y5ncu/SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("xv0y5ncu/SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw")
print(tok.decode(model.generate(
    **tok("The capital of France is", return_tensors="pt").to(model.device),
    max_new_tokens=20,
)[0], skip_special_tokens=True))
```

`import glq.hf_integration` registers `quant_method="glq"` with HF
Transformers; `from_pretrained` then swaps `nn.Linear` for `E8RHTLinear`
and uses the fused CUDA C kernel on inference. CPU falls back to a
naive dequantize-then-matmul.

Or serve the fastest GLQ checkpoint on vLLM (the trellis-3INST decode —
single-stream speed at bf16 parity, 1.9 GiB of weights):

```bash
pip install glq vllm      # glq ≥ 0.7.0 (trellis kernel storage layout)
vllm serve xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel --quantization glq
```

### Available pre-quantized checkpoints

A few popular checkpoints (all on the [**`xv0y5ncu`** HF org](https://huggingface.co/xv0y5ncu)):

| Repo | Base model | bpw | License | Footprint¹ | Best for |
|---|---|---|--:|--:|---|
| [`Gemma-4-E4B-it-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw) | Gemma-4-E4B (8B, multimodal) | 4.0 | Apache 2.0 | 5.8 GiB | a capable model on an 8–12 GB card |
| [`SmolLM3-3B-trellis-3inst-4bpw-kernel`](https://huggingface.co/xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel) | SmolLM3-3B | 4.0 trellis | Apache 2.0 | 1.9 GiB | **fastest GLQ decode** — single-stream at bf16 parity |
| [`SmolLM3-3B-GLQ-block-diagonal-3.5bpw`](https://huggingface.co/xv0y5ncu/SmolLM3-3B-GLQ-block-diagonal-3.5bpw) | SmolLM3-3B | 3.5 mix | Apache 2.0 | 1.8 GiB | small + fast, fits anything |
| [`Gemma-4-12B-it-GLQ-5.0bpw`](https://huggingface.co/xv0y5ncu/Gemma-4-12B-it-GLQ-5.0bpw) | Gemma-4-12B | 5.0 mix | Apache 2.0 | 6.9 GiB | 12B on a 24 GB card |
| [`gemma-4-26B-A4B-it-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw) | Gemma-4-26B-A4B (MoE) | 4.0 | Apache 2.0 | ~15 GiB | best quality-per-GB (MoE) |
| [`Gemma-4-31B-it-GLQ-5.0bpw-mix3-8`](https://huggingface.co/xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8) | Gemma-4-31B | 5.0 mix | Apache 2.0 | 16.5 GiB | a 31B on **one** 24–32 GB card |
| [`Devstral-Small-2-24B-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw) | Devstral-Small 24B | 4.0² | Apache 2.0 | ~20.5 GiB | coding / agentic |
| [`SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw`](https://huggingface.co/xv0y5ncu/SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw) | SmolLM2-360M | 4.0 | Apache 2.0 | 0.25 GiB | tiny / CI demo |

**21 checkpoints total** — the [HF org](https://huggingface.co/xv0y5ncu) also has SmolLM3
at 6 bpw and the Gemma-4 12B/31B/E4B family across **3–8 bpw** (incl. `e8p`
variants). Per-model **quality** (MMLU-Pro / AIME, paired vs bf16) and **throughput** are in
each model card and in [How GLQ compares](#how-glq-compares) and
[Quality & footprint](#quality--footprint) below.

<sub>¹ **Footprint** = resident weight memory after load (vLLM's `Model loading took … GiB`)
— the figure that decides whether a model fits a 24/32 GB card. For current (block-diagonal)
checkpoints it tracks the bpw budget — that is what lets a 31B fit one GPU. E8-KV compression
doesn't change it; its payoff is an up-to-~4× smaller KV cache → more context / concurrency in
the same VRAM.</sub>

<sub>² Quantized before block-diagonal FHT became the quantizer default: power-of-2 FHT padding
is stored as real bits, so the checkpoint holds more bits per weight than its nominal rate and
the footprint is correspondingly larger than 4 bpw implies. The `-block-diagonal-` repos above
are true-to-label re-quants.</sub>

### Quantize your own model

```bash
pip install 'glq[quantize]'    # adds transformers, datasets, etc.

glq-quantize \
    --model HuggingFaceTB/SmolLM2-360M \
    --output ./smollm2-glq-4bpw \
    --bpw 4 \
    --nsamples 128 \
    --device cuda
```

Other bit-widths: pass `--bpw 2` through `--bpw 8` (fractional like
`2.5` also works). `glq-quantize --help` lists every flag. For models
that don't fit in system RAM use `--streaming` (loads one layer at a
time from safetensors).

For **mixed-precision** allocation, run a two-pass flow: a profile
pass writes a per-layer `bpw_allocation.json`, then a quantize pass
applies it. See [`examples/quantize_mixed_precision.md`](examples/quantize_mixed_precision.md).

**Trellis (TCQ) quantization** — the fastest-decoding GLQ format and the
recommended pick at 2–4 bpw:

```bash
GLQ_TRELLIS_VARIANT=3inst glq-quantize \
    --model HuggingFaceTB/SmolLM3-3B \
    --output ./smollm3-trellis-3inst-4bpw \
    --codebook trellis --bpw 4 --nsamples 128
```

Always set `GLQ_TRELLIS_VARIANT=3inst` for new quantizations: the
lookup-free 3INST decode is what the fused fast path is built for, and
its quality measured equal-or-slightly-better than the legacy `hyb`
lookup-table variant in our paired tests. (`hyb` remains the env default
only for back-compat with existing hyb checkpoints.)

Trellis constraints differ from the shell/e8p paths: **integer bpw only
(2 / 3 / 4)** — mixed precision and fractional rates are rejected rather
than silently rounded — and the fused kernel needs layer dims with
`out % 32 == 0`, `in % 64 == 0` (standard transformer shapes qualify).
Models with per-layer embeddings (Gemma-4 E2B/E4B) are handled
automatically — the PLE table quantizes via the shell codebook (requires
glq ≥ 0.7.2). Use `--streaming` for Gemma-4 family models.
See [Trellis codebook](#trellis-codebook---codebook-trellis--qtip-derived-tcq)
for details.

## Docker image (NVIDIA GPU)

A prebuilt CUDA image ships everything needed to run GLQ models —
`glq`, PyTorch, vLLM, transformers, and lm-eval on CUDA 12.8:

```
ghcr.io/cnygaard/glq-env:latest     # CUDA 12.8 bundle: glq + vLLM + transformers + lm-eval
```

**Prerequisite — GPU access in Docker.** You need an NVIDIA GPU plus
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host; that's what makes the `--gpus all` flag pass
the GPU into the container. Verify it works:

```bash
docker run --rm --gpus all ghcr.io/cnygaard/glq-env:latest nvidia-smi
```

If that prints your GPU table, you're set. (No toolkit → `--gpus`
errors with "could not select device driver".)

**Produce output.** Mount a host directory for the model cache (the
image's `HF_HOME` is `/cache/hf`, so models persist across runs
instead of re-downloading), then generate:

```bash
docker run --rm --gpus all \
    -v "$HOME/.cache/huggingface:/cache/hf" \
    ghcr.io/cnygaard/glq-env:latest \
    python -c '
import glq.hf_integration, torch                      # registers GLQ with HF
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(
    mid, device_map="cuda", torch_dtype=torch.float16)
ids = tok("The capital of France is", return_tensors="pt").to("cuda")
print(tok.decode(model.generate(**ids, max_new_tokens=20)[0],
                 skip_special_tokens=True))
'
```

Expected output:

```
The capital of France is Paris. It is located in the north of the country.
```

The first run downloads the model into the mounted cache; later runs
reuse it. Swap `mid` for any GLQ checkpoint (see
[Available pre-quantized checkpoints](#available-pre-quantized-checkpoints)).

**Flag reference:**

| Flag | Why |
|---|---|
| `--gpus all` | binds **all** host GPUs into the container (needs the NVIDIA Container Toolkit). Use `--gpus '"device=0"'` to pick one. |
| `-v "$HOME/.cache/huggingface:/cache/hf"` | persists downloaded weights on the host (`HF_HOME=/cache/hf` inside) so they survive `--rm`. |
| `--rm` | remove the container when it exits (drop it to keep the container around). |

**Serving (vLLM) & an interactive shell.** The image bundles vLLM, so
you can serve an OpenAI-compatible endpoint — publish the port and
mount the cache:

```bash
# Plain chat — the model ships its own chat template, so nothing extra needed:
docker run --rm --gpus all -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/cache/hf" \
    ghcr.io/cnygaard/glq-env:latest \
    vllm serve xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw --max-model-len 64000
```

**Tool-calling + thinking.** Gemma-4's tool template is *not* in the model (its
bundled `chat_template.jinja` is plain chat) and *not* in the vLLM pip wheel, so
fetch it from vLLM's `examples/` first, then mount it:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.20.2/examples/tool_chat_template_gemma4.jinja \
    -o tool_chat_template_gemma4.jinja

docker run --rm --gpus all -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/cache/hf" \
    -v "$PWD/tool_chat_template_gemma4.jinja:/work/tool.jinja:ro" \
    ghcr.io/cnygaard/glq-env:latest \
    vllm serve xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw \
        --max-model-len 64000 \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --chat-template /work/tool.jinja \
        --default-chat-template-kwargs '{"enable_thinking": true}'
```

Pass `--chat-template` the **in-container** mount path (`/work/tool.jinja`), not
the host path. `--default-chat-template-kwargs '{"enable_thinking": true}'`
defaults Gemma-4 reasoning on. The `gemma4` parsers and all of these flags are
accepted by the image's bundled vLLM 0.20.2 and the model loads; note that
startup runs a multi-minute `torch.compile` + CUDA-graph capture before the
endpoint is ready. See the
[vLLM Gemma-4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
for the full tool-calling / reasoning reference.

> **Image vs. pip:** the image is a convenience bundle and may lag the PyPI
> package — `pip install -U glq` inside the container (or your own venv) always
> gets the newest release. The pip package is the source of truth; the image just
> saves you assembling a matching CUDA + vLLM + transformers stack.

For the long-context E8 KV-cache flags (`GLQ_KV_*`), pass them with
`-e` and see [E8 lattice cache](#e8-lattice-cache-vllm-v030) /
[Inline-dequant E8 KV](#inline-dequant-e8-kv-default-in-v051).
The image's default command is a shell (`docker run --rm -it --gpus all
ghcr.io/cnygaard/glq-env:latest`) if you'd rather poke around interactively.

## How GLQ compares

GLQ sits next to the other post-training quantizers — AWQ, GPTQ, NVIDIA's NVFP4
(TensorRT Model Optimizer), and Unsloth's dynamic mixed-precision. They optimize for
different things; here is the honest layout.

| | **GLQ** | AWQ | GPTQ | NVFP4 (ModelOpt) | Unsloth dynamic |
|---|---|---|---|---|---|
| Bits / weight | **2–8** + mixed | 4 (grouped) | 3–4 (grouped) | ~4 (W4A4) | mixed 2–8 (selective) |
| Group-size constraint | **none** | g64/g128 | g64/g128 | — | — |
| Core method | E8 lattice + RHT + LDLQ | activation-aware scale | block error-feedback | FP4 + per-tensor scale | per-layer bit allocation |
| Footprint at ~4-bit | **smallest** (no per-group scales/zeros) | + group scales/zeros | + group scales/zeros | + FP8 scales | varies |
| Speed on Blackwell | W4A16; **single-stream at bf16 parity** (trellis 3INST, 3B measured) | W4A16 (Marlin) | W4A16 (Marlin) | **fastest at batch** (native FP4) | n/a (GGUF) |
| Serving stack | **vLLM · HF** | vLLM · HF · TRT | vLLM · HF · TRT | vLLM · TRT-LLM | **llama.cpp / Ollama** (GGUF) |
| KV-cache compression | **built-in (~4×)** | — | — | fp8 KV | (llama.cpp KV) |
| Bit-exact deterministic kernels | **yes** | — | — | — | — |
| Fine-tuning (QLoRA) | — | — | — | — | **yes** |

<sub>A "—" means the feature is absent or not advertised by that method — we have not
tested the other methods for those properties.</sub>

**Where each wins** — pick by your constraint, not by a single "best":

- **GLQ** — the smallest footprint of the ~4-bit methods we measured, the widest bit-range (2–8) with
  per-layer mixed precision, deterministic kernels, and built-in KV compression. The pick
  when you are **memory-bound** — fit a bigger model or longer context on a 24–32 GB card —
  and serve on vLLM / HF.
- **NVFP4 (ModelOpt)** — **fastest on Blackwell** (native FP4 tensor cores). The pick when
  you have a Blackwell GPU with memory to spare and want raw decode speed; it trades a larger
  footprint than GLQ for that speed.
- **AWQ / GPTQ** — mature, ubiquitous 4-bit weight-only with fast Marlin kernels. The safe
  default on any GPU when a ~4-bit footprint is enough and you don't need <4 bpw.
- **Unsloth dynamic** — selective mixed-precision for the **GGUF / llama.cpp** stack, plus a
  strong QLoRA fine-tuning story. A different target (CPU/edge/Ollama, or training) than GLQ's
  GPU serving — GLQ also does per-layer mixed precision, but for the vLLM/HF stack.

Where GLQ speed stands after v0.7.1 (measured, SmolLM3-3B on an RTX PRO 6000, vLLM 0.25):
**single-stream (B=1) trellis-3INST decode is at bf16 parity** — 176 vs 180 tok/s — while
using a third of the weight memory. At batch, bf16 still leads (2,423 vs 4,887 tok/s at
concurrency 32) — the batched GEMM is the remaining gap. The shell/e8p codebooks decode
slower than trellis; their draw is the 2–8 bpw range and mixed precision. A
weight-quantization method's headline win remains **footprint** — the freed VRAM as KV /
longer-context headroom, and fitting models bf16 can't — but at 4 bpw trellis the
single-user speed cost of that footprint is now ~zero on the hardware we measured.

### Matched 4-bit head-to-head (measured)

Numbers behind the matrix: GLQ vs AWQ vs NVFP4 on **one** base model (`gemma-4-26B-A4B-it`, an
MoE), **all at ~4-bit** so footprint is directly comparable, on a single RTX PRO 6000 Blackwell
(vLLM 0.23, seed 0, thinking mode). bf16 is the uncompressed ceiling.

| Method | bits | Weights, exact on disk¹ | MMLU-Pro n=60² | AIME-2026 n=30² |
|---|--:|--:|--:|--:|
| bf16 (ceiling) | 16 | ~50 GB | 91.7% | 90.0% |
| **GLQ 4 bpw** | 4.0 | **14.97 GB** | **93.3%** | **90.0%** |
| AWQ 4-bit | ~4.25 | 17.19 GB | 86.7% | 83.3% |
| NVFP4 (W4A4) | ~4 | 18.78 GB | 86.7% | **90.0%** |

**GLQ has the smallest footprint, the top MMLU-Pro, *and* matches the bf16 ceiling on hard
reasoning** — AIME-2026 GLQ 90.0% = bf16 90.0% = NVFP4 90.0% > AWQ 83.3% (GLQ reproduces bf16's
27/30 exactly, at <1/3 the footprint); MMLU-Pro is saturated (all 86.7–93.3%, inside the ±8% n=60
band → a tie). The trade is **decode speed**: GLQ ran the same 30-problem AIME in **58 min** vs
~24 min for AWQ/NVFP4 and ~32 min for bf16 — its W4A16 kernel is L2-codebook-gather-bound on
Blackwell, where NVFP4's native FP4 tensor cores, AWQ's Marlin, and even bf16 all decode faster.
So GLQ's win is **footprint + quality, not speed** — the pick when you are memory-bound. Full
tables, truncations, chain lengths, and caveats:
[`benchmarks/_quant_compare_gemma4_26b.md`](benchmarks/_quant_compare_gemma4_26b.md).

<sub>¹ Exact safetensors weight bytes summed from tensor headers (tower-independent — all three ship
the identical 1.15 GB vision tower); loaded footprints track this ordering (GLQ ~14.0 < AWQ 15.6 <
NVFP4 17.1 GiB). ² Thinking mode, single-sample pass@1, seed-fixed subsets; n=60 MMLU-Pro 95% CI ≈
±8%, n=30 AIME ≈ ±15% — a fidelity comparison on one model / one GPU, not a leaderboard. NVFP4 is
W4A4 (4-bit activations) and defaults to fp8 KV, a speed edge the W4A16 methods don't take;
calibration differs per vendor. The paired GLQ-vs-bf16 view (incl. AIME-2024) is in
[Quality & footprint](#quality--footprint) below.</sub>

### Trellis 3INST vs bf16 vs NVFP4 (measured)

The same exercise for the **trellis-3INST** format on a dense model: SmolLM3-3B, one
RTX PRO 6000 Blackwell, vLLM 0.25.0, glq 0.7.1. Speed = `vllm bench sweep serve`
(random 128-in/256-out, ignore-eos, seed 42, mean of 3 runs); quality = wikitext-2 PPL +
AIME-2026 in thinking mode (32k budget, 8 samples/problem, avg@8).

| | **GLQ trellis-3INST 4 bpw** | bf16 | NVFP4 (W4A4)¹ |
|---|--:|--:|--:|
| Weights on disk | **1.9 GB** | 5.8 GB | 2.5 GB |
| Decode, 1 stream (tok/s) | **176** | 180 | 301 |
| Decode, 32 streams (tok/s) | 2,423 | 4,887 | 7,600 |
| PPL (wikitext-2) | 9.23 | 9.12 | n/a² |
| AIME-2026 avg@8 (thinking) | **41.7%** | 47.5% | 32.5% |
| 240-generation batch job, wall-clock³ | 52 min | 36 min | 32 min |

**The read**: at 4 bpw trellis, GLQ decodes **single-stream at bf16 parity** in a third of
the memory, and keeps most of the reasoning quality (−5.8 pts AIME vs bf16) where this
NVFP4 checkpoint's 4-bit activations cost −15 pts. NVFP4 is decisively faster — but its
FP4 tensor cores exist **only on Blackwell**; on the 24 GB 3090-class (sm_86) / 4090-class
(sm_89) cards GLQ targets, NVFP4 cannot use FP4 hardware, while GLQ's fp16-mma kernels have
no Blackwell dependency (core GLQ kernels validated on sm_86 / sm_89 / sm_120; the trellis
kernels to date on sm_120).

<sub>¹ [`Firworks/SmolLM3-3B-nvfp4`](https://huggingface.co/Firworks/SmolLM3-3B-nvfp4), a
community w4a4 checkpoint — results are scoped to it, not to NVFP4 at large. ² The
checkpoint does not load in HF transformers; quality measured via vLLM (AIME row).
³ 30 AIME problems × 8 samples at up to 32k tokens each, launched as one batch — fixed
load/compile overhead and the low-concurrency straggler tail compress the steady-state
speed ratios (3.1× at c=32 becomes 1.6× on the real job).</sub>

## Quality & footprint

### SmolLM3-3B at matched 4.5 bpw vs GPTQ

Blackwell RTX PRO 6000, 128 calibration samples,
`lm-evaluation-harness` limit=200/task (GSM8K n=500, MMLU 50/subtask).
GLQ 4.5 bpw uses two-pass mixed allocation (91 layers @ 4 bpw + 161 @
5 bpw, avg 4.64 bpw).

| Task                     | bf16   | **GLQ 4.5 bpw** | GPTQ W4 g128 |
|--------------------------|--------|-----------------|--------------|
| ARC-challenge (acc_n)    | 0.490  | **0.475**       | 0.420        |
| ARC-easy (acc_n)         | 0.745  | **0.735**       | 0.695        |
| HellaSwag (acc_n)        | 0.660  | 0.660           | **0.675**    |
| MMLU (acc)               | 0.617  | **0.603**       | 0.589        |
| TruthfulQA mc2           | 0.529  | **0.545**       | 0.515        |
| WinoGrande               | 0.655  | 0.660           | **0.670**    |
| WikiText-2 ppl ↓         | 10.67  | **10.90**       | 11.33        |
| GSM8K flex (n=500)       | 0.722  | **0.738**       | 0.688        |
| IFEval prompt-strict     | 0.310  | 0.310           | 0.285        |
| IFEval prompt-loose      | 0.325  | **0.330**       | 0.295        |
| IFEval inst-strict       | 0.478  | 0.472           | 0.453        |
| IFEval inst-loose        | 0.494  | 0.491           | 0.469        |

GLQ beats GPTQ on 10/12 metrics. WikiText-2 ppl gap to bf16: +2.2 %
(GLQ) vs +6.2 % (GPTQ). GSM8K flex matches bf16; GPTQ drops 0.034.

### Small models: SmolLM2-360M-Instruct at 4 bpw

GPTQ requires a group-size dividing the hidden dim; SmolLM2-360M's
hidden=960 is not divisible by 128, forcing `group_size=64` (~4.5 eff
bpw) and losing quality. GLQ has no group-size constraint.

| Method        | bpw stored | 5-task avg | % of bf16 |
|---------------|------|------------|-----------|
| bf16          | 16.0 | 0.557      | 100 %     |
| **GLQ 4-bit block-diagonal** (current) | 4.0 | — | **97.9 %**¹ |
| GLQ 4-bit legacy (padded) | ~6.4 eff¹ | 0.555 | 99.6 % |
| GPTQ W4 (g64) | ~4.5 | 0.486      | 87.2 %    |

5-task = ARC-e, HellaSwag, PIQA, WinoGrande, LAMBADA; 128 calibration
samples; L40S. GPTQ's LAMBADA collapses to 0.346; GLQ preserves 0.508.

<sub>¹ The original "4 bpw" checkpoint predates block-diagonal FHT and stored its
power-of-2 padding as real bits — effectively ~6.4 bits/weight, so its 99.6 % was
earned with extra storage. The true-4-bpw re-quant
([`SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw`](https://huggingface.co/xv0y5ncu/SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw))
scores 97.9 % of bf16 on the same 5 tasks (paired GLQ/bf16 runs on the vLLM stack)
— still well clear of GPTQ. The group-size constraint on GPTQ applies regardless.</sub>

### Gemma-4 family — GLQ vs bf16, paired (thinking mode)

Each GLQ checkpoint was run **head-to-head against its bf16 base on the same
questions**, in thinking mode (these are reasoning models), on a single RTX PRO 6000
Blackwell with vLLM 0.23. These are **small-n fidelity** comparisons (95% CI ≈ ±8%
MMLU-Pro, ±15% AIME), single-sample pass@1 — not leaderboard scores.

| Model (GLQ vs bf16) | bpw | Footprint GLQ / bf16 | MMLU-Pro n=60 | AIME-2024 n=30 |
|---|--:|--:|--:|--:|
| Gemma-4-31B-it | 5.0 mix | **16.5** / 57.9 GiB | 90.0% vs 86.7% | 90.0% vs 86.7% |
| Gemma-4-26B-A4B-it (MoE) | 4.0 | **~15** / ~50 GiB | 93.3% vs 91.7% | 93.3% vs 93.3% |
| Gemma-4-12B-it | 5.0 mix | **6.9** / 24 GiB | 81.7% vs 78.3% | 83.3% vs 93.3%† |

On these runs GLQ is **within noise of bf16** — ahead on MMLU-Pro for all three,
ahead/tied on AIME-2024 for the 31B and 26B-A4B. †At n=30 the 12B AIME-2024 gap
(bf16 +3 items) is not statistically significant; likewise the 31B's AIME-2026
(GLQ 83.3% vs bf16 90.0%, bf16 +2 items). Footprint is the consistent win — a 31B in
16.5 GiB fits one 24–32 GB card where bf16 (~58 GiB) needs three. See each model card
for the full paired tables, thinking budgets, and caveats.

For decode speed vs bf16 on the current stack, see
[How GLQ compares](#how-glq-compares) — single-stream trellis-3INST at bf16
parity, bf16 ahead at batch (measured numbers there).

## How it works

1. **E8 lattice codebook.** 65,536 vectors from the first seven shells
   of the E8 lattice in 8 dimensions. Each 8-weight group of the weight
   matrix is encoded as one 16-bit index into this codebook (so the
   primary stage is 2 bpw). For 3–8 bpw, additional 8-bit (256-entry)
   or 16-bit (E8) residual codebooks refine the primary's
   reconstruction error.

2. **Randomized Hadamard Transform.** Random sign flips followed by
   Fast Walsh-Hadamard Transform rotate both weights and Hessian.
   After RHT the Hessian is approximately diagonal, so plain Euclidean
   nearest-neighbour in the codebook is near-optimal under the
   Hessian-weighted proxy loss.

3. **LDLQ error feedback.** Block-LDL decomposition of the Hessian
   drives a sequential sweep — GPTQ-style, but over 8-D blocks instead
   of scalar columns. Each block's quantization error propagates
   forward to correct downstream blocks.

4. **Fused inference kernels.** Custom CUDA C and Triton kernels read
   codebook indices from HBM, gather the 8-D vectors from the
   L2-cached 1 MB codebook, and accumulate the matmul directly — the
   dense weight matrix is never materialized. GPU memory savings scale
   with the compression ratio.

## KV cache compression

GLQ ships two KV cache compressors. Either is opt-in — default
behaviour is unchanged.

### INT8 cache (HF transformers)

Per-channel absmax INT8 plus a small fp16 residual window for recent
tokens — KIVI-style. Halves the KV memory at long context.

```python
import glq.hf_integration
from glq.kv_cache import GLQQuantizedCache

cache = GLQQuantizedCache(model.config)
output = model.generate(**inputs, max_new_tokens=200,
                         past_key_values=cache)
```

Requires `transformers >= 4.45`. No external dependencies.

### E8 lattice cache (vLLM, v0.3.0+)

Drops vLLM's paged KV cache to **as little as ~25 % of fp16 footprint**
(recipe-dependent) using the same E8 lattice quantizer used for weights. Two fused Triton kernels
(read-side dequant-gather, write-side scatter) keep decode within
~20 % of un-fused throughput.

Measured on Gemma-4-E4B-it, RTX PRO 6000 Blackwell, vLLM 0.20:

| | fp16 baseline | E8 lattice |
|---|---|---|
| KV cache capacity @ 27.9 GiB | 303,984 tokens | **1,221,232 (4.02×)** at `e8_relaxed:1` |
| mmlu_pro n=240 accuracy | 71.25 % | **71.25 % (bit-identical)** at `e8_relaxed:2` |
| NIAH passkey @ ctx=16k / 32k / 64k / 130k | — | **40/40** at `e8_relaxed:2` (full 128k window) |
| `cudaLaunchKernel` per decode | 110,659 | **71,619 (−35 %)** at `e8_relaxed:2` |

Note the capacity row was measured with the `e8_relaxed:1` recipe and the
quality rows with `e8_relaxed:2` — the recipes trade capacity for fidelity,
so pick one and measure both for your model.

Activation:

```bash
GLQ_KV_QUANT=e8_relaxed:2 \
GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 \
GLQ_KV_E8_COMPRESSED_ALLOC=1 \
GLQ_KV_E8_FUSED_GATHER=1 GLQ_KV_E8_FUSED_WRITE=1 \
vllm serve google/gemma-4-E4B-it
```

The envs above use the **workspace path**: GLQ pre-decompresses the
referenced K/V into a scratch buffer, then calls vLLM's stock
attention. Because that buffer is built with a data-dependent
`block_table.unique()`, glq auto-forces `cudagraph_mode=PIECEWISE`
for this path (you'll see `[glq_vllm] E8 KV active → cudagraph_mode
forced ... to PIECEWISE` at startup; `--enforce-eager` is no longer
required as of v0.3.5). Weight-only GLQ still uses the default
`FULL_AND_PIECEWISE`. The v0.5 **inline-dequant path below** lifts
the PIECEWISE restriction and is the recommended path for
long-context / KV-bound serving.

Validated end-to-end on Gemma-4-E4B-it / Gemma-4-31B-it on vLLM
0.20.x.

### Inline-dequant E8 KV (default in v0.5.1)

The workspace path above pre-decompresses K/V into a scratch buffer
that vLLM's attention then re-reads — pure overhead, since each K/V
vector is read exactly once. The **inline-dequant path** instead
dequantizes the compressed E8 K/V *inside* a forked Triton attention
kernel (an 8-point FHT butterfly for the inverse Hadamard, plus
flash-decoding KV-split for long-context occupancy). There is no
workspace, and — because the read/write hooks are host-sync-clean —
the **FULL CUDA graph captures the whole decode**, eliminating the
per-token eager-dispatch overhead that dominated E8-KV decode.

**As of v0.5.1 this is the default** for the E8-KV path — the standard
bundle is all you need (no extra flag):

```bash
GLQ_KV_QUANT=e8_relaxed:2 \
GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 \
GLQ_KV_E8_COMPRESSED_ALLOC=1 \
GLQ_KV_E8_FUSED_GATHER=1 GLQ_KV_E8_FUSED_WRITE=1 \
vllm serve xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw
```

Opt out with `GLQ_KV_E8_INLINE_DEQUANT_V3=0` (reverts to the 65 K
workspace path) or `GLQ_KV_E8_FORCE_PIECEWISE=1` (keeps inline but
disables the FULL decode graph).

Decode throughput, SmolLM3-3B-GLQ-3.5bpw, RTX PRO 6000 Blackwell,
vLLM 0.20.2 — inline vs the pre-v0.5 E8-KV path (workspace,
PIECEWISE):

| | E8 KV before v0.5 | inline (v0.5) |
|---|---:|---:|
| decode B=1 | ~15 tok/s | **38 (2.5×)** |
| decode B=4 | ~37 | **127 (3.4×)** |
| decode @ ctx=16k, B=1 | ~15 | **36 (2.4×)** |

The speedup is the FULL-graph capture the inline path unlocks; it
brings E8-KV decode to roughly weight-only parity. On Gemma-4-E4B-it
(large heads, already compute-bound) decode is roughly unchanged,
but quality and long-context behaviour match.

**Quality is neutral.** On SmolLM3 the inline-FULL path is
*bit-identical* to PIECEWISE (MMLU-Pro n=120 and NIAH-16k match
exactly). On Gemma-4 it lands within vLLM's own run-to-run greedy
non-determinism — MMLU-Pro n=120, thinking, 16384-token budget:
PIECEWISE 0.742 vs inline-FULL 0.750 (a smaller gap than two
PIECEWISE runs differ from each other), NIAH-16k 10/10 both.

**Scope.** It covers the **4 bpw** KV recipe (`e8_relaxed:2`); other
recipes automatically fall back to the workspace path. It requires the
Triton attention backend (auto-forced when E8 KV is active). Validated
across the consumer GPU lineup — A10G (`sm_86`, 3090-class, 24 GB),
L40S (`sm_89`, 4090-class), and RTX PRO 6000 Blackwell (`sm_120`,
5090-class): the kernels compile and NIAH-16k + MMLU are correct on all
three, and FULL-vs-PIECEWISE is quality-neutral on Blackwell (the
consumer-card runs are shorter FULL-only smokes). Opt out per above.

## Advanced

### CUDA-graph decode wrapper

The B=1 autoregressive decode path is Python-dispatch-bound in eager
mode. `CUDAGraphWrapper` captures the fixed-shape decode and replays
it; benchmarks below are on SmolLM3-3B 3.5bpw, L40S.

| Mode       | GLQ 3.5 bpw | bf16   |
|------------|-------------|--------|
| Eager      | 25 tok/s    | 40     |
| CUDA graph | 37 tok/s    | 40     |

```python
from glq.cuda_graph import CUDAGraphWrapper
wrapper = CUDAGraphWrapper(model)
logits = wrapper(input_ids)   # first call captures; replays after
```

The wrapper falls back to eager for variable shapes (prefill, batch>1,
extra kwargs). For 24B models the matmul is compute-bound at B=1, so
graphs don't help (Devstral-24B GLQ 4 bpw: 6.6 tok/s eager vs 6.4
graphed).

### Tuning vLLM CUDA-graph capture sizes (v0.3.4+)

vLLM 0.20 captures both **FULL** model-forward graphs (single replay
per fixed shape) and **PIECEWISE** subgraphs split at attention. The
default capture set is derived from `max_num_seqs * 2`, so a
single-sequence harness only gets FULL captures for `[1, 2]`. For
batched serving, raise the list explicitly:

```python
from vllm import LLM
llm = LLM(model="xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw",
          compilation_config={
              "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
          })
```

Measured impact on Gemma-4-E4B-it-GLQ-4bpw, RTX PRO 6000 Blackwell,
256-token decode:

| Mode | B=1 tok/s | B=4 tok/s (total) |
|---|---:|---:|
| Eager | 14.4 | 35.0 |
| Piecewise + default capture `[1, 2]` | 39.4 | 132.7 |
| Piecewise + capture `[1, 2, 4, 8, 16]` | **40.0** | **157.3 (+18.5 %)** |

At B=1 the FULL graph was already captured (no change). At B=4 the
extended list keeps the FULL graph active where the default
degenerated to PIECEWISE-only, recovering ~6 tok/s per sequence.

Cost: ~10-20 MB VRAM per captured shape on 3B / E4B models (vLLM
prints the total at "Graph capturing finished in N s, took X GiB").
On 24-31B models budget ~100-200 MB per shape. Capture time is
~1 s per shape, one-time at LLM init.

### Bit widths

| bpw | Primary | Residual stages |
|-----|---------|-----------------|
| 2   | 16 b    | —                |
| 3   | 16 b    | + 8 b            |
| 4   | 16 b    | + 16 b           |
| 5   | 16 b    | + 16 b + 8 b     |
| 6   | 16 b    | + 16 b + 16 b    |
| 7   | 16 b    | + 16 b + 16 b + 8 b |
| 8   | 16 b    | + 16 b + 16 b + 16 b |

One global scale per layer; no group-size parameter. Non-power-of-2
hidden sizes use block-diagonal FHT (v0.2.9+) — e.g. 2688 is
decomposed as `2048 + 512 + 128` so on-disk storage matches the
nominal rate exactly.

The table above applies to the shell and e8p codebooks. The
[trellis codebook](#trellis-codebook---codebook-trellis--qtip-derived-tcq)
is a single K-bit trellis code with no residual stages — integer
2 / 3 / 4 bpw only.

### Trellis codebook (`--codebook trellis`) — QTIP-derived TCQ

`--codebook trellis` replaces the per-8-weight lattice lookup with **trellis-coded
quantization** (TCQ) over 256-weight sequences, following
[QTIP](https://github.com/Cornell-RelaxML/qtip) (Tseng et al., 2024): a Viterbi search
encodes each row against a tail-biting trellis, so neighbouring weights share state and
the effective codebook is exponentially larger than a flat lookup at the same rate. In our
SmolLM3-3B tests it is GLQ's best format at **2–4 bpw** — at 2 bpw it clearly beats the
e8p codebook (PPL 11.74 vs 13.21) — and at 4 bpw its decode is GLQ's fastest
(single-stream at bf16 parity, see
[the measured table](#trellis-3inst-vs-bf16-vs-nvfp4-measured)).

**Variants.** `GLQ_TRELLIS_VARIANT=3inst` decodes each 16-bit trellis state
*arithmetically* (a hash + two fp16 halves — "3 instructions", no lookup table), which is
what the fused fast path is built around; the default `hyb` uses a 512-entry lookup
table and exists for back-compat with earlier hyb checkpoints. Quality measured
equal-or-slightly-better for 3INST in our paired tests — **use 3INST for new quants**.

**What the v0.7.1 kernels do** (the 129 → 179 tok/s single-stream jump): the input
Hadamard transform and fp16 cast run *inside* every decode block instead of as separate
1-block launches; the five lowest FHT butterfly stages run as warp shuffles; and the
per-shard output transforms of fused QKV / gate-up layers batch into single launches.
All bit-exact — wikitext-2 PPL is unchanged to the fourth decimal across the entire
optimization series. Runtime opt-outs, should you ever need the unfused paths:
`GLQ_TRELLIS_FUSE_INPUT=0`, `GLQ_TRELLIS_BATCH_OUT_RHT=0`.

**Storage layout.** Trellis checkpoints store indices in the decoder-native "kernel"
layout (`trellis_layout: "kernel"` in the config) — loading them requires
**glq ≥ 0.7.0**; older versions abort with a layout error rather than decode garbage.
Rates are integer 2 / 3 / 4 bpw with a single trellis code per layer (no residual
stages, no mixed precision — both are rejected at quantize time). Quantization cost is
the Viterbi encode: ~35 min for a 3B on one GPU (CUDA-graph-cached).

### E8P codebook (`--codebook e8p`) — derivative of QuIP#

`--codebook e8p` is **derivative work that ports the E8P codebook and its
tensor-core decode kernels from
[QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)** (Tseng et al., 2024).
It swaps the default E8-shell codebook for QuIP#'s 2-bit **E8P** (padded-D̂8)
grid decoded on tensor cores (`mma.sync` GEMV), plus residual vector
quantization for the higher rates. The port is self-contained (no `quiptools`
dependency); full credit and the citation are in
[Acknowledgments](#acknowledgments).

| bpw | RVQ recipe (stages)         | added per stage |
|-----|-----------------------------|-----------------|
| 2   | `[E8P]`                     | 16-bit E8P (primary) |
| 3   | `[E8P, E81B]`               | + 8-bit E81B residual |
| 4   | `[E8P, E8P]`                | + 16-bit E8P residual |
| 5   | `[E8P, E8P, E81B]`          | + 8-bit E81B |
| 6   | `[E8P, E8P, E8P]`           | + 16-bit E8P |
| 7   | `[E8P, E8P, E8P, E81B]`     | + 8-bit E81B |
| 8   | `[E8P, E8P, E8P, E8P]`      | + 16-bit E8P |

Each E8P stage is a 16-bit tensor-core decode (`mma.sync` GEMV, +2 bpw); odd
bit-widths end in a single 8-bit **E81B** residual (WMMA lookup-matmul, +1 bpw).

```bash
glq-quantize --model HuggingFaceTB/SmolLM2-360M --output ./out \
    --codebook e8p --bpw 4 --nsamples 128
```

The entire linear — input RHT → E8P tensor-core decode (all RVQ stages) →
×Wscale → output RHT — runs as a **single fused CUDA op**
(`glq_fused_linear_e8p_cuda`), so vLLM captures B=1 decode in a **FULL CUDA
graph** exactly like the default path (HF inference is supported too). On
SmolLM3-3B 4 bpw (RTX PRO 6000 Blackwell, vLLM 0.23) that fused op runs B=1
decode at **~85 tok/s**; collapsing the per-linear dispatch into one opaque op
is what makes cudagraph a win here — the equivalent unfused multi-op path is
~7× slower under capture. **The full 2–8 bpw range serves on both HF and vLLM
as of v0.6.7** (the 5–8 bpw N-stage decode landed there); the fused op is
bit-exact against the unfused reference across all of 2–8 bpw (decode reproduces
the quantize-side weight to ~66–68 dB SQNR per bit-width).

### Devstral-24B tokenizer

`transformers` 5.x auto-routes Mistral/Devstral models through
`mistral_common`, which rejects the standard `tokenizer.json`. Use
`PreTrainedTokenizerFast` explicitly:

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

path = snapshot_download("xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw")
tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
tok.pad_token, tok.eos_token, tok.bos_token = "<pad>", "</s>", "<s>"
model = AutoModelForCausalLM.from_pretrained(
    "xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw",
    device_map="cuda", dtype="float16",
)
```

`examples/inference_hf.py` includes a `load_tokenizer()` helper that
handles this automatically.

### transformers compatibility

For models ≤ 1B parameters use `transformers >= 5.0`. Transformers
4.57.x has a weight-loading bug that produces garbage output for small
GLQ models. Larger models (3B+) work with both 4.x and 5.x.

## Inference kernels

`glq/inference_kernel.py` + `glq/csrc/glq_cuda.cu` provide CUDA C and
Triton kernels that compute `Y = X @ dequant(W)^T` without
materializing the weight matrix. Each kernel iterates over `N/8`
codebook blocks per output row, gathers 8-D vectors from the L2-cached
codebook, and accumulates the matmul directly against indices.

| Path | When | Notes |
|------|------|-------|
| **CUDA C Tensor Core** | B ≥ 2 (prefill) | inline PTX `mma.sync` against codebook-loaded registers; 3-5× faster than Triton |
| **CUDA C split-K matvec** | B = 1 (decode) | 4 rows/warp + `__shfl_xor_sync` reduction; 2.7× faster than Triton |
| **CUDA C shared-mem FHT** | RHT step | double-buffered butterfly; low 5 stages as warp shuffles (v0.7.1) |
| **Trellis 3INST fused linear** | trellis checkpoints, B = 1 | lookup-free decode with the input RHT + cast computed in-block — one kernel per linear (v0.7.1) |
| **Shard-batched output RHT** | trellis, fused QKV / gate-up | per-shard output transforms in one `grid.y` launch (v0.7.1) |
| **Triton fallback** | no `ninja`, or `n_pad > 32 768` | always available |

**Bit-exact determinism.** Every kernel reduces partial sums in a fixed
order (scratch-buffer split-K or in-block reduction) instead of
`atomicAdd` across k-splits, so running the
same prompt at B=1 decode or B=8 prefill produces identical logits
across runs — required for reproducible lm-eval scoring and
on-policy RL rollouts.

Direct kernel access:

```python
from glq.inference_kernel import glq_dequant_matmul
y = glq_dequant_matmul(x, Qidxs, codebook, Wscale,
                       Qidxs2=Qidxs2, codebook2=codebook2,
                       inv_resid_scale=inv_rs)  # 3/4 bpw two-stage
```

## Architecture

```
glq/
  codebook.py          # E8ShellCodebook: enumeration, encode/decode
  hadamard.py          # Fast Walsh-Hadamard Transform
  rht.py               # Randomized Hadamard Transform
  ldlq.py              # Block-LDL quantization with error feedback
  quantize_model.py    # Full model pipeline + CLI
  quantized_linear.py  # E8RHTLinear: drop-in nn.Linear replacement
  inference_kernel.py  # Triton kernels + CUDA dispatch
  csrc/glq_cuda.cu     # CUDA C kernels (split-K matvec, TC, FHT)
  hf_integration.py    # HuggingFace Transformers integration
  kv_cache.py          # INT8 quantized KV cache
  cuda_graph.py        # B=1 decode wrapper
glq_vllm/              # vLLM integration: weight + KV cache (v0.3.0+)
```

## Acknowledgments

This project builds on **[QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)**
(Tseng et al., 2024). The default E8-shell pipeline is *inspired by* its
Hadamard-incoherence + lattice-codebook formulation, and the optional
`--codebook e8p` path is **derivative work that ports QuIP#'s E8P (padded-D̂8)
codebook and its tensor-core decode / residual kernels** — the `grid_packed_abs`
codebook plus the `decode_matvec_e8p`, `decompress_packed_e8p`, and E81B
lookup-matmul kernels in [`glq/csrc/glq_e8p.cu`](glq/csrc/glq_e8p.cu). All credit
for the E8P codebook and those kernels belongs to the QuIP# authors; this
repository is an independent port, not an official QuIP# release. See the
[QuIP# repository](https://github.com/Cornell-RelaxML/quip-sharp) for the paper
and citation.

The `--codebook trellis` path is likewise **derivative of
[QTIP](https://github.com/Cornell-RelaxML/qtip)** (Tseng et al., 2024): the tail-biting
trellis formulation, the hybrid lookup codebook, the "3INST" lookup-free decode idea, and
the bit-unpack structure of the kernels in
[`glq/csrc/glq_trellis.cu`](glq/csrc/glq_trellis.cu) all originate there — GLQ grafts
them onto its own RHT + LDLQ pipeline and fused-linear kernel architecture. All credit
for TCQ and the 3INST decode belongs to the QTIP authors; see the
[QTIP repository](https://github.com/Cornell-RelaxML/qtip) for the paper and citation.

Other foundations:

- E8 lattice: Korkin & Zolotarev (1872); Gosset (1900); Conway & Sloane, *Sphere Packings, Lattices and Groups*; [Viazovska (2016)](https://arxiv.org/abs/1603.04246) — sphere-packing optimality in 8 dimensions.
- Block-feedback quantization: [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022).
- INT8 KV cache: [KIVI](https://arxiv.org/abs/2402.02750) (Liu et al., 2024).

## License

GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE).
