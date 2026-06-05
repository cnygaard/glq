# GLQ

Post-training weight quantization for LLMs using **E8 lattice codebooks**.

GLQ encodes each 8-weight group as a 16-bit index into a 65,536-entry
E8 lattice codebook. A Randomized Hadamard Transform (RHT) decorrelates
the Hessian so that Euclidean nearest-neighbour search is near-optimal
under the proxy loss. The result: 2–8 bpw weights with quality
comparable to QuIP# / better than GPTQ, and a fused CUDA kernel that
matmuls directly against the compressed indices without materializing
the weight matrix.

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
    "xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw")
print(tok.decode(model.generate(
    **tok("The capital of France is", return_tensors="pt").to(model.device),
    max_new_tokens=20,
)[0], skip_special_tokens=True))
```

`import glq.hf_integration` registers `quant_method="glq"` with HF
Transformers; `from_pretrained` then swaps `nn.Linear` for `E8RHTLinear`
and uses the fused CUDA C kernel on inference. CPU falls back to a
naive dequantize-then-matmul.

### Available pre-quantized checkpoints

| Repo | Base model | bpw | License | VRAM¹ | Tok/s² (b1 / b32) |
|---|---|---|---|--:|--:|
| [`xv0y5ncu/SmolLM2-135M-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/SmolLM2-135M-Instruct-GLQ-4bpw) | SmolLM2-135M-Instruct | 4.0 | Apache 2.0 | 0.18 | 152 / 4205 |
| [`xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw) | SmolLM2-360M-Instruct | 4.0 | Apache 2.0 | 0.33 | 135 / 2990 |
| [`xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw`](https://huggingface.co/xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw) | SmolLM3-3B | 3.5 (mixed) | Apache 2.0 | 2.4 | 35 / 654 |
| [`xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw) | Gemma-4-E4B-it | 4.0 | Apache 2.0 | 5.8 | 33 / 600 |
| [`xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw) | Devstral-Small 24B | 4.0 | Apache 2.0 | ~20.5 | 6.6 / — |
| [`xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-4bpw) | Nemotron-3-Nano-30B (Mamba-MoE) | 4.0 | Nemotron | — | — |

<sub>¹ **VRAM** = resident weight footprint at load (vLLM's `Model loading took … GiB`)
on a **g6e.xlarge** (NVIDIA L40S) — the figure that decides whether a model fits a
24/32 GB card; it tracks the bpw budget. Devstral ≈ 20.5 GiB is its HF-transformers
load; Nemotron-30B not measured here.<br>
² **Tok/s** = total decode throughput, weight-only GLQ, vLLM 0.20.2, short context
(256 generated tokens), same hardware. **b1** = single-stream, **b32** = 32 concurrent
sequences — a high-batch sample near the throughput knee, not a hard maximum.
Devstral-24B is HF-transformers single-stream (vLLM v1 deadlocks; no batched figure;
see [CUDA-graph decode wrapper](#cuda-graph-decode-wrapper)). Nemotron-3-Nano-30B is a
Mamba-MoE (vLLM-unsupported here, compute-bound) — not benchmarked. E8-KV compression
leaves these short-context numbers unchanged; its payoff is a ~4× smaller KV cache →
more context / concurrency in the same VRAM (see [KV cache compression](#kv-cache-compression)).</sub>

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

## Docker image (NVIDIA GPU)

A prebuilt CUDA image ships everything needed to run GLQ models —
`glq`, PyTorch, vLLM, transformers, and lm-eval on CUDA 12.8:

```
ghcr.io/cnygaard/glq-env:0.5.0     # also :latest, :0.5
```

**Prerequisite — GPU access in Docker.** You need an NVIDIA GPU plus
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host; that's what makes the `--gpus all` flag pass
the GPU into the container. Verify it works:

```bash
docker run --rm --gpus all ghcr.io/cnygaard/glq-env:0.5.0 nvidia-smi
```

If that prints your GPU table, you're set. (No toolkit → `--gpus`
errors with "could not select device driver".)

**Produce output.** Mount a host directory for the model cache (the
image's `HF_HOME` is `/cache/hf`, so models persist across runs
instead of re-downloading), then generate:

```bash
docker run --rm --gpus all \
    -v "$HOME/.cache/huggingface:/cache/hf" \
    ghcr.io/cnygaard/glq-env:0.5.0 \
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
# `tool.jinja` is your tool-calling chat template — keep it in the current
# directory; it is mounted into the container at /work/tool.jinja below.
docker run --rm --gpus all -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/cache/hf" \
    -v "$PWD/tool.jinja:/work/tool.jinja:ro" \
    ghcr.io/cnygaard/glq-env:latest \
    vllm serve xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw \
        --max-model-len 64000 \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --chat-template /work/tool.jinja
```

A minimal serve is just `vllm serve <glq-model>`; the flags above add 64k
context plus Gemma-4 tool-calling and reasoning parsing. Two gotchas: pass
`--chat-template` the **in-container** mount path (`/work/tool.jinja`), not the
host path; and the `gemma4` tool-call/reasoning parsers require a vLLM build
that registers them.

> **Image vLLM note:** in-image vLLM serving needs an image built from the
> **v0.5.2 Dockerfile fix or later** (vLLM now resolves its own matching-CUDA
> torch). The published `:0.5.0` / `:0.5.1` snapshots predate that fix and hit
> `ImportError: libcudart.so.13` under `--gpus all`; on those, use the HF
> generate path above (works), or `pip install glq` + your own vLLM. The pip
> package is unaffected on all versions.

For the long-context E8 KV-cache flags (`GLQ_KV_*`), pass them with
`-e` and see [E8 lattice cache](#e8-lattice-cache-vllm-v030) /
[Inline-dequant E8 KV](#inline-dequant-e8-kv-default-in-v051).
The image's default command is a shell (`docker run --rm -it --gpus all
ghcr.io/cnygaard/glq-env:latest`) if you'd rather poke around interactively.

## Results

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

| Method        | bpw  | 5-task avg | % of bf16 |
|---------------|------|------------|-----------|
| bf16          | 16.0 | 0.557      | 100 %     |
| **GLQ 4-bit** | 4.0  | **0.555**  | **99.6 %** |
| GPTQ W4 (g64) | ~4.5 | 0.486      | 87.2 %    |

5-task = ARC-e, HellaSwag, PIQA, WinoGrande, LAMBADA; 128 calibration
samples; L40S. GPTQ's LAMBADA collapses to 0.346; GLQ preserves 0.508.

### Throughput: SmolLM3-3B on vLLM

GLQ runs at near-bf16 throughput because compressed weights cut DRAM
bandwidth enough to roughly offset the dequantization cost.

| Method         | bpw  | Single req     | Batch=5        | vs bf16 |
|----------------|------|----------------|----------------|---------|
| bf16           | 16.0 | 39.4 tok/s     | 184 tok/s      | 100 %   |
| **GLQ 3.5bpw** | 3.5  | **37.1 tok/s** | **173 tok/s**  | **94 %** |
| GPTQ W4 (g128) | ~4.5 | 34.6 tok/s     | 172 tok/s      | 88 %    |

vLLM 0.18.1, L40S.

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

Drops vLLM's paged KV cache to **~25 % of fp16 footprint** using the
same E8 lattice quantizer used for weights. Two fused Triton kernels
(read-side dequant-gather, write-side scatter) keep decode within
~20 % of un-fused throughput.

Measured on Gemma-4-E4B-it, RTX PRO 6000 Blackwell, vLLM 0.20:

| | fp16 baseline | E8 lattice |
|---|---|---|
| KV cache capacity @ 27.9 GiB | 303,984 tokens | **1,221,232 (4.02×)** at `e8_relaxed:1` |
| mmlu_pro n=240 accuracy | 71.25 % | **71.25 % (bit-identical)** at `e8_relaxed:2` |
| NIAH passkey @ ctx=16k / 32k / 64k / 130k | — | **40/40** at `e8_relaxed:2` (full 128k window) |
| `cudaLaunchKernel` per decode | 110,659 | **71,619 (−35 %)** at `e8_relaxed:2` |

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

### Serving with sglang

A fork of sglang with GLQ support lives at
[`cnygaard/sglang`](https://github.com/cnygaard/sglang) on the
`glq-quantization` branch. It registers `"glq"` as a quantization
method and reuses the existing `glq.inference_kernel` CUDA extension
as a runtime dependency.

```bash
git clone -b glq-quantization https://github.com/cnygaard/sglang
cd sglang/python && pip install -e .

python -m sglang.launch_server \
    --model xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw \
    --tokenizer-path HuggingFaceTB/SmolLM2-360M-Instruct \
    --quantization glq \
    --attention-backend triton --sampling-backend pytorch
```

Requires the triton attention backend (flashinfer returns wrong
logprobs in echo/prefill mode). Default CUDA-graph capture is
supported (v0.3.2+). If you hit a graph-break in a model architecture
we haven't tested, pass `--disable-piecewise-cuda-graph` as a
fallback.

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
| **CUDA C shared-mem FHT** | RHT step | double-buffered butterfly; 1.6-3× faster than Triton |
| **Triton fallback** | no `ninja`, or `n_pad > 32 768` | always available |

**Bit-exact determinism.** Every kernel uses a scratch-buffer + fixed-
order reduction instead of `atomicAdd` across k-splits, so running the
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

Inspired by [QuIP#](https://arxiv.org/abs/2402.04396) (Tseng et al., 2024).

- E8 lattice: Korkin & Zolotarev (1872); Gosset (1900); Conway & Sloane, *Sphere Packings, Lattices and Groups*; [Viazovska (2016)](https://arxiv.org/abs/1603.04246) — sphere-packing optimality in 8 dimensions.
- Block-feedback quantization: [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022).
- INT8 KV cache: [KIVI](https://arxiv.org/abs/2402.02750) (Liu et al., 2024).

## License

Apache 2.0
