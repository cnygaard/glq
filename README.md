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

| Repo | Base model | bpw | License |
|---|---|---|---|
| [`xv0y5ncu/SmolLM2-135M-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/SmolLM2-135M-Instruct-GLQ-4bpw) | SmolLM2-135M-Instruct | 4.0 | Apache 2.0 |
| [`xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw) | SmolLM2-360M-Instruct | 4.0 | Apache 2.0 |
| [`xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw`](https://huggingface.co/xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw) | SmolLM3-3B | 3.5 (mixed) | Apache 2.0 |
| [`xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw) | Gemma-4-E4B-it | 4.0 | Apache 2.0 |
| [`xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw) | Devstral-Small 24B | 4.0 | Apache 2.0 |
| [`xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-4bpw`](https://huggingface.co/xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-4bpw) | Nemotron-3-Nano-30B (Mamba-MoE) | 4.0 | Nemotron |

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

| | fp16 baseline | E8 4 bpw |
|---|---|---|
| KV cache capacity @ 27.9 GiB | 303,984 tokens | **1,221,232 (4.02×)** |
| mmlu_pro n=240 accuracy | 71.25 % | **71.25 % (bit-identical)** |
| `cudaLaunchKernel` per decode | 110,659 | **71,619 (−35 %)** |

Activation:

```bash
GLQ_KV_QUANT=e8_relaxed:2 \
GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 \
GLQ_KV_E8_COMPRESSED_ALLOC=1 \
GLQ_KV_E8_FUSED_GATHER=1 GLQ_KV_E8_FUSED_WRITE=1 \
vllm serve google/gemma-4-E4B-it --enforce-eager
```

Limitations: vLLM 0.20.x only, `--enforce-eager` required, validated
end-to-end on Gemma-4-E4B-it. The codebook-NN kernel is still ~42 %
of CUDA time at 4 bpw — fusing that is the v0.4 target.

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
    --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --disable-piecewise-cuda-graph
```

Requires the triton attention backend (flashinfer returns wrong
logprobs in echo/prefill mode) and `--disable-piecewise-cuda-graph`
(torch.dynamo can't trace the pybind GLQ extension).

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

GLQ builds directly on [QuIP#](https://arxiv.org/abs/2402.04396) (Tseng
et al., 2024) — the RHT-incoherence + lattice-codebook recipe is
theirs, and this project would not exist without that paper. The
specific choices here (relaxed E8 codebook enumeration, N-stage residual
vector quantization, the fused CUDA C kernels) are derivative
engineering on top of QuIP#'s framework.

Additional foundations:

- LDLQ block-feedback follows [GPTQ](https://arxiv.org/abs/2210.17323)
  (Frantar et al., 2022).
- E8 lattice geometry from Conway & Sloane, *Sphere Packings, Lattices
  and Groups*.
- INT8 KV cache approach follows
  [KIVI](https://arxiv.org/abs/2402.02750) (Liu et al., 2024).

## License

Apache 2.0
