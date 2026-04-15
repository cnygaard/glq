# GLQ

Post-training weight quantization for LLMs using E8 lattice codebooks.

GLQ encodes weights into 8-dimensional E8 lattice points via nearest-neighbor lookup. A Randomized Hadamard Transform (RHT) makes the Hessian approximately diagonal so that Euclidean nearest-neighbor is near-optimal under the proxy loss.

## Results

> **Note on BPW labels:** The `BPW` column in the tables below shows the **nominal** quantization config for GLQ rows (e.g. "GLQ 4-bit" → 4.00). For AWQ/QuIP+GPTQ rows the value is effective bpw including group scales. For legacy GLQ measurements (pre-v0.2.9), actual storage is 1.3-1.7× the nominal due to power-of-2 FHT padding on non-power-of-2 hidden sizes — you can derive actual eff bpw from `Size (MB) / bf16 MB × 16`. With v0.2.9+ block-diagonal FHT, new quantizations store at true nominal bpw. See [`xv0y5ncu/SmolLM3-3B-GLQ-6bpw`](https://huggingface.co/xv0y5ncu/SmolLM3-3B-GLQ-6bpw) for a model with true 6.0 bpw at 99.6% of bf16.

**SmolLM3-3B-Base** on WikiText-2 (128 calibration samples, NVIDIA L40S):

| Method | BPW | Size (MB) | Perplexity | vs bf16 |
|--------|----------|-----------|------------|---------|
| bf16 | 16.00 | 6150 | 7.04 | 1.00x |
| GLQ 4-bit | 4.00 | 2531 | 7.19 | 1.02x |
| AWQ 4-bit† | 5.60 | 2152 | 8.15 | 1.16x |
| QuIP+GPTQ 4-bit† | 4.76 | 1829 | 8.17 | 1.16x |
| **GLQ 3.5-bit mixed** | **3.50** | **2282** | **7.20** | **1.02x** |
| GLQ 3-bit | 3.00 | — | 7.64 | 1.09x |
| GLQ 3-bit mixed (2+4) | 3.00 | 2035 | 7.65 | 1.09x |
| GLQ 2.5-bit mixed | 2.50 | 2031 | 8.08 | 1.15x |
| QuIP+GPTQ 3-bit† | 3.70 | 1423 | 9.30 | 1.32x |
| GLQ 2-bit | 2.00 | 1531 | 9.61 | 1.36x |

†AWQ and QuIP+GPTQ numbers are from an earlier measurement on A10G with 16 calibration samples and may not be directly comparable. GLQ numbers were re-measured on L40S with 128 samples.

Mixed-precision models use `--bpw <target> --min-bpw 2 --max-bpw 4` to automatically allocate per-layer bit widths via Hessian sensitivity profiling. Sensitive layers (attention projections in the middle of the network) get higher precision while robust layers are compressed further. GLQ 3.5-bit mixed matches uniform 4-bit quality (7.20 vs 7.19) at 10% less storage.

**Mistral-7B-v0.3** on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 | GPU MB |
|--------|-----|------------|---------|--------|
| bf16 | 16 | 4.20 | 1.00x | 14505 |
| GLQ 3-bit | 3 | 4.41 | 1.05x | 4436 |

**Nemotron-3-Nano-30B** (hybrid Mamba-Attention-MoE) on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 |
|--------|-----|------------|---------|
| bf16 | 16 | 7.72 | 1.00x |
| GLQ 3-bit | 3 | 8.09 | 1.05x |

**Llama-3.2-3B** on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 | GPU MB |
|--------|-----|------------|---------|--------|
| bf16 | 16 | 6.17 | 1.00x | 6137 |
| GLQ 3-bit | 3 | 6.78 | 1.10x | 3529 |
| GLQ 2-bit | 2 | 8.49 | 1.38x | 3526 |

**SmolLM3-3B-Base** 5-task accuracy via lm-evaluation-harness (acc_norm where available, 128 calibration samples, NVIDIA L40S):

| Method | BPW | ARC-c | ARC-e | HellaSwag | PIQA | WinoGrande | Avg |
|--------|----------|-------|-------|-----------|------|------------|-----|
| bf16 baseline | 16.00 | 0.540 | 0.793 | 0.758 | 0.786 | 0.668 | 0.709 |
| GLQ 3.5-bit mixed | 3.50 | 0.497 | 0.777 | 0.735 | 0.769 | 0.668 | 0.685 |
| GLQ 3-bit | 3.00 | 0.447 | 0.759 | 0.530 | 0.755 | 0.688 | 0.636 |
| GLQ 2-bit | 2.00 | 0.415 | 0.679 | 0.634 | 0.730 | 0.660 | 0.623 |

GLQ 3.5-bit mixed retains 96.6% of bf16 accuracy (nominal 4.6× weight-bpw ratio; actual storage ratio 2.70× on this legacy-padded measurement). WinoGrande holds perfectly (0.668 = bf16). GLQ 2-bit retains 87.9% (nominal 8× ratio; actual ~4×).

**SmolLM3-3B-Base** 4-bit method comparison (acc_norm where available, 128 calibration samples, NVIDIA L40S):

| Method | BPW | ARC-c | ARC-e | HellaSwag | PIQA | WinoGrande | Avg | tok/s | VRAM |
|--------|----------|-------|-------|-----------|------|------------|-----|-------|------|
| bf16 baseline | 16.00 | 0.540 | 0.793 | 0.758 | 0.786 | 0.668 | 0.709 | 30.8 | 5,875 MB |
| AutoRound W4 | 4.50 | 0.532 | 0.797 | 0.748 | 0.781 | 0.661 | 0.704 | — | — |
| GPTQ W4A16 | 4.50 | 0.538 | 0.785 | 0.740 | 0.781 | 0.648 | 0.698 | 8.5† | 7,487 MB |
| GLQ 4-bit | 4.00 | 0.522 | 0.776 | 0.746 | 0.780 | 0.672 | 0.699 | — | 5,044 MB |

GLQ 4-bit retains 98.6% of bf16 accuracy. AutoRound (99.3%) and GPTQ (98.5%) use group_size=128 (~4.5 eff bpw). The GLQ row above is a legacy-padded measurement (actual eff bpw ~6.6 on SmolLM3-3B); with v0.2.9+ block-diagonal FHT, new GLQ 4-bit quantizations store at true 4.00 bpw.

**SmolLM2-360M** on WikiText-2 (128 calibration samples, NVIDIA L40S):

| Method | BPW | Perplexity | vs bf16 |
|--------|----------|------------|---------|
| bf16 baseline | 16.00 | 11.47 | 1.00x |
| GLQ 4-bit | 4.00 | 11.77 | 1.03x |
| QuIP+GPTQ 4-bit | 4.75 | 12.06 | 1.05x |
| GLQ 3-bit | 3.00 | 13.16 | 1.15x |
| QuIP+GPTQ 3-bit | 3.69 | 14.84 | 1.29x |
| GLQ 2-bit | 2.00 | 17.94 | 1.56x |

GLQ uses a single global scale per layer rather than per-group scales. With v0.2.9+ block-diagonal FHT, true bit widths match the nominal rate exactly (legacy power-of-2 FHT added 1.3-1.7× padding overhead on non-power-of-2 dimensions). GLQ 4-bit (11.77) beats QuIP+GPTQ 4-bit (12.06) on perplexity.

**SmolLM2-360M-Instruct** 5-task accuracy via lm-evaluation-harness (128 calibration samples, NVIDIA L40S):

| Method | BPW | ARC-e | HellaSwag | PIQA | WinoGrande | LAMBADA | Avg | % of bf16 |
|--------|----------|-------|-----------|------|------------|---------|-----|-----------|
| bf16 baseline | 16.00 | 0.565 | 0.428 | 0.712 | 0.573 | 0.508 | 0.557 | 100% |
| **GLQ 4-bit** | **4.00** | 0.554 | 0.420 | 0.717 | 0.575 | 0.508 | **0.555** | **99.6%** |
| GPTQ W4 (g64) | ~4.5 | 0.473 | 0.386 | 0.681 | 0.542 | 0.346 | 0.486 | 87.2% |

GLQ 4-bit retains 99.6% of bf16 accuracy vs GPTQ's 87.2% on this 360M model. GPTQ with group_size=64 (required since hidden_size=960 is not divisible by 128) loses 13% of quality, with LAMBADA dropping from 0.508 to 0.346. GLQ preserves LAMBADA perfectly (0.508). The E8 lattice codebook approach handles small models significantly better than group-wise INT4.

### Inference performance

**SmolLM3-3B** vLLM 0.18.1 throughput on NVIDIA L40S:

| Method | BPW | Single request | Batch=5 | vs bf16 |
|--------|----------|---------------|---------|---------|
| bf16 | 16.0 | 39.4 tok/s | 184 tok/s | 100% |
| **GLQ 3.5bpw** | **3.5** | **37.1 tok/s** | **173 tok/s** | **94%** |
| GPTQ W4 (g128) | ~4.5 | 34.6 tok/s | 172 tok/s | 88% |

GLQ serves at 94% of bf16 speed while GPTQ reaches 88% at single-request. At batch=5 GPTQ closes most of its gap (88% → 93%), while GLQ holds at ~94% of bf16 — as batch size grows the matmul becomes more compute-bound and the memory-bandwidth advantage of compressed weights narrows.

**SmolLM3-3B** HuggingFace Transformers decode on NVIDIA L40S:

| Mode | GLQ 3.5bpw | bf16 | GLQ / bf16 |
|------|-----------|------|------------|
| Eager (default) | 25 tok/s | 40 tok/s | 63% |
| CUDA graph | 37 tok/s | 40 tok/s | 93% |

With CUDA graph capture, GLQ decode approaches bf16 throughput because the smaller quantized weights require less DRAM bandwidth.

**Devstral-24B** (Ministral3, 24B params) GLQ 4bpw on NVIDIA L40S — fits in ~22 GB (bf16 would need ~48 GB; this is a legacy power-of-2 FHT model, so effective bpw is above the nominal 4):

| Mode | tok/s |
|------|-------|
| HF eager | 6.6 tok/s |
| CUDA graph | 6.4 tok/s (compute-bound, no graph benefit) |

#### CUDA graph acceleration

CUDA graphs eliminate Python dispatch overhead between kernel launches (~60% of eager wall-clock time). The `CUDAGraphWrapper` captures a B=1 single-token forward pass and replays it without CPU involvement:

```python
import glq.hf_integration
from glq.cuda_graph import CUDAGraphWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./smollm2-glq-4bpw", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./smollm2-glq-4bpw")

wrapper = CUDAGraphWrapper(model)

# First call captures the graph; subsequent calls replay it
input_ids = tokenizer("Hello", return_tensors="pt").input_ids[:, -1:].to(model.device)
logits = wrapper(input_ids)  # ~37 tok/s vs ~25 tok/s eager (SmolLM3-3B on L40S)
```

The wrapper automatically falls back to eager execution for variable shapes (prefill, batch>1) or calls with extra kwargs (past_key_values, attention_mask). It only accelerates the fixed-shape B=1 seqlen=1 decode path.

## How it works

1. **E8 lattice codebook**: 65536 vectors from the first 7 shells of the E8 lattice. Each 8-weight group maps to a 16-bit index (2 bpw). For 3-8 bpw, N-stage residual vector quantization adds further 8-bit (256-entry) or 16-bit (65536-entry) codebooks per stage — see the Bit widths table for the stage schedule.

2. **Randomized Hadamard Transform (RHT)**: Random sign flips + Fast Walsh-Hadamard Transform applied to both weights and Hessian. This spreads weight magnitude evenly across dimensions, making the Hessian block-diagonal approximately proportional to identity. After RHT, Euclidean nearest-neighbor in the codebook is close to Hessian-optimal.

3. **LDLQ error feedback**: Block-LDL decomposition of the Hessian drives a sequential quantization sweep (like GPTQ but over 8-dim blocks instead of scalar columns). Quantization error from each block propagates forward to correct subsequent blocks.

4. **Fused inference kernels**: On CUDA, custom CUDA C and Triton kernels read codebook indices directly from HBM and gather from the L2-cached codebook (65536 x 8 fp16 = 1 MB) without ever materializing the full weight matrix. The CUDA C path uses inline PTX Tensor Core instructions (B>=2) and split-K matvec with warp shuffles (B=1). This provides real GPU memory savings proportional to the compression ratio.

## Install

Requires Python 3.10+ and PyTorch 2.0+. Install PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
# Full install (includes transformers, datasets, etc. for glq-quantize CLI):
pip install 'glq[quantize]'

# Or minimal install (inference only, no quantization dependencies):
pip install glq
```

Triton is bundled with PyTorch on CUDA and will be used automatically when available. On CPU, GLQ falls back to a naive dequantize-then-matmul path.

**Note on transformers version:** For small models (360M and below), use `transformers >= 5.0`. Transformers 4.57.x has a weight loading bug that produces garbage output for small GLQ models. Larger models (3B+) work with both 4.x and 5.x.

## Quickstart

### Quantizing a model

#### Command line

```bash
# 2-bit quantization (smallest model, ~1.5x perplexity)
glq-quantize \
    --model HuggingFaceTB/SmolLM2-360M \
    --output ./smollm2-glq-2bpw \
    --bpw 2 \
    --nsamples 128 \
    --device cuda

# 3-bit quantization (good balance of size and quality)
glq-quantize \
    --model HuggingFaceTB/SmolLM2-360M \
    --output ./smollm2-glq-3bpw \
    --bpw 3 \
    --nsamples 128 \
    --device cuda

# 4-bit quantization (near-lossless, ~1.03x perplexity)
glq-quantize \
    --model HuggingFaceTB/SmolLM2-360M \
    --output ./smollm2-glq-4bpw \
    --bpw 4 \
    --nsamples 128 \
    --device cuda
```

Mixed-precision quantization uses a two-pass workflow. Pass 1 profiles each layer's sensitivity at 2bpw and computes the optimal per-layer bit allocation. Pass 2 quantizes with the allocation:

```bash
# Pass 1: profile sensitivity and compute allocation (outputs bpw_allocation.json)
glq-quantize \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --output ./nemotron-30b-profile \
    --bpw 3.5 --min-bpw 2 --max-bpw 4 \
    --nsamples 128 \
    --streaming --trust-remote-code

# Pass 2: quantize with the per-layer allocation
glq-quantize \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --output ./nemotron-30b-glq-3.5bpw \
    --bpw-map ./nemotron-30b-profile/bpw_allocation.json \
    --nsamples 128 \
    --streaming --trust-remote-code
```

The allocator assigns more bits to sensitive layers (attention K/V projections in the middle of the network) and fewer to robust layers, achieving better quality than uniform quantization at the same average bpw.

For large models (30B+), use `--streaming` to load one layer at a time instead of the full model. This keeps memory constant at ~1 layer size instead of scaling with model size.

All CLI options:

```
glq-quantize --help
  --model              HuggingFace model ID or local path (required)
  --output             Output directory for quantized model (required)
  --bpw                Bits per weight: 2-8 or fractional like 2.5 (default: 2)
  --min-bpw            Minimum per-layer bpw for mixed-precision
  --max-bpw            Maximum per-layer bpw for mixed-precision
  --tune-iters         LDLQ refinement iterations (default: 0)
  --nsamples           Calibration samples from WikiText-2 (default: 128)
  --seqlen             Calibration sequence length (default: 2048)
  --device             cuda or cpu (default: cuda)
  --trust-remote-code  Allow custom model code from HF Hub
  --streaming          Load weights layer-by-layer from safetensors
                       (for models exceeding system RAM)
```

#### Python API

```python
from glq import quantize

quantize(
    model_name="HuggingFaceTB/SmolLM2-360M",
    output_dir="./smollm2-glq-4bpw",
    bpw=4,
    nsamples=128,
    device="cuda",
)
```

The `quantize()` function handles the full pipeline: load model, capture Hessians via calibration data, quantize each linear layer with E8+RHT+LDLQ, and save the result as a standard HuggingFace model directory (safetensors + config.json + tokenizer).

### Loading and running a quantized model

```python
import glq.hf_integration  # registers GLQ with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./smollm2-glq-4bpw",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./smollm2-glq-4bpw")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The `import glq.hf_integration` line registers GLQ as a quantization method with HuggingFace Transformers. After that, `from_pretrained` automatically:
1. Reads `quantization_config.quant_method = "glq"` from config.json
2. Replaces `nn.Linear` modules with `E8RHTLinear`
3. Loads the quantized weights (codebook indices + sign vectors)
4. Builds the E8 codebook and attaches it to all quantized layers

On CUDA, inference automatically uses CUDA C kernels (with Triton as fallback). On CPU, it falls back to dequantize-then-matmul.

### Devstral-24B tokenizer note

`transformers` 5.x auto-routes Mistral/Devstral models through `mistral_common`, which rejects the standard `tokenizer.json` format shipped in the quantized repo. `examples/inference_hf.py` includes a `load_tokenizer()` helper that handles the fallback automatically. For your own scripts:

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

Devstral-24B GLQ 4bpw uses ~22 GB of GPU memory (bf16 would need ~48 GB), so it fits on an L40S (48 GB) or A100 40 GB.

### Serving with sglang

A fork of sglang with GLQ support is maintained at [cnygaard/sglang](https://github.com/cnygaard/sglang) on the `glq-quantization` branch. It registers `"glq"` as a quantization method, reuses the existing `glq.inference_kernel` CUDA extension as a runtime dependency (no kernel port needed), and handles fused QKV / gate-up projections via a `GLQShardedParameter` with per-shard SV vectors.

```bash
git clone -b glq-quantization https://github.com/cnygaard/sglang
cd sglang/python && pip install -e .

python -m sglang.launch_server \
    --model xv0y5ncu/SmolLM2-360M-Instruct-GLQ-4bpw \
    --tokenizer-path HuggingFaceTB/SmolLM2-360M-Instruct \
    --quantization glq --disable-cuda-graph --disable-piecewise-cuda-graph \
    --attention-backend triton --sampling-backend pytorch
```

SmolLM2-360M-Instruct GLQ 4bpw via the native `LlamaForCausalLM` sglang path at `num_concurrent=16` batched lm-eval 5-task matches bf16 within 99.3-99.6% (the exact number shifts by ~0.1% with each fresh quant run). The `triton` attention backend is required — `flashinfer` returns wrong logprobs in echo/prefill mode. `--disable-piecewise-cuda-graph` is required because torch.dynamo can't trace through the pybind GLQ extension.

### INT8 KV cache

For long-context inference, GLQ provides an optional INT8 quantized KV cache that halves the memory used by keys and values. This is especially useful for large models at long sequence lengths where the KV cache dominates VRAM (e.g. 30B model at 4K+ context).

```python
import glq.hf_integration
from glq.kv_cache import GLQQuantizedCache
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./smollm2-glq-4bpw", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./smollm2-glq-4bpw")

cache = GLQQuantizedCache(model.config)
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200, past_key_values=cache)
```

The INT8 cache uses per-channel absmax quantization with no external dependencies (pure PyTorch). Recent tokens are kept in full precision (configurable via `residual_length`) while older tokens are quantized to INT8, following the [KIVI](https://arxiv.org/abs/2402.02750) approach. Requires transformers >= 4.45.

### Bit widths

| BPW | Stages | Bits per 8 weights | Storage |
|-----|--------|--------------------|---------|
| 2 | 1 | 16 | Global scale only |
| 3 | 2 | 16 + 8 | Global scale + residual scale |
| 4 | 2 | 16 + 16 | Global scale + residual scale |
| 5 | 3 | 16 + 16 + 8 | Global scale + 2 residual scales |
| 6 | 3 | 16 + 16 + 16 | Global scale + 2 residual scales |
| 7 | 4 | 16 + 16 + 16 + 8 | Global scale + 3 residual scales |
| 8 | 4 | 16 + 16 + 16 + 16 | Global scale + 3 residual scales |

All bit widths use a single global scale per layer (no group-size parameter). With v0.2.9+ block-diagonal FHT, true bit widths match the nominal rate exactly.

For 3+ bpw, GLQ uses N-stage residual vector quantization (RVQ): the primary codebook (65536 entries) encodes the bulk of the weight, and each additional stage encodes the residual error scaled by a learned factor. Each stage is either 8-bit (256-entry codebook) or 16-bit (65536-entry E8 codebook); see the table above for the per-bpw stage schedule.

### Block-diagonal FHT (v0.2.9+)

Non-power-of-2 hidden sizes are decomposed into sums of powers of 2 (e.g. `2688 = 2048 + 512 + 128`), with independent FHTs per block. This eliminates the padding waste of single-block power-of-2 FHT (which would pad 2688 → 4096, inflating storage by 1.5×). Enabled by default when quantizing; existing power-of-2 models fall back to the original single-block FHT path.

## Inference kernels

GLQ provides CUDA C and Triton kernel implementations (`glq/inference_kernel.py`, `glq/csrc/glq_cuda.cu`) that compute `Y = X @ dequant(W)^T` without materializing the full weight matrix.

### How it works

Instead of the naive approach (decode all indices into a dense bf16 matrix, then matmul), the kernel:

1. Iterates over `N/8` codebook blocks per output row
2. Loads int16 indices from HBM and gathers 8-element vectors from the L2-cached codebook
3. Accumulates dot products (matvec) or Tensor Core matmuls (prefill) against the gathered codebook vectors
4. Applies the global scale factor and writes the output

This means GPU memory holds only the compressed indices (2 bytes per 8 weights) rather than the full fp16 weight matrix (16 bytes per 8 weights) — an 8x reduction at 2 bpw.

### Kernel variants

- **CUDA C Tensor Core kernel** (`glq_matmul_tc_kernel` / `glq_matmul_tc_scratch_kernel`): For batch sizes >= 2 (prefill). Uses inline PTX `mma.sync.aligned.m16n8k16` with direct codebook-to-register loading — no shared memory staging. 3-5x faster than Triton TC for prefill.
- **CUDA C split-K matvec** (`glq_matvec_splitk_kernel` / `glq_matvec_splitk_scratch_kernel`): For B=1 (autoregressive decode). 4 rows per warp with `__shfl_xor_sync` reduction, 2D grid for K-split parallelism. 2.7x faster than Triton matvec.
- **CUDA C shared-memory FHT** (`glq_input_rht_kernel`, `glq_output_rht_kernel`): Double-buffered butterfly stages in shared memory for the Hadamard transform. 1.6-3x faster than Triton FHT.
- **Triton fallback kernels**: Used when CUDA C extension is unavailable (no ninja) or for dimensions exceeding shared memory limits (`n_pad > 32768`; the CUDA C FHT uses a two-pass path for `n_pad` up to 32768, and block-diagonal FHT decomposes non-power-of-2 dims into sub-blocks).

The CUDA C kernels currently implement two-stage RVQ (3/4 bpw) via a `HAS_STAGE2` compile-time constant. For 5-8 bpw (N-stage RVQ), inference falls back to the Triton/PyTorch path. The CUDA C path is selected automatically when available and supported.

### Bit-exact determinism

The B=1 matvec and B>=2 TC matmul launchers use a scratch-buffer + fixed-order reduction pipeline instead of `atomicAdd` across k-splits. Each CTA writes its partial sum to a unique `(k_split, b, m)` slot (no atomic), then a follow-up `glq_reduce_splits_kernel` sums across k-splits in deterministic loop order.

This means **every GLQ kernel is bit-exact run-to-run** on the same input. Running the same prompt at B=1 decode or B=8 batched prefill produces identical logits across runs, which is required for reproducible `lm-eval` scoring, RL on-policy rollouts, and CI regression tests. The adaptive split-K SM-saturation benefit is preserved — determinism costs only a small follow-up reduction kernel launch per matmul.

### Using the kernel directly

The kernel is used automatically by `E8RHTLinear.forward()` when running on CUDA with Triton available. You can also call it directly:

```python
from glq.inference_kernel import glq_dequant_matmul

# 2bpw: single codebook
y = glq_dequant_matmul(
    x,          # (B, N) input activations, fp16/fp32
    Qidxs,      # (M, N//8) codebook indices, int16
    codebook,    # (65536, 8) codebook vectors, fp16
    Wscale,      # float, global scale factor
)

# 3/4bpw: two-stage with residual codebook
y = glq_dequant_matmul(
    x, Qidxs, codebook, Wscale,
    Qidxs2=Qidxs2,              # (M, N//8) secondary indices, int16
    codebook2=codebook2,          # (K2, 8) secondary codebook, fp16
    inv_resid_scale=inv_rs,       # float, 1.0 / residual_scale
)
```

Falls back to naive dequantize+matmul on CPU or when Triton is not available.

### Requirements

- CUDA GPU
- Triton (bundled with `pip install torch` on CUDA, or `pip install 'glq[cuda]'`)
- PyTorch 2.0+

## Architecture

```
glq/
  codebook.py          # E8ShellCodebook: enumeration, encode/decode, make_small()
  hadamard.py          # Fast Walsh-Hadamard Transform
  rht.py               # Randomized Hadamard Transform (sign flips + FHT)
  ldlq.py              # Block-LDL quantization with error feedback
  quantize_model.py    # Full model quantization pipeline + CLI
  quantized_linear.py  # E8RHTLinear: drop-in nn.Linear replacement
  inference_kernel.py  # Dispatch + Triton fallback kernels
  csrc/glq_cuda.cu     # CUDA C kernels (split-K matvec, TC matmul, FHT)
  hf_integration.py    # HuggingFace Transformers integration
  kv_cache.py          # INT8 quantized KV cache (optional)
  cuda_graph.py        # CUDA graph wrapper for B=1 decode (~1.5x speedup)
```

## Acknowledgments

- The RHT incoherence approach follows [QuIP#](https://arxiv.org/abs/2402.04396) (Tseng et al., 2024)
- E8 lattice geometry from Conway & Sloane, *Sphere Packings, Lattices and Groups*
- LDLQ error feedback from [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)

## License

Apache 2.0
