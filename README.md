# GLQ

Post-training weight quantization for LLMs using E8 lattice codebooks.

GLQ encodes weights into 8-dimensional E8 lattice points via nearest-neighbor lookup. A Randomized Hadamard Transform (RHT) makes the Hessian approximately diagonal so that Euclidean nearest-neighbor is near-optimal under the proxy loss.

## Results

**SmolLM3-3B-Base** on WikiText-2 (128 calibration samples, NVIDIA A10G):

| Method | Eff. BPW | Size (MB) | Perplexity | vs bf16 | GPU MB | tok/s |
|--------|----------|-----------|------------|---------|--------|-------|
| bf16 | 16.00 | 6150 | 7.90 | 1.00x | 6151 | 33.9 |
| GLQ 4-bit | 4.00 | 1538 | 8.11 | 1.03x | - | - |
| AWQ 4-bit | 5.60 | 2152 | 8.15 | 1.03x | - | - |
| QuIP+GPTQ 4-bit | 4.76 | 1829 | 8.17 | 1.03x | - | - |
| GLQ 3-bit | 3.00 | 1153 | 8.91 | 1.13x | - | - |
| QuIP+GPTQ 3-bit | 3.70 | 1423 | 9.30 | 1.18x | - | - |
| GLQ 2-bit | 2.00 | 769 | 11.35 | 1.44x | 2540 | 13.4 |

**Mistral-7B-v0.3** on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 | GPU MB | tok/s |
|--------|-----|------------|---------|--------|-------|
| bf16 | 16 | 4.20 | 1.00x | 14505 | 28.1 |
| GLQ 3-bit | 3 | 4.41 | 1.05x | 4436 | 9.7 |

**Ministral-3-3B-Base-2512** on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 | GPU MB | tok/s |
|--------|-----|------------|---------|--------|-------|
| bf16 | 16 | 5.91 | 1.00x | 7348 | 37.0 |
| GLQ 3-bit | 3 | 6.47 | 1.09x | 3788 | 11.4 |

**Nemotron-3-Nano-30B** (hybrid Mamba-Attention-MoE) on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 |
|--------|-----|------------|---------|
| bf16 | 16 | 7.72 | 1.00x |
| GLQ 3-bit | 3 | 8.09 | 1.05x |

**Llama-3.2-3B** on WikiText-2 (16 calibration samples, NVIDIA A10G):

| Method | BPW | Perplexity | vs bf16 | GPU MB | tok/s |
|--------|-----|------------|---------|--------|-------|
| bf16 | 16 | 6.17 | 1.00x | 6137 | 37.6 |
| GLQ 3-bit | 3 | 6.78 | 1.10x | 3529 | 10.8 |
| GLQ 2-bit | 2 | 8.49 | 1.38x | 3526 | 11.0 |

**SmolLM2-360M** on WikiText-2 (128 calibration samples, NVIDIA A10G):

| Method | Eff. BPW | Perplexity | vs bf16 | GPU MB | tok/s |
|--------|----------|------------|---------|--------|-------|
| bf16 baseline | 16.00 | 11.48 | 1.00x | 724 | 37.6 |
| GLQ 4-bit | 4.00 | 11.82 | 1.03x | - | - |
| QuIP+GPTQ 4-bit | 4.75 | 12.06 | 1.05x | - | - |
| GLQ 3-bit | 3.00 | 13.38 | 1.17x | - | - |
| QuIP+GPTQ 3-bit | 3.69 | 14.84 | 1.29x | - | - |
| GLQ 2-bit | 2.00 | 17.70 | 1.54x | 356 | 15.5 |
| GPTQ 3-bit | 9.48 | 18.61 | 1.62x | - | - |

GLQ uses a single global scale per layer rather than per-group scales, so effective bit widths match the nominal rate exactly. GLQ 2-bit (17.70) beats GPTQ 3-bit (18.61) at less than 1/4 the storage. GLQ 4-bit (11.82) beats QuIP+GPTQ 4-bit (12.06) at lower effective bpw (4.00 vs 4.75).

## How it works

1. **E8 lattice codebook**: 65536 vectors from the first 7 shells of the E8 lattice. Each 8-weight group maps to a 16-bit index (2 bpw). For 3/4 bpw, a second-stage residual codebook adds 8 or 16 more bits.

2. **Randomized Hadamard Transform (RHT)**: Random sign flips + Fast Walsh-Hadamard Transform applied to both weights and Hessian. This spreads weight magnitude evenly across dimensions, making the Hessian block-diagonal approximately proportional to identity. After RHT, Euclidean nearest-neighbor in the codebook is close to Hessian-optimal.

3. **LDLQ error feedback**: Block-LDL decomposition of the Hessian drives a sequential quantization sweep (like GPTQ but over 8-dim blocks instead of scalar columns). Quantization error from each block propagates forward to correct subsequent blocks.

4. **Fused Triton inference kernel**: On CUDA, a custom Triton kernel reads codebook indices directly from HBM and gathers from the L2-cached codebook (65536 x 8 fp16 = 1 MB) without ever materializing the full weight matrix. This provides real GPU memory savings proportional to the compression ratio.

## Install

Requires Python 3.10+ and PyTorch 2.0+. Install PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
# Full install (includes transformers, datasets, etc. for glq-quantize CLI):
pip install 'glq[quantize]'

# Or minimal install (inference only, no quantization dependencies):
pip install glq
```

Triton is bundled with PyTorch on CUDA and will be used automatically when available. On CPU, GLQ falls back to a naive dequantize-then-matmul path.

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

All CLI options:

```
glq-quantize --help
  --model              HuggingFace model ID or local path (required)
  --output             Output directory for quantized model (required)
  --bpw                Bits per weight: 2, 3, or 4 (default: 2)
  --tune-iters         LDLQ refinement iterations (default: 0)
  --nsamples           Calibration samples from WikiText-2 (default: 16)
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

On CUDA, inference automatically uses the fused Triton kernel. On CPU, it falls back to dequantize-then-matmul.

### Bit widths

| BPW | Encoding | Bits per 8 weights | Storage |
|-----|----------|--------------------|---------|
| 2 | 16-bit codebook index | 16 | Global scale only |
| 3 | 16-bit primary + 8-bit residual index | 24 | Global scale + residual scale |
| 4 | 16-bit primary + 16-bit residual index | 32 | Global scale + residual scale |

All bit widths use a single global scale per layer (no group-size parameter), so effective bit widths match the nominal rate exactly.

For 3/4 bpw, GLQ uses a two-stage residual vector quantization (RVQ): the primary codebook (65536 entries) encodes the bulk of the weight, and a secondary codebook (256 entries for 3 bpw, 65536 for 4 bpw) encodes the residual error scaled by a learned factor.

## Triton inference kernel

The fused Triton kernel (`glq/inference_kernel.py`) is the core of GLQ's inference performance. It computes `Y = X @ dequant(W)^T` without materializing the full weight matrix.

### How it works

Instead of the naive approach (decode all indices into a dense bf16 matrix, then matmul), the kernel:

1. Iterates over `N/8` codebook blocks per output row
2. Loads int16 indices from HBM and gathers 8-element vectors from the L2-cached codebook
3. Accumulates dot products (matvec) or Tensor Core matmuls (prefill) against the gathered codebook vectors
4. Applies the global scale factor and writes the output

This means GPU memory holds only the compressed indices (2 bytes per 8 weights) rather than the full fp16 weight matrix (16 bytes per 8 weights) — an 8x reduction at 2 bpw.

### Kernel variants

- **Tensor Core matmul kernel** (`_glq_dequant_matmul_tc_kernel`): For batch sizes >= 2 (prefill). Processes pairs of codebook blocks to form K=16 tiles for `tl.dot` (mma.m16n8k16 Tensor Core instructions). Autotuned over BLOCK_B and BLOCK_M.
- **Matvec kernel** (`_glq_dequant_matvec_kernel`): For B=1 (autoregressive decode). Autotuned over BLOCK_M and num_warps for the memory-bound single-token case.
- **Fused RHT kernels** (`_input_rht_kernel`, `_output_rht_kernel`): Fuse pad + sign vector + Fast Hadamard Transform into single kernel launches, eliminating per-layer Python overhead.

All kernels support two-stage RVQ for 3/4 bpw via a `HAS_STAGE2` compile-time constant.

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
  inference_kernel.py  # Fused Triton dequant+matmul kernels
  hf_integration.py    # HuggingFace Transformers integration
```

## Acknowledgments

- The RHT incoherence approach follows [QuIP#](https://arxiv.org/abs/2402.04396) (Tseng et al., 2024)
- E8 lattice geometry from Conway & Sloane, *Sphere Packings, Lattices and Groups*
- LDLQ error feedback from [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)

## License

Apache 2.0
