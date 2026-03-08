# GLQ

Post-training weight quantization for LLMs using E8 lattice codebooks.

GLQ encodes weights into 8-dimensional E8 lattice points via nearest-neighbor lookup. A Randomized Hadamard Transform (RHT) makes the Hessian approximately diagonal so that Euclidean nearest-neighbor is near-optimal under the proxy loss.

## Results

**SmolLM3-3B-Base** on WikiText-2 (NVIDIA A10G):

| Method | Eff. BPW | Size (MB) | Perplexity | vs bf16 |
|--------|----------|-----------|------------|---------|
| bf16 | 16.00 | 6150 | 7.90 | 1.00x |
| GLQ 4-bit | 4.00 | 1538 | 8.11 | 1.03x |
| AWQ 4-bit | 5.60 | 2152 | 8.15 | 1.03x |
| QuIP+GPTQ 4-bit | 4.76 | 1829 | 8.17 | 1.03x |
| GLQ 3-bit | 3.00 | 1153 | 8.91 | 1.13x |
| QuIP+GPTQ 3-bit | 3.70 | 1423 | 9.30 | 1.18x |
| GLQ 2-bit | 2.00 | 769 | 11.35 | 1.44x |

GLQ uses a single global scale per layer rather than per-group scales, so effective bit widths match the nominal rate. Early results on one model — more benchmarks needed.

## How it works

1. **E8 lattice codebook**: 65536 vectors from the first 7 shells of the E8 lattice. Each 8-weight group maps to a 16-bit index (2 bpw). For 3/4 bpw, a second-stage residual codebook adds 8 or 16 more bits.

2. **Randomized Hadamard Transform (RHT)**: Random sign flips + Fast Walsh-Hadamard Transform applied to both weights and Hessian. This spreads weight magnitude evenly across dimensions, making the Hessian block-diagonal approximately proportional to identity. After RHT, Euclidean nearest-neighbor in the codebook is close to Hessian-optimal.

3. **LDLQ error feedback**: Block-LDL decomposition of the Hessian drives a sequential quantization sweep (like GPTQ but over 8-dim blocks instead of scalar columns). Quantization error from each block propagates forward to correct subsequent blocks.

## Install

Requires Python 3.10+ and PyTorch 2.0+. Install PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
# Core package (codebook + quantization):
pip install 'glq[quantize] @ git+https://github.com/cnygaard/glq.git'

# Or minimal install (no transformers/datasets):
pip install git+https://github.com/cnygaard/glq.git
```

Triton (for the fused codebook kernel) is bundled with PyTorch on CUDA and will be used automatically.

## Quickstart

### Command line

```bash
glq-quantize \
    --model HuggingFaceTB/SmolLM2-360M \
    --output ./smollm2-glq-2bpw \
    --bpw 2 \
    --nsamples 128 \
    --device cuda
```

### Python API

```python
from glq import quantize

quantize(
    model_name="HuggingFaceTB/SmolLM2-360M",
    output_dir="./smollm2-glq-2bpw",
    bpw=2,
    nsamples=128,
    device="cuda",
)
```

### Loading a quantized model

```python
import glq.hf_integration  # registers GLQ with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./smollm2-glq-2bpw", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./smollm2-glq-2bpw")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Bit widths

| BPW | Encoding | Overhead |
|-----|----------|----------|
| 2 | 16-bit index per 8 weights | Global scale only |
| 3 | 16-bit + 8-bit residual index per 8 weights | Global scale only |
| 4 | 16-bit + 16-bit residual index per 8 weights | Global scale only |

All bit widths use a single global scale per layer (no group-size parameter).

## Acknowledgments

- The RHT incoherence approach follows [QuIP#](https://arxiv.org/abs/2402.04396) (Tseng et al., 2024)
- E8 lattice geometry from Conway & Sloane, *Sphere Packings, Lattices and Groups*
- LDLQ error feedback from [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)

## License

Apache 2.0
