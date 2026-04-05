# GLQ Examples

Example scripts for quantizing and running GLQ models.

## Quantization

- [`quantize_model.py`](quantize_model.py) — Quantize any HuggingFace model to GLQ format

## Inference

- [`inference_hf.py`](inference_hf.py) — Load and generate with GLQ models via HuggingFace Transformers
- [`inference_vllm.py`](inference_vllm.py) — Serve GLQ models with vLLM for production throughput

## Quick start

```bash
# Small 3B model
python inference_hf.py --model xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw

# 24B Devstral (needs the PreTrainedTokenizerFast workaround, handled
# automatically by load_tokenizer() in inference_hf.py on transformers 5.x)
python inference_hf.py \
    --model xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw \
    --prompt "Write a Python function that computes the Fibonacci sequence" \
    --max-tokens 100
```

### Devstral tokenizer note

transformers 5.x auto-routes Mistral/Devstral models through `mistral_common`,
which rejects the standard `tokenizer.json` format shipped in our quantized
repos. `inference_hf.py::load_tokenizer` catches this and falls back to
`PreTrainedTokenizerFast(tokenizer_file=path + "/tokenizer.json")` — pointing
at the local snapshot downloaded by `snapshot_download`. No manual
intervention needed; just run the command above.

For your own scripts:

```python
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

path = snapshot_download("xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw")
tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
tok.pad_token = "<pad>"
tok.eos_token = "</s>"
tok.bos_token = "<s>"
```

## Requirements

```bash
pip install glq>=0.2.8
# For vLLM serving:
pip install vllm>=0.18.1
# Devstral-24B needs ~22 GB GPU memory (L40S 46 GB is sufficient)
```
