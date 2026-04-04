# GLQ Examples

Example scripts for quantizing and running GLQ models.

## Quantization

- [`quantize_model.py`](quantize_model.py) — Quantize any HuggingFace model to GLQ format

## Inference

- [`inference_hf.py`](inference_hf.py) — Load and generate with GLQ models via HuggingFace Transformers
- [`inference_vllm.py`](inference_vllm.py) — Serve GLQ models with vLLM for production throughput

## Requirements

```bash
pip install glq>=0.2.8
# For vLLM serving:
pip install vllm>=0.18.1
```
