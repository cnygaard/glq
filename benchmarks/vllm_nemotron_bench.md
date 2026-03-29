# Nemotron-30B vLLM Benchmark Results

## Hardware
- GPU: NVIDIA L40S (48 GB)
- Instance: g6e.2xlarge (AWS, eu-north-1)
- CUDA: 12.9
- vLLM: 0.16.0
- PyTorch: 2.9.1+cu128

## NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4

### Server command
```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len 1024 \
  --max-num-seqs 64
```

### Benchmark command
```bash
vllm bench serve \
  --backend vllm \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /opt/dlami/nvme/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

### Results (2026-03-29)
```
============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Benchmark duration (s):                  11.50
Total input tokens:                      1392
Total generated tokens:                  2660
Request throughput (req/s):              0.87
Output token throughput (tok/s):         231.24
Peak output token throughput (tok/s):    565.00
Peak concurrent requests:                10.00
Total token throughput (tok/s):          352.25
---------------Time to First Token----------------
Mean TTFT (ms):                          4677.94
Median TTFT (ms):                        4679.49
P99 TTFT (ms):                           4680.67
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.38
Median TPOT (ms):                        11.73
P99 TPOT (ms):                           14.91
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.55
Median ITL (ms):                         8.88
P99 ITL (ms):                            15.19
==================================================
```

- Model memory: 18.05 GiB
- Quantization: modelopt_fp4 (NVFP4, ~4bpw)
- CUDA graphs: enabled (full + piecewise)
- KV cache: fp8_e4m3
- Backend: Marlin for NVFP4 GEMM

### Notes
- max-num-seqs=64 required to avoid OOM during sampler warmup with CUDA graphs
- max-model-len > 1024 causes OOM during CUDA graph capture on L40S (48 GB)
- TTFT is high on first batch due to CUDA graph warmup
- ninja must be in PATH for torch.compile (symlink /opt/dlami/nvme/venv/bin/ninja to /usr/local/bin/ninja)

---

## GLQ Nemotron-3-Nano-30B-A3B-GLQ-3.5bpw

### Python API command
```python
# vllm serve CLI doesn't support plugin registration — use Python API
import glq_vllm
from vllm import LLM, SamplingParams

llm = LLM(model='xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-3.5bpw',
          tokenizer='/opt/dlami/nvme/nemotron_tokenizer',
          quantization='glq',
          trust_remote_code=True,
          gpu_memory_utilization=0.90,
          max_model_len=1024,
          enforce_eager=True)
```

Env vars required:
```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # GLQ params have non-serializable weight_loader
```

Tokenizer workaround:
```bash
# Download tokenizer, remove tokenizer_class from tokenizer_config.json
# (NemotronH uses TokenizersBackend which isn't available in standard transformers)
```

### Benchmark command
```bash
# Server (via wrapper that registers GLQ plugin before vLLM CLI):
python benchmarks/serve_glq.py serve xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-3.5bpw \
  --tokenizer /opt/dlami/nvme/nemotron_tokenizer \
  --quantization glq \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 1024 \
  --enforce-eager

# Benchmark:
vllm bench serve \
  --backend vllm \
  --model xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-3.5bpw \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /opt/dlami/nvme/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 5 \
  --tokenizer /opt/dlami/nvme/nemotron_tokenizer
```

### Results (2026-03-29, 5 prompts, max_model_len=1024, fused CUDA kernels)
```
============ Serving Benchmark Result ============
Successful requests:                     5
Failed requests:                         0
Benchmark duration (s):                  751.00
Total input tokens:                      411
Total generated tokens:                  1445
Request throughput (req/s):              0.01
Output token throughput (tok/s):         1.92
Peak output token throughput (tok/s):    5.00
Peak concurrent requests:                5.00
Total token throughput (tok/s):          2.47
---------------Time to First Token----------------
Mean TTFT (ms):                          20848.03
Median TTFT (ms):                        21288.03
P99 TTFT (ms):                           21288.42
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1644.82
Median TPOT (ms):                        1715.58
P99 TPOT (ms):                           2091.55
---------------Inter-token Latency----------------
Mean ITL (ms):                           1267.86
Median ITL (ms):                         1269.33
P99 ITL (ms):                            2232.98
==================================================
```

- Model memory: 27.2 GiB (Mamba layers dense, MoE compressed)
- KV cache: 11.47 GiB (382,976 tokens)
- Weight loading: 6.2s, process_weights_after_loading: ~160s
- Output speed: **1.92 tok/s** (fused CUDA kernel dequant+matmul, no CUDA graphs)
- Previous (Python dequant): 1.05 tok/s → **1.83× speedup from fused kernels**

### Comparison

| Metric | NVFP4 (4bpw) | GLQ (3.5bpw) | Ratio |
|--------|-------------|-------------|-------|
| Model memory | 18.05 GiB | 27.2 GiB | 1.5× more (Mamba dense) |
| Output tok/s | 231 | 1.92 | 120× slower |
| TPOT | 11 ms | 1,645 ms | 149× |
| TTFT | 4.7s | 20.8s | 4.4× |
| Quantization | FP4 + Marlin kernel | E8 lattice + fused CUDA kernel | — |
| CUDA graphs | Yes | No (enforce-eager) | — |
| Num prompts | 10 | 5 | — |

### Notes
- MoE apply() uses fused CUDA dequant+matmul kernels, loops only over active experts (top-2)
- Mamba layers dequanted to dense fp16 at load time (~160s process_weights_after_loading)
- CUDA graphs not yet supported (enforce-eager required)
- GLQ model memory is higher because Mamba layers are stored dense (not compressed)
- Remaining bottleneck: 23 Mamba dense layers + vLLM Python dispatch overhead
- Next optimization: CUDA graphs, shared input RHT across experts
