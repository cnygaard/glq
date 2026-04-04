#!/usr/bin/env python3
"""
Serve GLQ-quantized models with vLLM for production throughput.

GLQ registers as a vLLM quantization plugin via entry_points. After
`pip install glq`, vLLM automatically discovers the "glq" quantization
method in all processes including the v1 engine subprocess.

Performance (SmolLM3-3B GLQ 3.5bpw, L40S):
    Single request:  37.1 tok/s (94% of bf16)
    Batch of 5:     173   tok/s

Published models:
    - xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw          (3B, 3.5 bpw)
    - xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw (24B, 4 bpw)
"""

from vllm import LLM, SamplingParams


def serve(model_id, tokenizer_id=None, prompts=None, max_tokens=100):
    """Load a GLQ model in vLLM and generate completions."""
    tokenizer_id = tokenizer_id or model_id
    prompts = prompts or ["The capital of France is", "Write a Python fibonacci function:"]

    print(f"Loading {model_id} via vLLM...")
    llm = LLM(
        model=model_id,
        tokenizer=tokenizer_id,
        quantization="glq",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,  # disable torch.compile for GLQ custom kernels
    )

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nPrompt:  {output.prompt}")
        print(f"Output:  {output.outputs[0].text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GLQ inference with vLLM")
    parser.add_argument("--model", default="xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM3-3B",
                        help="Tokenizer ID (use base model for GLQ models)")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    serve(args.model, args.tokenizer, max_tokens=args.max_tokens)
