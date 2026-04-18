#!/usr/bin/env python3
"""
Run inference on GLQ-quantized models with HuggingFace Transformers.

GLQ models load via the standard from_pretrained() API after importing
glq.hf_integration. The quantized weights stay compressed in GPU memory
and are dequantized on-the-fly during matmul via fused CUDA C kernels.

Published models:
    - xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw          (3B, 3.5 bpw, public)
    - xv0y5ncu/Nemotron-3-Nano-30B-A3B-GLQ-4bpw (30B MoE, 4 bpw, public)
    - xv0y5ncu/Devstral-Small-2-24B-Instruct-GLQ-4bpw (24B, 4 bpw, public)
"""

import torch
import glq.hf_integration  # noqa: F401 — registers GLQ quantization method
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(tokenizer_id):
    """Load a tokenizer, working around the transformers 5.x + Mistral issue.

    transformers 5.x auto-routes Mistral/Devstral models through mistral_common,
    which rejects the standard tokenizer.json format shipped in our quantized
    repos. Fall back to PreTrainedTokenizerFast using the local tokenizer.json
    file when that happens.
    """
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    except (ValueError, ImportError) as e:
        if "tokenizer" not in str(e).lower() and "mistral" not in str(e).lower():
            raise
        from huggingface_hub import snapshot_download
        path = snapshot_download(tokenizer_id)
        tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
        tok.pad_token = "<pad>"
        tok.eos_token = "</s>"
        tok.bos_token = "<s>"
        return tok


def generate(model_id, tokenizer_id=None, prompt="The capital of France is", max_tokens=50):
    """Load a GLQ model and generate text."""
    tokenizer_id = tokenizer_id or model_id

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.float16,
    )
    tokenizer = load_tokenizer(tokenizer_id)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)

    print(f"\nPrompt: {prompt}")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    return model, tokenizer


def generate_with_cuda_graph(model_id, tokenizer_id=None, prompt="Write a fibonacci function:", max_tokens=128):
    """Load a GLQ model and generate with CUDA graph acceleration.

    CUDA graphs eliminate Python dispatch overhead between kernel launches,
    giving ~1.8x speedup on small models (3B). For large models (24B+)
    the GPU is already fully utilized, so graphs provide minimal benefit.
    """
    from glq.cuda_graph import CUDAGraphWrapper

    tokenizer_id = tokenizer_id or model_id

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.float16,
    )
    tokenizer = load_tokenizer(tokenizer_id)
    wrapper = CUDAGraphWrapper(model)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output = wrapper.generate(input_ids, max_new_tokens=max_tokens)

    print(f"\nPrompt: {prompt}")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")


def score_with_bucket_graph(model_id, tokenizer_id=None,
                             prompts=("The capital of France is",
                                      "The capital of Germany is",
                                      "The largest planet is",
                                      "Photosynthesis converts"),
                             bucket_B=4, bucket_S=32):
    """Score a small batch of prompts through ``CUDAGraphBucketWrapper``.

    The wrapper is an opt-in tool — it pays off when per-forward Python
    dispatch dominates wall time (small models, small batches) and when
    the chosen ``(B, seqlen)`` bucket closely fits the actual workload.
    See ``glq.cuda_graph.CUDAGraphBucketWrapper`` docstring for when it
    helps and when it hurts.
    """
    import time
    from glq.cuda_graph import CUDAGraphBucketWrapper

    tokenizer_id = tokenizer_id or model_id
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", dtype=torch.float16,
    )
    tokenizer = load_tokenizer(tokenizer_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to("cuda")
    if enc.input_ids.shape[0] > bucket_B or enc.input_ids.shape[1] > bucket_S:
        print(f"  note: inputs {tuple(enc.input_ids.shape)} exceed bucket "
              f"({bucket_B},{bucket_S}); wrapper will fall back to eager")

    wrapper = CUDAGraphBucketWrapper(
        model, buckets=[(bucket_B, bucket_S)], padding="left", verbose=True,
    )

    # First call captures; subsequent calls replay.
    torch.cuda.synchronize()
    t0 = time.time()
    out = wrapper(enc.input_ids, attention_mask=enc.attention_mask)
    torch.cuda.synchronize()
    t_first = (time.time() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        out = wrapper(enc.input_ids, attention_mask=enc.attention_mask)
    torch.cuda.synchronize()
    t_replay = (time.time() - t0) * 1000 / 5

    next_ids = out.logits[:, -1, :].argmax(dim=-1)
    print(f"\nBucket ({bucket_B},{bucket_S})  first-call {t_first:.1f}ms  "
          f"replay-avg {t_replay:.1f}ms")
    for p, tok_id in zip(prompts, next_ids.tolist()):
        print(f"  {p!r:50} -> {tokenizer.decode([tok_id])!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GLQ inference with HuggingFace Transformers")
    parser.add_argument("--model", default="xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer ID (defaults to model ID)")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Use CUDA graph acceleration for B=1 decode (Phase A)")
    parser.add_argument("--bucket-graph", action="store_true",
                        help="Demo shape-bucketed CUDA graphs for batched scoring (opt-in, "
                             "see CUDAGraphBucketWrapper docstring for when it helps)")
    parser.add_argument("--bucket-B", type=int, default=4, help="Bucket batch size for --bucket-graph")
    parser.add_argument("--bucket-S", type=int, default=32, help="Bucket seqlen for --bucket-graph")
    args = parser.parse_args()

    if args.bucket_graph:
        score_with_bucket_graph(
            args.model, args.tokenizer,
            bucket_B=args.bucket_B, bucket_S=args.bucket_S,
        )
    elif args.cuda_graph:
        generate_with_cuda_graph(args.model, args.tokenizer, args.prompt, args.max_tokens)
    else:
        generate(args.model, args.tokenizer, args.prompt, args.max_tokens)
