"""CUDA graph wrapper for GLQ model inference.

Captures a single-token decode step (B=1, seqlen=1) into a CUDA graph
and replays it without Python dispatch overhead. Works with HuggingFace
generate() by using StaticCache for fixed-shape KV buffers.

Usage:
    from glq.cuda_graph import CUDAGraphWrapper
    wrapper = CUDAGraphWrapper(model)
    # Works with generate():
    output = wrapper.generate(input_ids, max_new_tokens=128)
"""

import torch
from transformers import StaticCache


class CUDAGraphWrapper:
    """Wraps a causal LM for CUDA graph replay on B=1 single-token decode.

    Captures one decode step with StaticCache, then replays the graph
    on subsequent decode steps by copying input_ids and cache_position
    into static buffers.

    The generate() method handles prefill in eager mode, then switches
    to graph replay for all decode steps.
    """

    def __init__(self, model, max_cache_len=2048, warmup_iters=3):
        self.model = model
        self.max_cache_len = max_cache_len
        self.warmup_iters = warmup_iters
        self._graph = None
        self._static_input_ids = None
        self._static_cache_position = None
        self._static_logits = None
        self._cache = None

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, **kwargs):
        """Generate with CUDA graph decode. Prefill is eager, decode is graphed."""
        device = input_ids.device
        B, prompt_len = input_ids.shape
        if B != 1:
            return self.model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

        total_len = prompt_len + max_new_tokens
        if total_len > self.max_cache_len:
            self.max_cache_len = total_len

        # Create static cache
        cache = StaticCache(
            config=self.model.config,
            max_cache_len=self.max_cache_len,
            device=device,
            dtype=self.model.dtype,
        )

        # Prefill (eager) — process full prompt
        cache_position = torch.arange(prompt_len, device=device)
        out = self.model(
            input_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated = [next_token]

        # Warmup decode steps (eager) to trigger JIT, autotune
        pos = prompt_len
        for i in range(min(self.warmup_iters, max_new_tokens - 1)):
            cp = torch.tensor([pos], device=device)
            out = self.model(
                next_token,
                cache_position=cp,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            next_token = out.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)
            pos += 1
        torch.cuda.synchronize()

        remaining = max_new_tokens - 1 - min(self.warmup_iters, max_new_tokens - 1)
        if remaining <= 0:
            return torch.cat([input_ids] + generated, dim=1)

        # Capture CUDA graph for single decode step
        self._static_input_ids = next_token.clone()
        self._static_cache_position = torch.tensor([pos], device=device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.model(
                self._static_input_ids,
                cache_position=self._static_cache_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        self._static_logits = out.logits
        self._graph = graph

        # Replay for remaining tokens
        for i in range(remaining):
            self._static_input_ids.copy_(next_token)
            self._static_cache_position.fill_(pos)
            self._graph.replay()
            next_token = self._static_logits[:, -1:].argmax(dim=-1)
            generated.append(next_token.clone())
            pos += 1

        self._graph = None  # Free graph after generation
        return torch.cat([input_ids] + generated, dim=1)

    @torch.no_grad()
    def __call__(self, input_ids, **kwargs):
        """Fallback: eager forward for non-generate usage."""
        return self.model(input_ids, **kwargs).logits

    def reset(self):
        """Clear the captured graph."""
        self._graph = None
        self._static_input_ids = None
        self._static_cache_position = None
        self._static_logits = None
        self._cache = None
