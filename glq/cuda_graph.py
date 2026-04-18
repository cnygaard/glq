"""CUDA graph wrappers for GLQ model inference.

Two wrappers live here:

- ``CUDAGraphWrapper``: captures a single-token decode step (B=1, seqlen=1)
  into a CUDA graph and replays it. Uses ``StaticCache`` for fixed-shape KV
  buffers. Designed for ``model.generate``.

- ``CUDAGraphBucketWrapper``: captures one CUDA graph per ``(B, seqlen)``
  bucket for stateless prefill / batched scoring (``use_cache=False``).
  At call time, pads the input up to the nearest bucket and replays the
  corresponding graph. Used for lm-eval loglikelihood scoring via
  ``wrap_hflm`` which monkey-patches ``HFLM._model_call``.

Usage:
    from glq.cuda_graph import CUDAGraphWrapper, CUDAGraphBucketWrapper, wrap_hflm
    wrapper = CUDAGraphWrapper(model)                 # B=1 decode
    output = wrapper.generate(input_ids, max_new_tokens=128)

    wrap_hflm(hflm_instance)                          # batched scoring
    # lm_eval.simple_evaluate(model=hflm_instance, ...) now replays graphs
"""

from dataclasses import dataclass

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


# ─────────────────────────────────────────────────────────────────────────
# Shape-bucketed CUDA graph wrapper (stateless prefill / scoring)
# ─────────────────────────────────────────────────────────────────────────

# Default buckets cover the (B, seqlen) shapes lm-eval loglikelihood tends
# to produce at batch_size="auto" on 135M..8B models. Smallest-area bucket
# that fits the input is chosen, so unused buckets cost only VRAM (shared
# memory pool keeps peak low).
DEFAULT_BUCKETS = [
    (1, 128), (1, 512), (1, 2048),
    (4, 128), (4, 512),
    (16, 128), (16, 512), (16, 1024),
    (32, 256),
    (64, 128), (64, 512),
]


@dataclass
class _ModelOutput:
    """Tiny stand-in for HF ``CausalLMOutputWithPast`` — we only need
    ``.logits`` on the scoring path."""
    logits: torch.Tensor


class CUDAGraphBucketWrapper:
    """Capture one CUDA graph per ``(B, seqlen)`` bucket for stateless
    prefill / scoring. **Opt-in tool — not enabled by default.**

    At call time, pads the input to the smallest bucket that fits and
    replays the corresponding graph. Captures lazily on first use of each
    bucket, sharing a memory pool across all captures so peak VRAM stays
    near that of the largest bucket rather than summed.

    Falls back to eager ``model(...)`` for:
      * inputs larger than the biggest bucket,
      * ``use_cache=True`` (decode path — use ``CUDAGraphWrapper`` instead),
      * ``past_key_values is not None``.

    Use ``wrap_hflm(hflm, buckets=...)`` to plug this into lm-eval's HFLM
    backend transparently.

    **When this helps (verified):**
      * Small/fast models where per-forward Python dispatch is a large
        fraction of wall time. SmolLM2-135M 4bpw winogrande limit=5:
        4.16 s → 3.33 s (1.25×).
      * Fixed-shape batch serving where the actual ``(B, seqlen)`` closely
        matches the chosen bucket, and many replays amortize one capture.

    **When this does NOT help (and may hurt):**
      * Larger models where per-forward is already GPU-bound (SmolLM3-3B
        6bpw winogrande limit=5: 4.25 s → 10.99 s, 0.39×). The 3 warmup
        + 1 capture forwards cost more than the Python dispatch saved.
      * Variable-shape workloads where real inputs are much smaller than
        the bucket — wasted compute on padded positions (4× more
        attention work when padding a (10, 50) input to a (16, 128)
        bucket). SmolLM2-135M limit=200: 3.71 s → 4.99 s, 0.74×.
      * Small number of replays per bucket (capture cost dominates).

    In short: useful when you control the workload shape and know replays
    will vastly outnumber captures. Default ``DEFAULT_BUCKETS`` covers
    wide shape variety but is not a universal speedup — tune to the
    workload you actually run.
    """

    def __init__(
        self,
        model,
        buckets=None,
        padding="left",
        pad_token_id=None,
        warmup_iters=3,
        verbose=False,
    ):
        assert padding in ("left", "right"), padding
        self.model = model
        self.buckets = sorted(list(buckets or DEFAULT_BUCKETS))
        self.padding = padding
        if pad_token_id is None:
            cfg = model.config
            pad_token_id = (getattr(cfg, "pad_token_id", None)
                            or getattr(cfg, "eos_token_id", None)
                            or 0)
            if isinstance(pad_token_id, list):
                pad_token_id = pad_token_id[0]
        self.pad_token_id = int(pad_token_id)
        self.warmup_iters = warmup_iters
        self.verbose = verbose

        device = next(model.parameters()).device
        self._device = device
        self._max_B = max(b for b, _ in self.buckets)
        self._max_S = max(s for _, s in self.buckets)
        # Shared max-shape slabs so bucket captures can alias into them.
        self._static_input_ids = torch.zeros(
            self._max_B, self._max_S, dtype=torch.long, device=device)
        self._static_attn_mask = torch.zeros(
            self._max_B, self._max_S, dtype=torch.long, device=device)
        self._static_position_ids = torch.zeros(
            self._max_B, self._max_S, dtype=torch.long, device=device)

        # Shared graph memory pool — reuses device allocations across buckets
        # so peak VRAM is bounded by the largest bucket's activations.
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._graphs = {}  # bucket -> (CUDAGraph, logits_static_tensor)

    # ── helpers ──────────────────────────────────────────────────────

    def _select_bucket(self, B_real, S_real):
        cands = [(B, S) for (B, S) in self.buckets
                 if B >= B_real and S >= S_real]
        if not cands:
            return None
        return min(cands, key=lambda bs: (bs[0] * bs[1], bs[0], bs[1]))

    def _capture_bucket(self, bucket):
        B, S = bucket
        device = self._device

        # Prime the static slab with plausible content so warmup can run.
        self._static_input_ids[:B, :S].fill_(self.pad_token_id)
        self._static_attn_mask[:B, :S].fill_(1)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S).contiguous()
        self._static_position_ids[:B, :S].copy_(pos)

        # Warmup on a private stream so captured stream is clean.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup_iters):
                _ = self.model(
                    input_ids=self._static_input_ids[:B, :S],
                    attention_mask=self._static_attn_mask[:B, :S],
                    position_ids=self._static_position_ids[:B, :S],
                    use_cache=False,
                    return_dict=True,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            out = self.model(
                input_ids=self._static_input_ids[:B, :S],
                attention_mask=self._static_attn_mask[:B, :S],
                position_ids=self._static_position_ids[:B, :S],
                use_cache=False,
                return_dict=True,
            )
        self._graphs[bucket] = (graph, out.logits)
        if self.verbose:
            print(f"[CUDAGraphBucketWrapper] captured bucket {bucket} "
                  f"(logits.shape={tuple(out.logits.shape)})")

    # ── call path ────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        if input_ids.dim() != 2:
            return self.model(input_ids, attention_mask=attention_mask,
                              position_ids=position_ids, **kwargs)

        B_real, S_real = input_ids.shape
        bucket = self._select_bucket(B_real, S_real)
        pkv = kwargs.get("past_key_values", None)
        if (bucket is None
                or kwargs.get("use_cache", False)
                or pkv is not None):
            return self.model(input_ids, attention_mask=attention_mask,
                              position_ids=position_ids, **kwargs)

        if bucket not in self._graphs:
            self._capture_bucket(bucket)

        B_cap, S_cap = bucket
        graph, logits_static = self._graphs[bucket]
        device = self._device

        # Zero slab so padded positions are known-safe.
        self._static_input_ids[:B_cap, :S_cap].fill_(self.pad_token_id)
        self._static_attn_mask[:B_cap, :S_cap].zero_()

        input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=torch.long,
                                               non_blocking=True)
        else:
            attention_mask = torch.ones_like(input_ids)

        if self.padding == "right":
            self._static_input_ids[:B_real, :S_real].copy_(input_ids)
            self._static_attn_mask[:B_real, :S_real].copy_(attention_mask)
            pos1d = torch.arange(S_cap, device=device)
        else:  # "left"
            pad_len = S_cap - S_real
            # Slice end must be S_cap, not open-ended — the static slab is
            # sized to max_S, not to the bucket's S_cap.
            self._static_input_ids[:B_real, pad_len:S_cap].copy_(input_ids)
            self._static_attn_mask[:B_real, pad_len:S_cap].copy_(attention_mask)
            pos1d = (torch.arange(S_cap, device=device) - pad_len).clamp_min(0)
        self._static_position_ids[:B_cap, :S_cap].copy_(
            pos1d.unsqueeze(0).expand(B_cap, S_cap).contiguous())

        graph.replay()

        if self.padding == "right":
            logits = logits_static[:B_real, :S_real]
        else:
            logits = logits_static[:B_real, S_cap - S_real:]
        return _ModelOutput(logits=logits)

    def reset(self):
        """Clear captured graphs and free the memory pool reference."""
        self._graphs.clear()
        self._graph_pool = torch.cuda.graph_pool_handle()


def wrap_hflm(hflm_instance, buckets=None, padding="left"):
    """Swap an ``lm_eval.models.huggingface.HFLM``'s ``_model_call`` to go
    through a ``CUDAGraphBucketWrapper``, so lm-eval loglikelihood scoring
    replays graphs instead of re-dispatching Python for every layer.

    Returns the same HFLM instance for chaining.
    """
    wrapper = CUDAGraphBucketWrapper(
        hflm_instance.model, buckets=buckets, padding=padding)

    def _wrapped_model_call(inps, attn_mask=None, labels=None):
        # HFLM._model_call convention: returns a bare logits tensor.
        return wrapper(inps, attention_mask=attn_mask).logits

    hflm_instance._model_call = _wrapped_model_call
    hflm_instance._glq_bucket_wrapper = wrapper
    return hflm_instance
