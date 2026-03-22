"""CUDA graph wrapper for GLQ model inference.

Captures a single forward pass into a CUDA graph and replays it
without Python dispatch overhead. Eliminates the 60% wall-clock
overhead from Python between kernel launches (see nsys profiling).

Usage:
    from glq.cuda_graph import CUDAGraphWrapper
    wrapper = CUDAGraphWrapper(model)
    # First call captures the graph:
    logits = wrapper(input_ids)
    # Subsequent calls replay it:
    logits = wrapper(input_ids_next)
"""

import torch


class CUDAGraphWrapper:
    """Wraps a causal LM for CUDA graph replay on B=1 single-token decode.

    Captures one forward pass during the first call, then replays the graph
    on subsequent calls by copying input into the static buffer.

    Only works for fixed-shape inputs (B=1, seqlen=1). For variable shapes,
    falls back to eager execution.
    """

    def __init__(self, model, warmup_iters=3):
        self.model = model
        self.warmup_iters = warmup_iters
        self._graph = None
        self._static_input_ids = None
        self._static_output = None
        self._captured_shape = None

    @torch.no_grad()
    def __call__(self, input_ids, **kwargs):
        shape = input_ids.shape
        # Fall back to eager for non-captured shapes
        if self._graph is not None and shape == self._captured_shape and not kwargs:
            self._static_input_ids.copy_(input_ids)
            self._graph.replay()
            return self._static_output.clone()

        if (self._graph is None
                and input_ids.is_cuda
                and shape[0] == 1
                and shape[1] == 1
                and not kwargs):
            return self._capture(input_ids)

        return self.model(input_ids, **kwargs).logits

    def _capture(self, input_ids):
        device = input_ids.device

        # Warmup: run eager passes to trigger JIT compilation and autotune
        for _ in range(self.warmup_iters):
            _ = self.model(input_ids)
        torch.cuda.synchronize()

        # Allocate static buffers
        self._static_input_ids = input_ids.clone()
        self._captured_shape = input_ids.shape

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            out = self.model(self._static_input_ids)
        self._static_output = out.logits

        # Replay once to populate the output buffer (capture only records, doesn't execute)
        self._static_input_ids.copy_(input_ids)
        self._graph.replay()
        torch.cuda.synchronize()
        return self._static_output.clone()

    def reset(self):
        """Clear the captured graph (e.g. if model moves to a different device)."""
        if self._graph is not None:
            del self._graph
        self._graph = None
        self._static_input_ids = None
        self._static_output = None
        self._captured_shape = None
