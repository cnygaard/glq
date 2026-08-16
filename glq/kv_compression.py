"""KV-cache compression: what GLQ offers, and what it does not.

**Offered — vLLM's own fp8 KV cache.** Two serve flags, maintained upstream, so they keep
working across vLLM upgrades:

    --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window

fp8 halves the cache against fp16, so roughly twice the context fits in the same VRAM. The
skip list excludes sliding-window attention layers, which the gemma-4 family uses.

**Not offered — GLQ's own E8 lattice KV cache.** The env set below still exists in
`glq_vllm/__init__.py` and buys more (~2.7-4x), but it does not serve on vLLM 0.27.1: all
six stages announce themselves and then EngineCore dies on

    assert len(kv_cache_stride_order) == len(kv_cache_shape)

Measured on an L4, vLLM 0.27.1; last known good was 0.25.1. It is kept here as a definition
so the knowledge is not lost, and deliberately not wired to any flag or prompt — offering a
path that cannot start is worse than not mentioning it. Re-offer it only after a real serve
and generate, not after seeing the announcement lines, which print either way.
"""
from __future__ import annotations

#: vLLM's own flags, appended to `vllm serve`. `fp8` is one of the dtypes 0.27.1 accepts for
#: `--kv-cache-dtype`; `--kv-cache-dtype-skip-layers` takes a list and `sliding_window`
#: excludes the sliding-window attention layers from quantization.
FP8_KV_ARGS = ("--kv-cache-dtype", "fp8",
               "--kv-cache-dtype-skip-layers", "sliding_window")

#: What the user is choosing. Deliberately unquantified on speed and quality: this is vLLM's
#: implementation and GLQ has not measured it on this stack, so the honest statement is the
#: memory arithmetic — which is exact — plus a pointer to check their own use.
TRADEOFF = (
    "fp8 KV cache (vLLM's own)\n"
    "  for:     the cache is 8 bits per element instead of 16, so about twice the\n"
    "           context fits in the same VRAM\n"
    "  against: lower-precision attention history. GLQ has not measured the speed or\n"
    "           quality effect on its checkpoints, so treat it as a reason to check\n"
    "           your own use rather than a free win\n"
    "  note:    sliding-window attention layers are excluded (gemma-4 uses them)\n"
    "  Off by default; the weights are already compressed, and this trades a second thing."
)

#: Parked, not offered. See the module docstring.
E8_KV_ENV = {
    "GLQ_KV_QUANT": "e8_relaxed:2",
    "GLQ_KV_E8_SIDECAR": "1",
    "GLQ_KV_E8_SIDECAR_READ": "1",
    "GLQ_KV_E8_COMPRESSED_ALLOC": "1",
    "GLQ_KV_E8_FUSED_GATHER": "1",
    "GLQ_KV_E8_FUSED_WRITE": "1",
}


def serve_args(fp8_kv: bool) -> tuple[str, ...]:
    """The extra `vllm serve` arguments for this choice."""
    return tuple(FP8_KV_ARGS) if fp8_kv else ()


def shell_suffix(fp8_kv: bool) -> str:
    """The same, for appending to a printed `vllm serve` command."""
    return (" " + " ".join(FP8_KV_ARGS)) if fp8_kv else ""
