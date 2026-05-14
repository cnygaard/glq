"""GLQ quantization plugin for vLLM.

Usage (vLLM 0.18+ with entry_points — automatic):
    pip install glq  # entry point registers "glq" in all vLLM processes

Usage (vLLM 0.16 or manual):
    import glq_vllm  # registers "glq" quantization method

    from vllm import LLM
    llm = LLM(model="xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw", quantization="glq")
"""


def register():
    """Entry point called by vLLM's plugin loader in ALL processes (including engine core).

    Registered as vllm.general_plugins entry point in pyproject.toml.

    Also activates the E8 KV-cache monkey-patch when ``GLQ_KV_QUANT`` is
    set (e.g. ``GLQ_KV_QUANT=e8_relaxed:2`` for 4 bpw RVQ). The patch
    must be applied in the engine subprocess for vLLM v1; this entry
    point runs there too, which is why ``glqkv.enable()`` lives here
    rather than in user code.
    """
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
        from .config import GLQvLLMConfig
        register_quantization_config("glq")(GLQvLLMConfig)
        from .custom_ops import _ensure_registered
        _ensure_registered()
    except ImportError:
        pass

    # KV-cache E8 compression — opt-in via env vars (mutually exclusive,
    # GLQ_KV_BPW_MAP wins if both are set).
    #
    # Uniform bpw:
    #   GLQ_KV_QUANT=<quant_method>:<n_stages>
    #     e.g. GLQ_KV_QUANT=e8_relaxed:1  → 2 bpw
    #          GLQ_KV_QUANT=e8_relaxed:2  → 4 bpw (RVQ)
    #          GLQ_KV_QUANT=e8_relaxed:3  → 6 bpw (3-stage RVQ)
    #
    # Per-layer mixed bpw (from glq.cli.quantize_kv):
    #   GLQ_KV_BPW_MAP=/path/to/kv_bpw_map.json
    #   GLQ_KV_QUANT_METHOD=e8_relaxed                  # optional, default
    import os
    bpw_map_path = os.environ.get("GLQ_KV_BPW_MAP")
    spec = os.environ.get("GLQ_KV_QUANT")
    if bpw_map_path:
        try:
            import json
            raw = json.load(open(bpw_map_path))
            bpw_map = {int(k): int(v) for k, v in raw.items()}
            method = os.environ.get("GLQ_KV_QUANT_METHOD", "e8_relaxed")
            from . import kv_compression
            kv_compression.enable(quant_method=method, bpw_map=bpw_map)
            cfg = kv_compression.active_config()
            print(f"[glq_vllm] GLQ_KV_BPW_MAP={bpw_map_path!r} → "
                  f"KV cache patch activated (method={method}, "
                  f"n_layers={cfg.get('n_layers')}, "
                  f"avg_bpw={cfg.get('bpw_avg'):.2f}, "
                  f"hist={cfg.get('bpw_hist')})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[glq_vllm] failed to load GLQ_KV_BPW_MAP="
                  f"{bpw_map_path!r}: {e}", flush=True)
    elif spec:
        try:
            method, n_stages_s = spec.split(":")
            n_stages = int(n_stages_s)
            from . import kv_compression
            kv_compression.enable(quant_method=method, n_stages=n_stages)
            print(f"[glq_vllm] GLQ_KV_QUANT={spec!r} → "
                  f"KV cache monkey-patch activated "
                  f"(quant_method={method}, n_stages={n_stages})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[glq_vllm] failed to activate GLQ_KV_QUANT={spec!r}: {e}",
                  flush=True)

    # Phase 5.3 Stage 2b — optional sidecar E8PagedKVCache for validating
    # the byte layout under real vLLM traffic. Currently runs alongside
    # vLLM's fp16 cache (no memory saving yet — Stage 2c will free that).
    if (bpw_map_path or spec) and os.environ.get("GLQ_KV_E8_SIDECAR", "0") != "0":
        from . import kv_compression
        kv_compression.enable_sidecar()
        print("[glq_vllm] GLQ_KV_E8_SIDECAR=1 → E8 paged cache sidecar "
              "enabled (Stage 5.3-2b)", flush=True)
        # Stage 2c-1: also hook attention reads to use the sidecar.
        if os.environ.get("GLQ_KV_E8_SIDECAR_READ", "0") != "0":
            kv_compression.enable_sidecar_read()
            print("[glq_vllm] GLQ_KV_E8_SIDECAR_READ=1 → attention reads "
                  "routed through E8 sidecar (Stage 5.3-2c-1)", flush=True)


# Also register on import for backward compat (vLLM 0.16 / manual usage)
register()
