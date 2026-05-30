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
            # Stage 2c-2b: declare compressed page size to vLLM's
            # allocator. Requires sidecar + read hooks to be active so
            # vLLM's compressed buffer is never read or written
            # directly (sidecar is the source of truth).
            if os.environ.get("GLQ_KV_E8_COMPRESSED_ALLOC", "0") != "0":
                kv_compression.enable_compressed_allocation()
                print("[glq_vllm] GLQ_KV_E8_COMPRESSED_ALLOC=1 → vLLM "
                      "allocates paged buffer at compressed page size "
                      "(Stage 5.3-2c-2b)", flush=True)
            # Stage 3b: opt-in fused Triton dequant-gather kernel. The
            # plumbing reads GLQ_KV_E8_FUSED_GATHER at every attention
            # call so it can be toggled without restart (mainly useful
            # during A/B benchmarks).
            if os.environ.get("GLQ_KV_E8_FUSED_GATHER", "0") != "0":
                print("[glq_vllm] GLQ_KV_E8_FUSED_GATHER=1 → fused Triton "
                      "dequant-gather kernel active (Stage 5.3-3b)",
                      flush=True)
            # Stage 4a: opt-in fused scatter on the write path. Same
            # toggle pattern — env var is read at each attention call.
            if os.environ.get("GLQ_KV_E8_FUSED_WRITE", "0") != "0":
                print("[glq_vllm] GLQ_KV_E8_FUSED_WRITE=1 → fused Triton "
                      "scatter kernel active on write path (Stage 5.3-4a)",
                      flush=True)
            # v0.5 Phase 5.3a research probe: opt-in v3.0 inline-dequant
            # attention with a 4 K KV codebook. Kernel-level microbench
            # already showed this at 1.00× of v2.1; the env-gated probe
            # exists to confirm the kernel-level conclusion holds under
            # vLLM cudagraph + piecewise dispatch. Production default
            # remains the v0.3.5 workspace path.
            if os.environ.get("GLQ_KV_E8_INLINE_DEQUANT_V3", "0") != "0":
                print("[glq_vllm] GLQ_KV_E8_INLINE_DEQUANT_V3=1 → v3.0 "
                      "inline-dequant attention with 4 K codebook "
                      "(Phase 5.3a research probe, not production)",
                      flush=True)

        # v0.3.5: auto-force ``cudagraph_mode=PIECEWISE`` when E8 KV is
        # active. FULL captures wrap the whole model.forward including
        # our patched attention, which calls ``.unique()`` on
        # ``block_table.flatten()`` — illegal during CUDA-graph capture
        # (``cudaErrorStreamCaptureUnsupported``). PIECEWISE captures
        # split at ``vllm::unified_attention_with_output`` so our
        # patched attention runs in the eager break between subgraphs,
        # where ``.unique()`` is fine. End-user effect: drop
        # ``--enforce-eager`` for E8 KV. v0.3.3's enforce_eager=True
        # workaround is no longer required.
        #
        # Hook point: ``vllm.LLM.__init__`` rather than
        # ``CompilationConfig.__post_init__`` — the latter runs in the
        # parent process only; the v1 EngineCore subprocess rebuilds the
        # config from cross-process state, bypassing post-init. Injecting
        # the override into the ``compilation_config`` kwarg at LLM
        # construction time lets the value reach the subprocess via
        # vLLM's own arg passing.
        # Hook point: ``EngineArgs.create_engine_config`` runs **inside**
        # ``LLM.__init__`` after vLLM has loaded plugins (and thus
        # imported us). Patching there means our override is always in
        # place by the time the VllmConfig is materialised, regardless
        # of whether the user imported glq_vllm explicitly or relied on
        # the ``vllm.general_plugins`` entry_point.
        try:
            from vllm.engine.arg_utils import EngineArgs as _GlqEngineArgs
            from vllm.config.compilation import (
                CUDAGraphMode as _GlqCUDAGraphMode,
            )
        except ImportError:
            _GlqEngineArgs = None
            _GlqCUDAGraphMode = None
        if _GlqEngineArgs is not None and not getattr(
                _GlqEngineArgs, "_glq_kv_piecewise_patched", False):
            _orig_create_engine_config = _GlqEngineArgs.create_engine_config

            def _glq_kv_create_engine_config(self, *args, **kwargs):
                vllm_config = _orig_create_engine_config(
                    self, *args, **kwargs)
                # Force the Triton attention backend when E8 KV is active.
                # The entire E8-KV read/alloc path (sidecar read hook +
                # compressed get_kv_cache_shape + v3 inline kernels) is built
                # on vLLM's TritonAttentionBackend. Gemma-4 selects Triton
                # natively (heterogeneous head dims make FlashAttention
                # invalid), but uniform-head models (SmolLM3 head=128, Llama,
                # …) default to FlashAttention → our hooks never fire and the
                # compressed buffer is read/reshaped as fp16 → crash/garbage.
                # Only set it when the user hasn't pinned a backend.
                ac = getattr(vllm_config, "attention_config", None)
                if ac is not None and getattr(ac, "backend", None) is None:
                    try:
                        from vllm.v1.attention.backends.registry import (
                            AttentionBackendEnum,
                        )
                        ac.backend = AttentionBackendEnum.TRITON_ATTN
                        print("[glq_vllm] E8 KV active → attention backend "
                              "forced to TRITON_ATTN (E8-KV path requires "
                              "the Triton backend; uniform-head models would "
                              "otherwise pick FlashAttention)", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[glq_vllm] could not force TRITON_ATTN "
                              f"backend: {e}", flush=True)
                cc = getattr(vllm_config, "compilation_config", None)
                if cc is not None:
                    mode = getattr(cc, "cudagraph_mode", None)
                    full_modes = {
                        _GlqCUDAGraphMode.FULL,
                        _GlqCUDAGraphMode.FULL_DECODE_ONLY,
                        _GlqCUDAGraphMode.FULL_AND_PIECEWISE,
                    }
                    if mode in full_modes or mode is None:
                        cc.cudagraph_mode = _GlqCUDAGraphMode.PIECEWISE
                        from_label = (
                            mode.name if hasattr(mode, "name") else repr(mode)
                        )
                        print(
                            f"[glq_vllm] E8 KV active → cudagraph_mode "
                            f"forced from {from_label} to PIECEWISE "
                            "(FULL captures incompatible with .unique() "
                            "in _patched_unified_attention; fused-dequant "
                            "paged_attention in v0.4 will lift this)",
                            flush=True,
                        )
                return vllm_config

            _GlqEngineArgs.create_engine_config = _glq_kv_create_engine_config
            _GlqEngineArgs._glq_kv_piecewise_patched = True
            print(
                "[glq_vllm] E8 KV EngineArgs.create_engine_config hook "
                "installed (forces cudagraph_mode=PIECEWISE)",
                flush=True,
            )


# Also register on import for backward compat (vLLM 0.16 / manual usage)
register()
