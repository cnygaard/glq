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
        # Cap cudagraph capture sizes <= GLQ_MOE_BD_MAX_TOKENS for GLQ MoE models,
        # so vLLM's default (up to 512) doesn't try to capture the >256-token
        # non-capturable MoE fallback (cudaErrorStreamCaptureUnsupported).
        from . import _cudagraph_cap
        _cudagraph_cap.install()
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
            if os.environ.get("GLQ_KV_E8_INLINE_DEQUANT_V3", "1") != "0":
                print("[glq_vllm] v3 inline-dequant attention (4 K codebook, "
                      "FHT butterfly, KV-split, FULL cudagraph) — DEFAULT "
                      "E8-KV read path as of v0.5.1; "
                      "GLQ_KV_E8_INLINE_DEQUANT_V3=0 to opt out",
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
                # The forced PIECEWISE downgrade below exists because the
                # v0.3.5 WORKSPACE read path calls ``block_table.unique()``
                # (data-dependent → illegal in CUDA-graph capture). The v3
                # inline path (GLQ_KV_E8_INLINE_DEQUANT_V3=1) is host-sync-
                # clean (no .unique()/.item() in the read hook, and the write
                # hook is too), so vLLM CAN capture it in the FULL decode
                # graph — which removes the ~41 ms/tok eager-dispatch tax.
                #
                # v0.5 Phase 5.7v (validated 2026-05-31, SmolLM3 + Gemma-4
                # focused A/B gate): FULL is now the DEFAULT for the v3 path.
                #   - SmolLM3 (deterministic): FULL is BIT-IDENTICAL to
                #     PIECEWISE on MMLU n=120 / NIAH-16k / varlen, and runs
                #     2.5-3.5x faster (B=1 15→38, B=4 37→130 tok/s; longctx
                #     16k 15→36).
                #   - Gemma-4 (sliding-window + head 256/512): FULL lands
                #     within Gemma's OWN PIECEWISE run-to-run greedy non-
                #     determinism (MMLU acc 79 inside the [78,85] PIE-vs-PIE
                #     spread; a pre-existing vLLM+Gemma property, not GLQ);
                #     NIAH 10/10; tok/s FULL >= PIECEWISE.
                # Escape hatch: GLQ_KV_E8_FORCE_PIECEWISE=1 (or the legacy
                # GLQ_KV_E8_ALLOW_FULL_CUDAGRAPH=0) restores PIECEWISE. The
                # WORKSPACE path (V3 unset) always stays forced-PIECEWISE.
                _v3 = os.environ.get("GLQ_KV_E8_INLINE_DEQUANT_V3", "1") != "0"
                _force_pie = (
                    os.environ.get("GLQ_KV_E8_FORCE_PIECEWISE", "0") == "1"
                )
                _legacy_off = (
                    os.environ.get("GLQ_KV_E8_ALLOW_FULL_CUDAGRAPH") == "0"
                )
                _allow_full = _v3 and not _force_pie and not _legacy_off
                if cc is not None and _allow_full:
                    print("[glq_vllm] E8 KV v3 inline path → leaving "
                          "cudagraph_mode as-is (FULL decode-graph capture, "
                          "Phase 5.7v default; GLQ_KV_E8_FORCE_PIECEWISE=1 "
                          "to opt out)", flush=True)
                if cc is not None and not _allow_full:
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
                        _why = (
                            "GLQ_KV_E8_FORCE_PIECEWISE/ALLOW_FULL=0 escape hatch"
                            if _v3 else
                            "workspace read path uses .unique() (capture-unsafe)"
                        )
                        print(
                            f"[glq_vllm] E8 KV active → cudagraph_mode "
                            f"forced from {from_label} to PIECEWISE ({_why})",
                            flush=True,
                        )
                return vllm_config

            _GlqEngineArgs.create_engine_config = _glq_kv_create_engine_config
            _GlqEngineArgs._glq_kv_piecewise_patched = True
            print(
                "[glq_vllm] E8 KV EngineArgs.create_engine_config hook "
                "installed (v3 inline path → FULL decode graph by default; "
                "workspace path → PIECEWISE; GLQ_KV_E8_FORCE_PIECEWISE=1 "
                "to force PIECEWISE)",
                flush=True,
            )


# Also register on import for backward compat (vLLM 0.16 / manual usage)
register()
