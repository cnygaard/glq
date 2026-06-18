"""Cap CUDA-graph capture sizes for GLQ **MoE** models served on vLLM.

Why: the GLQ MoE kernels (block-diag / grouped) only capture batches up to
``GLQ_MOE_BD_MAX_TOKENS`` (default 256); above that the layer routes to a
per-expert Python loop (``.item()``/``.unique()`` host-syncs) that **cannot** be
stream-captured. vLLM's default ``cudagraph_capture_sizes`` go up to 512, so the
capture phase tries a >256 batch and dies with
``cudaErrorStreamCaptureUnsupported: operation not permitted when stream is
capturing`` (observed on both FULL and PIECEWISE, vLLM 0.22/0.23).

Fix: when a GLQ MoE model is configured, drop capture sizes above the cap (and
clamp ``max_cudagraph_capture_size``) so vLLM's out-of-the-box config captures
cleanly. Non-MoE GLQ (dense linears) is untouched — its matvec/TC path captures
at any size. Installed via the ``vllm.general_plugins`` entry point so it runs in
the API-server AND engine-core processes (both build a ``VllmConfig``).
"""
import os

_CAP_INSTALLED = False


def _looks_moe(cfg) -> bool:
    """True if a HF config object exposes a (truthy) expert-count marker."""
    if cfg is None:
        return False
    for name in dir(cfg):
        if "expert" in name.lower():
            v = getattr(cfg, name, None)
            if isinstance(v, int) and v > 0:
                return True
    for m in ("moe_intermediate_size", "n_routed_experts", "num_local_experts"):
        if getattr(cfg, m, None):
            return True
    return False


def _is_glq_moe(vllm_config) -> bool:
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or getattr(mc, "quantization", None) != "glq":
        return False
    hf = getattr(mc, "hf_config", None)
    cfgs = [hf, getattr(hf, "text_config", None)]
    get_tc = getattr(hf, "get_text_config", None)
    if callable(get_tc):
        try:
            cfgs.append(get_tc())
        except Exception:
            pass
    return any(_looks_moe(c) for c in cfgs)


def _cap(vllm_config) -> None:
    cap = int(os.environ.get("GLQ_MOE_BD_MAX_TOKENS", "256"))
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None:
        return
    sizes = getattr(cc, "cudagraph_capture_sizes", None)
    if not sizes:
        return
    capped = [s for s in sizes if s <= cap]
    if len(capped) == len(sizes):
        return  # already within the cap — no-op
    cc.cudagraph_capture_sizes = capped
    mx = getattr(cc, "max_cudagraph_capture_size", None)
    if mx is not None and (not capped or mx > cap):
        cc.max_cudagraph_capture_size = capped[-1] if capped else 0
    print(f"[glq_vllm] GLQ MoE: capped cudagraph_capture_sizes to <= {cap} "
          f"({len(sizes)} -> {len(capped)}); above that the MoE path is "
          f"non-capturable (cudaErrorStreamCaptureUnsupported). Set "
          f"GLQ_MOE_BD_MAX_TOKENS to change.", flush=True)


def install() -> None:
    """Monkey-patch ``VllmConfig.__post_init__`` to cap GLQ-MoE cudagraph sizes."""
    global _CAP_INSTALLED
    if _CAP_INSTALLED:
        return
    try:
        from vllm.config import VllmConfig
    except Exception:
        return
    if getattr(VllmConfig, "_glq_cudagraph_cap_installed", False):
        _CAP_INSTALLED = True
        return
    _orig = VllmConfig.__post_init__

    def _patched(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        try:
            if _is_glq_moe(self):
                _cap(self)
        except Exception as e:  # never break engine init over this
            print(f"[glq_vllm] cudagraph-cap hook skipped: {e}", flush=True)

    VllmConfig.__post_init__ = _patched
    VllmConfig._glq_cudagraph_cap_installed = True
    _CAP_INSTALLED = True
