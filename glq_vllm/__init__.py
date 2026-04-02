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
    """
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
        from .config import GLQvLLMConfig
        register_quantization_config("glq")(GLQvLLMConfig)
        from .custom_ops import _ensure_registered
        _ensure_registered()
    except ImportError:
        pass


# Also register on import for backward compat (vLLM 0.16 / manual usage)
register()
