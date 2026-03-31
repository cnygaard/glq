"""GLQ quantization plugin for vLLM.

Usage:
    import glq_vllm  # registers "glq" quantization method

    from vllm import LLM
    llm = LLM(model="xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw", quantization="glq")
"""

try:
    from vllm.model_executor.layers.quantization import register_quantization_config
    from .config import GLQvLLMConfig
    register_quantization_config("glq")(GLQvLLMConfig)
    # Register CUDA C kernels as torch custom ops for torch.compile + CUDA graphs
    from .custom_ops import _ensure_registered
    _ensure_registered()
except ImportError:
    pass  # vLLM not installed — dequant helpers still importable
