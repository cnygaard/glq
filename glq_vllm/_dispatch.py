"""Pure GLQ MoE dispatch-policy helpers.

Kept dependency-free (no torch / no vllm / no env reads) so the kernel-selection
policy is unit-testable on a CPU box without a GPU or vLLM installed. The methods
in ``fused_moe_method`` read the env and tensors, then delegate the *decision* here.
"""


def _grouped_enabled(env_val, grouped_min: int, num_tokens: int) -> bool:
    """Tri-state decision for the Stage-3 grouped-GEMM MoE path (GLQ_MOE_GROUPED).

    The grouped path sorts tokens by expert and runs one batched tensor-core GEMM
    per expert — the batched-decode throughput win (validated 26B-A4B: b32 6.9-8.4x
    over the per-(token,expert) block-diag matvec; b1 ~1.15x). It is numerically
    equivalent to but not bit-identical with the matvec oracle (TC accumulation),
    so by default b1 stays on the bit-exact block-diag path.

      "0"/"off"/"false"/"no"  -> never grouped (force block-diag; A/B isolation)
      "1"/"on"/"true"/"yes"   -> force grouped whenever the rest of the gate holds
                                 (including b1)
      None / "" / "auto"/else -> grouped only for batched MoE (num_tokens >=
                                 grouped_min); b1 keeps the block-diag matvec.
    """
    v = (env_val or "auto").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    return num_tokens >= grouped_min


#: Codebooks whose MoE experts can run on the CPU platform. Only trellis: its per-expert
#: fallback (fused_moe_method._apply_trellis) reaches E8RHTLinear._trellis_linear_apply,
#: which has a CPU branch, and every tensor it passes follows the packed buffers' device.
#: e8p and shell have no CPU expert path at all — their per-expert dequant asserts
#: sv.is_cuda — so they must refuse rather than fail deep inside a forward pass.
_CPU_MOE_CODEBOOKS = frozenset({"trellis"})


def moe_cpu_refusal(codebook: str) -> str | None:
    """Why this MoE cannot serve on the CPU platform, or None if it can.

    Unknown codebooks are refused: opting in is a decision that needs a CPU expert path
    behind it, and a wrong "yes" here surfaces as garbage output rather than an error.
    """
    if codebook in _CPU_MOE_CODEBOOKS:
        return None
    return (f"GLQ MoE on the CPU platform is supported for trellis checkpoints only; this "
            f"one is {codebook}, whose expert decode kernels are CUDA-only. Serve it on a "
            f"GPU, or pick a trellis checkpoint for CPU serving.")
