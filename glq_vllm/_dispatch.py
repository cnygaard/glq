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


def moe_cpu_fused_refusal(*, fused_shape_ok: bool, has_stage2: bool, unpadded: bool,
                          activation_type: int, ext_has_entry: bool,
                          force_fallback: bool, cpu_fused_enabled: bool) -> str | None:
    """Why ``glq_fused_moe_trellis_3inst_cpu`` cannot serve this layer, or None if it can.

    Every "no" here lands on ``fused_moe_method._apply_trellis``, which is correct on any
    shape — so this gate only ever trades speed for reach, never correctness. The reason
    string exists because the alternative (a silent drop onto the slower loop) is the kind
    of thing that gets diagnosed as "GLQ is slow on CPU" instead of as a missing gate.

    The limits, in the order they are checked:

    * **stage 2** — the CPU op has no ``packed2``/``inv_resid_scale2`` inputs, so 5-8 bpw
      stacked RVQ has no fused CPU MoE path at all.
    * **padding** — the op passes *logical* dims to its per-expert bracket, matching the
      invariant the dense CPU bracket asserts ("trellis layers are unpadded"). A padded
      layer would index the packed tiles wrongly.
    * **shape** — ``m % 32``/``n % 64``/``R`` bounds, already computed at load as
      ``glq_trellis_fused_ok``; the CPU kernel splits work as ``m / 32`` exactly like CUDA.
    * **activation** — the op is gated-only (``w13_out == 2 * intermediate``); ids 0/1/2 are
      silu / gelu-tanh / relu², and 3+ are the ``*_no_mul`` variants.
    * **extension** — a wheel predating the symbol must fall back, not raise.
    """
    if has_stage2:
        return ("stage-2 residual (5-8 bpw stacked RVQ): the fused CPU MoE op decodes one "
                "stage only")
    if not unpadded:
        return ("padded shapes: the fused CPU MoE op assumes trellis' unpadded layout "
                "(n_pad == in_features, m_pad == out_features)")
    if not fused_shape_ok:
        return ("shapes the CPU trellis kernel cannot take (needs m_pad % 32 == 0, "
                "n_pad % 64 == 0 and 2 <= R <= 4)")
    if not 0 <= activation_type < 3:
        return (f"activation id {activation_type}: the fused CPU MoE op is gated-only "
                f"(0 silu, 1 gelu-tanh, 2 relu^2)")
    if not ext_has_entry:
        return ("the CPU extension has no glq_fused_moe_trellis_3inst_cpu entry "
                "(older glq wheel, or the extension failed to load)")
    if force_fallback:
        return "GLQ_MOE_FORCE_FALLBACK is set"
    if not cpu_fused_enabled:
        return "GLQ_FUSED_TRELLIS_CPU=0 is set"
    return None
