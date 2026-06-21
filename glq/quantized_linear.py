"""E8RHTLinear — quantized linear layer with E8 shell codebook + RHT."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hadamard import fast_hadamard_transform

try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    _triton_available = False

# Kill-switch for the single-call fused E8P linear op (glq_fused_linear_e8p_cuda).
# Default on; flip to False to fall back to the multi-op E8P apply (used by the
# bit-exact A/B test and as a production escape hatch).
_GLQ_FUSED_E8P_ENABLED = True


def _pack_block_meta(block_sizes):
    """Pack (col_offset, bs, log_bs, _) per sub-block as CPU int32 (N, 4).

    C++ reinterprets as ``int4*`` for one coalesced load per CTA in
    ``glq_*_rht_multiblock_kernel``.
    """
    n = len(block_sizes)
    meta = torch.zeros((n, 4), dtype=torch.int32, device="cpu")
    offset = 0
    for i, bs in enumerate(block_sizes):
        meta[i, 0] = offset
        meta[i, 1] = bs
        meta[i, 2] = int(math.log2(bs))
        # meta[i, 3] = 0 (unused padding for int4 alignment)
        offset += bs
    return meta


# ────────────────────────────────────────────────────────────────
# Fused RHT kernels: pad+SV+FHT and FHT+SU+unpad in one launch
# ────────────────────────────────────────────────────────────────

if _triton_available:

    @triton.jit
    def _input_rht_kernel(
        x_ptr,          # (B, in_features) fp16/fp32 input
        sv_ptr,         # (n_pad,) fp16 sign vector
        out_ptr,        # (B, n_pad) fp32 output — x_rht
        in_features,
        stride_x,       # x.stride(0)
        rsqrt_n,        # 1 / sqrt(n_pad)
        N: tl.constexpr,
        LOG_N: tl.constexpr,
    ):
        """Fused pad + SV multiply + FHT in one kernel launch.

        Replaces: x.float() -> F.pad -> * SV -> FHT -> x_rht
        """
        row = tl.program_id(0)
        offs = tl.arange(0, N)
        base = out_ptr + row * N

        # Load x with zero-padding beyond in_features, cast to fp32
        x = tl.load(x_ptr + row * stride_x + offs,
                     mask=offs < in_features, other=0.0).to(tl.float32)

        # Apply SV signs
        sv = tl.load(sv_ptr + offs).to(tl.float32)
        x = x * sv

        # FHT butterfly stages (in-place via global memory + barrier)
        for k in tl.static_range(LOG_N):
            tl.store(base + offs, x)
            tl.debug_barrier()
            x_partner = tl.load(base + (offs ^ (1 << k)))
            lo = (offs & (1 << k)) == 0
            x = tl.where(lo, x + x_partner, x_partner - x)

        # Store normalized result
        tl.store(base + offs, x * rsqrt_n)

    @triton.jit
    def _output_rht_kernel(
        y_rht_ptr,      # (B, m_pad) fp32 input — y_rht from dequant
        su_ptr,         # (m_pad,) fp16 sign vector
        out_ptr,        # (B, out_features) output in target dtype
        out_features,
        stride_y,       # y_rht.stride(0)
        stride_out,     # out.stride(0)
        rsqrt_m,        # 1 / sqrt(m_pad)
        OUTPUT_FP16: tl.constexpr,
        M: tl.constexpr,
        LOG_M: tl.constexpr,
    ):
        """Fused FHT + SU multiply + unpad + cast in one kernel launch.

        Replaces: FHT(y_rht) -> * SU -> [:, :out_features] -> .to(dtype)
        """
        row = tl.program_id(0)
        offs = tl.arange(0, M)
        base = y_rht_ptr + row * stride_y

        # Load y_rht
        x = tl.load(base + offs)

        # FHT butterfly stages
        for k in tl.static_range(LOG_M):
            tl.store(base + offs, x)
            tl.debug_barrier()
            x_partner = tl.load(base + (offs ^ (1 << k)))
            lo = (offs & (1 << k)) == 0
            x = tl.where(lo, x + x_partner, x_partner - x)

        # Normalize, apply SU, and store with unpadding + dtype cast
        su = tl.load(su_ptr + offs).to(tl.float32)
        x = x * rsqrt_m * su
        mask = offs < out_features
        if OUTPUT_FP16:
            tl.store(out_ptr + row * stride_out + offs, x.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + row * stride_out + offs, x, mask=mask)


class E8RHTLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with E8+RHT quantized weights.

    Stores quantized weights as codebook indices + RHT sign vectors.
    On forward, dequantizes in the RHT domain and uses Hadamard transforms
    on input/output to avoid full weight materialization.

    Buffers (loaded from safetensors):
        Qidxs: (m_pad, n_pad // 8) int16 — primary codebook indices
        SU: (m_pad,) float16 — row sign vector (±1)
        SV: (n_pad,) float16 — column sign vector (±1)
        Wscale: () float32 — global scale factor
        Qidxs2: (m_pad, n_pad // 8) int16 — secondary indices (3/4bpw), optional
        inv_resid_scale: () float32 — 1/resid_scale (3/4bpw), optional

    Shared state (set after loading):
        codebook: E8ShellCodebook instance (primary)
        codebook2: E8ShellCodebook instance (secondary, for 3/4bpw)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_diagonal: bool = False, codebook_type: str = "e8_shell"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # E8P uses int64 tensor-core-packed weights + a pow2 RHT pad (n_pad÷64,
        # m_pad÷16). Detected here so the right buffers get registered; the
        # forward early-returns into _forward_e8p, leaving the shell path intact.
        self._is_e8p = (codebook_type == "e8p")
        if self._is_e8p:
            block_diagonal = False

        if block_diagonal:
            from .hadamard import _block_decompose
            self.blocks_m = _block_decompose(out_features)
            self.blocks_n = _block_decompose(in_features)
            self.m_pad = sum(self.blocks_m)  # = out_features (no padding)
            self.n_pad = sum(self.blocks_n)  # = in_features (no padding)
        else:
            # Legacy: pad to next power of 2
            self.m_pad = 1 << (out_features - 1).bit_length() if out_features > 0 else 1
            self.n_pad = 1 << (in_features - 1).bit_length() if in_features > 0 else 1
            self.blocks_m = [self.m_pad]
            self.blocks_n = [self.n_pad]
        self.block_diagonal = block_diagonal

        if self._is_e8p:
            self._init_e8p_buffers(bias)
            return

        # Quantized weight storage
        self.register_buffer('Qidxs', torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('SU', torch.ones(self.m_pad, dtype=torch.float16))
        self.register_buffer('SV', torch.ones(self.n_pad, dtype=torch.float16))
        self.register_buffer('Wscale', torch.ones((), dtype=torch.float32))

        # Multi-stage buffers (2bpw layers leave as zeros; missing keys handled on load).
        self.register_buffer('Qidxs2', torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('inv_resid_scale', torch.zeros((), dtype=torch.float32))
        # Stages 3-4 for 5-8bpw
        self.register_buffer('Qidxs3', torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('inv_resid_scale2', torch.zeros((), dtype=torch.float32))
        self.register_buffer('Qidxs4', torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('inv_resid_scale3', torch.zeros((), dtype=torch.float32))
        self._has_stage2 = False
        self._n_stages = 1

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Codebook references — set after loading via set_codebook()
        self.codebook = None
        self.codebook2 = None

        # Cached scalar values (set in set_codebook, avoids GPU→CPU sync per forward)
        self._wscale_float = 1.0
        self._inv_rs_float = 0.0
        self._inv_rs2_float = 0.0
        self._inv_rs3_float = 0.0

        # Eager block-diagonal metadata (CPU int64; read host-side by the C++
        # wrapper's launch loop). device="cpu" is explicit so HF's meta-device
        # init context doesn't promote these to meta. Kept as plain attrs so
        # they're not persisted to state_dict or moved to GPU by .to(cuda).
        self._blocks_n_tensor = torch.tensor(self.blocks_n, dtype=torch.int64, device="cpu")
        self._blocks_m_tensor = torch.tensor(self.blocks_m, dtype=torch.int64, device="cpu")

        # Packed metadata for the multiblock FHT kernel: one int4 per sub-block
        # {col_offset, bs, log_bs, _padding}. Built on CPU here; pushed to GPU
        # lazily on first forward (cached by device).
        self._blocks_n_meta_cpu = _pack_block_meta(self.blocks_n)
        self._blocks_m_meta_cpu = _pack_block_meta(self.blocks_m)
        self._blocks_n_meta_gpu = None  # cached (device, tensor)
        self._blocks_m_meta_gpu = None

        # Cached empty placeholders (lazy device-keyed init in forward) so that
        # repeated `torch.empty(0, ...)` calls don't allocate during graph capture.
        self._empty_i16 = None
        self._empty_f16 = None

    def _init_e8p_buffers(self, bias):
        """Register the E8P int64 tensor-core-packed buffers (used instead of the
        shell's int16 Qidxs). The stage-2 buffer (Qidxs2_e8p for 4bpw / Qidxs2_e81b
        for 3bpw) is registered lazily at load from the checkpoint's exact shape so
        2/3bpw layers carry no dead weight."""
        mp16 = max(self.m_pad // 16, 1)
        nb64 = max(self.n_pad // 64, 1)
        self.register_buffer('Qidxs_e8p', torch.zeros(mp16, nb64, 8, 4, dtype=torch.int64))
        self.register_buffer('SU', torch.ones(self.m_pad, dtype=torch.float16))
        self.register_buffer('SV', torch.ones(self.n_pad, dtype=torch.float16))
        self.register_buffer('Wscale', torch.ones((), dtype=torch.float32))
        self.register_buffer('inv_resid_scale', torch.zeros((), dtype=torch.float32))
        # Stage-2 buffers as 0-size placeholders. from_pretrained's meta-assign path
        # (set_module_tensor_to_device) only populates buffers that already exist, so
        # these MUST be registered here (not lazily in _load_from_state_dict). The
        # recipe uses exactly one — Qidxs2_e8p (4bpw) or Qidxs2_e81b (3bpw) — and the
        # checkpoint carries only that key, so the unused one stays 0-size (no dead
        # weight). Non-meta load_state_dict resizes the used one to the checkpoint shape.
        self.register_buffer('Qidxs2_e8p', torch.zeros(0, dtype=torch.int64))
        self.register_buffer('Qidxs2_e81b', torch.zeros(0, dtype=torch.int64))
        if bias:
            self.register_buffer('bias', torch.zeros(self.out_features, dtype=torch.float16))
        else:
            self.bias = None
        self.codebook = None        # E8PCodebook (grid_packed_abs)
        self.codebook2 = None       # E8PCodebook (4bpw) or E81BCodebook (3bpw)
        self._wscale_float = 1.0
        self._inv_rs_float = 0.0
        self._e8p_has_stage2 = False   # cached stage-flags (set on first forward / set_codebook)
        self._e8p_has_e81b = False
        self._e8p_grid_dev = None   # cached grid_packed_abs on the compute device
        self._e81b_grid_dev = None  # cached e81b_grid on the compute device

    @property
    def weight(self):
        """Proxy so code checking weight.device works (e.g. Mamba).

        Returns a zero-element tensor on the same device as the quantized
        weights, without allocating any real memory.
        """
        dev = self.SV.device if self._is_e8p else self.Qidxs.device
        return torch.empty(0, dtype=torch.float16, device=dev)

    def set_codebook(self, codebook, codebook2=None):
        """Attach the shared codebook(s) (called after weight loading)."""
        self.codebook = codebook
        self.codebook2 = codebook2
        if getattr(codebook, 'is_e8p', False):
            self._is_e8p = True
        if self._is_e8p:
            # E8P caches grid_packed_abs / e81b_grid on the compute device lazily
            # in forward; only the two scalars need resolving here.
            self._e8p_grid_dev = None
            self._e81b_grid_dev = None
            if self.Wscale.device.type == "meta":
                self._wscale_float = None
                return
            self._wscale_float = self.Wscale.item()
            self._inv_rs_float = self.inv_resid_scale.item()
            return
        # Skip scalar caching for meta tensors (offloaded layers);
        # values will be resolved lazily in forward() if needed.
        if self.Wscale.device.type == "meta":
            self._has_stage2 = codebook2 is not None
            self._wscale_float = None
            self._inv_rs_float = None
            return
        # Detect number of stages from inv_resid_scale buffers
        self._has_stage2 = (
            codebook2 is not None
            and self.inv_resid_scale is not None
            and self.inv_resid_scale.abs().item() > 0
        )
        self._n_stages = 1
        if self._has_stage2:
            self._n_stages = 2
            if self.inv_resid_scale2.abs().item() > 0:
                self._n_stages = 3
                if self.inv_resid_scale3.abs().item() > 0:
                    self._n_stages = 4
        # Cache scalar values to avoid GPU→CPU sync on every forward pass
        self._wscale_float = self.Wscale.item()
        self._inv_rs_float = self.inv_resid_scale.item() if self._has_stage2 else 0.0
        self._inv_rs2_float = self.inv_resid_scale2.item() if self._n_stages >= 3 else 0.0
        self._inv_rs3_float = self.inv_resid_scale3.item() if self._n_stages >= 4 else 0.0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Adapt buffer sizes to match checkpoint on load.

        Handles both legacy (power-of-2 padded) and block-diagonal (unpadded)
        checkpoints by resizing buffers to match the checkpoint's actual shapes.
        Also handles 2bpw models that omit Qidxs2/inv_resid_scale.
        """
        if self._is_e8p:
            # E8P buffers are stored at their exact pow2-padded shape — no runtime
            # padding. Register whichever stage-2 buffer the recipe used (present in
            # the checkpoint) at its exact shape, and default inv_resid_scale to 0
            # for 2bpw so a full load doesn't flag it missing.
            _full = (prefix + 'Qidxs_e8p') in state_dict
            # Resize the pre-registered 0-size stage-2 buffer to the checkpoint shape
            # so super()'s copy matches (non-meta path). The meta-assign path replaces
            # the buffer object wholesale, so this is a no-op there.
            for suffix in ('Qidxs2_e8p', 'Qidxs2_e81b'):
                key = prefix + suffix
                if key in state_dict:
                    buf = getattr(self, suffix)
                    t = state_dict[key]
                    if buf.device.type != 'meta' and buf.shape != t.shape:
                        buf.resize_(t.shape)
            if _full and (prefix + 'inv_resid_scale') not in state_dict:
                state_dict[prefix + 'inv_resid_scale'] = torch.zeros((), dtype=torch.float32)
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata,
                strict, missing_keys, unexpected_keys, error_msgs)
            return
        # Pad if checkpoint is smaller than current buffers (legacy compat)
        for suffix in ('Qidxs', 'Qidxs2', 'Qidxs3', 'Qidxs4'):
            key = prefix + suffix
            if key in state_dict:
                t = state_dict[key]
                target = (self.m_pad, self.n_pad // 8)
                if t.shape[0] < target[0] or t.shape[1] < target[1]:
                    padded = torch.zeros(target, dtype=t.dtype, device=t.device)
                    padded[:t.shape[0], :t.shape[1]] = t
                    state_dict[key] = padded
        su_key = prefix + 'SU'
        if su_key in state_dict and state_dict[su_key].shape[0] < self.m_pad:
            t = state_dict[su_key]
            padded = torch.ones(self.m_pad, dtype=t.dtype, device=t.device)
            padded[:t.shape[0]] = t
            state_dict[su_key] = padded
        sv_key = prefix + 'SV'
        if sv_key in state_dict and state_dict[sv_key].shape[0] < self.n_pad:
            t = state_dict[sv_key]
            padded = torch.ones(self.n_pad, dtype=t.dtype, device=t.device)
            padded[:t.shape[0]] = t
            state_dict[sv_key] = padded
        # Only inject zero placeholders for optional buffers that are BOTH:
        # (a) missing from the checkpoint, AND (b) this is a full-model load
        # (not a per-key load). Detect per-key loads by checking if the
        # primary 'Qidxs' key is in the state_dict (it always is for full loads).
        _primary_key = prefix + 'Qidxs'
        _is_full_load = _primary_key in state_dict
        if _is_full_load:
            for suffix, default_fn in [
                ('Qidxs2', lambda: torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16)),
                ('inv_resid_scale', lambda: torch.zeros((), dtype=torch.float32)),
                ('Qidxs3', lambda: torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16)),
                ('inv_resid_scale2', lambda: torch.zeros((), dtype=torch.float32)),
                ('Qidxs4', lambda: torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16)),
                ('inv_resid_scale3', lambda: torch.zeros((), dtype=torch.float32)),
            ]:
                key = prefix + suffix
                if key not in state_dict:
                    state_dict[key] = default_fn()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)

    def _pad_if_needed(self):
        """Adapt buffer dimensions on first forward.

        Handles accelerate dispatch (which bypasses _load_from_state_dict).
        Detects block-diagonal checkpoints (non-power-of-2 buffer shapes)
        and reconfigures m_pad/n_pad/blocks accordingly.
        """
        actual_m = self.Qidxs.shape[0]
        actual_n = self.Qidxs.shape[1] * 8
        # Detect block-diagonal: buffer dims differ from init AND are not power-of-2
        if actual_m != self.m_pad or actual_n != self.n_pad:
            is_pow2_m = actual_m > 0 and (actual_m & (actual_m - 1)) == 0
            is_pow2_n = actual_n > 0 and (actual_n & (actual_n - 1)) == 0
            if not (is_pow2_m and is_pow2_n):
                from .hadamard import _block_decompose
                self.m_pad = actual_m
                self.n_pad = actual_n
                self.block_diagonal = True
                self.blocks_m = _block_decompose(actual_m)
                self.blocks_n = _block_decompose(actual_n)
                # Rebuild Phase A/B metadata: __init__ built these from the
                # pow2 default if _detect_block_diagonal missed the checkpoint
                # (e.g. local path). Stale metadata would cause OOB reads in
                # the fused block-diag kernel once we reach it.
                self._blocks_n_tensor = torch.tensor(self.blocks_n, dtype=torch.int64, device="cpu")
                self._blocks_m_tensor = torch.tensor(self.blocks_m, dtype=torch.int64, device="cpu")
                self._blocks_n_meta_cpu = _pack_block_meta(self.blocks_n)
                self._blocks_m_meta_cpu = _pack_block_meta(self.blocks_m)
                self._blocks_n_meta_gpu = None
                self._blocks_m_meta_gpu = None
                return
        if self.Qidxs.shape[0] == self.m_pad:
            return
        dev = self.Qidxs.device
        target = (self.m_pad, self.n_pad // 8)
        for name in ('Qidxs', 'Qidxs2', 'Qidxs3', 'Qidxs4'):
            old = getattr(self, name)
            if old.shape[0] < target[0] or old.shape[1] < target[1]:
                new = torch.zeros(target, dtype=old.dtype, device='cpu')
                new[:old.shape[0], :old.shape[1]] = old.cpu()
                self.register_buffer(name, new.to(dev))
        if self.SU.shape[0] < self.m_pad:
            new = torch.ones(self.m_pad, dtype=self.SU.dtype, device='cpu')
            new[:self.SU.shape[0]] = self.SU.cpu()
            self.register_buffer('SU', new.to(dev))
        if self.SV.shape[0] < self.n_pad:
            new = torch.ones(self.n_pad, dtype=self.SV.dtype, device='cpu')
            new[:self.SV.shape[0]] = self.SV.cpu()
            self.register_buffer('SV', new.to(dev))
        torch.cuda.empty_cache()

    def _ensure_codebook_device(self):
        """Move codebook(s) to match Qidxs device (lazy, once).

        Uses in-place device move to preserve sharing across modules.
        All E8RHTLinear modules share the same codebook object, so moving
        it once moves it for all.
        """
        if self.codebook is not None and self.codebook.codebook.device != self.Qidxs.device:
            self.codebook._move_to_device(self.Qidxs.device)
        if self.codebook2 is not None and self.codebook2.codebook.device != self.Qidxs.device:
            self.codebook2._move_to_device(self.Qidxs.device)

    def _forward_e8p(self, x: torch.Tensor) -> torch.Tensor:
        """E8P-RVQ forward: input_rht (CUDA) → N× tensor-core decode → output_rht.

        Reuses glq's CUDA RHT ops (no Python Hadamard). B=1 uses the TC-GEMV
        (decode_matvec_e8p) and WMMA (lookupmatmul_e81b_k8) matvec kernels; B>1
        decompresses the weight and runs a dense matmul (prefill / PPL).
        """
        # Resolve cached scalars / grids / stage-flags once (lazily for layers on
        # meta at set_codebook time). After this the shared core has no host syncs.
        if self._wscale_float is None:
            self._wscale_float = self.Wscale.item()
            self._inv_rs_float = self.inv_resid_scale.item()
        if self._e8p_grid_dev is None or self._e8p_grid_dev.device != x.device:
            self._e8p_grid_dev = self.codebook.grid_packed_abs.to(x.device)
            if self.codebook2 is not None and hasattr(self.codebook2, 'e81b_grid'):
                self._e81b_grid_dev = self.codebook2.e81b_grid.to(x.device)
            self._e8p_has_stage2 = self.Qidxs2_e8p.numel() > 0
            self._e8p_has_e81b = self.Qidxs2_e81b.numel() > 0

        shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        y = self._e8p_linear_apply(
            x2d, self.SV, self.SU, self.Qidxs_e8p, self.Qidxs2_e8p, self.Qidxs2_e81b,
            self._e8p_grid_dev, self._e81b_grid_dev, self._wscale_float, self._inv_rs_float,
            self._e8p_has_stage2, self._e8p_has_e81b, self.in_features, self.out_features,
            self.n_pad, self.m_pad, bias=self.bias, out_dtype=x.dtype)
        return y.reshape(*shape[:-1], self.out_features)

    @staticmethod
    def _e8p_linear_apply(x2d, SV, SU, Qidxs_e8p, Qidxs2_e8p, Qidxs2_e81b, grid, e81b_grid,
                          wscale, inv_rs, has_e8p2, has_e81b, in_features, out_features,
                          n_pad, m_pad, bias=None, out_dtype=torch.float16):
        """Capture-safe E8P linear core: input_rht → N× TC-GEMV decode → ×Wscale → output_rht.

        x2d: (B, in_features) → (B, out_features) in out_dtype, bias applied. Prefers the
        registered torch.ops.glq.* ops (traceable / cudagraph-capturable) over the pybind
        extension. Shared by E8RHTLinear._forward_e8p (HF) and glq_vllm's e8p apply (serving)
        so both run one validated path. `has_e8p2` (4bpw E8P residual) and `has_e81b` (3bpw
        E81B residual) are independent + mutually exclusive per layer."""
        from . import inference_kernel as _ik
        _ik._try_load_cuda_ext()
        cuda = _ik._glq_cuda
        _ops = (torch.ops.glq if (hasattr(torch.ops, "glq")
                and hasattr(torch.ops.glq, "decode_matvec_e8p")) else None)
        if _ops is not None:
            _input_rht, _output_rht = _ops.input_rht, _ops.output_rht
            _decode, _decompress = _ops.decode_matvec_e8p, _ops.decompress_packed_e8p
            _e81b_mm, _e81b_dec = _ops.lookupmatmul_e81b_k8, _ops.decompress_e81b_packed
        else:
            _input_rht, _output_rht = cuda.glq_input_rht_cuda, cuda.glq_output_rht_cuda
            _decode, _decompress = cuda.glq_decode_matvec_e8p, cuda.glq_decompress_packed_e8p
            _e81b_mm, _e81b_dec = cuda.glq_lookupmatmul_e81b_k8, cuda.glq_decompress_e81b_packed

        B = x2d.shape[0]
        dev = x2d.device
        log_n, log_m = int(math.log2(n_pad)), int(math.log2(m_pad))
        rsqrt_n, rsqrt_m = 1.0 / math.sqrt(n_pad), 1.0 / math.sqrt(m_pad)

        # Fast path: the entire linear (input_rht + N-stage decode/matmul + ×Wscale
        # + output_rht) as ONE opaque op — one torch.ops dispatch, traced as a single
        # node so vLLM's cudagraph capture matches the shell's fused_linear instead of
        # a multi-op chain inductor pads with copies. Covers stage-1 (2bpw) + E8P
        # stage-2 (4bpw); E81B (3bpw) falls through to the multi-op path below.
        if (_ops is not None and _GLQ_FUSED_E8P_ENABLED
                and hasattr(_ops, "fused_linear_e8p") and not has_e81b):
            q2 = Qidxs2_e8p if has_e8p2 else Qidxs2_e8p.new_empty(0)
            rs = float(inv_rs) if has_e8p2 else 0.0
            y = _ops.fused_linear_e8p(
                x2d.half().contiguous(), SV, SU, Qidxs_e8p, q2, grid,
                float(wscale), rs, in_features, out_features,
                n_pad, m_pad, log_n, log_m).to(out_dtype)
            if bias is not None:
                y = y + bias.unsqueeze(0).to(out_dtype)
            return y

        # 1. input RHT: pad + SV signs + FHT  →  (B, n_pad) fp32
        x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=dev)
        _input_rht(x2d.half().contiguous(), SV, x_rht, in_features, in_features, rsqrt_n, n_pad, log_n)

        # 2. decode + matmul in the RHT domain  →  y_rht (B, m_pad) fp32
        if B == 1:
            xh = x_rht[0].half()
            y_rht = _decode(xh, Qidxs_e8p, grid)
            if has_e8p2 and inv_rs != 0.0:
                y_rht = y_rht + inv_rs * _decode(xh, Qidxs2_e8p, grid)
            elif has_e81b and inv_rs != 0.0:
                X8 = torch.zeros(8, n_pad, dtype=torch.float16, device=dev)
                X8[0] = xh
                Z = torch.zeros(8, m_pad, dtype=torch.float32, device=dev)
                _e81b_mm(X8, Qidxs2_e81b, e81b_grid, Z)
                y_rht = y_rht + inv_rs * Z[0]
            y_rht = y_rht.unsqueeze(0)
        else:
            # Anchor the decompressed dense weights to the activation device.
            # The decompress custom op's fake derives its output device from
            # ``weights_compressed.device``; under vLLM's fullgraph dynamo trace
            # that param fake can mis-propagate (one operand lands cpu), so the
            # ``W + inv_rs*W2`` add hits a cuda/cpu mismatch even though every
            # buffer is genuinely on-device at runtime. ``.to(dev)`` (a no-op on
            # the live tensors) forces a consistent fake device == x2d's.
            W = _decompress(Qidxs_e8p, grid).float().to(dev)
            if has_e8p2 and inv_rs != 0.0:
                W = W + inv_rs * _decompress(Qidxs2_e8p, grid).float().to(dev)
            elif has_e81b and inv_rs != 0.0:
                Y = torch.zeros(m_pad, n_pad, dtype=torch.float16, device=dev)
                _e81b_dec(Qidxs2_e81b, e81b_grid, Y)
                W = W + inv_rs * Y.float()
            y_rht = x_rht @ W.T

        # 3. scale + output RHT: FHT + SU signs + unpad  →  (B, out_features)
        y_rht = (y_rht * wscale).contiguous()
        y = torch.empty(B, out_features, dtype=torch.float16, device=dev)
        _output_rht(y_rht, SU, y, out_features, m_pad, log_m, rsqrt_m)
        y = y.to(out_dtype)
        if bias is not None:
            y = y + bias.unsqueeze(0).to(out_dtype)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with RHT-domain matmul.

        1. Pad input, apply SV signs, FHT → transform to RHT domain
        2. Fused dequant+matmul in RHT domain (Triton) or decode+matmul (fallback)
        3. FHT on y_rht, apply SU signs, unpad → inverse RHT on output
        """
        if self._is_e8p:
            return self._forward_e8p(x)
        self._pad_if_needed()
        self._ensure_codebook_device()
        # Lazy-resolve cached scalars (for layers that were on meta at set_codebook time)
        if self._wscale_float is None:
            self._wscale_float = self.Wscale.item()
            self._has_stage2 = (
                self.codebook2 is not None
                and self.inv_resid_scale is not None
                and self.inv_resid_scale.abs().item() > 0
            )
            self._inv_rs_float = self.inv_resid_scale.item() if self._has_stage2 else 0.0

        shape = x.shape
        x = x.reshape(-1, self.in_features)
        B = x.shape[0]
        dtype = x.dtype
        # The B>=2 Tensor Core branch in glq_fused_linear{,_block_diag}_cuda
        # scales with B via dynamic scratch alloc and a 3-D grid; a static
        # B cap here was vestigial and starved lm-eval / prefill (flattened
        # B=seq*batch easily exceeds 64).
        use_fused = x.is_cuda and _triton_available
        _nvtx = torch.cuda.nvtx if x.is_cuda else None
        has_stage2 = self._has_stage2

        # Try fully-fused C++ path: input_rht + dequant_matmul + output_rht in 1 call
        from . import inference_kernel as _ik
        n_pad = self.n_pad
        m_pad = self.m_pad
        _is_pow2 = not self.block_diagonal
        if (use_fused and n_pad <= 32768 and m_pad <= 32768
                and self._n_stages <= 4
                and _ik._try_load_cuda_ext()
                and hasattr(_ik._glq_cuda, 'glq_fused_linear_cuda')):
            if self._empty_i16 is None or self._empty_i16.device != x.device:
                self._empty_i16 = torch.empty(0, dtype=torch.int16, device=x.device)
                self._empty_f16 = torch.empty(0, dtype=torch.float16, device=x.device)
            _empty_i16 = self._empty_i16
            _empty_f16 = self._empty_f16
            n_stages = self._n_stages
            cb2_tensor = self.codebook2.codebook_half if has_stage2 else _empty_f16
            # Stages 3+ always use the primary (full) E8 codebook, matching the
            # Python fallback at `_extra_stages` below. Indices into a 65536-entry
            # codebook → pass cb3_size=0 path (global codebook gather) — no smem
            # staging for stages 3/4.
            primary_cb = self.codebook.codebook_half
            q3_tensor = self.Qidxs3 if n_stages >= 3 else _empty_i16
            cb3_tensor = primary_cb if n_stages >= 3 else _empty_f16
            q4_tensor = self.Qidxs4 if n_stages >= 4 else _empty_i16
            cb4_tensor = primary_cb if n_stages >= 4 else _empty_f16
            irs2_float = self._inv_rs2_float
            irs3_float = self._inv_rs3_float
            # Prefer torch.ops.glq.* dispatch when registered (by glq_vllm,
            # if installed) so torch.compile can trace these kernels as
            # opaque ops. Falls back to direct pybind11 otherwise — same
            # binding, just visible to dynamo.
            _have_custom = (hasattr(torch.ops, "glq")
                            and hasattr(torch.ops.glq, "fused_linear"))
            x_half = x.half().contiguous()
            q2_arg = self.Qidxs2 if has_stage2 else _empty_i16
            if _is_pow2:
                args = (
                    x_half, self.SV, self.SU,
                    self.Qidxs, primary_cb,
                    self._wscale_float,
                    self.in_features, self.out_features,
                    n_pad, m_pad,
                    int(math.log2(n_pad)), int(math.log2(m_pad)),
                    q2_arg, cb2_tensor, self._inv_rs_float,
                    q3_tensor, cb3_tensor, irs2_float,
                    q4_tensor, cb4_tensor, irs3_float,
                )
                if _have_custom:
                    y = torch.ops.glq.fused_linear(*args)
                else:
                    y = _ik._glq_cuda.glq_fused_linear_cuda(*args)
            elif hasattr(_ik._glq_cuda, 'glq_fused_linear_block_diag_cuda'):
                # Lazily push packed metadata to x.device (cache per device)
                if self._blocks_n_meta_gpu is None or self._blocks_n_meta_gpu.device != x.device:
                    self._blocks_n_meta_gpu = self._blocks_n_meta_cpu.to(x.device, non_blocking=True)
                    self._blocks_m_meta_gpu = self._blocks_m_meta_cpu.to(x.device, non_blocking=True)
                args = (
                    x_half, self.SV, self.SU,
                    self.Qidxs, primary_cb,
                    self._wscale_float,
                    self.in_features, self.out_features,
                    n_pad, m_pad,
                    self._blocks_n_tensor, self._blocks_m_tensor,
                    self._blocks_n_meta_gpu, self._blocks_m_meta_gpu,
                    q2_arg, cb2_tensor, self._inv_rs_float,
                    q3_tensor, cb3_tensor, irs2_float,
                    q4_tensor, cb4_tensor, irs3_float,
                )
                if _have_custom and hasattr(torch.ops.glq,
                                             "fused_linear_block_diag"):
                    y = torch.ops.glq.fused_linear_block_diag(*args)
                else:
                    y = _ik._glq_cuda.glq_fused_linear_block_diag_cuda(*args)
            else:
                y = None  # fall through to fallback
            if y is not None:
                if dtype != torch.float16:
                    y = y.to(dtype)
                if self.bias is not None:
                    y = y + self.bias.unsqueeze(0).to(dtype)
                return y.reshape(*shape[:-1], self.out_features)

        # Fallback: 3 separate kernel calls
        # Transform input to RHT domain
        if _nvtx: _nvtx.range_push("input_rht")
        if _is_pow2 and use_fused:
            log_n = int(math.log2(n_pad))
            x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=x.device)
            if n_pad <= 32768 and _ik._try_load_cuda_ext():
                _ik._glq_cuda.glq_input_rht_cuda(
                    x.half().contiguous(), self.SV, x_rht,
                    self.in_features, self.in_features,
                    1.0 / math.sqrt(n_pad), n_pad, log_n)
            else:
                _input_rht_kernel[(B,)](
                    x, self.SV, x_rht,
                    self.in_features, x.stride(0),
                    1.0 / math.sqrt(n_pad),
                    N=n_pad, LOG_N=log_n,
                    num_warps=8,
                )
        else:
            from .hadamard import block_diagonal_fht
            x_f = x.float()
            x_pad = F.pad(x_f, (0, n_pad - self.in_features))
            sv = self.SV.float()
            x_rht = block_diagonal_fht(x_pad * sv.unsqueeze(0), self.blocks_n)

        if _nvtx: _nvtx.range_pop()  # input_rht

        # Dequant + matmul in RHT domain
        if _nvtx: _nvtx.range_push("dequant_matmul")
        n_stages = self._n_stages
        if x_rht.is_cuda and _triton_available and n_stages <= 2:
            # Triton/CUDA path: up to 2 stages
            from .inference_kernel import glq_dequant_matmul
            cb2_tensor = self.codebook2.codebook_half if has_stage2 else None
            cb_packed = getattr(self.codebook, 'codebook_packed', None)
            y_rht = glq_dequant_matmul(
                x_rht, self.Qidxs,
                self.codebook.codebook_half,
                self._wscale_float,
                Qidxs2=self.Qidxs2 if has_stage2 else None,
                codebook2=cb2_tensor,
                inv_resid_scale=self._inv_rs_float,
                codebook_packed=cb_packed,
            )
        else:
            # PyTorch fallback: supports N stages
            W_rht = self.codebook.decode(self.Qidxs.long().reshape(-1))
            W_rht = W_rht.reshape(m_pad, n_pad)
            if has_stage2:
                cb2 = self.codebook2 if self.codebook2 is not None else self.codebook
                W_rht2 = cb2.decode(self.Qidxs2.long().reshape(-1))
                W_rht2 = W_rht2.reshape(m_pad, n_pad)
                W_rht = W_rht + W_rht2 * self.inv_resid_scale.item()
            # Stages 3-4 (5-8bpw)
            _extra_stages = [
                (self.Qidxs3, self.inv_resid_scale2),
                (self.Qidxs4, self.inv_resid_scale3),
            ]
            for qidxs_i, inv_rs_i in _extra_stages:
                if inv_rs_i.abs().item() == 0:
                    break
                cb_i = self.codebook  # all extra stages use full codebook
                W_i = cb_i.decode(qidxs_i.long().reshape(-1)).reshape(m_pad, n_pad)
                W_rht = W_rht + W_i * inv_rs_i.item()
            y_rht = x_rht @ W_rht.T * self.Wscale.float()

        if _nvtx: _nvtx.range_pop()  # dequant_matmul

        # Inverse RHT on output
        if _nvtx: _nvtx.range_push("output_rht")
        if _is_pow2 and use_fused:
            log_m = int(math.log2(m_pad))
            if m_pad <= 32768 and _ik._try_load_cuda_ext():
                y = torch.empty(B, self.out_features, dtype=torch.float16, device=x.device)
                _ik._glq_cuda.glq_output_rht_cuda(
                    y_rht, self.SU, y,
                    self.out_features, m_pad, log_m,
                    1.0 / math.sqrt(m_pad))
                if dtype != torch.float16:
                    y = y.to(dtype)
            else:
                output_fp16 = (dtype == torch.float16)
                y = torch.empty(B, self.out_features, dtype=dtype, device=x.device)
                _output_rht_kernel[(B,)](
                    y_rht, self.SU, y,
                    self.out_features, y_rht.stride(0), y.stride(0),
                    1.0 / math.sqrt(m_pad),
                    OUTPUT_FP16=output_fp16,
                    M=m_pad, LOG_M=log_m,
                    num_warps=8,
                )
            if self.bias is not None:
                y = y + self.bias.unsqueeze(0).to(dtype)
        else:
            from .hadamard import block_diagonal_fht
            su = self.SU.float()
            y = block_diagonal_fht(y_rht, self.blocks_m) * su.unsqueeze(0)
            y = y[:, :self.out_features]
            if self.bias is not None:
                y = y + self.bias.float().unsqueeze(0)
            y = y.to(dtype)

        if _nvtx: _nvtx.range_pop()  # output_rht

        return y.reshape(*shape[:-1], self.out_features)

    def dequantize(self) -> torch.Tensor:
        """
        Full weight dequantization for debugging/validation.
        Returns (out_features, in_features) dense weight matrix.
        """
        self._pad_if_needed()
        self._ensure_codebook_device()
        W_rht = self.codebook.decode(self.Qidxs.long().reshape(-1))
        W_rht = W_rht.reshape(self.m_pad, self.n_pad)
        if self._has_stage2:
            W_rht2 = self.codebook2.decode(self.Qidxs2.long().reshape(-1))
            W_rht2 = W_rht2.reshape(self.m_pad, self.n_pad)
            W_rht = W_rht + W_rht2 * self.inv_resid_scale.item()
        W_rht = W_rht * self.Wscale.float()

        # Inverse RHT
        sv = self.SV.float()
        su = self.SU.float()

        W_t = fast_hadamard_transform(W_rht.clone())
        W_t = W_t * sv.unsqueeze(0)

        W_t = W_t.T
        W_t = fast_hadamard_transform(W_t.clone())
        W_t = W_t.T

        W_t = W_t * su.unsqueeze(1)

        return W_t[:self.out_features, :self.in_features]

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, m_pad={self.m_pad}, n_pad={self.n_pad}')


# ────────────────────────────────────────────────────────────────
# E8RHTEmbedding — quantized embedding with per-row gather
# ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _dequant_embedding_rows(
    input_ids: torch.Tensor,
    qidxs: torch.Tensor,
    sv: torch.Tensor,
    wscale: torch.Tensor,
    codebook: torch.Tensor,
    qidxs2: torch.Tensor | None,
    inv_resid_scale: torch.Tensor | None,
    codebook2: torch.Tensor | None,
    n_pad: int,
    embedding_dim: int,
    embed_scale: float = 1.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-row gather + GLQ dequant + inverse RHT + optional embed_scale.

    Single source of truth for both ``E8RHTEmbedding.forward`` (HF path)
    and ``glq_vllm.embedding_method.GLQEmbeddingMethod.embedding`` (vLLM
    path). The math is the inverse of the per-row quantization done at
    ``quantize_model`` time:

        W_t = FHT_cols(W * SV)              # forward quant
        W   = SV * FHT_cols(W_t)            # we apply this here

    Stage-2 (3+ bpw) adds a per-row residual scaled by
    ``inv_resid_scale`` before the inverse RHT. ``codebook`` is the
    65536-entry E8Shell table; indices are stored as int16 and
    reinterpreted as uint16 to recover values in ``[0, 65536)``.

    Codebook tensors are lazily moved to ``input_ids.device`` (the
    codebook may live on CPU shared across many modules).
    """
    dev = qidxs.device
    flat_ids = input_ids.reshape(-1).to(dev)

    cb1 = codebook.to(dev) if codebook.device != dev else codebook
    rows1 = qidxs.index_select(0, flat_ids)                  # [B, n_pad/8]
    idx1 = rows1.reshape(-1).long() & 0xFFFF                 # int16→uint16
    deq1 = cb1.index_select(0, idx1)                         # [B*(n_pad/8), 8]
    deq = deq1.reshape(flat_ids.shape[0], n_pad).float()

    if qidxs2 is not None and codebook2 is not None and inv_resid_scale is not None:
        cb2 = codebook2.to(dev) if codebook2.device != dev else codebook2
        rows2 = qidxs2.index_select(0, flat_ids)
        idx2 = rows2.reshape(-1).long() & 0xFFFF
        deq2 = cb2.index_select(0, idx2).reshape(flat_ids.shape[0], n_pad).float()
        # Per-row residual scale [vocab] → broadcast over n_pad axis
        inv_rs = inv_resid_scale.index_select(0, flat_ids).unsqueeze(-1)
        deq = deq + deq2 * inv_rs.float()

    # Per-row Wscale → broadcast over n_pad axis, then inverse RHT.
    ws = wscale.index_select(0, flat_ids).unsqueeze(-1)
    deq = deq * ws.float()
    deq = fast_hadamard_transform(deq)
    deq = deq * sv.float()
    if embed_scale != 1.0:
        deq = deq * embed_scale

    # Trim padded n_pad → embedding_dim, cast to output dtype.
    if out_dtype is None:
        out_dtype = sv.dtype
    out = deq[..., :embedding_dim].to(out_dtype)
    return out.reshape(*input_ids.shape, embedding_dim)


class E8RHTEmbedding(nn.Module):
    """nn.Embedding equivalent with E8 + right-side-only RHT compressed weight.

    Used for Gemma-4 ``embed_tokens_per_layer``: a [vocab × num_layers·ple_dim]
    table that's ~4 GB at bf16 for E2B. Storing it via per-row GLQ + on-demand
    dequant at lookup cuts that to ~1 GB on disk and in GPU.

    Quantization (in ``quantize_model``): right-side RHT only — SV scale plus
    column FHT. Rows stay independent so a single ``forward(input_ids)`` only
    needs to dequant the rows actually requested.

    Storage matches ``E8RHTLinear`` for state-dict compatibility:
        ``Qidxs`` [vocab, n_pad/8] int16, ``SV`` [n_pad] fp16,
        ``Wscale`` scalar, optional ``Qidxs2`` + ``inv_resid_scale`` for 3+ bpw.

    ``SU`` is registered as a 1-element ones tensor purely so legacy code that
    iterates state-dict keys doesn't choke; it's never read.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 embed_scale: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # ``Gemma4TextScaledWordEmbedding`` (and analogues) multiply lookups
        # by sqrt(embedding_dim) inside their forward. We replicate that
        # post-dequant so substituted modules preserve the original output
        # magnitude. Default 1.0 = plain ``nn.Embedding``.
        self.embed_scale = float(embed_scale)
        # n_pad must be a power of 2 for the non-block FHT path. The PLE
        # weight in Gemma-4 has embedding_dim = num_layers * ple_dim, which is
        # already a power of 2 for E2B (35 * 256 → not pow2; we'll round up).
        self.n_pad = 1 << (embedding_dim - 1).bit_length() if embedding_dim > 0 else 1

        self.register_buffer('Qidxs',
            torch.zeros(num_embeddings, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('SV', torch.ones(self.n_pad, dtype=torch.float16))
        # SU kept ones-shaped + 1-elem for state-dict round-trip; never used.
        self.register_buffer('SU', torch.ones(1, dtype=torch.float16))
        # Per-row Wscale (vocab-sized): chunked quant calibrates one Wscale
        # per chunk and we map row → chunk via index. A single scalar fails
        # because chunks from different parts of the embedding have different
        # row-magnitude statistics → severe quality loss when one scalar is
        # forced on all rows.
        self.register_buffer('Wscale',
            torch.ones(num_embeddings, dtype=torch.float32))
        # Optional residual stage (3+ bpw) — also per-row scale
        self.register_buffer('Qidxs2',
            torch.zeros(num_embeddings, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('inv_resid_scale',
            torch.zeros(num_embeddings, dtype=torch.float32))
        self.codebook = None
        self.codebook2 = None
        self._n_stages = 1
        self._rsqrt_n = self.n_pad ** -0.5

    def set_codebook(self, codebook, codebook2=None):
        """Attach the shared E8ShellCodebook(s). Mirrors E8RHTLinear API."""
        self.codebook = codebook
        self.codebook2 = codebook2
        if self.Wscale.device.type == "meta":
            return
        # Detect 2-stage usage from any non-zero per-row inv_resid_scale.
        self._n_stages = 1
        if codebook2 is not None and self.inv_resid_scale.abs().any().item():
            self._n_stages = 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Inject zero defaults for stage-2 buffers absent from 2bpw checkpoints."""
        primary = prefix + 'Qidxs'
        if primary in state_dict:
            for suffix, default_fn in [
                ('Qidxs2', lambda: torch.zeros(self.num_embeddings,
                                              self.n_pad // 8, dtype=torch.int16)),
                ('inv_resid_scale', lambda: torch.zeros((), dtype=torch.float32)),
            ]:
                key = prefix + suffix
                if key not in state_dict:
                    state_dict[key] = default_fn()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather + dequant per-row. ``input_ids`` shape: [...]; returns [..., embedding_dim]."""
        if self.codebook is None:
            raise RuntimeError("E8RHTEmbedding.set_codebook() must be called first")

        out_dtype = getattr(self, "_compute_dtype", self.SV.dtype)
        cb1 = self.codebook.codebook
        cb2 = self.codebook2.codebook if (
            self._n_stages >= 2 and self.codebook2 is not None) else None
        return _dequant_embedding_rows(
            input_ids,
            self.Qidxs, self.SV, self.Wscale, cb1,
            self.Qidxs2 if self._n_stages >= 2 else None,
            self.inv_resid_scale if self._n_stages >= 2 else None,
            cb2,
            n_pad=self.n_pad,
            embedding_dim=self.embedding_dim,
            embed_scale=self.embed_scale,
            out_dtype=out_dtype,
        )

    def extra_repr(self) -> str:
        return (f'num_embeddings={self.num_embeddings}, '
                f'embedding_dim={self.embedding_dim}, n_pad={self.n_pad}')
