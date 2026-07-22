"""Shippable QTIP trellis-coded-quantization codebook for GLQ.

Trellis-coded quantization (TCQ, QTIP / arXiv 2406.11235, Cornell RelaxML — the QuIP#/e8p
group) reaches effective quantization dimension 256 with a lookup-free / small-LUT decode,
beating GLQ's 8-D E8 lattice at low bit-rates. Validated end-to-end for GLQ: SmolLM3-3B
Wikitext-2 PPL trellis-2bit **11.94 < e8p-2bit 13.79** (bf16 9.12); layer proxy +1.32 dB.

This module promotes the validated scratch (`benchmarks/_qtip_{trellis,ldlq}.py`) into the
package and adds the **shippable compressed storage** — the exact byte layout QTIP's CUDA
`decompress_matvec` kernel consumes — so a stored checkpoint decodes bit-exactly both here
(pure torch, the kernel's oracle) and, in S1, on-GPU. The trellis reuses GLQ's RHT + LDLQ
error-feedback verbatim (GLQ's `block_LDL` is provably identical to QTIP's BlockLDLQ); only
the codebook changes.

Variants (both ship with CUDA kernels, glq/csrc/glq_trellis.cu):
  * ``"hyb"``   — ``quantlut_sym`` V=2 tlut_bits=9: decodes via a 512-entry tlut replicated in
                  shared memory. ~1.1 dB over e8p at 2 bpw.
  * ``"3inst"`` — lookup-free V=1 (``decode_3inst``): quality ≈ HYB at 2 & 4 bpw, decodes
                  ARITHMETICALLY (no tlut, zero dynamic smem) on the ``<R, IS_3INST>`` kernel
                  instantiations — removes the smem codebook-gather from the B=1 decode path.

QTIP is GPL-3.0, same license as GLQ; the Viterbi/pack/permute logic is ported from its
``lib/codebook/bitshift.py``, ``lib/algo/{ldlq,finetune}.py``. See
``~/.claude/plans/hazy-churning-shannon.md``.
"""
import math
import os
import threading
import warnings
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from glq.ldlq import block_LDL
from glq.rht import RHT

TD = 16  # QTIP tile dim td_x == td_y == 16 → 256-element tiles

# ---------------------------------------------------------------------------
# Encoder-side CUDA-graph capture of the Viterbi (offline quant speedup).
# The 127-step Viterbi loop (`bitshift_codebook.viterbi`) is launch-bound: nsys shows ~96%
# of CUDA-API time is kernel-launch + memcpy DISPATCH, GPU busy only ~17%. Capturing the
# whole `viterbi` as one CUDA graph per (T, B, has_overlap) collapses that dispatch. It is
# ENCODE-ONLY (decode uses `decode_layer`, never the Viterbi), deterministic (min/argmin/
# gather, no atomics), and a pure speed change → replay is `torch.equal` to eager, which is
# the gate. CPU / kill-switch fall back to the unchanged eager `viterbi`.
_GLQ_TRELLIS_CUDAGRAPH_ENABLED = True   # module kill-switch (mirrors _GLQ_FUSED_TRELLIS_ENABLED)
# Layers with B above this fall back to eager: they hold a large `from_state`/`cost` graph
# pool (4 MB·B at K=2) and are more GPU-bound, so they lose the least from staying eager.
_GLQ_TRELLIS_CUDAGRAPH_MAX_B = int(os.environ.get("GLQ_TRELLIS_CUDAGRAPH_MAX_B", "256"))
_VitGraph = namedtuple("_VitGraph", "graph x_buf overlap_buf out_buf")


def _trellis_cudagraph_on():
    return (_GLQ_TRELLIS_CUDAGRAPH_ENABLED
            and not os.environ.get("GLQ_TRELLIS_NO_CUDAGRAPH")
            and torch.cuda.is_available())

# MMA-fragment intra-tile reorder (QTIP lib/algo/ldlq.py:10-13). Applied to the 256-element
# tile BEFORE trellis-quantizing (for_kernel) so decoded lanes land in tensor-core fragment
# order; INV restores natural (row-major 16×16) order for the reconstructed weight.
_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
_INV_PERMUTE = torch.zeros(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)


# ---------------------------------------------------------------------------
# lookup-free decode helpers (3inst / 1mad / 2mad) — QTIP bitshift.py
# ---------------------------------------------------------------------------
def decode_1mad(x):
    x = x.to(torch.int64) & ((1 << 32) - 1)
    x = (x * 34038481 + 76625530) & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255) - 510
    return (y.to(torch.float32)) / 147.800537109375


def decode_2mad(x):
    x = x.to(torch.int64) & ((1 << 32) - 1)
    x = (x * 264435761 + 1013904223) & ((1 << 32) - 1)
    x = (((x * 1664525) >> 32) + x) & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255) - 510
    return (y.to(torch.float32)) / 147.800537109375


def decode_3inst(x):

    def bfe16_to_fp16(v):
        v[torch.where(v >= 2 ** 15)] -= 2 ** 16
        return torch.tensor(v.to(torch.int16).numpy().view(np.float16))

    a, b, fpmask = 89226354, 64248484, 996162400
    x = x.to(torch.int64) & ((1 << 32) - 1)
    x = x * a + b
    mask = (1 << 15) + ((1 << 12) - 1)
    mask = (mask << 16) + mask
    res = (mask & x) ^ fpmask
    return (bfe16_to_fp16(res >> 16) + bfe16_to_fp16(res & ((1 << 16) - 1))).float()


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


def _make_hyb_tlut(tlut_bits=9, seed=0):
    """kmeans-fit the HYB (V=2) tunable LUT on 2-D Gaussian samples (encoder-only).

    Deterministic given ``seed``; ALL layers share one tlut, which is stored in the
    checkpoint so the decoder never needs scipy/kmeans. Mirrors QTIP's quantlut_sym init.
    """
    import scipy.cluster.vq  # lazy — encode-time only

    torch.manual_seed(seed)
    init = torch.randn(2 ** tlut_bits, 2)
    data = torch.randn(1 << 20, 2)
    centroids = torch.tensor(scipy.cluster.vq.kmeans(data.numpy(), init.numpy())[0])
    return (centroids / centroids.std(unbiased=False)) * 0.9682458365518543


# ---------------------------------------------------------------------------
# bitshift trellis codebook (QTIP bitshift.py, tqdm/train paths stripped)
# ---------------------------------------------------------------------------
class bitshift_codebook(nn.Module):

    def __init__(self, L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", tlut=None):
        super().__init__()
        self.idx_dtype = torch.int32
        self.L, self.K, self.V = L, K, V
        self.tlut_bits, self.decode_mode = tlut_bits, decode_mode

        if decode_mode == "3inst":
            assert V == 1
            self.register_buffer("lut", decode_3inst(torch.arange(2 ** L)).unsqueeze(0))
            self.tlut = None
        elif decode_mode == "1mad":
            assert V == 1
            self.register_buffer("lut", decode_1mad(torch.arange(2 ** L)).unsqueeze(0))
            self.tlut = None
        elif decode_mode == "2mad":
            assert V == 1
            self.register_buffer("lut", decode_2mad(torch.arange(2 ** L)).unsqueeze(0))
            self.tlut = None
        elif decode_mode == "quantlut_sym":
            assert V == 2 and tlut_bits > 0
            if tlut is None:
                tlut = _make_hyb_tlut(tlut_bits)              # encoder: fit once
            self.register_buffer("tlut", tlut.float())
            self.register_buffer("lut", quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())
        else:
            raise ValueError(decode_mode)

        self.fakeinf = torch.tensor(torch.inf)
        self.register_buffer("sumdelta", (torch.arange(2 ** (K * V)) << (L - K * V)).view(1, 1, -1))
        self.register_buffer("state", torch.arange(2 ** L).unsqueeze(0))
        self.register_buffer(
            "state_cand",
            (self.state >> (K * V))[0, ::2 ** (K * V)].unsqueeze(-1) + self.sumdelta)
        self.register_buffer("recons_state", self.recons(self.state))
        # arange(2^KV) for the overlap tail-biting mask, as a device BUFFER — the original
        # `torch.arange(...).to(X.device)` is a CPU→CUDA copy, which is ILLEGAL during CUDA-graph
        # capture and silently forced the overlap pass onto the eager fallback.
        self.register_buffer("_kv_arange", torch.arange(2 ** (K * V)).view(1, 1, -1))

        # Viterbi CUDA-graph cache: (T, B, has_overlap) -> _VitGraph. Plain attrs (NOT buffers)
        # so `.to(device)` never touches them. Lazily populated on first CUDA `quantize_seq`.
        # The lock is REQUIRED: MoE experts quantize on a ThreadPoolExecutor sharing this one
        # codebook instance, and CUDA-graph capture is process-global/exclusive — concurrent
        # captures (or replays racing on the shared static buffers) would corrupt/abort.
        self._vit_graphs = {}
        self._vit_graph_pool = None
        self._vit_lock = threading.Lock()

    def recons(self, encoded):
        return self.lut[:, encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, thing):
        state_err = (self.recons_state - thing.unsqueeze(-1)).square().sum(dim=0)
        cand_cost = torch.gather(
            cost.unsqueeze(-2).expand(-1, self.state_cand.shape[1], -1), -1,
            self.state_cand.expand(len(cost), -1, 2 ** (self.K * self.V)))
        best = torch.min(cand_cost, dim=-1)
        cost = state_err + best.values.unsqueeze(-1).expand(
            -1, -1, 2 ** (self.K * self.V)).reshape(state_err.shape)
        prev_state = torch.gather(
            self.state_cand.expand(thing.shape[1], -1, -1), -1, best.indices.unsqueeze(-1))[..., 0]
        return prev_state, cost

    def viterbi(self, X, overlap=None):
        T, B = X.shape
        assert T % self.V == 0
        cost = (self.recons_state - X[:self.V].unsqueeze(-1)).square().sum(dim=0)
        if overlap is not None:
            mask = torch.ones(B, 2 ** self.L, device=X.device) * self.fakeinf
            allow = (overlap << (self.K * self.V)).unsqueeze(-1) + self._kv_arange
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, self.fakeinf)
        from_state = torch.zeros(T // self.V, B, 2 ** (self.L - self.K * self.V),
                                 dtype=self.state.dtype, device=self.state.device)
        for i in range(1, T // self.V):
            from_state[i], cost = self.update(cost, X[i * self.V:(i + 1) * self.V])
        if overlap is not None:
            mask = torch.ones(B, 2 ** self.L, device=X.device) * self.fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, self.fakeinf)
        final_state = torch.zeros(T // self.V, B, dtype=self.idx_dtype, device=X.device)
        final_state[T // self.V - 1] = torch.argmin(cost, dim=-1)
        for i in range(T // self.V - 1, 0, -1):
            final_state[i - 1] = torch.gather(
                from_state[i], -1,
                (final_state[i].to(torch.int64).unsqueeze(-1)) >> (self.K * self.V))[..., 0]
        return final_state

    # -- CUDA-graph capture of `viterbi` (see module header) ------------------
    @torch.no_grad()
    def _capture_viterbi(self, X, overlap, warmup_iters=3):
        """Warm (compile `update` + prime allocator) on a side stream, then capture the whole
        `viterbi` into the shared graph pool. Returns a _VitGraph with the static I/O buffers."""
        if self._vit_graph_pool is None:
            self._vit_graph_pool = torch.cuda.graph_pool_handle()
        x_buf = torch.empty_like(X)
        overlap_buf = torch.empty_like(overlap) if overlap is not None else None
        x_buf.copy_(X)
        if overlap_buf is not None:
            overlap_buf.copy_(overlap)
        # Side-stream warmup: the FIRST call compiles @torch.compile(update) for this B and
        # primes the caching allocator; capture-during-compile would be illegal.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self.viterbi(x_buf, overlap_buf)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._vit_graph_pool):
            out_buf = self.viterbi(x_buf, overlap_buf)
        return _VitGraph(graph, x_buf, overlap_buf, out_buf)

    @torch.no_grad()
    def _viterbi_graphed(self, X, overlap=None):
        """Replay a cached CUDA graph of `viterbi` (lazy-capture on miss). Bit-exact to eager.

        The whole capture + copy-in → replay → clone-out is under `_vit_lock`: capture is
        process-global/exclusive (a sibling MoE-expert thread launching any kernel mid-capture
        would corrupt it), and the static x_buf/out_buf are shared across chunks/threads. The
        `.clone()` (enqueued inside the lock) reads this replay's result before the next
        copy-in can overwrite `out_buf`. The lock cost is ~nil — the ThreadPool only existed to
        overlap host dispatch, which the graph has already collapsed to one launch."""
        key = (int(X.shape[0]), int(X.shape[1]), overlap is not None)  # (T, B, has_overlap)
        with self._vit_lock:
            if key not in self._vit_graphs:
                try:
                    self._vit_graphs[key] = self._capture_viterbi(X, overlap)
                except Exception as e:
                    # Fall back to eager for THIS shape only (a None sentinel, not a retry) and
                    # WARN — a silent global disable previously hid a real capture bug.
                    warnings.warn(
                        f"trellis Viterbi CUDA-graph capture failed for {key}; using eager for "
                        f"this shape. {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
                    self._vit_graphs[key] = None
            entry = self._vit_graphs[key]
            if entry is None:                      # capture failed for this shape → eager
                return self.viterbi(X, overlap)
            entry.x_buf.copy_(X)
            if entry.overlap_buf is not None:
                entry.overlap_buf.copy_(overlap)
            entry.graph.replay()
            return entry.out_buf.clone()

    def free_viterbi_graphs(self):
        """Release captured Viterbi graphs + their pool (call after quantization to reclaim VRAM)."""
        self._vit_graphs.clear()
        self._vit_graph_pool = None

    def quantize_seq(self, X, overlap=None):
        T, NO = X.shape
        bs = min(2 ** (24 - self.L), NO)
        pad_amt = math.ceil(NO / bs) * bs - NO
        X = torch.nn.functional.pad(X, (0, pad_amt))
        T, N = X.shape
        X = X.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt)).reshape(N // bs, bs)
        Qidxs = torch.zeros(N // bs, T // self.V, bs, dtype=self.idx_dtype, device=X.device)
        # CUDA-graph the per-chunk Viterbi when on-device; CPU/kill-switch/oversize B use eager.
        use_graph = (X.is_cuda and _trellis_cudagraph_on() and bs <= _GLQ_TRELLIS_CUDAGRAPH_MAX_B)
        _vit = self._viterbi_graphed if use_graph else self.viterbi
        for i in range(len(X)):
            Qidxs[i] = _vit(X[i], overlap=None if overlap is None else overlap[i])
        return Qidxs.transpose(0, 1).reshape(T // self.V, N)[:, :NO]

    def quantize(self, X):
        X = X.T.contiguous().to(torch.float16)
        T = X.shape[0]
        roll_X = torch.roll(X, T // (2 * self.V) * self.V, 0)
        state = self.quantize_seq(roll_X, overlap=None)
        overlap = state[T // (2 * self.V)] >> self.K * self.V
        state = self.quantize_seq(X, overlap=overlap)
        hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
        return hatX.T.contiguous().to(X.device), state.T.contiguous().to(X.device)

    def pack_trellis(self, trellis):
        """(B, T//V) int trellis states → (B, ceil(T*K/16)) int16 (tail-biting, MSB-first)."""
        B, T = trellis.shape
        bf = torch.zeros(B, T * self.K * self.V + self.L - self.K * self.V,
                         dtype=bool, device=trellis.device)
        bf[:, :self.L] = (trellis[:, 0].unsqueeze(-1) & (2 ** torch.arange(
            self.L, device=trellis.device).flip(dims=(-1,))).unsqueeze(0)) > 0
        K_mask = 2 ** torch.arange(self.K * self.V, device=trellis.device).flip(dims=(-1,)).unsqueeze(0)
        for i in range(1, T):
            assert ((trellis[:, i - 1] & ((1 << (self.L - self.K * self.V)) - 1)) ==
                    (trellis[:, i] >> (self.K * self.V))).all()
            bf[:, (self.L + (i - 1) * self.K * self.V):(self.L + i * self.K * self.V)] = (
                (trellis[:, i] & ((1 << (self.K * self.V)) - 1)).unsqueeze(-1) & K_mask) > 0
        bf = bf[:, :-(self.L - self.K * self.V)]
        pad_amt = math.ceil(T * self.K * self.V / 16) * 16 - T * self.K * self.V
        bf = torch.nn.functional.pad(bf, (0, pad_amt)).reshape(
            -1, (T * self.K * self.V + pad_amt) // 16, 16)
        uint_mask = (2 ** torch.arange(16, dtype=torch.int32, device=bf.device)).flip(
            dims=(-1,)).unsqueeze(0).unsqueeze(0)
        return (bf.to(torch.int32) * uint_mask).sum(dim=-1).to(torch.uint16).view(torch.int16)

    def unpack_trellis(self, packed, T):
        """Inverse of ``pack_trellis``: (B, ceil(T*K/16)) int16 → (B, T//V) int trellis."""
        packed = packed.view(torch.uint16).to(torch.int32)
        uint_mask = (2 ** torch.arange(16, dtype=torch.int32, device=packed.device)).flip(
            dims=(-1,)).unsqueeze(0).unsqueeze(0)
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(T * self.K / 16) * 16 - T * self.K
        bf = bf.reshape(-1, (T * self.K + pad_amt))[:, :T * self.K]
        bf = torch.concat([bf, bf[:, :self.L - self.K * self.V]], dim=-1)
        L_mask = (2 ** torch.arange(self.L, dtype=torch.int32, device=packed.device).flip(dims=(-1,))).unsqueeze(0)
        K_mask = (2 ** torch.arange(self.K * self.V, dtype=torch.int32, device=packed.device).flip(dims=(-1,))).unsqueeze(0)
        trellis = torch.zeros(bf.shape[0], T // self.V, dtype=torch.int32, device=bf.device)
        trellis[:, 0] = (bf[:, :self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, T // self.V):
            trellis[:, i] = ((trellis[:, i - 1] << (self.K * self.V)) & ((1 << self.L) - 1)) + \
                (bf[:, self.L + (i - 1) * self.K * self.V:self.L + i * self.K * self.V].int() * K_mask).sum(dim=-1)
        return trellis


# ---------------------------------------------------------------------------
# GLQ-facing codebook wrapper (duck-typed like E8ShellCodebook: codesz, quantize)
# ---------------------------------------------------------------------------
class TrellisCodebook:
    """GLQ-duck-typed (codesz=16) wrapper around QTIP's bitshift trellis codebook.

    Encoder (``tlut=None``, HYB): kmeans-fits the tlut once. Decoder (``tlut`` given):
    reconstructs the codebook deterministically from the stored tlut — no scipy/kmeans.
    """

    is_trellis = True

    def __init__(self, variant="hyb", K=2, device="cpu", tlut=None):
        if variant == "3inst":
            self.cb = bitshift_codebook(L=16, K=K, V=1, tlut_bits=0, decode_mode="3inst")
        elif variant == "hyb":
            self.cb = bitshift_codebook(L=16, K=K, V=2, tlut_bits=9,
                                        decode_mode="quantlut_sym", tlut=tlut)
        else:
            raise ValueError(variant)
        self.cb = self.cb.to(device)
        self.cb.fakeinf = self.cb.fakeinf.to(device)   # plain attr, not moved by .to()
        self.variant = variant
        self.K, self.V, self.L = K, self.cb.V, self.cb.L
        self.tlut_bits = self.cb.tlut_bits
        self.tlut = self.cb.tlut                        # (2**tlut_bits, V) fp32, or None
        # Both variants ship in the kernel (for_kernel/MMA-fragment) storage layout: HYB decodes
        # on the tlut CUDA kernels, 3inst on the lookup-free <R, IS_3INST=true> kernels
        # (glq_trellis.cu). Checkpoints stamp `trellis_layout: "kernel"` in config.json; the
        # loader refuses natural-layout 3inst checkpoints from before this flip (they must be
        # re-quantized — a kernel-layout decoder would silently scramble them).
        self.has_kernel = variant in ("hyb", "3inst")
        self.codesz = TD
        self.device = device
        # QTIP scale: Wr /= rms(Wr)/(rms(lut)*0.9)  ⇔  GLQ Wscale = rms(W)*opt_scale
        self.opt_scale = 1.0 / (self.cb.lut.double().square().mean().sqrt().item() * 0.9)

    def quantize_tiles(self, tiles):
        """(B, 256) fp32 tiles → (hatX (B,256), state (B, 256//V)). No permute."""
        return self.cb.quantize(tiles)

    def recons(self, state):
        return self.cb.recons(state)

    def pack_trellis(self, state):
        return self.cb.pack_trellis(state)

    def unpack_trellis(self, packed, T):
        return self.cb.unpack_trellis(packed, T)


# ---------------------------------------------------------------------------
# for_kernel byte tile-flip (QTIP lib/algo/finetune.py:196-201) + its inverse
# ---------------------------------------------------------------------------
def _kernel_flip_perm(m, n, K, nbytes, device):
    """Flat byte permutation of the finetune tile-flip: kernel[i] = natural[perm[i]]."""
    idx = torch.arange(nbytes, dtype=torch.int64, device=device)
    return idx.view(-1, 2).flip((-1,)).reshape(
        m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8, K).permute(0, 2, 4, 3, 1, 5).flip((-1,)).contiguous().flatten()


def kernel_tile_flip(packed, m, n, K, forward=True):
    """Reorder packed int16 bytes into (forward) / out of (inverse) MMA-fragment order.

    QTIP's kernel expects the packed trellis with a specific byte shuffle; the inverse
    (needed by the pure-torch decode) is derived by tracking the byte-index permutation.
    """
    shp = packed.shape
    b = packed.contiguous().view(torch.uint8).flatten()
    perm = _kernel_flip_perm(m, n, K, b.numel(), b.device)
    if forward:
        out = b[perm]
    else:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        out = b[inv]
    return out.contiguous().view(torch.int16).reshape(shp)


# ---------------------------------------------------------------------------
# layer pack / unpack / decode — the compressed-storage round-trip
# ---------------------------------------------------------------------------
def pack_layer(cb, Qidxs, m, n, has_kernel=True):
    """(m, n//V) trellis states → packed int16 [(m//16)*(n//16), ceil(256*K/16)].

    Regroups per-column-block indices into per-tile sequences (QTIP finetune.py:192-194)
    then bit-packs; ``has_kernel`` additionally applies the MMA-fragment byte flip.
    """
    td, V = TD, cb.V
    packed = cb.pack_trellis(
        Qidxs.reshape(m // td, td, n // td, td // V).transpose(1, 2).reshape(-1, td * td // V))
    if has_kernel:
        packed = kernel_tile_flip(packed, m, n, cb.K, forward=True)
    return packed


def unpack_layer(cb, packed, m, n, has_kernel=True):
    """Inverse of ``pack_layer`` at the per-tile level → trellis states (num_tiles, 256//V)."""
    if has_kernel:
        packed = kernel_tile_flip(packed, m, n, cb.K, forward=False)
    return cb.unpack_trellis(packed, TD * TD)          # (num_tiles, 256//V)


def decode_layer(cb, packed, m, n, has_kernel=True):
    """Pure-torch reference decode: packed int16 → normalized hatW (m, n).

    This is the exact oracle the CUDA ``decompress_matvec`` kernel (S1) must match
    bit-for-bit. Un-flips the byte layout, unpacks the tail-biting trellis, reconstructs
    each 256-tile, undoes the ``_PERMUTE`` (kernel layout), and re-tiles to (m, n).
    """
    td = TD
    trellis = unpack_layer(cb, packed, m, n, has_kernel)          # (num_tiles, 256//V)
    rec = cb.recons(trellis)                                      # (V, num_tiles, 256//V)
    nt = trellis.shape[0]
    tiles = rec.permute(1, 2, 0).reshape(nt, td * td)            # (num_tiles, 256) weight = step*V + v
    if has_kernel:
        tiles = tiles[:, _INV_PERMUTE.to(tiles.device)]
    return tiles.reshape(m // td, n // td, td, td).transpose(1, 2).reshape(m, n)


# ---------------------------------------------------------------------------
# LDLQ (Hessian error feedback) with the trellis codebook — GLQ block_LDL
# ---------------------------------------------------------------------------
def trellis_ldlq(W, H, cb, Wscale=None, for_kernel=True):
    """LDLQ reverse sweep, block 16, tile-trellis codebook.

    Returns (hatWr_norm (m,n)  — NORMALIZED reconstruction (pre-Wscale),
             Qidxs (m, n//V)   — trellis states in the (permuted, if for_kernel) tile order,
             Wscale (float)).
    GLQ's block_LDL == QTIP's BlockLDLQ; feedback reads only rows below the block, so QTIP's
    diagonal-zeroing is a no-op ⇒ this is QTIP BlockLDLQ with GLQ machinery.
    """
    dev = cb.device
    W, H = W.float().to(dev), H.float().to(dev)
    m, n = W.shape
    b = TD
    assert n % b == 0 and m % b == 0, f"trellis needs m,n %16 (got {m}x{n})"
    perm = _PERMUTE.to(dev)
    inv_perm = _INV_PERMUTE.to(dev)

    damp = 0.01 * torch.diag(H).mean()
    L, _ = block_LDL(H + damp * torch.eye(n, device=dev), block_size=b)

    if Wscale is None:
        Wscale = W.pow(2).mean().sqrt().item() * cb.opt_scale
    Wr = W / Wscale
    hatWr = torch.zeros_like(Wr)
    R = Wr.clone()
    Qidxs = torch.zeros(m, n // cb.V, dtype=torch.int32, device=dev)
    for k in reversed(range(n // b)):
        kb, ke = k * b, (k + 1) * b
        feedback = R[:, ke:] @ L[ke:, kb:ke] if ke < n else 0.0
        WXWX = Wr[:, kb:ke] + feedback                        # (m, 16)
        tiles = WXWX.reshape(m // b, b * b)                   # (m/16, 256)
        if for_kernel:
            tiles = tiles[:, perm]
        hatX, state = cb.quantize_tiles(tiles)                # (m/16,256),(m/16,256//V)
        if for_kernel:
            hatX = hatX[:, inv_perm]
        hatWr[:, kb:ke] = hatX.reshape(m, b).to(hatWr.dtype)
        R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]
        Qidxs[:, (b // cb.V) * k:(b // cb.V) * (k + 1)] = state.reshape(m, b // cb.V)
    return hatWr, Qidxs, Wscale


def quantize_layer_trellis_rht(W, H, cb, block_diagonal=True, for_kernel=None):
    """RHT + trellis-LDLQ, mirroring ``quantize_layer_e8_shell_rht``.

    Returns (W_hat (m,n) in original weight space, artifacts dict with the compressed
    trellis + per-codebook metadata needed to reconstruct at inference).

    ``for_kernel`` selects the stored tile layout (the MMA-fragment ``_PERMUTE`` + byte
    flip). It MUST match the layout the decoder uses — i.e. the codebook's ``has_kernel``.
    It therefore DEFAULTS to ``cb.has_kernel`` so store-layout == decode-layout for every
    variant automatically; the driver relies on this. Both shipped variants (HYB + 3inst)
    are has_kernel=True → kernel layout. Pass explicitly only in tests.
    """
    if for_kernel is None:
        for_kernel = cb.has_kernel
    dev = cb.device
    m, n = W.shape
    Wf, Hf = W.float().to(dev), H.float().to(dev)
    damp = 0.01 * torch.mean(torch.diag(Hf))
    Hf = Hf + damp * torch.eye(n, device=dev)
    rht = RHT(m, n, device=dev, block_diagonal=block_diagonal, apply_left=True, e8p=False)
    Wt = rht.transform_weights(Wf)
    Ht = rht.transform_hessian(Hf)
    assert Wt.shape == (m, n), f"trellis needs no RHT padding; got {tuple(Wt.shape)} for {m}x{n}"
    hatWr_norm, Qidxs, Wscale = trellis_ldlq(Wt, Ht, cb, for_kernel=for_kernel)
    W_hat = rht.inverse_transform_weights(hatWr_norm * Wscale)
    # Artifacts are the per-layer E8RHTLinear-trellis buffers — TENSORS ONLY (the driver
    # .cpu()'s every value + writes it to the state_dict). Codebook metadata (variant/K/V)
    # lives in config.json (K is re-derived from the packed shape at load).
    artifacts = {
        "trellis_packed": pack_layer(cb, Qidxs, m, n, has_kernel=for_kernel).cpu(),
        "SU": rht.su.detach().to(torch.float16).cpu(),          # (m,) RHT row signs
        "SV": rht.sv.detach().to(torch.float16).cpu(),          # (n,) RHT col signs
        "Wscale": torch.tensor(float(Wscale), dtype=torch.float32),
    }
    if cb.tlut is not None:
        artifacts["tlut"] = cb.tlut.detach().to(torch.float16).cpu()
    return W_hat, artifacts
