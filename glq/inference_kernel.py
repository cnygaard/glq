"""Triton fused dequant+matmul kernels for GLQ inference.

Computes Y = X @ W^T * Wscale where W is stored as codebook indices,
without materializing the full weight matrix in GPU memory.

For 2bpw: W[i, j:j+8] = codebook[Qidxs[i, j//8]]
For 3/4bpw: W[i, j:j+8] = cb1[idx1[i, j//8]] + cb2[idx2[i, j//8]] * inv_resid_scale

Kernels:
  _glq_dequant_matvec_kernel: B=1 decode (autotuned BLOCK_M)
  _glq_dequant_matmul_tc_kernel: B>=2 prefill (Tensor Core tl.dot with K=16)
  _splitk_matvec_kernel: B=1 split-K for small grids (more CTAs → better SM util)
  _splitk_matmul_tc_kernel: B>=2 split-K Tensor Core variant
"""

import torch

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    _triton_available = False

# CUDA C extension: loaded lazily on first B=1 call to avoid 30s JIT penalty on import
_glq_cuda = None
_cuda_ext_available = None  # None = not tried, True/False = result
_cuda_ext_error = None      # why the build failed, verbatim — see cuda_ext_status()


#: Where to send someone whose bundled CUDA wheels turn out not to be enough. The landing
#: page rather than a versioned .run URL — that one rots on every CUDA release.
CUDA_DOWNLOADS_URL = "https://developer.nvidia.com/cuda-downloads"


def _venv_cuda_home(search_roots=None):
    """The CUDA root inside the venv, or None if the wheels are not installed.

    `cuda-toolkit[nvcc,cccl]` puts the compiler at `<site-packages>/nvidia/cu13/bin/nvcc` —
    not on PATH, and not anywhere torch looks. Without pointing torch at it the build dies
    with `OSError: CUDA_HOME environment variable is not set` before reaching a compiler,
    which makes installing the toolchain at all pointless. Measured in a clean container.

    The CUDA major is torch's choice, so glob `cu*` rather than naming one.
    """
    import glob
    import os
    import site

    if search_roots is None:
        search_roots = list(site.getsitepackages())
    for root in search_roots:
        for nvcc in sorted(glob.glob(os.path.join(root, "nvidia", "*", "bin", "nvcc"))):
            return os.path.dirname(os.path.dirname(nvcc))
    return None


def _cudart_link_dir(search_roots=None, cache_dir=None):
    """Return a directory that makes `-lcudart` resolvable, or None if it already is.

    The pip CUDA wheels install

        nvidia/cu13/lib/libcudart.so.13

    with no bare `libcudart.so` and no `lib64/`. torch's cpp_extension emits
    `-L$CUDA_HOME/lib64 -lcudart`, and `-lcudart` only ever matches `libcudart.so`, so the
    link fails with `cannot find -lcudart` even though the library is right there. A system
    CUDA install (`/usr/local/cuda/lib64/libcudart.so`) ships the symlink and needs none of
    this — hence the early return, so we never shadow a real toolkit.

    Measured in an ubuntu:24.04 container, 2026-08-15.
    """
    import glob
    import os
    import site

    if search_roots is None:
        search_roots = list(site.getsitepackages())
    if cache_dir is None:
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "glq", "cudalibs")

    # A real toolkit already satisfies `-lcudart`; injecting the wheels' copy ahead of it
    # would mix CUDA minors (the box: system 13.2 + wheel 13.0). Leave it alone.
    try:
        from torch.utils.cpp_extension import CUDA_HOME as _torch_cuda_home
    except Exception:                                 # pragma: no cover - torch always has it
        _torch_cuda_home = None
    if _torch_cuda_home:
        for libdir in ("lib64", "lib"):
            if os.path.exists(os.path.join(_torch_cuda_home, libdir, "libcudart.so")):
                return None

    versioned = []
    for root in search_roots:
        # nvidia/<cu13|cu14|…>/lib/libcudart.so.<major> — the CUDA major is torch's choice,
        # so glob it rather than naming one.
        versioned += glob.glob(os.path.join(root, "nvidia", "*", "lib", "libcudart.so.*"))
    if not versioned:
        return None                                   # system CUDA, or no CUDA at all

    real = sorted(versioned)[0]
    if os.path.exists(os.path.join(os.path.dirname(real), "libcudart.so")):
        return None                                   # dev symlink already present

    try:
        os.makedirs(cache_dir, exist_ok=True)
        link = os.path.join(cache_dir, "libcudart.so")
        if os.path.islink(link) or os.path.exists(link):
            if os.path.realpath(link) == os.path.realpath(real):
                return cache_dir                      # idempotent: already correct
            os.remove(link)                           # stale, e.g. after a torch upgrade
        os.symlink(real, link)
        return cache_dir
    except OSError:
        return None                                   # read-only HOME: fall through to fail


def repair_cuda_wheel_layout(search_roots=None):
    """Give the pip CUDA wheels the on-disk layout every CUDA build system expects.

    `_cudart_link_dir()` repairs GLQ's own link line. It cannot repair anyone else's, and on
    a machine whose only CUDA is the pip wheels, everyone else has the identical problem.
    Measured in an ubuntu:24.04 container: glq installed, GLQ's kernels built, `--verify`
    green — and then vLLM refused to start, because it JIT-compiles flashinfer:

        c++ … -L …/nvidia/cu13/lib64 -L …/nvidia/cu13/lib64/stubs -lcudart -lcuda
        /usr/bin/ld: cannot find -lcudart: No such file or directory

    Two things a real CUDA install has and the wheels do not:

      * `lib/libcudart.so` — the unversioned dev symlink, the only name `-lcudart` matches
      * `lib64/`          — the directory build systems pass to `-L`; the wheels use `lib/`

    Both are created here, once, in the venv: symlinks beside files pip already installed.
    Shimming per-consumer does not scale past the consumers we happen to control.

    Idempotent, and it never replaces something that already exists — if a wheel starts
    shipping the symlink, or the root is a real toolkit, this is a no-op. A read-only
    site-packages is a degrade, not a failure.
    """
    import glob
    import os
    import site
    import sys

    if search_roots is None:
        # Only ever a venv, and only by default. Under `sudo pip install glq`, or root in a
        # container, `site.getsitepackages()` is a system prefix — /usr/lib/python3/
        # dist-packages — which glq has no business creating symlinks in unasked. The venv
        # is the thing install.sh creates and therefore owns. An explicit `search_roots` is
        # a deliberate instruction and is honoured either way.
        if sys.prefix == sys.base_prefix:
            return
        search_roots = list(site.getsitepackages())

    for root in search_roots:
        for libdir in sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib"))):
            # libcudart.so.13 and libcudart.so.13.0.96 both map to libcudart.so; link the
            # shortest, which is the soname the versioned files themselves chain to.
            by_base = {}
            for path in glob.glob(os.path.join(libdir, "lib*.so.*")):
                base = path.split(".so.")[0] + ".so"
                if len(path) < len(by_base.get(base, path * 2)):
                    by_base[base] = path
            for base, real in sorted(by_base.items()):
                if os.path.lexists(base):
                    continue                      # already correct, or not ours to replace
                try:
                    os.symlink(os.path.basename(real), base)
                except OSError:
                    pass                          # read-only venv: nothing more we can do

            lib64 = os.path.join(os.path.dirname(libdir), "lib64")
            if not os.path.lexists(lib64):
                try:
                    os.symlink("lib", lib64)
                except OSError:
                    pass


def cuda_ext_status():
    """(available, error) for the fused CUDA extension, resolving the lazy build first.

    `error` is the verbatim build failure — the compiler's own words — or None on success.
    Without it a failed JIT build is indistinguishable from one that was never attempted,
    and the first thing the user sees is an AttributeError on a `None` handle several
    layers downstream. Used by `glq-setup --verify`, which cannot tell from
    `torch.cuda.is_available()` whether GLQ's kernels can actually run here.
    """
    _try_load_cuda_ext()
    return _cuda_ext_available, _cuda_ext_error


def require_cuda_ext(symbol=None):
    """Return the loaded extension, raising a *named* error if it cannot serve `symbol`.

    Call sites used to reach straight through — `_ik._glq_cuda.<entry>` — so a failed build
    surfaced as `AttributeError: 'NoneType' object has no attribute …` deep inside a forward
    pass, with the actual cause (a missing header, no nvcc, no ninja) discarded far earlier.

    Two distinct failures, two distinct messages:
      * the extension never built  -> the recorded build error + where to get a toolkit
      * it built but lacks `symbol` -> a stale build; the JIT does not hash headers, so it
        has to be cleared by hand (CLAUDE.md).
    """
    available, error = cuda_ext_status()
    if not available:
        raise RuntimeError(
            "GLQ CUDA extension is not available, so this kernel cannot run.\n"
            f"{error}"
            "GLQ compiles its kernels on first use and needs a CUDA toolchain matching the "
            f"torch build. Install the full toolkit from {CUDA_DOWNLOADS_URL}, or use a "
            "release that ships prebuilt kernels.")
    if symbol is not None and not hasattr(_glq_cuda, symbol):
        raise RuntimeError(
            f"GLQ CUDA extension is loaded but has no '{symbol}' — it predates this kernel. "
            "The JIT build does not hash headers, so a stale build survives a source change; "
            "clear it with:  rm -rf ~/.cache/torch_extensions/*/glq_cuda")
    return _glq_cuda


def _try_load_cuda_ext():
    """Lazy-load the CUDA C dequant kernel. Returns True if available."""
    global _cuda_ext_available, _glq_cuda
    if _cuda_ext_available is not None:
        return _cuda_ext_available
    try:
        import os, shutil, sys

        # Step 1 of the ladder: the extension compiled in CI and shipped in the wheel. When
        # it is here, none of the toolchain machinery below runs — no nvcc, no ninja, no
        # ~1 min first-use compile, and none of the ways that compile fails on a machine
        # that has never built CUDA before. Step 2 (JIT) remains for torch/arch combinations
        # no wheel matches, and for sdist installs.
        try:
            from glq import _C as _prebuilt          # noqa: F401
        except Exception:                            # noqa: BLE001 - absent is the norm
            _prebuilt = None
        if _prebuilt is not None:
            _glq_cuda = _prebuilt
            _cuda_ext_available = True
            globals()['_cuda_ext_error'] = None
            return True

        # Ensure ninja is in PATH (venv bin may not be in subprocess PATH)
        if shutil.which('ninja') is None:
            venv_bin = os.path.join(sys.prefix, 'bin')
            if os.path.exists(os.path.join(venv_bin, 'ninja')):
                os.environ['PATH'] = venv_bin + ':' + os.environ.get('PATH', '')
        # Point torch at the venv-local CUDA if nothing else has. Precedence: a toolkit torch
        # already resolved (a system install) outranks the wheels, then an explicit
        # CUDA_HOME, then the pip toolchain — mixing a wheel's CUDA minor with the toolkit
        # torch was built against is how you get subtle link errors.
        #
        # `cpp.CUDA_HOME` is resolved **once, when torch is imported**, and `_join_cuda_home`
        # reads that constant rather than the environment. So assigning os.environ here is
        # necessary (nvcc's own subprocesses read it) but not sufficient: without also
        # updating the constant, torch still raises `CUDA_HOME environment variable is not
        # set` with the compiler sitting right there in site-packages.
        # Self-heal the wheel layout before building: a venv that predates this, or one glq
        # was pip-installed into without install.sh, still has the missing symlinks.
        try:
            repair_cuda_wheel_layout()
        except Exception:                                 # noqa: BLE001 - never fatal
            pass

        # The test is "can this CUDA_HOME compile", not "is it set". torch's
        # `_find_cuda_home()` falls back to `/usr/local/cuda` whenever that path merely
        # *exists*, and on a CUDA **runtime** install it does exist — with include/ and
        # lib64/ but no nvcc. Measured in nvidia/cuda:12.9.1-runtime-ubuntu24.04, and true of
        # any host carrying the runtime without the toolkit. Keyed on `is None`, glq would
        # leave that toolchain-less path in place and never reach the nvcc in site-packages.
        import torch.utils.cpp_extension as _cpp
        if not _cpp.CUDA_HOME or not os.path.exists(
                os.path.join(_cpp.CUDA_HOME, 'bin', 'nvcc')):
            _cuda_home = os.environ.get('CUDA_HOME') or _venv_cuda_home()
            if _cuda_home:
                os.environ['CUDA_HOME'] = _cuda_home
                os.environ['PATH'] = (os.path.join(_cuda_home, 'bin') + ':'
                                      + os.environ.get('PATH', ''))
                _cpp.CUDA_HOME = _cuda_home

        _load_ext = _cpp.load
        _csrc = os.path.join(os.path.dirname(__file__), 'csrc')
        cu_file = os.path.join(_csrc, 'glq_cuda.cu')
        cpp_file = os.path.join(_csrc, 'glq_bindings.cpp')
        e8p_file = os.path.join(_csrc, 'glq_e8p.cu')          # E8P TC-GEMV decode (--codebook e8p)
        trellis_file = os.path.join(_csrc, 'glq_trellis.cu')  # QTIP TCQ decode (--codebook trellis)
        if not os.path.exists(cu_file):
            _cuda_ext_available = False
            return False
        sources = ([cu_file, cpp_file]
                   + ([e8p_file] if os.path.exists(e8p_file) else [])
                   + ([trellis_file] if os.path.exists(trellis_file) else []))
        # The pip CUDA wheels ship libcudart.so.<major> but no `libcudart.so`, so torch's
        # implicit `-lcudart` cannot resolve. Supply the missing symlink; a no-op when a
        # system CUDA is in use.
        _ldflags = []
        _link_dir = _cudart_link_dir()
        if _link_dir:
            _ldflags.append(f'-L{_link_dir}')

        _glq_cuda = _load_ext(
            'glq_cuda',
            sources=sources,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            extra_ldflags=_ldflags,
            verbose=False,
        )
        _cuda_ext_available = True
        globals()['_cuda_ext_error'] = None   # never report available-with-a-stale-error
    except Exception:
        # Keep the reason. A bare `except: available = False` discards the one piece of
        # information that makes this fixable — the compiler said exactly what was missing
        # (a header, a library, nvcc itself), and the user is otherwise left with a
        # NoneType AttributeError from a call site that has no idea why.
        global _cuda_ext_error
        import traceback
        _cuda_ext_error = traceback.format_exc()
        _cuda_ext_available = False
        import warnings
        warnings.warn(
            "GLQ CUDA extension failed to build — falling back to the slower path, and "
            "kernel-only features (trellis serving under vLLM) will not work.\n"
            f"{_cuda_ext_error}"
            "GLQ compiles its kernels on first use and needs a CUDA toolchain matching "
            "the torch build. The bundled CUDA wheels were not sufficient here; install "
            f"the full toolkit from {CUDA_DOWNLOADS_URL} and try again.",
            RuntimeWarning, stacklevel=2)
    return _cuda_ext_available


if _triton_available:

    # ────────────────────────────────────────────────────────────────
    # Matvec kernel (B=1 decode) — autotuned BLOCK_M
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N_BLOCKS'],
    )
    @triton.jit
    def _glq_dequant_matvec_kernel(
        # Pointers
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        # 2-stage
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        # Dimensions
        M,
        N_BLOCKS,  # N // 8
        Wscale,
        # Strides
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        # Config
        HAS_STAGE2: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Fused dequant+matvec for B=1: y = dequant(Qidxs) @ x * Wscale."""
        pid = tl.program_id(0)
        m_start = pid * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        d_range = tl.arange(0, 8)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for j in range(N_BLOCKS):
            # Load x[j*8 : j*8+8]
            x_vec = tl.load(x_ptr + j * 8 + d_range).to(tl.float32)

            # Load indices: (BLOCK_M,)
            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask,
                other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            # Gather codebook: (BLOCK_M, 8)
            cb_vecs = tl.load(
                codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            if HAS_STAGE2:
                indices2 = tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask,
                    other=0,
                )
                indices2 = (indices2.to(tl.int32) & 0xFFFF)
                cb_vecs2 = tl.load(
                    codebook2_ptr + indices2[:, None] * stride_cb2_k + d_range[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                cb_vecs = cb_vecs + cb_vecs2 * inv_resid_scale

            # Dot: (BLOCK_M, 8) . (8,) -> (BLOCK_M,)
            acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

        acc *= Wscale
        tl.store(y_ptr + m_range, acc, mask=m_mask)

    # ────────────────────────────────────────────────────────────────
    # Tensor Core matmul kernel (B>=2) — tl.dot with K=16
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 16}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N_BLOCKS'],
    )
    @triton.jit
    def _glq_dequant_matmul_tc_kernel(
        # Pointers
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        # 2-stage pointers (optional)
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        # Dimensions
        B,
        M,
        N,
        N_BLOCKS,  # N // 8
        Wscale,
        # Strides
        stride_x_b,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        stride_y_b,
        # Config
        HAS_STAGE2: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Fused dequant+matmul with Tensor Core tl.dot (K=16).

        Processes 2 consecutive codebook blocks per iteration to form K=16
        tiles suitable for mma.m16n8k16 Tensor Core instructions.

        For each pair of blocks (j, j+1):
          x_tile: (BLOCK_B, 16) — 16 contiguous input values
          w_tile: (BLOCK_M, 16) — concat of 2 codebook gathers
          acc += tl.dot(x_tile, w_tile^T)
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)

        b_start = pid_b * BLOCK_B
        m_start = pid_m * BLOCK_M

        b_range = b_start + tl.arange(0, BLOCK_B)
        m_range = m_start + tl.arange(0, BLOCK_M)
        b_mask = b_range < B
        m_mask = m_range < M

        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

        # Precompute dimension helpers for K=16 tile construction
        d16 = tl.arange(0, 16)
        is_hi = d16 >= 8                           # which dims come from block j+1
        d_local = tl.where(is_hi, d16 - 8, d16)   # local dim within 8-d codebook entry

        # Process pairs of codebook blocks for K=16
        for j in range(0, N_BLOCKS, 2):
            # --- x tile: (BLOCK_B, 16) from two consecutive 8-blocks ---
            x_tile = tl.load(
                x_ptr + b_range[:, None] * stride_x_b + (j * 8 + d16[None, :]),
                mask=b_mask[:, None] & ((j * 8 + d16[None, :]) < N),
                other=0.0,
            ).to(tl.float16)

            # --- Primary indices for blocks j and j+1 ---
            idx0 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask, other=0,
            ).to(tl.int32) & 0xFFFF)

            j1_valid = (j + 1) < N_BLOCKS
            idx1 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + (j + 1) * stride_q_k,
                mask=m_mask & j1_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            # --- w tile: (BLOCK_M, 16) by concatenating two 8-d codebook gathers ---
            # For d in 0..7:  codebook[idx0[m], d]
            # For d in 8..15: codebook[idx1[m], d-8]
            idx_sel = tl.where(is_hi[None, :], idx1[:, None], idx0[:, None])
            w_tile = tl.load(
                codebook_ptr + idx_sel * stride_cb_k + d_local[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float16)

            if HAS_STAGE2:
                # Secondary codebook indices
                idx2_0 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_1 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + (j + 1) * stride_q2_k,
                    mask=m_mask & j1_valid, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_sel = tl.where(is_hi[None, :], idx2_1[:, None], idx2_0[:, None])
                w2_tile = tl.load(
                    codebook2_ptr + idx2_sel * stride_cb2_k + d_local[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)

                # Combine in fp32 then cast back to fp16
                w_tile = (w_tile.to(tl.float32) + w2_tile * inv_resid_scale).to(tl.float16)

            # --- Tensor Core matmul: (BLOCK_B, 16) @ (16, BLOCK_M) ---
            acc += tl.dot(x_tile, tl.trans(w_tile))

        acc *= Wscale
        tl.store(
            y_ptr + b_range[:, None] * stride_y_b + m_range[None, :],
            acc,
            mask=b_mask[:, None] & m_mask[None, :],
        )

    # ────────────────────────────────────────────────────────────────
    # Split-K matvec (B=1) — distributes N reduction across CTAs
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        ],
        key=['M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _splitk_matvec_kernel(
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        M,
        N_BLOCKS,
        Wscale,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        HAS_STAGE2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K dequant+matvec: each CTA processes BLOCKS_PER_SPLIT blocks,
        atomicAdds partial sums. Grid: (M/BLOCK_M, ceil(N_BLOCKS/BPS))."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_start = pid_m * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        d_range = tl.arange(0, 8)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for j_off in tl.static_range(BLOCKS_PER_SPLIT):
            j = j_start + j_off
            j_mask = j < N_BLOCKS

            x_vec = tl.load(x_ptr + j * 8 + d_range,
                            mask=j_mask, other=0.0).to(tl.float32)

            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_mask, other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            cb_vecs = tl.load(
                codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
                mask=m_mask[:, None] & j_mask, other=0.0,
            ).to(tl.float32)

            if HAS_STAGE2:
                indices2 = tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask & j_mask, other=0,
                )
                indices2 = (indices2.to(tl.int32) & 0xFFFF)
                cb_vecs2 = tl.load(
                    codebook2_ptr + indices2[:, None] * stride_cb2_k + d_range[None, :],
                    mask=m_mask[:, None] & j_mask, other=0.0,
                ).to(tl.float32)
                cb_vecs = cb_vecs + cb_vecs2 * inv_resid_scale

            acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

        tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)

    # ────────────────────────────────────────────────────────────────
    # Split-K Tensor Core matmul (B>=2) — 3D grid
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
        ],
        key=['M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _splitk_matmul_tc_kernel(
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        B,
        M,
        N,
        N_BLOCKS,
        Wscale,
        stride_x_b,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        stride_y_b,
        HAS_STAGE2: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K dequant+matmul with Tensor Cores.
        Grid: (B/BLOCK_B, M/BLOCK_M, ceil(N_BLOCKS/BPS))."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        b_start = pid_b * BLOCK_B
        m_start = pid_m * BLOCK_M

        b_range = b_start + tl.arange(0, BLOCK_B)
        m_range = m_start + tl.arange(0, BLOCK_M)
        b_mask = b_range < B
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

        d16 = tl.arange(0, 16)
        is_hi = d16 >= 8
        d_local = tl.where(is_hi, d16 - 8, d16)

        # Process pairs of blocks for K=16 TC tiles
        for pair_off in tl.static_range(BLOCKS_PER_SPLIT // 2):
            j = j_start + pair_off * 2
            j_valid = j < N_BLOCKS

            x_tile = tl.load(
                x_ptr + b_range[:, None] * stride_x_b + (j * 8 + d16[None, :]),
                mask=b_mask[:, None] & ((j * 8 + d16[None, :]) < N) & j_valid,
                other=0.0,
            ).to(tl.float16)

            idx0 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            j1_valid = (j + 1) < N_BLOCKS
            idx1 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + (j + 1) * stride_q_k,
                mask=m_mask & j1_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            idx_sel = tl.where(is_hi[None, :], idx1[:, None], idx0[:, None])
            w_tile = tl.load(
                codebook_ptr + idx_sel * stride_cb_k + d_local[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float16)

            if HAS_STAGE2:
                idx2_0 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask & j_valid, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_1 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + (j + 1) * stride_q2_k,
                    mask=m_mask & j1_valid, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_sel = tl.where(is_hi[None, :], idx2_1[:, None], idx2_0[:, None])
                w2_tile = tl.load(
                    codebook2_ptr + idx2_sel * stride_cb2_k + d_local[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)

                w_tile = (w_tile.to(tl.float32) + w2_tile * inv_resid_scale).to(tl.float16)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        # 2D atomic add
        scaled = acc * Wscale
        addrs = b_range[:, None] * stride_y_b + m_range[None, :]
        tl.atomic_add(y_ptr + addrs, scaled, mask=b_mask[:, None] & m_mask[None, :])

    # ────────────────────────────────────────────────────────────────
    # Packed uint32 split-K matvec (B=1) — 4x less L2 traffic
    # E8 codebook coords are {-3,-2.5,...,2.5,3} = (nibble-6)*0.5
    # Pack 8 nibbles into 1 uint32: 16B gather → 4B gather
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        ],
        key=['M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _packed_splitk_matvec_kernel(
        x_ptr,
        qidxs_ptr,
        packed_cb_ptr,
        y_ptr,
        M,
        N_BLOCKS,
        Wscale,
        stride_q_m,
        stride_q_k,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K matvec with packed uint32 codebook (4B per entry vs 16B)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_start = pid_m * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        d_range = tl.arange(0, 8)
        shifts = d_range * 4
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for j_off in tl.static_range(BLOCKS_PER_SPLIT):
            j = j_start + j_off
            j_mask = j < N_BLOCKS

            x_vec = tl.load(x_ptr + j * 8 + d_range,
                            mask=j_mask, other=0.0).to(tl.float32)

            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_mask, other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            packed = tl.load(packed_cb_ptr + indices,
                             mask=m_mask & j_mask, other=0)
            nibbles = (packed[:, None] >> shifts[None, :]) & 0xF
            cb_vecs = nibbles.to(tl.float32) * 0.5 - 3.0

            acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

        tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)

    # ────────────────────────────────────────────────────────────────
    # Packed uint32 split-K TC matmul (B>=2)
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
        ],
        key=['M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _packed_splitk_matmul_tc_kernel(
        x_ptr,
        qidxs_ptr,
        packed_cb_ptr,
        y_ptr,
        B, M, N, N_BLOCKS, Wscale,
        stride_x_b, stride_q_m, stride_q_k, stride_y_b,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K TC matmul with packed uint32 codebook."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        b_start = pid_b * BLOCK_B
        m_start = pid_m * BLOCK_M

        b_range = b_start + tl.arange(0, BLOCK_B)
        m_range = m_start + tl.arange(0, BLOCK_M)
        b_mask = b_range < B
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

        d16 = tl.arange(0, 16)
        is_hi = d16 >= 8
        d_local = tl.where(is_hi, d16 - 8, d16)
        shifts = d_local * 4

        for pair_off in tl.static_range(BLOCKS_PER_SPLIT // 2):
            j = j_start + pair_off * 2
            j_valid = j < N_BLOCKS

            x_tile = tl.load(
                x_ptr + b_range[:, None] * stride_x_b + (j * 8 + d16[None, :]),
                mask=b_mask[:, None] & ((j * 8 + d16[None, :]) < N) & j_valid,
                other=0.0,
            ).to(tl.float16)

            idx0 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            j1_valid = (j + 1) < N_BLOCKS
            idx1 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + (j + 1) * stride_q_k,
                mask=m_mask & j1_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            packed0 = tl.load(packed_cb_ptr + idx0, mask=m_mask & j_valid, other=0)
            packed1 = tl.load(packed_cb_ptr + idx1, mask=m_mask & j1_valid, other=0)
            packed_sel = tl.where(is_hi[None, :], packed1[:, None], packed0[:, None])
            nibbles = (packed_sel >> shifts[None, :]) & 0xF
            w_tile = (nibbles.to(tl.float32) * 0.5 - 3.0).to(tl.float16)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        scaled = acc * Wscale
        addrs = b_range[:, None] * stride_y_b + m_range[None, :]
        tl.atomic_add(y_ptr + addrs, scaled, mask=b_mask[:, None] & m_mask[None, :])


# Cached SM count to avoid repeated GPU queries
_num_sms: int = 0
_BLOCKS_PER_SPLIT = 64


def _get_num_sms(device) -> int:
    global _num_sms
    if _num_sms == 0:
        _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms


def glq_dequant_matmul(
    x: torch.Tensor,
    Qidxs: torch.Tensor,
    codebook: torch.Tensor,
    Wscale: float,
    Qidxs2: torch.Tensor = None,
    codebook2: torch.Tensor = None,
    inv_resid_scale: float = 0.0,
    codebook_packed: torch.Tensor = None,
    Qidxs3: torch.Tensor = None,
    codebook3: torch.Tensor = None,
    inv_resid_scale2: float = 0.0,
    Qidxs4: torch.Tensor = None,
    codebook4: torch.Tensor = None,
    inv_resid_scale3: float = 0.0,
) -> torch.Tensor:
    """Fused dequant+matmul: Y = X @ dequant(Qidxs)^T * Wscale.

    Auto-selects the fastest kernel variant based on shape and GPU:
    - Split-K when grid undersaturates SMs (1.5-2.4x speedup)
    - Packed uint32 codebook for small matrices (additional 1.3x speedup)

    Args:
        x: (B, N) input activations
        Qidxs: (M, N//8) primary codebook indices, int16
        codebook: (K, 8) primary codebook vectors, fp16/fp32
        Wscale: global scale factor
        Qidxs2: (M, N//8) secondary indices for 3/4bpw, or None
        codebook2: (K2, 8) secondary codebook for 3/4bpw, or None
        inv_resid_scale: 1.0 / resid_scale for 3/4bpw
        codebook_packed: (K,) uint32 packed codebook, or None
        Qidxs3, codebook3, inv_resid_scale2:
            Stage-3 RVQ (5/6bpw) — codebook3 reuses the primary E8
            codebook in practice. ``None`` disables stage 3.
        Qidxs4, codebook4, inv_resid_scale3:
            Stage-4 RVQ (7/8bpw). ``None`` disables stage 4.

    Returns:
        Y: (B, M) output in fp32
    """
    if not x.is_cuda or (not _triton_available and not _try_load_cuda_ext()):
        return _fallback_dequant_matmul(
            x, Qidxs, codebook, Wscale, Qidxs2, codebook2, inv_resid_scale
        )

    B, N = x.shape
    M, N_BLOCKS = Qidxs.shape
    assert N_BLOCKS == N // 8, f"Qidxs shape {Qidxs.shape} incompatible with N={N}"
    assert codebook.shape[1] == 8

    x_fp16 = x.half().contiguous()
    cb = codebook.half().contiguous()

    has_stage2 = Qidxs2 is not None and codebook2 is not None
    has_stage3 = Qidxs3 is not None and codebook3 is not None
    has_stage4 = Qidxs4 is not None and codebook4 is not None

    if has_stage2:
        q2 = Qidxs2.contiguous()
        cb2 = codebook2.half().contiguous()
    else:
        # Dummy pointers (won't be accessed when HAS_STAGE2=False)
        q2 = Qidxs
        cb2 = cb

    # CUDA C kernels: B=1 split-K matvec, B>1 inline PTX TC (mma.m16n8k16)
    # Both 2.7-3.3× faster than Triton
    # Use torch.ops.glq.* when registered (torch.compile compatible),
    # otherwise fall back to direct pybind11 calls.
    if _try_load_cuda_ext():
        _empty_i16 = torch.empty(0, dtype=torch.int16, device=x.device)
        _empty_f16 = torch.empty(0, dtype=torch.float16, device=x.device)
        _empty_i32 = torch.empty(0, dtype=torch.int32, device=x.device)
        _use_ops = hasattr(torch.ops, 'glq') and hasattr(torch.ops.glq, 'dequant_matvec')
        if has_stage3:
            q3 = Qidxs3.contiguous()
            cb3 = codebook3.half().contiguous()
        else:
            q3, cb3 = _empty_i16, _empty_f16
        if has_stage4:
            q4 = Qidxs4.contiguous()
            cb4 = codebook4.half().contiguous()
        else:
            q4, cb4 = _empty_i16, _empty_f16
        irs2 = inv_resid_scale2 if has_stage3 else 0.0
        irs3 = inv_resid_scale3 if has_stage4 else 0.0
        if B == 1:
            _q2 = q2 if has_stage2 else _empty_i16
            _cb2 = cb2 if has_stage2 else _empty_f16
            _irs = inv_resid_scale if has_stage2 else 0.0
            if _use_ops:
                y = torch.ops.glq.dequant_matvec(x_fp16[0], Qidxs, cb, Wscale, _q2, _cb2, _irs, _empty_i32)
            else:
                y = _glq_cuda.glq_dequant_matvec_cuda(x_fp16[0], Qidxs, cb, Wscale, _q2, _cb2, _irs, _empty_i32)
            return y.unsqueeze(0)  # (M,) → (1, M)
        else:
            if (not has_stage2 and not has_stage3 and not has_stage4
                    and codebook_packed is not None):
                if _use_ops:
                    y = torch.ops.glq.dequant_matmul_packed(x_fp16, Qidxs, codebook_packed, Wscale)
                else:
                    y = _glq_cuda.glq_dequant_matmul_packed_cuda(x_fp16, Qidxs, codebook_packed, Wscale)
                return y
            _q2 = q2 if has_stage2 else _empty_i16
            _cb2 = cb2 if has_stage2 else _empty_f16
            _irs = inv_resid_scale if has_stage2 else 0.0
            if _use_ops:
                y = torch.ops.glq.dequant_matmul(
                    x_fp16, Qidxs, cb, Wscale, _q2, _cb2, _irs, _empty_i32,
                    q3, cb3, irs2,
                    q4, cb4, irs3,
                )
            else:
                y = _glq_cuda.glq_dequant_matmul_cuda(
                    x_fp16, Qidxs, cb, Wscale, _q2, _cb2, _irs, _empty_i32,
                    q3, cb3, irs2,
                    q4, cb4, irs3,
                )
            return y

    # Decide whether to use split-K based on estimated grid saturation.
    # BLOCK_M=64 is typical for both matvec and TC paths.
    num_sms = _get_num_sms(x.device)
    block_m_est = 64
    if B == 1:
        est_grid = triton.cdiv(M, block_m_est)
    else:
        block_b_est = 32
        est_grid = triton.cdiv(B, block_b_est) * triton.cdiv(M, block_m_est)

    use_splitk = (est_grid < num_sms) and (N_BLOCKS >= _BLOCKS_PER_SPLIT)

    # Packed uint32 codebook: 4B gather instead of 16B.
    # Benchmarks show packed is 2-6% faster across all shapes (ALU decode
    # cost is negligible vs L2 bandwidth savings on L40S).
    # Only for 2bpw (no 2-stage support in packed kernels).
    use_packed = (
        use_splitk
        and codebook_packed is not None
        and not has_stage2
    )

    if use_packed:
        # Packed split-K path: 4B gather instead of 16B
        y = torch.zeros(B, M, dtype=torch.float32, device=x.device)
        bps = _BLOCKS_PER_SPLIT
        cb_packed = codebook_packed.contiguous()

        if B == 1:
            n_splits = triton.cdiv(N_BLOCKS, bps)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), n_splits)
            _packed_splitk_matvec_kernel[grid](
                x_fp16[0], Qidxs, cb_packed, y[0],
                M, N_BLOCKS, Wscale,
                Qidxs.stride(0), Qidxs.stride(1),
                BLOCKS_PER_SPLIT=bps,
            )
        else:
            bps_even = bps if bps % 2 == 0 else bps - 1
            n_splits = triton.cdiv(N_BLOCKS, bps_even)
            grid = lambda meta: (
                triton.cdiv(B, meta['BLOCK_B']),
                triton.cdiv(M, meta['BLOCK_M']),
                n_splits,
            )
            _packed_splitk_matmul_tc_kernel[grid](
                x_fp16, Qidxs, cb_packed, y,
                B, M, N, N_BLOCKS, Wscale,
                x_fp16.stride(0),
                Qidxs.stride(0), Qidxs.stride(1),
                y.stride(0),
                BLOCKS_PER_SPLIT=bps_even,
            )
    elif use_splitk:
        # Split-K: zero-init output for atomic accumulation
        y = torch.zeros(B, M, dtype=torch.float32, device=x.device)
        bps = _BLOCKS_PER_SPLIT

        if B == 1:
            n_splits = triton.cdiv(N_BLOCKS, bps)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), n_splits)
            _splitk_matvec_kernel[grid](
                x_fp16[0],
                Qidxs, cb, y[0],
                q2, cb2, inv_resid_scale,
                M, N_BLOCKS, Wscale,
                Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
                q2.stride(0), q2.stride(1), cb2.stride(0),
                HAS_STAGE2=has_stage2,
                BLOCKS_PER_SPLIT=bps,
            )
        else:
            # Ensure bps is even for K=16 TC pairing
            bps_even = bps if bps % 2 == 0 else bps - 1
            n_splits = triton.cdiv(N_BLOCKS, bps_even)
            grid = lambda meta: (
                triton.cdiv(B, meta['BLOCK_B']),
                triton.cdiv(M, meta['BLOCK_M']),
                n_splits,
            )
            _splitk_matmul_tc_kernel[grid](
                x_fp16, Qidxs, cb, y,
                q2, cb2, inv_resid_scale,
                B, M, N, N_BLOCKS, Wscale,
                x_fp16.stride(0),
                Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
                q2.stride(0), q2.stride(1), cb2.stride(0),
                y.stride(0),
                HAS_STAGE2=has_stage2,
                BLOCKS_PER_SPLIT=bps_even,
            )
    elif B == 1:
        # Original matvec path
        y = torch.empty(B, M, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        _glq_dequant_matvec_kernel[grid](
            x_fp16[0],
            Qidxs, cb, y[0],
            q2, cb2, inv_resid_scale,
            M, N_BLOCKS, Wscale,
            Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
            q2.stride(0), q2.stride(1), cb2.stride(0),
            HAS_STAGE2=has_stage2,
        )
    else:
        # Original TC matmul path
        y = torch.empty(B, M, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']), triton.cdiv(M, meta['BLOCK_M']))
        _glq_dequant_matmul_tc_kernel[grid](
            x_fp16, Qidxs, cb, y,
            q2, cb2, inv_resid_scale,
            B, M, N, N_BLOCKS, Wscale,
            x_fp16.stride(0),
            Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
            q2.stride(0), q2.stride(1), cb2.stride(0),
            y.stride(0),
            HAS_STAGE2=has_stage2,
        )

    return y


def _fallback_dequant_matmul(
    x: torch.Tensor,
    Qidxs: torch.Tensor,
    codebook: torch.Tensor,
    Wscale: float,
    Qidxs2: torch.Tensor = None,
    codebook2: torch.Tensor = None,
    inv_resid_scale: float = 0.0,
) -> torch.Tensor:
    """Naive fallback: materialize W then matmul."""
    M, n_blocks = Qidxs.shape
    N = n_blocks * 8

    # Convert int16 to unsigned indices (0-65535)
    idx = (Qidxs.long() & 0xFFFF).reshape(-1)
    W = codebook[idx].reshape(M, N)

    if Qidxs2 is not None and codebook2 is not None:
        idx2 = (Qidxs2.long() & 0xFFFF).reshape(-1)
        W2 = codebook2[idx2].reshape(M, N)
        W = W + W2 * inv_resid_scale

    return x.float() @ W.float().T * Wscale
