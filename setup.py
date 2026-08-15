"""Ahead-of-time build of `glq._C`, the fused CUDA extension.

GLQ has always compiled its kernels on the user's machine, at first use, via
`torch.utils.cpp_extension.load()`. Container testing found five independent ways that
fails on a machine which has never built CUDA before — no ninja, no nvcc, missing CCCL
headers, an unresolvable `-lcudart`, and a `CUDA_HOME` torch cached at import — none of
which reproduce on a box with a system CUDA install. Shipping the compiled object makes all
five structurally impossible for anyone a wheel matches.

The JIT path stays: it is step 2 of the ladder in `glq.inference_kernel._try_load_cuda_ext`,
for torch/arch combinations no wheel matches and for sdist installs.

Deliberately guarded rather than required. `pip install glq` on a CPU-only box, and building
the sdist anywhere without a compiler, must still produce a working (slow) install — so a
missing toolchain skips the extension instead of failing the build. Set GLQ_BUILD_EXT=1 to
turn that into a hard error, which is what CI wants: a wheel that silently shipped without
its kernels is worse than a red build.
"""
import os
import sys

from setuptools import setup

#: The same four translation units the JIT path compiles — keep in step with
#: `_try_load_cuda_ext`, which is the other place this list exists.
#:
#: Relative and /-separated, which setuptools requires: an absolute path fails the build with
#: "setup script specifies an absolute path ... must *always* be /-separated paths relative
#: to the setup.py directory". os.path.join(HERE, ...) is the natural thing to write here and
#: is wrong.
SOURCES = [f"glq/csrc/{name}" for name in
           ("glq_cuda.cu", "glq_bindings.cpp", "glq_e8p.cu", "glq_trellis.cu")]

#: "auto" (default): build it if we can. "1": require it. "0": never.
MODE = os.environ.get("GLQ_BUILD_EXT", "auto")


def _point_cuda_home_at_the_wheels():
    """Resolve CUDA_HOME from the pip CUDA wheels, before torch caches its own answer.

    `torch.utils.cpp_extension` computes `CUDA_HOME` once, at import, so this has to run
    first. Its fallbacks are, in order: $CUDA_HOME, `dirname(dirname(which nvcc))`, then
    `/usr/local/cuda` if that path exists.

    The second fallback is a trap when the toolchain came from pip. Putting nvcc on PATH by
    symlinking it next to `python` makes torch infer the *python* prefix as CUDA_HOME, and
    the build then dies on `fatal error: cuda_runtime.h: No such file or directory` — the
    headers live beside the real nvcc, in `nvidia/cu*/include`. Naming the wheel directory
    outright avoids guessing.

    A real toolkit already on PATH wins: torch was compiled against it.
    """
    import glob
    import shutil
    import site

    if os.environ.get("CUDA_HOME") or shutil.which("nvcc"):
        return
    for root in site.getsitepackages():
        for nvcc in sorted(glob.glob(os.path.join(root, "nvidia", "*", "bin", "nvcc"))):
            home = os.path.dirname(os.path.dirname(nvcc))
            os.environ["CUDA_HOME"] = home
            os.environ["PATH"] = os.path.join(home, "bin") + os.pathsep + os.environ.get("PATH", "")
            return


def _cuda_toolchain():
    """(BuildExtension, CUDAExtension) if this machine can compile CUDA, else None.

    The test is a compiler on disk, not `CUDA_HOME is not None`. torch's `_find_cuda_home()`
    falls back to `/usr/local/cuda` whenever that directory merely exists, and a CUDA
    *runtime* install has it with `include/` and `lib64/` and no nvcc at all — measured in
    nvidia/cuda:12.9.1-runtime-ubuntu24.04. Believing it there yields a build that dies well
    into the compile rather than a clean skip here.
    """
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except Exception as exc:                                          # noqa: BLE001
        return None, f"torch is not importable ({type(exc).__name__}: {exc})"
    if not CUDA_HOME:
        return None, "torch reports no CUDA_HOME"
    if not os.path.exists(os.path.join(CUDA_HOME, "bin", "nvcc")):
        return None, f"no nvcc under CUDA_HOME={CUDA_HOME} (a runtime install, not a toolkit)"
    return (BuildExtension, CUDAExtension), None


ext_modules, cmdclass = [], {}

if MODE != "0":
    _point_cuda_home_at_the_wheels()          # must precede any cpp_extension import
    toolchain, why_not = _cuda_toolchain()
    if toolchain is None:
        if MODE == "1":
            raise SystemExit(
                f"GLQ_BUILD_EXT=1 but the CUDA extension cannot be built: {why_not}.\n"
                "This is set in CI on purpose — a wheel without its kernels is worse than a "
                "failed build, because it looks fine until the first forward pass.")
        print(f"glq: building without the CUDA extension ({why_not}); "
              f"kernels will JIT-compile on first use", file=sys.stderr)
    else:
        BuildExtension, CUDAExtension = toolchain
        ext_modules = [
            CUDAExtension(
                name="glq._C",
                sources=SOURCES,
                # Matches the JIT path's flags exactly; a wheel that behaves differently from
                # the fallback it replaces would be a subtle source of "works on my machine".
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
            )
        ]
        # `use_ninja` inherits MAX_JOBS, which CI sets to the runner's core count.
        cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}

# Everything else — name, version, deps, entry points, package data — stays in
# pyproject.toml. This file exists only to add the extension.
setup(ext_modules=ext_modules, cmdclass=cmdclass)
