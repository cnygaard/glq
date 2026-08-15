#!/usr/bin/env bash
# Prepare a manylinux container to compile glq._C.
#
# Runs once per wheel, inside cibuildwheel's build environment. Everything here is the same
# ground install.sh covers on a user's machine, for the same reasons — the container has no
# CUDA at all, so the toolchain comes from NVIDIA's published pip wheels.
set -euo pipefail

: "${GLQ_TORCH_VERSION:?set GLQ_TORCH_VERSION in the workflow}"

# The build runs with --no-isolation (see CIBW_BUILD_FRONTEND), so the backend and its
# helpers have to be present here rather than in a pip-managed isolated environment.
echo "== build backend"
pip install --no-cache-dir "setuptools>=64" wheel ninja

echo "== torch ${GLQ_TORCH_VERSION} (pinned: the extension links its C++ ABI)"
pip install --no-cache-dir "torch==${GLQ_TORCH_VERSION}"

# A SEPARATE pip call, at the version torch already pinned. Resolved together, pip picks the
# newest cuda-toolkit, finds it incompatible with torch's pin, backtracks, and silently
# downgrades torch — measured 2.13.0 -> 2.10.0, which would then build against an ABI the
# wheel does not declare.
ver="$(python -c 'import importlib.metadata as m; print(m.version("cuda-toolkit"))')"
echo "== cuda-toolkit[nvcc,cccl]==${ver} (torch's own pin)"
pip install --no-cache-dir --upgrade "cuda-toolkit[nvcc,cccl]==${ver}"

# The wheels ship lib/libcudart.so.NN with no unversioned libcudart.so and no lib64/, so
# `-L$CUDA_HOME/lib64 -lcudart` cannot resolve. Same repair install.sh performs in a user's
# venv; without it the link step fails exactly as it does on a fresh machine.
echo "== normalising the CUDA wheel layout"
python - <<'PY'
import glob, os, site, sys

roots = list(site.getsitepackages()) + [site.getusersitepackages()]
made = 0
for root in roots:
    for libdir in sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib"))):
        by_base = {}
        for path in glob.glob(os.path.join(libdir, "lib*.so.*")):
            base = path.split(".so.")[0] + ".so"
            if len(path) < len(by_base.get(base, path * 2)):
                by_base[base] = path
        for base, real in sorted(by_base.items()):
            if not os.path.lexists(base):
                os.symlink(os.path.basename(real), base)
                made += 1
        lib64 = os.path.join(os.path.dirname(libdir), "lib64")
        if not os.path.lexists(lib64):
            os.symlink("lib", lib64)
            made += 1
print(f"   {made} symlink(s) created", file=sys.stderr)
PY

# Deliberately NOT symlinking nvcc onto PATH. torch derives CUDA_HOME as
# `dirname(dirname(which nvcc))`, so an nvcc sitting next to `python` makes it infer the
# *python* prefix, and the build then dies on `fatal error: cuda_runtime.h: No such file or
# directory` — the headers live beside the real nvcc, under nvidia/cu*/include. setup.py
# names the wheel directory outright instead (`_point_cuda_home_at_the_wheels`), before
# torch.utils.cpp_extension is imported and caches its answer.
nvcc_path="$(python - <<'PY'
import glob, os, site
for root in site.getsitepackages():
    hits = sorted(glob.glob(os.path.join(root, "nvidia", "*", "bin", "nvcc")))
    if hits:
        print(hits[0])
        break
PY
)"
[ -n "$nvcc_path" ] || { echo "no nvcc in any CUDA wheel — cannot build" >&2; exit 1; }
echo "== nvcc: $nvcc_path"
"$nvcc_path" --version | tail -2
ls "$(dirname "$(dirname "$nvcc_path")")/include/cuda_runtime.h" \
  || { echo "nvcc present but cuda_runtime.h is not — the runtime headers are missing" >&2; exit 1; }
