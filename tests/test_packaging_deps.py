"""Everything the inference path imports at module scope must be a declared dependency.

`pip install glq` has to be enough to *serve* a checkpoint. It usually looks like it is,
because the two normal installs drag the missing pieces in by accident: vLLM depends on
numpy, and so does transformers. A core-only install has neither, and the failure lands as
a ModuleNotFoundError from inside a decode call rather than from pip.

That is how `numpy` was missed. `glq/trellis.py` imports it at module scope, trellis has
been the default codebook since 0.8.8, and `glq_vllm` imports `trellis_rvq_recipe` while
loading weights — so a core install could not serve the default format. Found in a pristine
`ubuntu:24.04` container, where torch printed "Failed to initialize NumPy" and nothing else
complained. The same class of bug is recorded in pyproject's `hf` extra comment.

This is deliberately mechanical rather than a list of known names: the point is to catch the
*next* one at edit time instead of in a container.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tomllib

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Modules reachable while loading and running a quantized model — no quantizer, no HF
#: integration (transformers lives in the `hf`/`quantize` extras), no installer.
INFERENCE_PATH = [
    "glq/__init__.py",
    "glq/trellis.py",
    "glq/quantized_linear.py",
    "glq/rht.py",
    "glq/ldlq.py",
    "glq/hadamard.py",
    "glq/inference_kernel.py",
    "glq/inference_kernel_cpu.py",
]

#: Distribution name -> the module it provides, where they differ.
_MODULE_TO_DIST = {"torch": "torch", "numpy": "numpy", "triton": "triton"}


def _core_dependencies() -> set[str]:
    """`[project] dependencies` as bare distribution names.

    tomllib is stdlib from 3.11 and glq's floor is 3.12, so it needs no fallback.
    """
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        pyproject = tomllib.load(fh)
    # "torch>=2.0" -> "torch"
    return {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].strip()
            for d in pyproject["project"]["dependencies"]}


def _module_scope_imports(relpath: str) -> set[str]:
    """Top-level (not function-local) imports, as top-level package names.

    Function-local imports are deliberately ignored: they are how this codebase makes an
    optional dependency optional — `glq/trellis.py` imports scipy inside the encode-time
    function that needs it, so a serving install never touches it.
    """
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=relpath)

    found: set[str] = set()
    for node in tree.body:                       # tree.body only == module scope
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("relpath", INFERENCE_PATH)
def test_module_scope_imports_are_declared(relpath):
    declared = _core_dependencies()
    for module in sorted(_module_scope_imports(relpath)):
        if module in sys.stdlib_module_names or module in ("glq", "glq_vllm"):
            continue
        dist = _MODULE_TO_DIST.get(module, module)
        assert dist in declared, (
            f"{relpath} imports {module!r} at module scope but {dist!r} is not in "
            f"pyproject's [project] dependencies ({sorted(declared)}). Either declare it, "
            f"or move the import into the function that needs it — a `pip install glq` "
            f"user hits this as a ModuleNotFoundError mid-decode.")


def test_numpy_is_declared():
    """The specific one this file was written for; kept explicit so the reason survives."""
    assert "numpy" in _core_dependencies(), (
        "glq.trellis imports numpy at module scope and glq_vllm imports glq.trellis while "
        "loading weights, so a core install must ship numpy")


def test_scipy_stays_out_of_the_core_dependencies():
    """The complement: scipy is encode-time only and imported inside its function, so a
    serving install must not be made to carry it."""
    assert "scipy" not in _core_dependencies()
    assert "scipy" not in _module_scope_imports("glq/trellis.py")
