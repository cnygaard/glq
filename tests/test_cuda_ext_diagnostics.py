"""When the CUDA extension fails to build, the reason has to survive the attempt.

`_try_load_cuda_ext()` JIT-compiles `glq/csrc/*` on first use. Its failure path is a bare
`except Exception: _cuda_ext_available = False` — the build error is caught and thrown away.
Nothing is logged, nothing is recorded, and `_glq_cuda` stays `None`.

That silence is not a cosmetic problem. Measured in an `ubuntu:24.04` container with a GPU
attached (2026-08-15), a user following the documented `curl … | bash` one-liner hits it in
sequence:

    1. no nvcc at all               → build impossible
    2. + cuda-toolkit[nvcc]         → fatal error: nv/target: No such file or directory
    3. + cuda-toolkit[nvcc,cccl]    → ld: cannot find -lcudart

Every one of those is a precise, actionable message from the compiler. All three were
discarded. What the user actually saw was, several layers away and much later:

    AttributeError: 'NoneType' object has no attribute
                    'glq_fused_linear_trellis_3inst_yrht_cuda'

Recovering the three real errors took repeated manual re-runs against a machine that had to
be rebuilt each time, because the information had been destroyed at the point it was known.

So: record the reason, and warn. The tests fake the build failure rather than causing a real
one — they must run on any machine, including CPU-only CI, and a real failure is neither
reproducible nor necessary to pin this behaviour.
"""
from __future__ import annotations

import os
import warnings

import pytest

torch = pytest.importorskip("torch")

import glq.inference_kernel as ik  # noqa: E402

#: A verbatim line from the container run, so the assertions pin something real.
NVCC_ERROR_LINE = "fatal error: nv/target: No such file or directory"
BUILD_ERROR = f"Error building extension 'glq_cuda'\n{NVCC_ERROR_LINE}"


@pytest.fixture(autouse=True)
def reset_loader_state():
    """The loader memoises its verdict in module globals; each test needs a fresh attempt.

    `_cuda_ext_error` has to be reset too, or a failure recorded by an earlier test leaks
    into a later success case — which is how these tests first went green individually and
    red as a suite.
    """
    saved = (ik._cuda_ext_available, ik._glq_cuda, ik._cuda_ext_error)
    ik._cuda_ext_available, ik._glq_cuda, ik._cuda_ext_error = None, None, None
    yield
    ik._cuda_ext_available, ik._glq_cuda, ik._cuda_ext_error = saved


def _make_build_fail(monkeypatch, exc=None):
    """Point torch's extension builder at a raiser.

    `_try_load_cuda_ext` does `from torch.utils.cpp_extension import load` *inside* the
    function, so patching the attribute on the module is enough — it is looked up at call
    time, not at import time.
    """
    import torch.utils.cpp_extension as cpp

    def _raise(*_a, **_k):
        raise exc or RuntimeError(BUILD_ERROR)

    monkeypatch.setattr(cpp, "load", _raise)
    # The scenario this file documents is a machine WITH a GPU whose build failed — that is
    # where the compiler's reason is the actionable thing. A machine with no CUDA device is
    # a supported CPU install and gets a different message, pinned by
    # test_a_machine_with_no_cuda_device_is_not_told_its_build_failed.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def _make_build_succeed(monkeypatch):
    import torch.utils.cpp_extension as cpp
    sentinel = object()
    monkeypatch.setattr(cpp, "load", lambda *_a, **_k: sentinel)
    return sentinel


# --------------------------------------------------------------- the recorded reason

def test_a_failed_build_records_the_compiler_error(monkeypatch):
    """The whole point: after a failed build, something must still know why."""
    _make_build_fail(monkeypatch)

    assert ik._try_load_cuda_ext() is False

    available, error = ik.cuda_ext_status()
    assert available is False
    assert error, "the build failed and no reason was retained"
    assert NVCC_ERROR_LINE in error, f"the compiler's own message was lost; got: {error!r}"


def test_the_reason_survives_repeated_calls(monkeypatch):
    """The verdict is cached, and the diagnosis must be cached with it.

    The extension is loaded lazily on the first matmul, but `glq-setup --verify` and any
    error message a user sees are asked much later. If the reason only exists during the
    failing call, every consumer after the first gets a bare 'unavailable'.
    """
    _make_build_fail(monkeypatch)
    ik._try_load_cuda_ext()

    for _ in range(3):
        available, error = ik.cuda_ext_status()
        assert available is False
        assert error and NVCC_ERROR_LINE in error


def test_the_build_is_attempted_only_once(monkeypatch):
    """Caching the failure must not turn into retrying a ~30 s compile on every call."""
    calls = []
    import torch.utils.cpp_extension as cpp

    def _raise(*_a, **_k):
        calls.append(1)
        raise RuntimeError(BUILD_ERROR)

    monkeypatch.setattr(cpp, "load", _raise)

    for _ in range(3):
        ik._try_load_cuda_ext()
    assert len(calls) == 1, f"rebuilt {len(calls)} times; the failure verdict is not cached"


# ------------------------------------------------------------------------- the warning

def test_the_user_is_warned_rather_than_silently_downgraded(monkeypatch):
    """A silent fallback reads as 'working, just slow' — or, on the trellis vLLM path,
    as an AttributeError with no stated cause. Say so at the moment it happens."""
    _make_build_fail(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ik._try_load_cuda_ext()

    messages = [str(w.message) for w in caught]
    assert messages, "the build failed and nothing was warned"
    assert any(NVCC_ERROR_LINE in m for m in messages), (
        f"warned, but without the compiler's reason — the actionable part. got: {messages}")


def test_the_warning_names_the_component_so_it_is_searchable(monkeypatch):
    """A user pasting the warning into a search box should land on GLQ, not on torch."""
    _make_build_fail(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ik._try_load_cuda_ext()

    joined = " ".join(str(w.message) for w in caught).lower()
    assert "glq" in joined
    assert "cuda extension" in joined or "cuda_ext" in joined


def test_it_warns_once_not_on_every_call(monkeypatch):
    """Paired with the cache: a warning per matmul would flood a serving log."""
    _make_build_fail(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            ik._try_load_cuda_ext()

    relevant = [w for w in caught if NVCC_ERROR_LINE in str(w.message)]
    assert len(relevant) == 1, f"warned {len(relevant)} times for one failure"


# ------------------------------------------------------------------------ the happy path

def test_a_successful_build_reports_no_error(monkeypatch):
    """The status accessor has to be trustworthy in both directions, or callers will
    learn to ignore it."""
    sentinel = _make_build_succeed(monkeypatch)

    assert ik._try_load_cuda_ext() is True
    available, error = ik.cuda_ext_status()
    assert available is True
    assert error is None
    assert ik._glq_cuda is sentinel


def test_a_successful_build_warns_about_nothing(monkeypatch):
    _make_build_succeed(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ik._try_load_cuda_ext()

    assert not [w for w in caught if "cuda extension" in str(w.message).lower()]


def test_the_failure_points_at_the_official_toolkit(monkeypatch):
    """When the bundled CUDA wheels are not enough, say where the real toolkit lives.

    The wheels are the default because they need no root and follow torch's CUDA major on
    their own. They are not guaranteed sufficient — the layout differs from a system
    install — so the failure path has to name the supported fallback. The landing page,
    not a versioned .run URL: the latter rots on every CUDA release.
    """
    _make_build_fail(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ik._try_load_cuda_ext()

    joined = " ".join(str(w.message) for w in caught)
    assert "developer.nvidia.com/cuda-downloads" in joined, (
        f"no route forward offered after a failed build: {joined!r}")


def test_a_machine_with_no_cuda_device_is_not_told_its_build_failed(monkeypatch):
    """CPU-only is a supported install, not a broken GPU one: GLQ serves trellis there —
    dense and MoE — on its own CPU kernels. Handing that user a compiler traceback and a
    CUDA-toolkit link sends them to fix something that was never going to exist, and the
    old wording went further and said vLLM trellis serving 'will not work', which is the
    thing they are doing. The compiler's reason is still recorded for anyone diagnosing."""
    _make_build_fail(monkeypatch)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ik._try_load_cuda_ext()

    joined = " ".join(str(w.message) for w in caught)
    assert joined, "silence is worse: the user still needs to know which path is running"
    assert "no CUDA device" in joined and "CPU" in joined
    assert "developer.nvidia.com" not in joined, "no toolkit errand on a CPU-only machine"
    assert NVCC_ERROR_LINE not in joined, "a compiler reason is noise where no build was expected"
    assert NVCC_ERROR_LINE in (ik._cuda_ext_error or ""), "but it must still be recorded"


# --------------------------------------------------- the libcudart.so the wheels omit
#
# Measured on the box and in containers (2026-08-15):
#
#   pip layout:     nvidia/cu13/lib/libcudart.so.13     <- versioned only, and lib/ not lib64/
#   system CUDA:    /usr/local/cuda/lib64/libcudart.so  <- dev symlink present
#
# torch's cpp_extension emits `-L$CUDA_HOME/lib64 -lcudart`, and `-lcudart` resolves only
# against a bare `libcudart.so`. With wheels alone the compile succeeds and the link dies:
#
#   /usr/bin/ld: cannot find -lcudart: No such file or directory
#
# So GLQ supplies the missing symlink in a cache dir and adds it to the link path. These
# tests fake the wheel layout in tmp_path: hermetic, no CUDA required, and they encode the
# layout facts above rather than a machine's current state.

def _fake_wheel_layout(root, soname="libcudart.so.13", cu="cu13"):
    lib = root / "site-packages" / "nvidia" / cu / "lib"
    lib.mkdir(parents=True)
    (lib / soname).write_bytes(b"")          # stand-in for the real shared object
    return root / "site-packages", lib / soname


def test_the_shim_supplies_the_symlink_the_wheels_omit(tmp_path):
    sp, real = _fake_wheel_layout(tmp_path)

    link_dir = ik._cudart_link_dir([str(sp)], str(tmp_path / "cache"))

    assert link_dir, "libcudart.so.13 was present but no link directory was produced"
    link = os.path.join(link_dir, "libcudart.so")
    assert os.path.exists(link), f"{link} missing — `-lcudart` will still fail"
    assert os.path.realpath(link) == str(real)


def test_the_shim_follows_whatever_soname_is_installed(tmp_path):
    """cu14 must work with no code change — the CUDA major is torch's to choose."""
    sp, real = _fake_wheel_layout(tmp_path, soname="libcudart.so.14", cu="cu14")

    link_dir = ik._cudart_link_dir([str(sp)], str(tmp_path / "cache"))

    assert link_dir
    assert os.path.realpath(os.path.join(link_dir, "libcudart.so")) == str(real)


def test_no_shim_when_the_wheels_already_ship_a_dev_symlink(tmp_path):
    """Don't manufacture a link that already exists — that is how you shadow a good
    system CUDA with a worse one."""
    sp, real = _fake_wheel_layout(tmp_path)
    (real.parent / "libcudart.so").symlink_to(real)

    assert ik._cudart_link_dir([str(sp)], str(tmp_path / "cache")) is None


def test_no_shim_when_a_system_cuda_already_provides_one(tmp_path, monkeypatch):
    """A machine can have both, and the GPU box does.

    It runs CUDA 13.2 under /usr/local (which torch compiles against, CUDA_HOME points
    there) *and* carries the pip wheels as torch dependencies, whose cudart is 13.0.
    Injecting `-L<wheel>` would put the 13.0 cudart ahead of the 13.2 the rest of the
    build uses — a silent version mix, on precisely the machine every GPU suite runs on.
    If a real toolkit can already satisfy `-lcudart`, do nothing.
    """
    sp, _ = _fake_wheel_layout(tmp_path)
    sys_cuda = tmp_path / "usr" / "local" / "cuda"
    (sys_cuda / "lib64").mkdir(parents=True)
    (sys_cuda / "lib64" / "libcudart.so").write_bytes(b"")

    import torch.utils.cpp_extension as cpp
    monkeypatch.setattr(cpp, "CUDA_HOME", str(sys_cuda), raising=False)

    assert ik._cudart_link_dir([str(sp)], str(tmp_path / "cache")) is None, (
        "shadowed a working system CUDA with the pip wheels")


def test_the_shim_still_applies_when_cuda_home_lacks_the_symlink(tmp_path, monkeypatch):
    """CUDA_HOME may itself be the wheel dir (torch resolves it there when no system CUDA
    exists) — that is the container case, and it must still get the shim."""
    sp, real = _fake_wheel_layout(tmp_path)

    import torch.utils.cpp_extension as cpp
    monkeypatch.setattr(cpp, "CUDA_HOME", str(real.parent.parent), raising=False)

    assert ik._cudart_link_dir([str(sp)], str(tmp_path / "cache"))


def test_no_shim_when_there_is_no_wheel_cudart_at_all(tmp_path):
    """A system-CUDA machine (the box) has no nvidia/cu*/lib at all. Must be a no-op,
    not a crash — this path runs on every first matmul."""
    assert ik._cudart_link_dir([str(tmp_path)], str(tmp_path / "cache")) is None


def test_the_shim_is_idempotent(tmp_path):
    """Called on every process start; a stale or duplicate link must not error."""
    sp, real = _fake_wheel_layout(tmp_path)
    cache = str(tmp_path / "cache")

    first = ik._cudart_link_dir([str(sp)], cache)
    second = ik._cudart_link_dir([str(sp)], cache)

    assert first == second
    assert os.path.realpath(os.path.join(second, "libcudart.so")) == str(real)


def test_the_link_dir_actually_reaches_the_compiler(monkeypatch, tmp_path):
    """Assert the mechanism, not just the helper.

    A shim that is built and then never passed to the builder fixes nothing, and every
    test above would still pass.
    """
    import torch.utils.cpp_extension as cpp
    captured = {}

    def _capture(*_a, **kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr(cpp, "load", _capture)
    monkeypatch.setattr(ik, "_cudart_link_dir", lambda *_a, **_k: str(tmp_path / "shim"))

    ik._try_load_cuda_ext()

    flags = " ".join(captured.get("extra_ldflags") or [])
    assert str(tmp_path / "shim") in flags, (
        f"the shim never reached the linker; extra_ldflags={captured.get('extra_ldflags')!r}")


# ------------------------------------------------ pointing torch at the venv-local nvcc
#
# `cuda-toolkit[nvcc,cccl]` installs the compiler into
#
#     <site-packages>/nvidia/cu13/bin/nvcc
#
# which is on nobody's PATH and is not where torch looks. Measured in a clean container
# (2026-08-15) with the toolchain correctly installed and torch un-downgraded:
#
#     derived cuda-toolkit version: 13.0.3.0
#     torch after toolkit: 2.13.0+cu130      <- good, the split worked
#     CUDA_HOME: None
#     OSError: CUDA_HOME environment variable is not set.
#
# The build failed in 1.1 s — it never reached a compiler. So installing the toolchain
# achieves nothing unless something also tells torch where it went. glq already does exactly
# this for ninja a few lines up; CUDA needs the same treatment.

def test_the_venv_cuda_root_is_discovered(tmp_path):
    root = tmp_path / "site-packages" / "nvidia" / "cu13"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "nvcc").write_bytes(b"")

    assert ik._venv_cuda_home([str(tmp_path / "site-packages")]) == str(root)


def test_no_venv_cuda_root_when_there_is_no_nvcc(tmp_path):
    (tmp_path / "site-packages" / "nvidia" / "cu13" / "lib").mkdir(parents=True)
    assert ik._venv_cuda_home([str(tmp_path / "site-packages")]) is None


def test_the_build_sets_cuda_home_when_it_is_unset(monkeypatch, tmp_path):
    """Without this the toolchain install is inert and the build dies before compiling."""
    root = tmp_path / "nvidia" / "cu13"
    (root / "bin").mkdir(parents=True)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(ik, "_venv_cuda_home", lambda *_a, **_k: str(root))
    _make_build_succeed(monkeypatch)

    ik._try_load_cuda_ext()

    assert os.environ.get("CUDA_HOME") == str(root)
    assert str(root / "bin") in os.environ.get("PATH", "")


def test_an_existing_cuda_home_is_never_overridden(monkeypatch, tmp_path):
    """A system CUDA, or one the user set deliberately, outranks the wheels — the same
    precedence the libcudart shim uses, and for the same reason: mixing CUDA minors."""
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-13.2")
    monkeypatch.setattr(ik, "_venv_cuda_home", lambda *_a, **_k: str(tmp_path / "wheels"))
    _make_build_succeed(monkeypatch)

    ik._try_load_cuda_ext()

    assert os.environ["CUDA_HOME"] == "/usr/local/cuda-13.2"


# -------------------------------------------- the environment is not what torch reads
#
# Setting os.environ['CUDA_HOME'] is necessary but NOT sufficient, and the difference cost a
# full container round trip. torch resolves the variable **once, at import time**, into a
# module-level constant, and that constant is what the build consults:
#
#     torch/utils/cpp_extension.py
#         CUDA_HOME = _find_cuda_home()                 # import time, ONE shot
#         def _join_cuda_home(*paths):
#             if CUDA_HOME is None:
#                 raise OSError('CUDA_HOME environment variable is not set. …')
#
# So by the time glq runs, the answer is already cached and no amount of os.environ writing
# changes it. Measured on the real `curl … | bash` path against released 0.8.5, in an
# ubuntu:24.04 container: the toolchain landed correctly (cuda-toolkit 13.0.3.0,
# nvidia-cuda-nvcc 13.0.88, nvcc present at
# `…/site-packages/nvidia/cu13/bin/nvcc` — exactly where `_venv_cuda_home()` globs) and the
# build still died with `OSError: CUDA_HOME environment variable is not set`.
#
# The earlier probe that "proved" this worked had exported CUDA_HOME in the shell, before
# python started — so torch picked it up at import and glq's code was never under test.
# These assert the mechanism that actually decides the build.

def test_torch_sees_the_cuda_home_the_build_sets(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp

    root = tmp_path / "nvidia" / "cu13"
    (root / "bin").mkdir(parents=True)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(cpp, "CUDA_HOME", None, raising=False)
    monkeypatch.setattr(ik, "_venv_cuda_home", lambda *_a, **_k: str(root))
    _make_build_succeed(monkeypatch)

    ik._try_load_cuda_ext()

    assert cpp.CUDA_HOME == str(root), (
        "os.environ['CUDA_HOME'] is not what torch reads — it caches the value at import "
        "time, so the variable must be pushed into the module constant as well")


def test_a_cuda_home_torch_already_resolved_is_not_replaced(monkeypatch, tmp_path):
    """The wheels must never shadow a real toolkit: torch was *compiled* against that one.

    "Real" means it has a compiler in it — see the runtime-image test below for why the
    directory merely existing is not enough.
    """
    import torch.utils.cpp_extension as cpp

    system = tmp_path / "usr" / "local" / "cuda"
    (system / "bin").mkdir(parents=True)
    (system / "bin" / "nvcc").write_bytes(b"")
    monkeypatch.setattr(cpp, "CUDA_HOME", str(system), raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(ik, "_venv_cuda_home", lambda *_a, **_k: str(tmp_path / "wheels"))
    _make_build_succeed(monkeypatch)

    ik._try_load_cuda_ext()

    assert cpp.CUDA_HOME == str(system)


def test_a_cuda_home_with_no_compiler_in_it_is_replaced(monkeypatch, tmp_path):
    """A CUDA *runtime* install is not a toolchain, and torch cannot tell the difference.

        def _find_cuda_home():
            …
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):   # <- existence, not usability
                cuda_home = None

    Measured in `nvidia/cuda:12.9.1-runtime-ubuntu24.04`: `/usr/local/cuda` is present with
    `include/` and `lib64/`, `nvcc` is MISSING, and there is no `libcudart.so`. So torch
    resolves a CUDA_HOME that cannot compile anything, and a guard keyed on `is None` sits
    there while the perfectly good nvcc in site-packages goes unused. Same on any host with
    the CUDA runtime but no toolkit — a very common shape.
    """
    import torch.utils.cpp_extension as cpp

    runtime_only = tmp_path / "usr" / "local" / "cuda"
    (runtime_only / "lib64").mkdir(parents=True)          # libs, but no bin/nvcc
    wheels = tmp_path / "nvidia" / "cu13"
    (wheels / "bin").mkdir(parents=True)

    monkeypatch.setattr(cpp, "CUDA_HOME", str(runtime_only), raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(ik, "_venv_cuda_home", lambda *_a, **_k: str(wheels))
    _make_build_succeed(monkeypatch)

    ik._try_load_cuda_ext()

    assert cpp.CUDA_HOME == str(wheels), (
        "torch resolved a toolkit-less /usr/local/cuda and glq left it there, so the build "
        "has no compiler even though the venv ships one")


def test_a_cuda_home_exported_after_torch_imported_still_reaches_torch(monkeypatch):
    """`os.environ['CUDA_HOME'] = …` in a notebook, or any import ordering that puts torch
    first, lands in the same trap — the export is real and torch never sees it."""
    import torch.utils.cpp_extension as cpp

    monkeypatch.setattr(cpp, "CUDA_HOME", None, raising=False)
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-13.2")
    monkeypatch.setattr(ik, "_venv_cuda_home", lambda *_a, **_k: None)
    _make_build_succeed(monkeypatch)

    ik._try_load_cuda_ext()

    assert cpp.CUDA_HOME == "/usr/local/cuda-13.2"


# ------------------------------------------- asking for a symbol that may not be there
#
# `glq_vllm/linear_method.py:671` reached straight into the extension:
#
#     _yrht = _ik._glq_cuda.glq_fused_linear_trellis_3inst_yrht_cuda
#
# With the build failed, `_glq_cuda` is None, and the user's first symptom — several layers
# from the cause, mid-forward-pass — is:
#
#     AttributeError: 'NoneType' object has no attribute
#                     'glq_fused_linear_trellis_3inst_yrht_cuda'
#
# That is what a container reported this morning after the toolchain was missing. The fix is
# one accessor that turns both failure shapes into a sentence naming the cause.

def test_requiring_a_symbol_reports_the_build_failure(monkeypatch):
    _make_build_fail(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        ik.require_cuda_ext("glq_fused_linear_trellis_3inst_yrht_cuda")

    message = str(excinfo.value)
    assert NVCC_ERROR_LINE in message, f"the build reason is missing from the error: {message}"
    assert "developer.nvidia.com/cuda-downloads" in message
    assert "NoneType" not in message


def test_requiring_a_missing_symbol_names_it_and_the_rebuild(monkeypatch):
    """A *stale* extension is the other failure: it built, but predates the kernel being
    asked for. CLAUDE.md's rule — the JIT does not hash headers, so a stale build must be
    cleared by hand — is exactly what the user needs told here."""
    class _Partial:
        pass
    monkeypatch.setattr(ik, "_cuda_ext_available", True)
    monkeypatch.setattr(ik, "_glq_cuda", _Partial())
    monkeypatch.setattr(ik, "_cuda_ext_error", None)

    with pytest.raises(RuntimeError) as excinfo:
        ik.require_cuda_ext("glq_fused_linear_trellis_3inst_yrht_cuda")

    message = str(excinfo.value)
    assert "glq_fused_linear_trellis_3inst_yrht_cuda" in message
    assert "torch_extensions" in message, f"no rebuild instruction given: {message}"


def test_requiring_a_present_symbol_returns_the_extension(monkeypatch):
    class _Complete:
        glq_fused_linear_trellis_3inst_yrht_cuda = staticmethod(lambda *a: None)

    ext = _Complete()
    monkeypatch.setattr(ik, "_cuda_ext_available", True)
    monkeypatch.setattr(ik, "_glq_cuda", ext)
    monkeypatch.setattr(ik, "_cuda_ext_error", None)

    assert ik.require_cuda_ext("glq_fused_linear_trellis_3inst_yrht_cuda") is ext
    assert ik.require_cuda_ext() is ext          # no symbol named: just the module


def test_status_before_any_attempt_does_not_lie(monkeypatch):
    """Untried is not the same as failed. Reporting 'unavailable' before the lazy build has
    run would make `--verify` fail on a perfectly good install."""
    sentinel = _make_build_succeed(monkeypatch)

    available, error = ik.cuda_ext_status()
    assert available is True, "status() must resolve the lazy build rather than guess"
    assert error is None
    assert ik._glq_cuda is sentinel


# ------------------------------------- making the pip CUDA wheels look like a real toolkit
#
# `_cudart_link_dir()` fixes GLQ's *own* link line. It cannot fix anybody else's, and on a
# machine whose only CUDA is the pip wheels, everybody else has the same problem. Measured
# in an ubuntu:24.04 container (2026-08-15): GLQ installed, GLQ's kernels built, the
# self-check passed — and vLLM then failed to start, because it JIT-compiles flashinfer:
#
#     c++ … -L …/nvidia/cu13/lib64 -L …/nvidia/cu13/lib64/stubs -lcudart -lcuda
#     /usr/bin/ld: cannot find -lcudart: No such file or directory
#
# Two independent gaps in the wheels' layout, both of which a real CUDA install has:
#   * `lib/libcudart.so`  — the dev symlink `-lcudart` resolves through
#   * `lib64/`            — the directory every CUDA build system passes to -L
#
# Repairing the venv once covers GLQ, flashinfer, and anything else compiled later; shimming
# per-consumer does not scale past the consumer you happen to control.

def _wheel_layout(tmp_path, major="13"):
    lib = tmp_path / "nvidia" / f"cu{major}" / "lib"
    lib.mkdir(parents=True)
    (lib / f"libcudart.so.{major}").write_bytes(b"")
    return tmp_path / "nvidia" / f"cu{major}"


def test_the_dev_symlink_is_created_so_lcudart_resolves(tmp_path):
    root = _wheel_layout(tmp_path)
    ik.repair_cuda_wheel_layout([str(tmp_path)])
    link = root / "lib" / "libcudart.so"
    assert link.is_symlink() or link.exists()
    assert os.path.realpath(link) == os.path.realpath(root / "lib" / "libcudart.so.13")


def test_lib64_is_created_because_build_systems_pass_it(tmp_path):
    """flashinfer's link line is `-L$CUDA_HOME/lib64`, and the wheels have no lib64 at all."""
    root = _wheel_layout(tmp_path)
    ik.repair_cuda_wheel_layout([str(tmp_path)])
    lib64 = root / "lib64"
    assert lib64.exists(), "no lib64 — every -L$CUDA_HOME/lib64 consumer still fails"
    assert (lib64 / "libcudart.so").exists()


def test_repairing_twice_changes_nothing(tmp_path):
    """It runs on every install and every lazy load; it must not thrash or raise."""
    root = _wheel_layout(tmp_path)
    ik.repair_cuda_wheel_layout([str(tmp_path)])
    before = sorted(p.name for p in (root / "lib").iterdir())
    ik.repair_cuda_wheel_layout([str(tmp_path)])
    assert sorted(p.name for p in (root / "lib").iterdir()) == before


def test_a_real_toolkit_layout_is_left_alone(tmp_path):
    """If the wheels already ship the symlink, do not replace it — and never touch a system
    CUDA, which glq does not own."""
    root = _wheel_layout(tmp_path)
    real = root / "lib" / "libcudart.so"
    real.write_bytes(b"not-a-symlink")
    ik.repair_cuda_wheel_layout([str(tmp_path)])
    assert real.read_bytes() == b"not-a-symlink"


def test_no_cuda_wheels_is_not_an_error(tmp_path):
    ik.repair_cuda_wheel_layout([str(tmp_path)])          # must not raise


def test_a_read_only_venv_is_not_an_error(tmp_path):
    """Some deployments run from a read-only site-packages. Degrade, do not crash."""
    root = _wheel_layout(tmp_path)
    os.chmod(root / "lib", 0o555)
    try:
        ik.repair_cuda_wheel_layout([str(tmp_path)])      # must not raise
    finally:
        os.chmod(root / "lib", 0o755)


def test_the_repair_refuses_to_touch_a_system_prefix(tmp_path, monkeypatch):
    """glq must not write into a system site-packages just because it was imported there.

    Under `sudo pip install glq`, or root in a container, `site.getsitepackages()` is
    /usr/lib/python3/dist-packages — not a directory an inference library should be creating
    symlinks in unasked. The venv case is the one install.sh creates and therefore owns.
    """
    import site as site_mod
    import sys as sys_mod

    root = _wheel_layout(tmp_path)
    monkeypatch.setattr(sys_mod, "prefix", "/usr", raising=False)
    monkeypatch.setattr(sys_mod, "base_prefix", "/usr", raising=False)   # i.e. not a venv
    monkeypatch.setattr(site_mod, "getsitepackages", lambda: [str(tmp_path)])

    ik.repair_cuda_wheel_layout()

    assert not (root / "lib" / "libcudart.so").exists(), "wrote into a system prefix"
    assert not (root / "lib64").exists(), "wrote into a system prefix"


def test_the_repair_runs_inside_a_venv(tmp_path, monkeypatch):
    """The case install.sh creates, and the only one glq owns."""
    import site as site_mod
    import sys as sys_mod

    root = _wheel_layout(tmp_path)
    monkeypatch.setattr(sys_mod, "prefix", "/home/u/.glq/venv", raising=False)
    monkeypatch.setattr(sys_mod, "base_prefix", "/usr", raising=False)   # venv
    monkeypatch.setattr(site_mod, "getsitepackages", lambda: [str(tmp_path)])

    ik.repair_cuda_wheel_layout()

    assert (root / "lib" / "libcudart.so").exists()
    assert (root / "lib64").exists()


def test_an_explicit_root_is_still_honoured_outside_a_venv(tmp_path, monkeypatch):
    """The guard is about the *default*. install.sh and the tests name a tree deliberately,
    and that intent should not be second-guessed."""
    import sys as sys_mod

    root = _wheel_layout(tmp_path)
    monkeypatch.setattr(sys_mod, "prefix", "/usr", raising=False)
    monkeypatch.setattr(sys_mod, "base_prefix", "/usr", raising=False)

    ik.repair_cuda_wheel_layout([str(tmp_path)])

    assert (root / "lib" / "libcudart.so").exists()


# ------------------------------------------------- step 1 of the ladder: a prebuilt wheel
#
# The wheel ships `glq._C`, compiled in CI for every arch we support. When it is present the
# user must never pay a JIT compile, and must never be exposed to the five ways that compile
# fails — the whole point of shipping binaries. JIT stays as step 2 for torch/arch
# combinations no wheel matches.

def test_a_prebuilt_extension_is_used_and_nothing_is_compiled(monkeypatch):
    import sys as sys_mod
    import types
    import torch.utils.cpp_extension as cpp

    prebuilt = types.ModuleType("glq._C")
    monkeypatch.setitem(sys_mod.modules, "glq._C", prebuilt)

    def _must_not_run(*_a, **_k):
        raise AssertionError("JIT-compiled despite a prebuilt glq._C being importable")

    monkeypatch.setattr(cpp, "load", _must_not_run)

    assert ik._try_load_cuda_ext() is True
    assert ik._glq_cuda is prebuilt
    available, error = ik.cuda_ext_status()
    assert available is True and error is None


def test_the_jit_still_runs_when_there_is_no_prebuilt_extension(monkeypatch):
    """sdist installs, unmatched torch minors and unmatched arches all land here."""
    import sys as sys_mod

    monkeypatch.setitem(sys_mod.modules, "glq._C", None)   # import raises ImportError
    sentinel = _make_build_succeed(monkeypatch)

    assert ik._try_load_cuda_ext() is True
    assert ik._glq_cuda is sentinel


# ------------------------------- why the *prebuilt* extension did not load, when it exists
#
# The ladder's first step originally did `except Exception: _prebuilt = None`, which is the
# same silent discard the JIT path was fixed for. It matters more here, not less: a wheel
# that ships a `_C.so` which cannot load — wrong torch ABI, a missing runtime library — is
# indistinguishable from a wheel that ships no extension at all, and the fallback JIT then
# fails for an unrelated reason and reports only that. Measured while validating a wheel on
# fedora:44, where exactly this happened and the first diagnosis was wrong.

def test_a_prebuilt_that_cannot_load_records_why(monkeypatch):
    import sys as sys_mod

    monkeypatch.setitem(sys_mod.modules, "glq._C", None)   # import raises
    _make_build_fail(monkeypatch)

    available, error = ik.cuda_ext_status()

    assert available is False
    assert "glq._C" in error, (
        f"the prebuilt extension failed to import and the reason was discarded; "
        f"error only says: {error!r}")
    assert NVCC_ERROR_LINE in error, "the JIT reason must survive too — both failed, say both"


def test_a_working_jit_does_not_report_the_missing_prebuilt(monkeypatch):
    """Absent is the normal case (sdist installs, unmatched arches). Reporting it as an
    error every time would train people to ignore the field that matters."""
    import sys as sys_mod

    monkeypatch.setitem(sys_mod.modules, "glq._C", None)
    _make_build_succeed(monkeypatch)

    available, error = ik.cuda_ext_status()

    assert available is True
    assert error is None
