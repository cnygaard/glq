"""vLLM must load torch's own cuDNN, whatever `LD_LIBRARY_PATH` carries.

Measured on an AWS Deep Learning AMI (g7e.12xlarge, 2x RTX PRO 6000, 2026-11-16): the AMI's
`/etc/profile.d/dlami.sh` puts `/usr/local/cuda-13.3/lib` on `LD_LIBRARY_PATH`, and that
directory holds cuDNN 9.20.0. torch 2.14.0 ships cuDNN 9.24.0 in the `nvidia-cudnn-cu13` wheel.
`LD_LIBRARY_PATH` is searched before a library's own RUNPATH, so cuDNN 9.24's runtime-compiled
engines resolved `libcudnn_graph.so.9` from 9.20, and cuDNN refused the mix:

    Unable to load any of {libcudnn_engines_runtime_compiled.so.9.24.0, ...}
    RuntimeError: ... cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED

vLLM reaches that during its encoder profile, through the vision tower's Conv3d, so glq-code
with GLM-5.3-Flash stopped before it was ready. The same Conv3d alone, on that box:

    LD_LIBRARY_PATH as the AMI sets it                      fails as above
    torch's cuDNN directory prepended, the rest unchanged    runs, cuDNN 9.24.0
    LD_LIBRARY_PATH unset                                    runs, cuDNN 9.24.0

`child_env` therefore puts torch's cuDNN first, and only when another cuDNN is on the path.
"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import glq.supervisor as S


def _cudnn_dir(root, name):
    """A directory holding a cuDNN library, as the AMI's and the wheel's both do."""
    d = root / name
    d.mkdir(parents=True)
    (d / "libcudnn.so.9").write_bytes(b"")
    (d / "libcudnn_graph.so.9").write_bytes(b"")
    return str(d)


@pytest.fixture
def dirs(tmp_path):
    other = tmp_path / "mpi"
    other.mkdir()
    return SimpleNamespace(
        system=_cudnn_dir(tmp_path, "cuda-13.3/lib"),
        wheel=_cudnn_dir(tmp_path, "site-packages/nvidia/cudnn/lib"),
        other=str(other))


def _env(monkeypatch, ld_path, wheel):
    if ld_path is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", ld_path)
    monkeypatch.setattr(S, "_wheel_cudnn_dir", lambda: wheel, raising=False)
    monkeypatch.setattr(S, "flashinfer_env", lambda: {})
    return S.child_env()


# ---- the rule --------------------------------------------------------------------------------

def test_a_foreign_cudnn_on_the_path_puts_torchs_first(monkeypatch, dirs):
    env = _env(monkeypatch, f"{dirs.system}:{dirs.other}", dirs.wheel)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [dirs.wheel, dirs.system, dirs.other]


def test_the_rest_of_the_path_is_kept_in_order(monkeypatch, dirs):
    """Prepend, never replace: the AMI's path also carries CUPTI, Open MPI and NCCL."""
    env = _env(monkeypatch, f"{dirs.other}:{dirs.system}", dirs.wheel)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [dirs.wheel, dirs.other, dirs.system]


def test_torchs_directory_already_on_the_path_moves_first_once(monkeypatch, dirs):
    env = _env(monkeypatch, f"{dirs.system}:{dirs.wheel}", dirs.wheel)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [dirs.wheel, dirs.system]


def test_no_other_cudnn_on_the_path_leaves_it_alone(monkeypatch, dirs):
    env = _env(monkeypatch, dirs.other, dirs.wheel)
    assert env["LD_LIBRARY_PATH"] == dirs.other


def test_an_unset_path_stays_unset(monkeypatch, dirs):
    """Unset, the loader takes cuDNN's own RUNPATH before ld.so.cache, so torch's wins."""
    env = _env(monkeypatch, None, dirs.wheel)
    assert "LD_LIBRARY_PATH" not in env


def test_a_torch_without_a_wheel_cudnn_leaves_the_path_alone(monkeypatch, dirs):
    """A torch built against the system cuDNN needs that cuDNN found."""
    env = _env(monkeypatch, f"{dirs.system}:{dirs.other}", None)
    assert env["LD_LIBRARY_PATH"] == f"{dirs.system}:{dirs.other}"


def test_the_cpu_backend_is_left_alone(monkeypatch, dirs):
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{dirs.system}:{dirs.other}")
    monkeypatch.setattr(S, "_wheel_cudnn_dir", lambda: dirs.wheel, raising=False)
    env = S.child_env(device="cpu", ram_bytes=16 * 1024**3, weights_bytes=1024**3)
    assert env["LD_LIBRARY_PATH"] == f"{dirs.system}:{dirs.other}"


# ---- finding torch's cuDNN -------------------------------------------------------------------

def test_the_wheel_directory_is_found_from_the_package(dirs):
    pkg = os.path.dirname(dirs.wheel)                      # .../nvidia/cudnn
    spec = SimpleNamespace(submodule_search_locations=[pkg])
    assert S._wheel_cudnn_dir(find_spec=lambda name: spec) == dirs.wheel


def test_no_cudnn_wheel_means_none(tmp_path):
    def missing(name):
        raise ModuleNotFoundError(name)
    assert S._wheel_cudnn_dir(find_spec=missing) is None
    empty = SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    assert S._wheel_cudnn_dir(find_spec=lambda name: empty) is None


# ---- the mechanism, on a GPU box that has the trap ---------------------------------------------

_CONV3D = """
import torch
m = torch.nn.Conv3d(3, 1152, kernel_size=(2, 14, 14), stride=(2, 14, 14)).cuda().bfloat16()
y = m(torch.randn(1, 3, 2, 224, 224, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()
print("CONV3D_OK", torch.backends.cudnn.version())
"""


def _conv3d(env):
    return subprocess.run([sys.executable, "-c", _CONV3D], env=env, capture_output=True,
                          text=True, timeout=300)


def test_the_vision_conv_runs_in_the_child_env_where_the_plain_env_fails():
    """The vision tower's Conv3d (GLM-5.3's patch embedding shape), in a fresh process. Only
    meaningful on a box with the trap, which the control proves: under the plain environment it
    must fail with the sub-library error, and under `child_env` it must run."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if S._wheel_cudnn_dir() is None:
        pytest.skip("this torch does not ship its own cuDNN")
    control = _conv3d(dict(os.environ))
    if control.returncode == 0:
        pytest.skip("no foreign cuDNN breaks torch's here: nothing to fix on this box")
    assert "SUBLIBRARY_LOADING_FAILED" in control.stderr, control.stderr[-2000:]

    fixed = _conv3d(S.child_env())
    assert fixed.returncode == 0, fixed.stderr[-2000:]
    assert "CONV3D_OK" in fixed.stdout
