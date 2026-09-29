"""Which checkpoint weight does vLLM 0.30.0 fail to resolve to a parameter?

vLLM 0.30.0's ``MergedColumnParallelLinear.load_weights`` does::

    param = getattr(self, name, self)          # linear.py:984 -- default is SELF
    param.weight_loader(param, loaded_weight, shard_id)

so "this layer has no parameter called `name`" surfaces as::

    AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'

from ``param_data = param.data`` one frame deeper -- naming neither the weight nor the layer.
vLLM logs a weight name only *after* a successful load, so the failing one is never printed and
the traceback cannot identify it.

This wraps ``load_weights`` and reports **UNRESOLVED** the moment ``getattr`` misses, before the
call that crashes. It also prints every resolved name, so a second unresolved weight after the
first is fixed is immediately visible rather than needing another bisection round.

``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is forced: vLLM spawns EngineCore as a separate process and
re-imports there, so a monkeypatch applied in the parent would simply not exist where the loader
actually runs.

Read-only with respect to the repo and to site-packages -- it patches in memory for one run.

    python benchmarks/_vllm030_loader_probe.py --model <repo> [--expect-gib 78] [...]

Any unrecognised arguments are forwarded to run_model.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Must precede any vllm import: the loader has to run in THIS process for the patch to apply.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 0.30.0 defaults PLE cpu-offload ON, which selects the pinned-host table GLQ cannot serve.
os.environ.setdefault("VLLM_PLE_CPU_OFFLOAD", "0")

_MARK = "GLQPROBE"


def _install() -> None:
    from vllm.model_executor.layers import linear as _linear

    targets = [
        getattr(_linear, n, None)
        for n in ("MergedColumnParallelLinear", "QKVParallelLinear",
                  "ColumnParallelLinear", "RowParallelLinear", "ReplicatedLinear")
    ]
    for cls in targets:
        if cls is None or "load_weights" not in cls.__dict__:
            continue          # only wrap classes that define their own load_weights
        original = cls.__dict__["load_weights"]
        if getattr(original, "_glqprobe", False):
            continue

        def _wrapped(self, weights, _orig=original, _cls=cls):
            def _tap():
                for name, w in weights:
                    # Mirror vLLM's own resolution, but WITHOUT the `self` default, so a miss
                    # is visible as a miss instead of becoming an AttributeError two frames on.
                    try:
                        if "." in name:
                            sub, _, attr = name.rpartition(".")
                            resolved = getattr(self.get_submodule(sub), attr, None)
                        else:
                            resolved = getattr(self, name, None)
                    except Exception as e:                      # get_submodule can raise
                        resolved, e = None, e
                    shard = getattr(w, "shard_id", None)
                    if resolved is None:
                        print(f"{_MARK} UNRESOLVED cls={_cls.__name__} "
                              f"prefix={getattr(self, 'prefix', '?')} name={name!r} "
                              f"shape={tuple(w.shape)} shard_id={shard} "
                              f"registered={sorted(dict(self.named_parameters(recurse=False)))}",
                              flush=True)
                    else:
                        print(f"{_MARK} ok cls={_cls.__name__} "
                              f"prefix={getattr(self, 'prefix', '?')} name={name!r} "
                              f"shape={tuple(w.shape)} shard_id={shard}", flush=True)
                    yield name, w
            return _orig(self, _tap())

        _wrapped._glqprobe = True
        setattr(cls, "load_weights", _wrapped)
        print(f"{_MARK} patched {cls.__name__}.load_weights", flush=True)


def main() -> int:
    import glq_vllm  # noqa: F401  ensure the plugin's register() has run
    _install()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    argv = [a for a in sys.argv[1:]]
    if not any(a == "--runtime" for a in argv):
        argv += ["--runtime", "vllm"]
    if not any(a == "--quant" for a in argv):
        argv += ["--quant", "glq"]
    sys.argv = ["run_model.py"] + argv

    import runpy
    try:
        runpy.run_path(os.path.join(here, "run_model.py"), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    except BaseException as e:                                   # noqa: BLE001
        print(f"{_MARK} run raised {type(e).__name__}: {e}", flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
