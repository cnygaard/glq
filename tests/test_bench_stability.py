"""Make a decode number report its own stability.

`_two_point` times one 1-token run and one N-token run and subtracts, which correctly
removes prefill but also ADDS the two measurements' variance: the difference of two noisy
samples is noisier than either. With `--decode 8` only 7 decode steps carry that noise, so
every figure so far has been a point estimate with no error bar.

Three cheap fixes, all pinned here:

* more decode steps -- endpoint noise is roughly constant while the span grows linearly,
  so relative error falls ~1/N;
* repeats, so there IS a spread to report rather than a bare number;
* a longer warmup. On CPU the 78 GiB of weights arrive by lazy page-fault through mmap, and
  a 4-token warmup cannot fault in the working set, so the timed run still pays first-touch
  cost. That is a bias, not just noise.
"""
from __future__ import annotations

import importlib.util
import os
import sys


HERE = os.path.dirname(__file__)
DRIVER = os.path.join(HERE, "..", "benchmarks", "run_model.py")


def _driver(tag):
    sys.path.insert(0, os.path.join(HERE, ".."))
    spec = importlib.util.spec_from_file_location(f"rm_stab_{tag}", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_reports_median_and_range():
    """Median, not mean: one scheduling hiccup should not drag the headline number."""
    mod = _driver("s1")
    med, lo, hi = mod.summarize([10.0, 11.0, 30.0])
    assert med == 11.0 and lo == 10.0 and hi == 30.0


def test_summarize_single_sample_has_zero_width():
    mod = _driver("s2")
    assert mod.summarize([7.5]) == (7.5, 7.5, 7.5)


def test_repeats_actually_repeat_the_timed_pair():
    """Each repeat is a 1-token and an N-token timing; 3 repeats means 6 calls, plus one
    warmup. A `--repeats` that silently timed once would report a spread of zero and look
    more trustworthy than it is."""
    mod = _driver("s3")
    calls = []

    def gen(batch, n):
        calls.append(n)
        return (0.1 * n, batch * n)          # (seconds, tokens)

    mod._two_point(gen, 1, decode=8, repeats=3, warmup=16)
    assert calls[0] == 16, "warmup should run first, at the warmup length"
    assert calls.count(1) == 3 and calls.count(8) == 3


def test_warmup_defaults_longer_than_the_old_four():
    """4 tokens was the old value and is far too short to fault in a 78 GiB model."""
    mod = _driver("s4")
    import inspect
    sig = inspect.signature(mod._two_point)
    assert sig.parameters["warmup"].default >= 16


def test_two_point_still_returns_a_usable_rate():
    mod = _driver("s5")

    def gen(batch, n):
        return (0.5 + 0.25 * n, batch * n)   # 0.5s prefill + 0.25s/token

    tps, ttft_ms, ntok, degraded, spread = mod._two_point(gen, 1, decode=9, repeats=2)
    assert not degraded
    assert abs(tps - 4.0) < 1e-6, tps       # (9-1) tokens / (8 * 0.25 s)
    assert spread == (4.0, 4.0)


def test_every_caller_unpacks_the_full_return():
    """`_two_point` has two call sites in two runners, and adding the spread to its return
    broke the vLLM one while the HF one was being tested -- a ValueError that only fires
    with a GPU and a served model, i.e. never in CI. Check the arity statically instead.
    """
    import ast
    src = open(DRIVER).read()
    tree = ast.parse(src)
    returns = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_two_point"]
    assert len(returns) == 1
    width = max(len(n.value.elts) for n in ast.walk(returns[0])
                if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple))

    sites = [n for n in ast.walk(tree)
             if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call)
             and getattr(n.value.func, "id", None) == "_two_point"]
    assert len(sites) == 2, f"expected a call in each runner, found {len(sites)}"
    for s in sites:
        target = s.targets[0]
        assert isinstance(target, ast.Tuple) and len(target.elts) == width, (
            f"line {s.lineno} unpacks {len(getattr(target, 'elts', []))} of {width}")
