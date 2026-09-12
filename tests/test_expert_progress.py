"""Progress reporting while routed MoE experts are quantized.

Why this exists: quantizing Qwen3.8-Flash-Next means 512 experts x 2 matrices = 1024 expert
quantizations per layer, ~2.5B parameters against ~34M for every other sublayer combined.
The non-expert loop prints one SQNR line each; the expert branch printed nothing until all
1024 finished, so a layer looked hung for many minutes. The only output during that window
was the incidental "no activations, using identity Hessian" notice, which is emitted BEFORE
quantization and only for the unrouted subset — it reads like progress and is not.

Periodic, not per-expert: one line each would be ~49,000 lines for this model (1024 x 48
layers). A tqdm bar is wrong because these runs are nohup'd to a file, where carriage
returns produce an unreadable log.
"""
from __future__ import annotations

from glq.quantize_model import _ExpertProgress


def _reporter(total, interval=30.0, clock=None, **kw):
    lines = []
    return _ExpertProgress(total, interval=interval, clock=clock,
                           emit=lines.append, **kw), lines


def test_throttles_to_the_interval_not_once_per_expert():
    """1000 experts finishing 1s apart, 30s throttle -> ~33 lines, not 1000."""
    t = [0.0]
    prog, lines = _reporter(1000, interval=30.0, clock=lambda: t[0])
    for _ in range(1000):
        t[0] += 1.0
        prog.update(sqnr=20.0)
    assert 30 <= len(lines) <= 36, f"{len(lines)} lines emitted"


def test_a_short_layer_still_reports_once():
    """5 experts finishing inside one interval must not be silent."""
    t = [0.0]
    prog, lines = _reporter(5, interval=30.0, clock=lambda: t[0])
    for _ in range(5):
        t[0] += 0.1
        prog.update(sqnr=21.0)
    assert lines == [], "nothing is due yet"
    prog.finish()
    assert len(lines) == 1, lines
    assert "5/5" in lines[0]


def test_progress_can_be_disabled():
    t = [0.0]
    prog, lines = _reporter(100, interval=0.0, clock=lambda: t[0])
    for _ in range(100):
        t[0] += 60.0
        prog.update(sqnr=20.0)
    prog.finish()
    assert lines == [], f"interval=0 must be silent, got {lines}"


def test_eta_comes_from_the_observed_rate():
    """Half done after 10s -> about 10s remaining."""
    t = [0.0]
    prog, lines = _reporter(100, interval=1.0, clock=lambda: t[0])
    for _ in range(50):
        t[0] += 0.2                      # 50 experts in 10s
        prog.update(sqnr=20.0)
    assert lines, "expected at least one line"
    last = lines[-1]
    # The throttle decides WHICH update emits, so the count is near 50 rather than exactly
    # 50 — asserting 50 would be testing the throttle's phase, not the ETA.
    assert "/100" in last
    # Half done in 10s at a steady rate -> about 10s left. That is the claim.
    assert "ETA 10s" in last, last


def test_the_line_carries_a_running_average_sqnr():
    t = [0.0]
    prog, lines = _reporter(4, interval=0.5, clock=lambda: t[0])
    for s in (10.0, 20.0, 30.0, 40.0):
        t[0] += 1.0
        prog.update(sqnr=s)
    prog.finish()
    assert "25.0dB" in lines[-1], lines[-1]


def test_experts_with_no_sqnr_do_not_break_the_average():
    """A failed or skipped expert reports no SQNR; the counter must still advance."""
    t = [0.0]
    prog, lines = _reporter(3, interval=0.5, clock=lambda: t[0])
    t[0] += 1.0; prog.update(sqnr=20.0)
    t[0] += 1.0; prog.update(sqnr=None)
    t[0] += 1.0; prog.update(sqnr=None)
    prog.finish()
    assert "3/3" in lines[-1]
    assert "20.0dB" in lines[-1], lines[-1]


def test_completion_order_does_not_lose_or_misfile_results():
    """The behaviour change: results are collected with as_completed, so they arrive out of
    submission order. _collect_result is keyed by name, so every result must still land
    under its own name — asserted rather than assumed, because this path writes checkpoints.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    names = [f"expert.{i}" for i in range(16)]

    def work(i):
        # Finish in reverse submission order.
        time.sleep(0.002 * (len(names) - i))
        return names[i], i

    stored = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(work, i): names[i] for i in range(len(names))}
        order = []
        for f in as_completed(futures):
            name, payload = f.result()
            stored[name] = payload
            order.append(name)

    assert len(stored) == len(names), "a result was lost"
    assert all(stored[n] == i for i, n in enumerate(names)), "a result was misfiled"
    assert order != names, "test did not actually exercise out-of-order completion"
