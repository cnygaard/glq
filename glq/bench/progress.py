"""Readable, timestamped progress for long benchmark runs (nohup/tee friendly).

Two pieces:
  - ``log_ts(msg)`` — a flushed, timestamped, greppable line (``[glq-bench HH:MM:SS]``).
  - ``pbar_factory(desc)`` — a ``use_tqdm=`` callable for vLLM's ``generate``/``chat``.
    vLLM only calls ``pbar.update()`` when a request *finishes*, so a single long
    generation (the classic "stuck at 28/30" AIME case) emits nothing. We instead
    suppress the carriage-return bar and run a background **heartbeat** thread that
    logs ``n/total | elapsed | last-completion ago`` every ``GLQ_BENCH_HEARTBEAT_SEC``
    (default 30) — a flat ``n`` with growing elapsed means a long generation in
    flight; if the heartbeat itself stops, the process is truly wedged.

All stdlib; imported lazily by the adapters so the module stays CPU-import-safe.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_STREAM = sys.stderr


def _fmt(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def log_ts(msg: str) -> None:
    """Emit a flushed, timestamped, greppable progress line."""
    print(f"[glq-bench {time.strftime('%H:%M:%S')}] {msg}", file=_STREAM, flush=True)


def heartbeat_interval() -> float:
    try:
        return max(1.0, float(os.environ.get("GLQ_BENCH_HEARTBEAT_SEC", "30")))
    except ValueError:
        return 30.0


def _make_heartbeat_tqdm():
    """Build the HeartbeatTqdm class lazily (tqdm is a vLLM dep, not a CPU dep)."""
    from tqdm import tqdm

    class HeartbeatTqdm(tqdm):
        """A tqdm whose CR bar is silenced; progress is logged as timestamped lines
        on a timer (so long single generations stay visible) and on each completion."""

        def __init__(self, *args, glq_desc: str = "", hb_interval: float = 30.0, **kw):
            self._glq_desc = glq_desc
            self._hb_interval = hb_interval
            self._hb_last_complete = time.time()
            self._hb_stop = threading.Event()
            kw["file"] = open(os.devnull, "w")  # silence the CR bar; keep n/format_dict
            kw.setdefault("disable", False)
            super().__init__(*args, **kw)
            self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._hb_thread.start()

        def _line(self) -> str:
            el = self.format_dict.get("elapsed", 0.0)
            tot = self.total if self.total is not None else "?"
            phase = f" [{self.desc.strip(': ')}]" if self.desc else ""
            msg = f"{self._glq_desc}{phase}: {self.n}/{tot} done | elapsed {_fmt(el)}"
            if self.n and self.total and self.n < self.total:
                msg += f" | last completion {_fmt(time.time() - self._hb_last_complete)} ago"
            return msg

        def _heartbeat(self) -> None:
            while not self._hb_stop.wait(self._hb_interval):
                if self.n != self.total:
                    log_ts(self._line())

        def update(self, n=1):
            r = super().update(n)
            self._hb_last_complete = time.time()
            log_ts(self._line())
            return r

        def close(self):
            if not self._hb_stop.is_set():
                self._hb_stop.set()
                log_ts(self._line())
                try:
                    self.fp.close()
                except Exception:  # noqa: BLE001
                    pass
            super().close()

    return HeartbeatTqdm


def pbar_factory(desc: str, interval: float | None = None):
    """Return a ``use_tqdm=`` callable for vLLM that logs timestamped heartbeats."""
    interval = heartbeat_interval() if interval is None else interval
    cls = _make_heartbeat_tqdm()

    def factory(*args, **kw):
        return cls(*args, glq_desc=desc, hb_interval=interval, **kw)

    return factory
