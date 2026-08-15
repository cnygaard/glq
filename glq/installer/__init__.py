"""First-run setup for GLQ: discover checkpoints, size them against the GPU, wire up serving.

`install.sh` is deliberately a thin bootstrap — it creates a venv, installs glq, and hands
over to this package. Everything with a decision in it lives here instead of in bash, so it
can be unit-tested (see `tests/test_installer_*.py`) rather than only exercised by running
the installer on a real box.

Stdlib only. This runs immediately after `pip install glq`, before any extra has been
chosen, so it cannot import requests, transformers, torch or gradio.
"""
from __future__ import annotations

__all__ = ["discovery", "hardware", "recommend"]
