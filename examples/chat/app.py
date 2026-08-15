"""Run the GLQ Gradio chat UI from a repo checkout.

The implementation lives in `glq/chat.py` so that it ships in the wheel — `examples/` is
not packaged, and the installer needs a command that exists on a pip-only machine. It is
installed as `glq-chat`; this shim keeps `python examples/chat/app.py` working too.
"""
from glq.chat import main

if __name__ == "__main__":
    raise SystemExit(main())
