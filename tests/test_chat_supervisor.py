"""`glq-chat` owns the vLLM server's lifetime, so the GPU is free when you stop chatting.

Before this, a new user ran three commands across two terminals: `vllm serve …` in one,
`glq-chat` in another, then typed a URL into a browser. Running them out of order gave a
chat window with an empty model dropdown, because the UI was built whether or not a server
existed. And the documented serve command passed no `--gpu-memory-utilization`, so vLLM took
its default 0.9 — on a 24 GB card serving a 1.8 GB model that reserves ~21.6 GB of KV pool
and holds it until the process dies. For someone who also wants to play a game on that card,
that is not a background service, it is the whole GPU.

vLLM has no lazy load and no idle unload (Ollama has both), so "leave it running" is not
available to us. One process therefore owns the whole lifetime: start it, use it, and on
exit — including the crash path — take it down.

The supervisor lives apart from `glq/chat.py` so these tests need neither gradio nor a GPU:
every process and probe is injected.
"""
from __future__ import annotations

import inspect
import io
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

import glq.supervisor as sup_mod
from glq.supervisor import VllmSupervisor


class FakeProc:
    """Just enough of Popen: it stays alive until terminated, and records how it died."""

    def __init__(self, argv, alive_for=0, output=(), env=None):
        self.argv = argv
        self.env = env
        self.pid = 4242                      # a pgid-leader stand-in for group-kill tests
        self.spawn_kwargs = {}
        self.terminated = self.killed = False
        self.waited = False
        self._alive_for = alive_for          # polls before it "exits" on its own; 0 = forever
        self._polls = 0
        self.stdout = iter(output)

    def poll(self):
        self._polls += 1
        if self._alive_for and self._polls > self._alive_for:
            return 1                          # exited, non-zero
        return None                           # still running

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        if self.ignores_terminate and not self.killed:
            raise TimeoutError("did not die")
        return 0

    ignores_terminate = False


def _sup(*, healthy_after=1, alive_for=0, output=(), **kw):
    """A supervisor whose clock, sleep, probe and process factory are all fakes."""
    spawned = []
    calls = {"probe": 0}

    def spawn(argv, **kw):
        # A real child writes to the handle it is handed. Modelling that is what lets these
        # tests cover the progress reporting and the failure quoting, both of which now read
        # the child's log rather than a pipe.
        fh = kw.get("stdout")
        if fh is not None and hasattr(fh, "write"):
            for line in output:
                fh.write(line)
            fh.flush()
        p = FakeProc(argv, alive_for=alive_for, output=output, env=kw.get("env"))
        p.spawn_kwargs = kw
        spawned.append(p)
        return p

    def probe(_base_url):
        calls["probe"] += 1
        return calls["probe"] >= healthy_after

    clock = {"t": 0.0}
    # Never the real ~/.glq/vllm.log: these tests must not write to the user's home.
    kw.setdefault("log_path", Path(tempfile.mkdtemp()) / "vllm.log")
    sup = VllmSupervisor(
        model="xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
        port=8000,
        spawn=spawn, probe=probe,
        sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        monotonic=lambda: clock["t"],
        **kw)
    return sup, spawned


# ------------------------------------------------------------------ attach vs start

def test_an_already_serving_endpoint_is_reused_not_duplicated():
    """Repeat runs must be instant, and a second vLLM on the same port would fail anyway."""
    sup, spawned = _sup(healthy_after=1)          # healthy on the very first probe

    started = sup.start()

    assert started is False, "reported starting a server when one was already up"
    assert spawned == [], "spawned a second vLLM against a live endpoint"


def test_it_starts_vllm_when_nothing_is_serving():
    sup, spawned = _sup(healthy_after=3)           # unhealthy twice, then up

    started = sup.start()

    assert started is True
    assert len(spawned) == 1
    argv = spawned[0].argv
    assert "serve" in argv and sup.model in argv
    assert "--quantization" in argv and "glq" in argv
    assert "8000" in argv


def test_it_never_spawns_when_told_to_attach_only():
    """`--no-serve` keeps the old behaviour for anyone running their own server."""
    sup, spawned = _sup(healthy_after=99, serve=False)

    with pytest.raises(RuntimeError, match="no server"):
        sup.start()
    assert spawned == []


# ------------------------------------------------------- the GPU has another job to do

def test_it_caps_the_kv_pool_so_the_card_is_not_seized():
    """vLLM defaults to 0.9 of total VRAM for the KV pool and never gives it back. On a
    machine that also runs games, the default is the bug."""
    sup, spawned = _sup(healthy_after=2, gpu_memory_utilization=0.35)

    sup.start()

    argv = spawned[0].argv
    assert "--gpu-memory-utilization" in argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.35"


def test_a_default_utilisation_is_supplied_rather_than_left_to_vllm():
    sup, spawned = _sup(healthy_after=2)
    sup.start()
    assert "--gpu-memory-utilization" in spawned[0].argv


# --------------------------------------------------------------------- teardown

def test_the_child_is_stopped_when_the_caller_finishes():
    sup, spawned = _sup(healthy_after=2)
    sup.start()

    sup.stop()

    assert spawned[0].terminated, "vLLM was left running; the GPU stays reserved"


def test_the_child_is_stopped_even_if_the_caller_raises():
    """The VRAM-release guarantee has to hold on the crash path, which is where it matters."""
    sup, spawned = _sup(healthy_after=2)

    with pytest.raises(ZeroDivisionError):
        with sup:
            raise ZeroDivisionError("the UI fell over")

    assert spawned[0].terminated


def test_a_child_that_ignores_terminate_is_killed():
    sup, spawned = _sup(healthy_after=2)
    sup.start()
    spawned[0].ignores_terminate = True

    sup.stop()

    assert spawned[0].killed, "a wedged vLLM would hold the GPU forever"


def test_attaching_to_someone_elses_server_does_not_stop_it():
    """We did not start it, so it is not ours to kill."""
    sup, spawned = _sup(healthy_after=1)
    sup.start()
    sup.stop()
    assert spawned == []


# ------------------------------------------------------------ failure is legible

def test_a_child_that_never_becomes_healthy_reports_its_own_output():
    """'failed to start' is useless. vLLM said why — quote it."""
    sup, _ = _sup(healthy_after=10**6, timeout=5.0,
                  output=["INFO loading weights\n", "ERROR: port 8000 already in use\n"])

    with pytest.raises(RuntimeError) as e:
        sup.start()

    assert "port 8000 already in use" in str(e.value)


def test_a_child_that_exits_early_is_not_waited_on():
    """If vLLM dies in 2 seconds, do not sit through the whole health timeout."""
    sup, _ = _sup(healthy_after=10**6, alive_for=2, timeout=10_000.0,
                  output=["ERROR: CUDA out of memory\n"])

    with pytest.raises(RuntimeError) as e:
        sup.start()

    assert "exited" in str(e.value).lower()
    assert "out of memory" in str(e.value)


# ==================================================================== the command

# The supervisor above is only useful if `glq-chat` actually uses it. These tests drive
# `main()` with the process, the probe and the UI all replaced, so they need neither gradio
# nor a GPU — which also means `glq.chat` must stay importable without the `chat` extra.



class _FakeSup:
    """Records lifecycle order, because "UI built before the server answered" is the exact
    bug this replaces: it produced a chat window with an empty model dropdown."""

    def __init__(self, events, **kw):
        self.events, self.kw = events, kw
        # the real supervisor resolves None to a planned window; 8192 stands in for it
        self.max_model_len = kw.get("max_model_len") or 8192

    def __enter__(self):
        self.events.append("start")
        return self

    def __exit__(self, *_exc):
        self.events.append("stop")
        return False


def _run_chat(monkeypatch, argv, *, display=None, cfg=None, launch=None, models=("m",)):
    import glq.chat as chat

    events, made, launched = [], [], []

    monkeypatch.setattr(chat, "_installed_config", lambda: dict(cfg or {}))
    monkeypatch.setattr(chat, "_served_models", lambda _c: list(models))
    # These tests are about the supervisor's process lifetime, not about HTTP. Both of the
    # `chat` extra's packages are stubbed out so the suite runs on an install that has
    # neither — which is exactly what CI has (`pip install .[hub]`), and what made twenty of
    # these tests fail there while passing on a development venv that happened to have them.
    monkeypatch.setattr(chat, "missing_chat_deps", lambda: [])
    monkeypatch.setattr(chat, "_openai_client", lambda *_a, **_kw: object())

    def sup(**kw):
        made.append(kw)
        return _FakeSup(events, **kw)

    monkeypatch.setattr(chat, "VllmSupervisor", sup)

    def build(base_url, model_list, **_kw):
        events.append("build_ui")
        demo = types.SimpleNamespace()
        demo.launch = lambda **kw: (launched.append(kw), events.append("launch"),
                                    launch and launch())[0]
        return demo

    monkeypatch.setattr(chat, "build_ui", build)

    for var in ("DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    if display:
        monkeypatch.setenv(display, ":0")

    return chat, events, made, launched


def test_the_server_is_started_before_the_ui_and_stopped_after_it(monkeypatch):
    chat, events, made, _ = _run_chat(monkeypatch, ["--model", "org/ckpt"])

    rc = chat.main(["--model", "org/ckpt"])

    assert rc == 0
    assert events == ["start", "build_ui", "launch", "stop"]
    assert made[0]["model"] == "org/ckpt"


def test_the_server_is_stopped_even_if_the_ui_falls_over(monkeypatch):
    """Otherwise a crashed UI leaves vLLM holding the card with nothing driving it."""
    def boom():
        raise RuntimeError("gradio could not bind port 7860")

    chat, events, _, _ = _run_chat(monkeypatch, [], launch=boom)

    with pytest.raises(RuntimeError):
        chat.main(["--model", "org/ckpt"])

    assert events[-1] == "stop"


def test_no_serve_keeps_the_old_attach_only_behaviour(monkeypatch):
    chat, _, made, _ = _run_chat(monkeypatch, [])
    chat.main(["--no-serve"])
    assert made[0]["serve"] is False


def test_the_model_comes_from_the_installer_config_when_not_given(monkeypatch):
    """`glq-chat` with no arguments is the whole point — the installer already chose."""
    chat, _, made, _ = _run_chat(monkeypatch, [], cfg={"model": "org/from-config"})
    chat.main([])
    assert made[0]["model"] == "org/from-config"


def test_it_says_which_model_to_pick_rather_than_starting_a_nameless_server(monkeypatch):
    chat, events, _, _ = _run_chat(monkeypatch, [], cfg={})
    rc = chat.main([])
    assert rc != 0
    assert events == [], "started a server without knowing what to serve"


def test_the_vllm_port_is_taken_from_the_base_url(monkeypatch):
    """One source of truth: the config records base_url, and the supervisor needs the port.
    Deriving it means they cannot disagree."""
    chat, _, made, _ = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt", "--base-url", "http://127.0.0.1:8123/v1"])
    assert made[0]["port"] == 8123


def test_the_kv_pool_size_is_passed_through(monkeypatch):
    chat, _, made, _ = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt", "--gpu-memory-utilization", "0.3"])
    assert made[0]["gpu_memory_utilization"] == 0.3


def test_the_browser_opens_when_there_is_a_display(monkeypatch):
    chat, _, _, launched = _run_chat(monkeypatch, [], display="DISPLAY")
    chat.main(["--model", "org/ckpt"])
    assert launched[0]["inbrowser"] is True


def test_the_browser_is_left_alone_on_a_headless_box(monkeypatch):
    """Over SSH there is no browser to open; printing the URL is the whole interface."""
    chat, _, _, launched = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt"])
    assert launched[0]["inbrowser"] is False


def test_no_browser_wins_over_a_display(monkeypatch):
    chat, _, _, launched = _run_chat(monkeypatch, [], display="WAYLAND_DISPLAY")
    chat.main(["--model", "org/ckpt", "--no-browser"])
    assert launched[0]["inbrowser"] is False


def test_a_terminated_chat_still_takes_the_server_down_with_it(monkeypatch):
    """Ctrl-C unwinds through the context manager, but `kill <pid>` and a closed terminal
    do not: the default SIGTERM/SIGHUP disposition ends this process without running any
    cleanup, leaving vLLM holding the card with nothing driving it."""
    chat, _, _, _ = _run_chat(monkeypatch, [])

    installed = {}
    monkeypatch.setattr(chat.signal, "signal",
                        lambda sig, handler: installed.__setitem__(sig, handler))

    chat.main(["--model", "org/ckpt"])

    for sig in (chat.signal.SIGTERM, chat.signal.SIGHUP):
        handler = installed.get(sig)
        assert handler is not None, f"{sig.name} would orphan vLLM"
        with pytest.raises(SystemExit):
            handler(sig, None)


# ==================================================================== saying something

# Measured: `glq-chat` prints two lines and then goes silent for minutes while vLLM
# downloads GiB of weights, builds kernels, loads the model and captures CUDA graphs. The
# user sees a dead terminal and no way to tell "working" from "hung" — the single worst
# thing a first-run experience can do, and the reason Ollama shows a progress bar.
#
# The child had plenty to say the whole time. We were throwing it away: its output went to
# a pipe that nothing read until the process failed. That is also a latent hang — a pipe
# holds ~64 KiB, and a child that fills it blocks forever on write.


def _sup_log(**kw):
    """`_sup`, plus the path its child logs to."""
    sup, spawned = _sup(**kw)
    return sup, spawned, sup.log_path


def test_the_child_writes_to_a_file_not_an_unread_pipe():
    """A pipe nobody reads is a hang: vLLM blocks on write once ~64 KiB has accumulated,
    which is well within what it logs while loading a model."""
    sup, spawned, log = _sup_log(healthy_after=2)
    kwargs = {}

    def spawn(argv, **kw):
        kwargs.update(kw)
        return FakeProc(argv)

    sup._spawn = spawn
    sup._probe = lambda _u: True                      # skip the wait; we want the kwargs
    sup._probe = lambda _u, c=[0]: (c.append(1), len(c) > 2)[1]
    sup.start()

    import subprocess as sp
    assert kwargs.get("stdout") is not sp.PIPE, "child output goes to an unread pipe"
    assert hasattr(kwargs.get("stdout"), "write"), "child output is not going to a file"


def test_it_says_where_the_log_is():
    """When the wait is long, "what is it doing?" needs an answer the user can go and read."""
    out = io.StringIO()
    sup, _, log = _sup_log(healthy_after=2, out=out)
    sup.start()
    assert str(log) in out.getvalue()


def test_it_warns_that_the_first_run_is_slow_before_going_quiet():
    """Expectation-setting is most of the fix: minutes of silence you were warned about is
    patience, the same silence unannounced is a hang."""
    out = io.StringIO()
    sup, _, _ = _sup_log(healthy_after=2, out=out)
    sup.start()
    text = out.getvalue().lower()
    assert "minute" in text, f"nothing prepares the user for a multi-minute wait:\n{text}"


def test_it_reports_progress_while_it_waits():
    """The heart of it: something must appear on screen during the wait."""
    out = io.StringIO()
    sup, _, log = _sup_log(healthy_after=12, out=out, timeout=600.0)

    # vLLM says what it is doing; simulate it writing as we poll.
    lines = iter(["INFO Starting vLLM engine\n",
                  "INFO Loading safetensors checkpoint shards: 40%\n",
                  "INFO Capturing CUDA graphs: 90%\n"])
    real_probe = sup._probe

    def probe(url):
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write(next(lines))
        except StopIteration:
            pass
        return real_probe(url)

    sup._probe = probe
    sup.start()

    text = out.getvalue()
    progress = [ln for ln in text.splitlines() if "s]" in ln]
    assert len(progress) >= 2, f"no progress while waiting:\n{text}"


def test_the_progress_quotes_what_vllm_is_actually_doing():
    """"Still working" is a guess. vLLM knows, and its own words are better than ours."""
    out = io.StringIO()
    sup, _, log = _sup_log(healthy_after=8, out=out, timeout=600.0)
    real_probe = sup._probe

    def probe(url):
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write("INFO Loading safetensors checkpoint shards: 40%\n")
        return real_probe(url)

    sup._probe = probe
    sup.start()
    assert "Loading safetensors" in out.getvalue()


def test_the_ready_line_says_how_long_it_took():
    out = io.StringIO()
    sup, _, _ = _sup_log(healthy_after=6, out=out, timeout=600.0)
    sup.start()
    ready = [ln for ln in out.getvalue().splitlines() if "ready" in ln]
    assert ready and any(ch.isdigit() for ch in ready[0]), (
        f"the ready line does not report elapsed time: {ready}")


def test_a_failure_points_at_the_log_file():
    """The tail we print is 25 lines; the reason is often further up."""
    sup, _, log = _sup_log(healthy_after=10**6, timeout=5.0,
                           output=["ERROR: port 8000 already in use\n"])

    with pytest.raises(RuntimeError) as e:
        sup.start()

    assert str(log) in str(e.value)
    assert "port 8000 already in use" in str(e.value)


def test_we_do_not_reprint_the_url_gradio_already_prints():
    """gradio announces its own local and public URLs. Printing our own copy of the local one
    just puts the same address on screen twice — the fix for it going missing was to stop
    block-buffering gradio's stdout, not to duplicate its output."""
    import glq.chat as chat
    src = inspect.getsource(chat.main)
    assert "opening the chat on" not in src, "still printing our own copy of the chat URL"


def test_verbose_streams_the_server_log(monkeypatch):
    """When the one-line summary is not enough, --verbose puts vLLM's own output on screen
    instead of making the user find the log."""
    chat, _, made, _ = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt", "--verbose"])
    assert made[0]["verbose"] is True


# ============================================================ the shareable public link

# `share=True` asks gradio for an HTTPS tunnel (`*.gradio.live`) so the chat can be opened
# from a phone, another machine, or sent to someone else — no port forwarding, no firewall
# rules, which is the whole reason it is on by default.
#
# It is also a public, unauthenticated URL to a GPU: anyone holding it can send prompts.
# `--no-share` keeps everything on localhost.

def test_the_public_link_is_on_by_default(monkeypatch):
    chat, _, _, launched = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt"])
    assert launched[0]["share"] is True


def test_no_share_keeps_it_on_this_machine(monkeypatch):
    chat, _, _, launched = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt", "--no-share"])
    assert launched[0]["share"] is False


def test_the_user_is_told_the_link_is_public(monkeypatch, capsys):
    """Sharing a GPU with the internet should never be a surprise, even when it is the
    default and even when it is what the user wanted."""
    chat, _, _, _ = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt"])
    captured = capsys.readouterr()          # one call: it drains the buffer
    text = (captured.out + captured.err).lower()
    assert "public" in text or "anyone" in text


def test_stdout_is_unbuffered_before_gradio_prints_the_link(monkeypatch):
    """gradio prints "Running on public URL: https://….gradio.live" to stdout itself, and we
    cannot know that URL in advance — the tunnel assigns it. Measured on a box: with stdout
    redirected, gradio's URL lines block-buffer and do not appear until the process exits,
    so the one thing the user is waiting for is invisible for the whole session."""
    chat, _, _, _ = _run_chat(monkeypatch, [])

    calls = {}

    class _Stdout:
        def reconfigure(self, **kw):
            calls.update(kw)

        def write(self, s):
            return len(s)

        def flush(self):
            pass

    monkeypatch.setattr(chat.sys, "stdout", _Stdout())
    chat.main(["--model", "org/ckpt"])

    assert calls.get("line_buffering") is True, (
        "gradio's own output stays block-buffered, so the share URL never appears")


def test_progress_skips_warning_noise_for_the_line_that_says_something():
    """Measured on a box: the last line in the log was a wrapped `warnings.warn(...)`
    fragment, so the progress line repeated it verbatim ten times while vLLM was quietly
    loading. A stale fragment repeated on a timer reads as stuck — which is the failure this
    whole feature exists to prevent."""
    out = io.StringIO()
    sup, _, log = _sup_log(
        healthy_after=8, out=out, timeout=600.0,
        output=["INFO 08-16 [gpu_model_runner.py] Starting to load model org/ckpt\n",
                "  warnings.warn('resource_tracker: There appear to be %d '\n"])

    sup.start()

    progress = [ln for ln in out.getvalue().splitlines() if "s]" in ln]
    assert progress, "no progress lines at all"
    assert not any("warnings.warn" in ln for ln in progress), (
        f"reported a warning fragment instead of what vLLM was doing:\n{progress}")
    assert any("Starting to load model" in ln for ln in progress), (
        f"the informative line never reached the user:\n{progress}")


def test_progress_says_when_the_child_has_gone_quiet():
    """Repeating the same line every few seconds hides the difference between "still working"
    and "stopped saying anything ten minutes ago"."""
    out = io.StringIO()
    sup, _, _ = _sup_log(healthy_after=40, out=out, timeout=600.0,
                         output=["INFO Starting to load model\n"])

    sup.start()

    text = out.getvalue()
    assert "no new output" in text, f"a silent child looks identical to a busy one:\n{text}"


# ==================================================== sizing the KV pool for the model

# A fixed fraction cannot serve both a 1.8 GB SmolLM3 and a 15 GB MoE. Measured on a 23 GB
# L4: `gpu_memory_utilization=0.45` gives a 10.4 GB budget, the 26B GLQ checkpoint's weights
# are ~15 GB, so vLLM had nothing left for KV blocks and died with
#
#     ValueError: No available memory for the cache blocks.
#
# The installer ranks checkpoints against *total* VRAM, so it will happily recommend a model
# that a fixed 45% then starves. The pool has to be sized from the checkpoint.

GIB = 1024 ** 3


def test_a_small_model_still_leaves_most_of_the_card_free():
    """The original point of not using vLLM's 0.9: a 1.8 GB model on a 24 GB card should not
    reserve 21 GB of it."""
    util = sup_mod.plan_gpu_memory_utilization(weights_bytes=int(1.8 * GIB),
                                               vram_bytes=int(23 * GIB))
    assert util <= 0.45, f"a tiny model still seizes the card: {util}"


def test_a_big_model_gets_room_for_its_weights_and_a_cache():
    """The 26B-on-an-L4 case that failed: the budget must cover the weights with enough left
    over for KV blocks, or vLLM refuses to start."""
    weights = int(15 * GIB)
    vram = int(23 * GIB)
    util = sup_mod.plan_gpu_memory_utilization(weights_bytes=weights, vram_bytes=vram)

    assert util * vram > weights, "the budget does not even cover the weights"
    assert util * vram - weights >= 2 * GIB, "no usable KV cache left"


def test_it_never_asks_for_more_of_the_card_than_exists():
    util = sup_mod.plan_gpu_memory_utilization(weights_bytes=int(40 * GIB),
                                               vram_bytes=int(23 * GIB))
    assert 0 < util <= 0.95, f"asked for {util} of the GPU"


def test_an_unknown_size_falls_back_to_the_documented_default():
    """`--model` can point anywhere, and the size lookup is a network call that may fail.
    Guessing large would seize the card; the old constant is the safe answer."""
    assert sup_mod.plan_gpu_memory_utilization(
        weights_bytes=None, vram_bytes=int(23 * GIB)) == sup_mod.DEFAULT_GPU_MEMORY_UTILIZATION
    assert sup_mod.plan_gpu_memory_utilization(
        weights_bytes=int(2 * GIB), vram_bytes=None) == sup_mod.DEFAULT_GPU_MEMORY_UTILIZATION


def test_an_explicit_flag_always_wins(monkeypatch):
    """--gpu-memory-utilization is the escape hatch; sizing must not override it."""
    sup, spawned = _sup(healthy_after=2, gpu_memory_utilization=0.31)
    sup.start()
    argv = spawned[0].argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.31"


def test_the_cache_block_failure_says_which_knob_to_turn():
    """vLLM's own message names the flag but not a value, and the user cannot see the sizing
    we computed. Say both."""
    sup, _, log = _sup_log(
        healthy_after=10**6, timeout=5.0,
        output=["ValueError: No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization`\n"])

    with pytest.raises(RuntimeError) as e:
        sup.start()

    assert "--gpu-memory-utilization" in str(e.value)


def test_the_chat_tells_the_supervisor_how_big_the_checkpoint_is(monkeypatch):
    """The sizing is useless if the numbers never reach the supervisor. This is the wiring."""
    chat, _, made, _ = _run_chat(monkeypatch, [])
    monkeypatch.setattr(chat, "_checkpoint_bytes", lambda repo: 15 * GIB)
    monkeypatch.setattr(chat, "_vram_bytes", lambda: 23 * GIB)

    chat.main(["--model", "org/big-moe"])

    assert made[0]["weights_bytes"] == 15 * GIB
    assert made[0]["vram_bytes"] == 23 * GIB


def test_naming_a_fraction_skips_the_size_lookup(monkeypatch):
    """`--gpu-memory-utilization` is the offline/oddball escape hatch; it should not then
    make an HTTP call to size something it was just told about."""
    chat, _, made, _ = _run_chat(monkeypatch, [])

    def boom(_repo):
        raise AssertionError("looked up the checkpoint size despite an explicit fraction")

    monkeypatch.setattr(chat, "_checkpoint_bytes", boom)
    chat.main(["--model", "org/big-moe", "--gpu-memory-utilization", "0.8"])
    assert made[0]["weights_bytes"] is None


def test_the_context_is_capped_so_a_chat_does_not_demand_a_256k_cache():
    """Measured on an L4: vLLM defaults max_model_len to the model's own maximum, and
    gemma-4's is 262144 — 6.15 GiB of KV for a *single* request. Even with the weights
    comfortably resident it refused to start:

        6.15 GiB KV cache is needed, which is larger than the available KV cache memory
        (0.42 GiB). Based on the available memory, the estimated maximum model length is 1984

    A chat UI does not need a quarter-million-token window, and asking for one costs the
    whole card."""
    sup, spawned = _sup(healthy_after=2)
    sup.start()
    argv = spawned[0].argv
    assert "--max-model-len" in argv, f"no context cap; vLLM will take the model's max\n{argv}"
    assert int(argv[argv.index("--max-model-len") + 1]) <= 32768


def test_the_context_cap_can_be_raised():
    sup, spawned = _sup(healthy_after=2, max_model_len=65536)
    sup.start()
    argv = spawned[0].argv
    assert argv[argv.index("--max-model-len") + 1] == "65536"


def test_the_overhead_allowance_matches_what_vllm_actually_took():
    """The first sizing attempt left 0.42 GiB of KV where it had budgeted 4.1 GiB — vLLM's
    non-weight footprint on this model was ~3.7 GiB, not the 2 GiB assumed. Budget for what
    was measured, or the pool is nominally fine and empty in practice."""
    weights = int(13.9 * GIB)
    vram = int(22.5 * GIB)
    util = sup_mod.plan_gpu_memory_utilization(weights_bytes=weights, vram_bytes=vram)
    spare = util * vram - weights
    assert spare >= 5.5 * GIB, (
        f"only {spare / GIB:.1f} GiB for overhead + KV; measured overhead alone is ~3.7 GiB")


# ============================================================ KV-cache compression

# GLQ offers *vLLM's own* fp8 KV cache — two serve flags, maintained upstream — rather than
# GLQ's E8 lattice cache, which announces all six of its stages on vLLM 0.27.1 and then dies
# in EngineCore on `kv_cache_stride_order`. An option that cannot start is worse than no
# option, and one built on upstream flags does not rot when vLLM moves.

def test_the_fp8_kv_cache_is_off_unless_asked_for():
    """It trades attention precision. Nobody should pay that without choosing to."""
    sup, spawned = _sup(healthy_after=2)
    sup.start()
    assert "--kv-cache-dtype" not in spawned[0].argv


def test_asking_for_it_passes_vllm_s_own_flags():
    """Verified against vLLM 0.27.1: `cache_dtype` accepts 'fp8', and
    `kv_cache_dtype_skip_layers` is a list[str] — so these are real flags, not ours."""
    sup, spawned = _sup(healthy_after=2, fp8_kv=True)
    sup.start()
    argv = spawned[0].argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--kv-cache-dtype-skip-layers") + 1] == "sliding_window"


def test_the_sliding_window_layers_are_skipped():
    """gemma-4 uses sliding-window attention; those layers stay unquantized."""
    from glq.kv_compression import FP8_KV_ARGS
    assert "sliding_window" in FP8_KV_ARGS


def test_glq_s_own_e8_cache_is_never_wired_to_anything():
    """It does not serve on vLLM 0.27.1. The definition is kept so the knowledge survives,
    but nothing may turn it on until a real serve-and-generate says it works."""
    import glq.chat as chat
    import glq.supervisor as sup_module
    from glq.installer import __main__ as installer

    for module in (chat, sup_module, installer):
        src = inspect.getsource(module)
        assert "GLQ_KV_QUANT" not in src, f"{module.__name__} can enable the E8 KV cache"
        assert "E8_KV_ENV" not in src, f"{module.__name__} reaches for the E8 env set"


def test_the_chat_can_turn_the_fp8_cache_on(monkeypatch):
    chat, _, made, _ = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt", "--fp8-kv-cache"])
    assert made[0]["fp8_kv"] is True


def test_the_chat_takes_the_installer_s_choice_as_its_default(monkeypatch):
    """Answering the installer's question once should not have to be repeated on every run."""
    chat, _, made, _ = _run_chat(monkeypatch, [], cfg={"model": "org/ckpt", "fp8_kv": True})
    chat.main([])
    assert made[0]["fp8_kv"] is True


def test_no_fp8_kv_cache_overrides_the_installer_s_choice(monkeypatch):
    chat, _, made, _ = _run_chat(monkeypatch, [], cfg={"model": "org/ckpt", "fp8_kv": True})
    chat.main(["--no-fp8-kv-cache"])
    assert made[0]["fp8_kv"] is False


# ==================================================================== the chat layout

# The controls — checkpoint dropdown, temperature, max tokens — sat above the conversation,
# taking permanent vertical space on every screen. Two fixes: fold them away, and drop the
# dropdown entirely when there is nothing to pick between, which is the normal case because
# `glq-chat` starts exactly one server with exactly one model.

def test_the_model_picker_is_hidden_when_there_is_one_model():
    from glq.chat import show_model_picker
    assert show_model_picker(["org/only-one"]) is False
    assert show_model_picker([]) is False


def test_the_model_picker_appears_when_a_server_offers_a_choice():
    from glq.chat import show_model_picker
    assert show_model_picker(["org/a", "org/b"]) is True


def test_the_controls_are_folded_away_rather_than_stacked_above_the_chat():
    """A chat window should be mostly chat. Gradio renders `additional_inputs` where they
    are created, so creating them in the open layout is what put them on screen."""
    import glq.chat as chat
    src = inspect.getsource(chat.build_ui)
    accordion = src.index("gr.Accordion")
    for control in ("gr.Slider", "gr.Dropdown"):
        assert control in src
        assert src.index(control) > accordion, (
            f"{control} is created before the accordion, so it renders in the open layout")


# The slider's ceiling has to come from the server's context window. `max_tokens` caps the
# *output*, but the window holds prompt + history + output — so offering the whole window as
# output is offering a setting that cannot succeed once anything is typed.

def test_the_output_cap_leaves_room_for_the_conversation():
    from glq.chat import max_tokens_ceiling
    for window in (2048, 8192, 32768):
        assert max_tokens_ceiling(window) < window, (
            f"a {window}-token window offers {max_tokens_ceiling(window)} tokens of output, "
            f"leaving nothing for the prompt")


def test_the_ceiling_tracks_the_window_it_was_given():
    from glq.chat import max_tokens_ceiling
    assert max_tokens_ceiling(32768) > max_tokens_ceiling(8192)


def test_a_tiny_window_still_allows_a_usable_answer():
    from glq.chat import max_tokens_ceiling
    assert max_tokens_ceiling(512) >= 256


# ==================================================================== sampling defaults

# gemma-4's card specifies temperature 1.0, top_p 0.95, top_k 64 across all use cases. The
# chat was sending temperature=0.7 and omitting the other two — overriding the model's own
# recommendation on one axis while deferring to it on the others, which is the worst of both.

def test_the_recommended_sampling_is_the_cards_numbers():
    from glq.chat import RECOMMENDED_SAMPLING
    assert RECOMMENDED_SAMPLING == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def test_top_k_travels_in_extra_body_because_it_is_not_an_openai_field():
    """`top_k` is not part of the OpenAI schema; the client drops unknown kwargs unless they
    go through extra_body, so a top_k passed the obvious way silently does nothing."""
    from glq.chat import completion_kwargs
    kwargs = completion_kwargs(model="org/ckpt", messages=[], temperature=1.0,
                               top_p=0.95, top_k=64, max_tokens=512)
    assert kwargs["extra_body"]["top_k"] == 64
    assert "top_k" not in kwargs


def test_the_standard_fields_stay_where_the_api_expects_them():
    from glq.chat import completion_kwargs
    kwargs = completion_kwargs(model="org/ckpt", messages=[], temperature=1.0,
                               top_p=0.95, top_k=64, max_tokens=512)
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["max_tokens"] == 512
    assert kwargs["stream"] is True


def test_top_k_disabled_is_not_sent_at_all():
    """vLLM reads the model's own generation_config for anything the request omits, so
    "no top_k" has to mean absent, not 0."""
    from glq.chat import completion_kwargs
    kwargs = completion_kwargs(model="org/ckpt", messages=[], temperature=0.6,
                               top_p=0.95, top_k=0, max_tokens=512)
    assert not kwargs.get("extra_body"), "top_k=0 should send nothing, not top_k=0"


# ============================================================ missing chat dependencies
#
# `gradio` and `openai` live in the `chat` extra, and both are imported lazily so the rest of
# glq works without them. But main() only reached them *inside* `with supervisor:` — after
# vLLM had loaded the weights. On a plain `pip install glq` that means several minutes of
# model load, then a bare ModuleNotFoundError, then the teardown throwing the load away.
#
# CI found it before a user did: the test job installs `.[hub]`, not `.[chat]`, so twenty
# supervisor tests died on `No module named 'openai'` — they were only passing locally
# because a development venv happens to have it.

def test_missing_deps_are_named(monkeypatch):
    import glq.chat as chat
    monkeypatch.setattr(chat.importlib.util, "find_spec",
                        lambda name: None if name == "openai" else object())
    assert chat.missing_chat_deps() == ["openai"]


def test_nothing_missing_is_an_empty_list(monkeypatch):
    import glq.chat as chat
    monkeypatch.setattr(chat.importlib.util, "find_spec", lambda _name: object())
    assert chat.missing_chat_deps() == []


def test_the_check_happens_before_the_gpu_is_touched(monkeypatch, capsys):
    """The whole point: fail in a second, not after a six-minute load."""
    chat, events, _made, _launched = _run_chat(monkeypatch, ["--model", "org/ckpt"])
    monkeypatch.setattr(chat, "missing_chat_deps", lambda: ["gradio", "openai"])

    rc = chat.main(["--model", "org/ckpt"])

    assert rc != 0
    assert events == [], f"vLLM must not be started: {events}"


def test_the_message_says_how_to_fix_it(monkeypatch, capsys):
    chat, _events, _made, _launched = _run_chat(monkeypatch, ["--model", "org/ckpt"])
    monkeypatch.setattr(chat, "missing_chat_deps", lambda: ["gradio"])

    chat.main(["--model", "org/ckpt"])

    # One call: readouterr() drains the buffer, so a second call returns empty strings.
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "gradio" in out
    assert "glq[chat]" in out


# ==================================================== flashinfer on Blackwell without nvcc
#
# Measured on an RTX PRO 6000 (sm_120), vLLM 0.27.1, in a container with no CUDA toolkit:
# GLQ's own prebuilt kernel loads (EXT_OK:True) and decodes correctly, but EngineCore dies
# before generating because vLLM's sampler backend JIT-compiles for `120f` and cannot:
#
#     flashinfer/jit/cpp_ext.py: RuntimeError: Could not find nvcc and default
#     cuda_home='/usr/local/cuda' doesn't exist
#
# Setting VLLM_USE_FLASHINFER_SAMPLER=0 in the same container produced "Paris. Paris is the
# most visited city". FlashInfer ships prebuilt kernels for older archs, which is why this
# never appeared on sm_86/sm_89 — and why the fallback must NOT fire there, since disabling
# a working fast sampler would be a silent regression for everyone else.

def test_blackwell_without_a_toolkit_falls_back():
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap="12.0", have_nvcc=False) == {
        "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def test_blackwell_with_a_toolkit_is_left_alone():
    """nvcc present: flashinfer can build, so keep the faster sampler."""
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap="12.0", have_nvcc=True) == {}


def test_older_cards_are_left_alone_even_without_nvcc():
    """sm_89 ships prebuilt flashinfer kernels; disabling the sampler there would cost
    speed to fix a problem that does not exist."""
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap="8.9", have_nvcc=False) == {}


def test_unknown_capability_changes_nothing():
    """No driver, no nvidia-smi, or an unparseable answer: do not silently alter vLLM's
    sampler on a guess."""
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap=None, have_nvcc=False) == {}
    assert flashinfer_env(compute_cap="banana", have_nvcc=False) == {}


def test_newer_than_blackwell_also_falls_back():
    """The bound is >=12.0, not ==12.0: the next arch will have the same gap."""
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap="13.0", have_nvcc=False) == {
        "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def test_the_child_env_carries_the_fallback(monkeypatch):
    """The mechanism, not just the helper: what the supervisor hands to `vllm serve`."""
    import glq.supervisor as S
    monkeypatch.setattr(S, "flashinfer_env", lambda: {"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    env = S.child_env()
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PYTHONUNBUFFERED"] == "1"


# ============================================================ gradio needs a writable CWD
#
# Measured in the distro matrix on ubuntu:26.04, with the repo mounted read-only at /glq:
#
#     Could not create share link. [Errno 13] Permission denied: '.gradio'.
#     This can happen when the current working directory is read-only.
#
# gradio writes a `.gradio` directory into the *current* directory, so where the user
# happened to be standing decides whether the share tunnel works. `~/.glq` already holds
# config.json and vllm.log, so it is the obvious place to stand.

def test_the_glq_home_is_used_when_writable(tmp_path, monkeypatch):
    import glq.chat as chat
    monkeypatch.setenv("GLQ_HOME", str(tmp_path / "home"))
    assert chat.writable_workdir() == tmp_path / "home"
    assert (tmp_path / "home").is_dir(), "must be created, not merely named"


def test_an_unwritable_home_falls_back_to_tmp(tmp_path, monkeypatch):
    """A read-only or un-creatable GLQ_HOME must not stop the chat starting: losing the
    share link is bad, refusing to run at all is worse."""
    import glq.chat as chat
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    monkeypatch.setenv("GLQ_HOME", str(blocked / "under-a-file"))
    assert chat.writable_workdir() == Path(tempfile.gettempdir())


def test_the_chat_stands_in_that_directory_before_launching(monkeypatch, tmp_path):
    """The mechanism: gradio only sees the CWD at launch time."""
    chat, events, _made, _launched = _run_chat(monkeypatch, ["--model", "org/ckpt"])
    monkeypatch.setenv("GLQ_HOME", str(tmp_path / "h"))
    seen = []
    monkeypatch.setattr(chat.os, "chdir", lambda p: seen.append(str(p)))

    chat.main(["--model", "org/ckpt"])

    assert seen, "never moved out of whatever directory the user was standing in"
    assert seen[-1] == str(tmp_path / "h")


# ================================================================== how long to wait for vLLM
#
# The supervisor gave up after a hardcoded 900 s. That is generous for a 1.8 GiB checkpoint on
# an idle box and too short in two real cases: a large MoE loading from cold storage, and the
# distro matrix, where several containers start vLLM at once — measured, 12 legs of a 44-leg
# run failed with "never got a server" purely from contention, at 5-way concurrency on 16
# cores. Neither is a bug to fix by waiting silently; both need the waiting to be the caller's
# choice.

def test_the_default_is_unchanged(monkeypatch):
    chat, _events, made, _launched = _run_chat(monkeypatch, ["--model", "org/ckpt"])
    chat.main(["--model", "org/ckpt"])
    assert made[0]["timeout"] == chat.DEFAULT_READY_TIMEOUT


def test_the_flag_reaches_the_supervisor(monkeypatch):
    chat, _events, made, _launched = _run_chat(monkeypatch, ["--model", "org/ckpt"])
    chat.main(["--model", "org/ckpt", "--ready-timeout", "1200"])
    assert made[0]["timeout"] == 1200.0


def test_a_nonsense_timeout_is_refused(monkeypatch):
    """Zero or negative would mean 'give up before asking', which reads as an instant,
    inexplicable startup failure."""
    chat, _events, _made, _launched = _run_chat(monkeypatch, ["--model", "org/ckpt"])
    with pytest.raises(SystemExit):
        chat.main(["--model", "org/ckpt", "--ready-timeout", "0"])


# ============================================ the child must be able to find its own tools
#
# Running a venv binary by absolute path — ~/.glq/venv/bin/vllm serve … — does NOT put the
# venv's bin/ on PATH; only `source activate` does. So anything that shells out to a sibling
# tool cannot find it.
#
# Measured on an RTX PRO 6000: FlashInfer ships no prebuilt sampler for sm_120, JIT-compiles
# at first sample, shells out to ninja, and dies with
#
#     FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
#
# taking EngineCore with it — while ~/.glq/venv/bin/ninja existed all along, installed by
# install.sh for exactly this purpose. nvcc was present too, so the 0.8.7 fallback did not
# fire and could not have helped.

def test_the_venv_bin_directory_is_on_the_child_path():
    import glq.supervisor as S
    env = S.child_env()
    first = env["PATH"].split(os.pathsep)[0]
    assert first == os.path.dirname(sys.executable), (
        "vLLM shells out to ninja by name; without the venv's bin/ first on PATH the JIT "
        "build fails even though ninja is installed")


def test_the_existing_path_is_kept(monkeypatch):
    """Prepend, never replace: the child still needs the system's own tools."""
    import glq.supervisor as S
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = S.child_env()
    assert env["PATH"].endswith("/usr/bin:/bin")
    assert os.path.dirname(sys.executable) in env["PATH"]


def test_the_directory_is_not_duplicated(monkeypatch):
    import glq.supervisor as S
    bindir = os.path.dirname(sys.executable)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin")
    assert S.child_env()["PATH"].count(bindir) == 1


def test_a_missing_ninja_also_triggers_the_sampler_fallback():
    """nvcc alone is not sufficient — the JIT needs ninja too, and this is exactly the case
    the sm_120 box hit: nvcc present, ninja unreachable, engine dead."""
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap="12.0", have_nvcc=True, have_ninja=False) == {
        "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def test_a_complete_toolchain_leaves_flashinfer_alone():
    from glq.supervisor import flashinfer_env
    assert flashinfer_env(compute_cap="12.0", have_nvcc=True, have_ninja=True) == {}


# ==================================================== progress-aware wait + group teardown

def test_a_growing_download_keeps_the_wait_alive_past_the_timeout():
    """A 14 GB checkpoint on a slow link writes NOTHING to the log — the only sign of life
    is the HF cache growing. Measured (gemma-4-26B, unauthenticated, RTX PRO 6000): the
    engine sat healthy in snapshot_download for 15 minutes and the old fixed deadline shot
    it. The timeout must be a no-progress window, not a stopwatch."""
    dl = {"bytes": 0}
    sup, spawned = _sup(healthy_after=400, timeout=60.0,
                        download_bytes=lambda: dl["bytes"])
    inner = sup._sleep

    def sleep_and_download(s):
        inner(s)
        dl["bytes"] += 50_000_000            # the download is moving

    sup._sleep = sleep_and_download
    sup.start()                              # 400 polls x 2s >> 60s: only progress saves it
    assert spawned and not spawned[0].terminated


def test_fresh_log_output_also_resets_the_window():
    """Loading, compiling and graph capture all log but can individually exceed a short
    window — a child that keeps talking is not stuck."""
    sup, _ = _sup(healthy_after=400, timeout=60.0, download_bytes=lambda: 0)
    inner = sup._sleep

    def sleep_and_chat(s):
        inner(s)
        with open(sup.log_path, "a") as fh:
            fh.write("INFO still loading\n")

    sup._sleep = sleep_and_chat
    sup.start()


def test_true_silence_still_times_out_at_the_window():
    sup, _ = _sup(healthy_after=10**6, timeout=60.0, download_bytes=lambda: 0)
    with pytest.raises(RuntimeError, match="progress"):
        sup.start()


def test_the_timeout_error_names_the_unauthenticated_download():
    """The failure that motivated all of this: rate-limited anonymous download, silent log,
    timeout. The error must say the fix, not just the symptom."""
    sup, _ = _sup(healthy_after=10**6, timeout=60.0, download_bytes=lambda: 0,
                  output=["Warning: You are sending unauthenticated requests to the "
                          "HF Hub. Please set a HF_TOKEN.\n"])
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        sup.start()


def test_the_progress_line_reports_download_bytes_when_the_log_is_silent():
    out = io.StringIO()
    dl = {"bytes": 0}
    sup, _ = _sup(healthy_after=200, timeout=10_000.0, out=out,
                  report_every=30.0, download_bytes=lambda: dl["bytes"])
    inner = sup._sleep

    def sleep_and_download(s):
        inner(s)
        dl["bytes"] += 100_000_000

    sup._sleep = sleep_and_download
    sup.start()
    assert "downloading weights" in out.getvalue()


def test_the_child_is_started_in_its_own_session():
    """The group is what holds the VRAM, so the group must be killable as a unit — that
    starts with giving the child its own session at spawn."""
    sup, spawned = _sup(healthy_after=2)
    sup.start()
    assert spawned[0].spawn_kwargs.get("start_new_session") is True


def test_stop_ends_the_whole_process_group_not_just_the_child():
    """SIGTERM to the API server orphaned an EngineCore holding 17 GiB while stop()
    claimed the GPU was free. Signal the group."""
    seen = []
    sup, spawned = _sup(healthy_after=2,
                        killpg=lambda pgid, sig: seen.append((pgid, sig)),
                        getpgid=lambda pid: pid)
    sup.start()
    sup.stop()
    assert seen and seen[0][0] == spawned[0].pid


def test_group_kill_escalates_when_terminate_is_ignored():
    seen = []
    sup, spawned = _sup(healthy_after=2,
                        killpg=lambda pgid, sig: seen.append((pgid, sig)),
                        getpgid=lambda pid: pid)
    sup.start()
    spawned[0].ignores_terminate = True
    sup.stop()
    import signal as _signal
    assert [s for _, s in seen] == [_signal.SIGTERM, _signal.SIGKILL]


def test_a_child_that_does_not_lead_its_group_gets_plain_terminate():
    """Never killpg a group we do not own — an attached or oddly-spawned child falls back
    to the old single-process terminate."""
    sup, spawned = _sup(healthy_after=2,
                        killpg=lambda pgid, sig: (_ for _ in ()).throw(AssertionError),
                        getpgid=lambda pid: pid + 1)
    sup.start()
    sup.stop()
    assert spawned[0].terminated


# ------------------------------------------------------- chat-scale concurrency cap

def test_the_server_is_capped_to_chat_scale_concurrency():
    """vLLM defaults max_num_seqs to 1024 — sized for a batch server, absurd for a
    single-user chat — and on hybrid-GDN models every decode sequence needs its own Mamba
    cache block, so the default turns a modest KV pool into a refusal to start. Measured
    (Qwen3.8-27B, 96 GB card, 0.45 utilization): 'max_num_seqs (1024) exceeds available
    Mamba cache blocks (399)'."""
    sup, spawned = _sup(healthy_after=2)
    sup.start()
    argv = spawned[0].argv
    assert "--max-num-seqs" in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "16"


def test_max_num_seqs_is_overridable():
    sup, spawned = _sup(healthy_after=2, max_num_seqs=4)
    sup.start()
    argv = spawned[0].argv
    assert argv[argv.index("--max-num-seqs") + 1] == "4"


def test_the_chat_wires_max_num_seqs_through(monkeypatch):
    chat, _, made, _ = _run_chat(monkeypatch, [])
    chat.main(["--model", "org/ckpt", "--max-num-seqs", "8"])
    assert made and made[0].get("max_num_seqs") == 8


# ------------------------------------------------- download progress: total + handoff

def _sup_with_download(dl, **kw):
    out = io.StringIO()
    sup, spawned = _sup(healthy_after=200, timeout=10_000.0, out=out,
                        report_every=30.0, download_bytes=lambda: dl["bytes"], **kw)
    inner = sup._sleep

    def sleep_and_download(s):
        inner(s)
        dl["bytes"] = min(dl["bytes"] + dl.get("step", 0), dl.get("cap", 1 << 60))

    sup._sleep = sleep_and_download
    return sup, out


def test_the_download_line_shows_the_total_when_the_checkpoint_size_is_known():
    """The size is already looked up for pool sizing — "17.8 GiB so far" with no
    denominator reads as a hang to someone expecting 14 and watching 17.8."""
    dl = {"bytes": 0, "step": 200_000_000}
    sup, out = _sup_with_download(dl, weights_bytes=4 * 2**30)
    sup.start()
    assert "/ 4.0 GiB" in out.getvalue()


def test_the_announcement_names_the_download_size_up_front():
    dl = {"bytes": 0, "step": 200_000_000}
    sup, out = _sup_with_download(dl, weights_bytes=4 * 2**30)
    sup.start()
    assert "~4.0 GiB" in out.getvalue()


def test_a_finished_download_says_so_instead_of_sticking(monkeypatch):
    """Measured: the line froze at "17.8 GiB so far" for the whole load phase — the
    download was done, and the label never moved again."""
    dl = {"bytes": 0, "step": 2 * 2**30, "cap": 4 * 2**30}
    sup, out = _sup_with_download(dl, weights_bytes=4 * 2**30)
    sup.start()
    assert "weights downloaded (4.0 GiB)" in out.getvalue()


def test_fresh_log_output_takes_the_line_back_from_the_download():
    """Once the engine logs again (loading, compiling, graph capture), its own words are
    the truth — the stale download figure must not shadow them."""
    dl = {"bytes": 0, "step": 100_000_000, "cap": 300_000_000}
    sup, out = _sup_with_download(dl)
    inner = sup._sleep
    tick = {"n": 0}

    def sleep_download_then_chat(s):
        inner(s)
        tick["n"] += 1
        if tick["n"] > 10:                        # download over; the engine talks again
            with open(sup.log_path, "a") as fh:
                fh.write("INFO loading weights from disk\n")

    sup._sleep = sleep_download_then_chat
    sup.start()
    assert "loading weights from disk" in out.getvalue()


# ------------------------------------------------- headroom-tiered context window

GIB = 2**30


def test_an_l4_serving_the_26b_stays_at_the_floor():
    """No regression on the cards the 8192 default was designed for: 23 GiB card,
    14.4 GiB weights → pool 0.89·23 ≈ 20.5 GiB, headroom ≈ 2.1 GiB after overhead;
    16384×8×24.6 KiB ≈ 3.1 GiB does not fit."""
    got = sup_mod.plan_max_model_len(weights_bytes=int(14.4 * GIB),
                                     vram_bytes=23 * GIB, model_max_len=262144)
    assert got == 8192


def test_a_96gib_card_reaches_the_top_tier():
    """96 GiB, same 26B: pool floors at 0.45 → 43 GiB, headroom ≈ 24 GiB;
    65536×8×24.6 KiB ≈ 12.3 GiB fits with room to spare."""
    got = sup_mod.plan_max_model_len(weights_bytes=int(14.4 * GIB),
                                     vram_bytes=96 * GIB, model_max_len=262144)
    assert got == 65536


def test_the_declared_maximum_clamps_the_tier():
    got = sup_mod.plan_max_model_len(weights_bytes=int(14.4 * GIB),
                                     vram_bytes=96 * GIB, model_max_len=32768)
    assert got == 32768


def test_a_small_declared_maximum_wins_over_the_floor():
    """SmolLM2 declares 8192; a huge card must not talk vLLM into refusing to start."""
    got = sup_mod.plan_max_model_len(weights_bytes=1 * GIB, vram_bytes=96 * GIB,
                                     model_max_len=8192)
    assert got == 8192


@pytest.mark.parametrize("kw", [
    dict(weights_bytes=None, vram_bytes=96 * GIB, model_max_len=262144),
    dict(weights_bytes=14 * GIB, vram_bytes=None, model_max_len=262144),
    dict(weights_bytes=14 * GIB, vram_bytes=96 * GIB, model_max_len=None),
])
def test_any_unknown_input_means_the_conservative_floor(kw):
    """Never tier up blind: an oversized guess makes vLLM refuse to start, which is
    strictly worse than a small window."""
    assert sup_mod.plan_max_model_len(**kw) == 8192


def test_the_floor_is_a_parameter_for_glq_code():
    got = sup_mod.plan_max_model_len(weights_bytes=None, vram_bytes=None,
                                     model_max_len=None, floor=16384)
    assert got == 16384


def test_the_supervisor_plans_the_window_when_not_pinned():
    sup, _ = _sup(healthy_after=2, max_model_len=None, model_max_len=262144,
                  weights_bytes=int(14.4 * GIB), vram_bytes=96 * GIB)
    assert sup.max_model_len == 65536
    sup.start()


def test_an_explicit_window_is_used_verbatim_with_no_planning():
    sup, spawned = _sup(healthy_after=2, max_model_len=4096)
    sup.start()
    argv = spawned[0].argv
    assert argv[argv.index("--max-model-len") + 1] == "4096"


def test_the_announcement_says_when_the_window_was_sized_from_headroom():
    out = io.StringIO()
    sup, _ = _sup(healthy_after=2, out=out, max_model_len=None, model_max_len=262144,
                  weights_bytes=int(14.4 * GIB), vram_bytes=96 * GIB)
    sup.start()
    assert "sized from KV headroom" in out.getvalue()
    out2 = io.StringIO()
    sup2, _ = _sup(healthy_after=2, out=out2, max_model_len=8192)
    sup2.start()
    assert "sized from KV headroom" not in out2.getvalue()



# ------------------------------------------------- chat wiring for the sized window

def test_the_chat_defaults_to_a_planned_window(monkeypatch):
    chat, _, made, _ = _run_chat(monkeypatch, [])
    monkeypatch.setattr(chat, "_model_max_len", lambda repo: 262144)
    chat.main(["--model", "org/ckpt"])
    assert made[0]["max_model_len"] is None
    assert made[0]["model_max_len"] == 262144


def test_a_pinned_window_skips_the_config_lookup(monkeypatch):
    """Same principle as the pool plan: an explicit flag costs no network round trip."""
    chat, _, made, _ = _run_chat(monkeypatch, [])
    monkeypatch.setattr(chat, "_model_max_len",
                        lambda repo: (_ for _ in ()).throw(AssertionError("looked up")))
    chat.main(["--model", "org/ckpt", "--max-model-len", "4096"])
    assert made[0]["max_model_len"] == 4096
    assert made[0].get("model_max_len") is None


def test_the_slider_ceiling_follows_the_planned_window(monkeypatch):
    """max_tokens_ceiling is half the window; with auto sizing the chosen value lives on
    the supervisor, not in args."""
    chat, _, made, _ = _run_chat(monkeypatch, [])
    monkeypatch.setattr(chat, "_model_max_len", lambda repo: 262144)
    seen = {}

    def build(base_url, models, *, max_model_len):
        seen["mml"] = max_model_len
        demo = types.SimpleNamespace()
        demo.launch = lambda **kw: None
        return demo

    monkeypatch.setattr(chat, "build_ui", build)
    chat.main(["--model", "org/ckpt"])
    assert seen["mml"] == 8192, "must read the supervisor's chosen window"
