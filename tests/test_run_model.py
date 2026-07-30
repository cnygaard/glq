"""CPU unit tests for the pure helpers in benchmarks/run_model.py.

The harness itself needs a GPU, but everything that decides whether a run is *valid* —
parsing vLLM's footprint/capture lines, the degeneracy heuristic, the footprint tolerance
gate — is pure string/number work and belongs in the CPU subset. The log fixtures below are
verbatim lines from a real SmolLM2-360M trellis-4bpw run on an L4 (vLLM 0.25.1), so a change
to vLLM's log format fails here rather than silently reporting `None` footprint forever.
"""

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "benchmarks" / "run_model.py"
_spec = importlib.util.spec_from_file_location("glq_run_model", _SRC)
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)


# Verbatim from /opt/dlami/nvme/l4_smoke.log — vLLM prefixes engine-core lines with the pid.
REAL_LOG = """\
(EngineCore pid=6368) INFO 07-27 09:17:38 [default_loader.py:430] Loading weights took 0.14 seconds
(EngineCore pid=6368) INFO 07-27 09:17:39 [model_runner.py:302] Model loading took 0.24 GiB and 37.268289 seconds
(EngineCore pid=6368) INFO 07-27 09:19:05 [kv_cache_utils.py:2146] GPU KV cache size: 470,688 tokens
(EngineCore pid=6368) INFO 07-27 09:19:08 [model_runner.py:722] Graph capturing finished in 3 secs, took 0.06 GiB
"""


class TestParseWeights:
    def test_parses_real_line(self):
        assert rm.parse_weights_gib(REAL_LOG) == pytest.approx(0.24)

    def test_absent_returns_none_never_a_fallback(self):
        """A missing line must be None, not an nvidia-smi delta: under vLLM that measures
        the gpu_memory_utilization KV pool (util x total), not the weights."""
        assert rm.parse_weights_gib("INFO nothing relevant here\n") is None

    def test_large_multi_digit_footprint(self):
        log = "INFO [model_runner.py:302] Model loading took 14.97 GiB and 120.5 seconds"
        assert rm.parse_weights_gib(log) == pytest.approx(14.97)


class TestParseKv:
    def test_parses_comma_grouped_tokens(self):
        assert rm.parse_kv_tokens(REAL_LOG) == 470688

    def test_absent_returns_none(self):
        assert rm.parse_kv_tokens("nothing\n") is None


class TestGraphCapture:
    def test_detects_capture_and_returns_gib(self):
        assert rm.parse_graph_gib(REAL_LOG) == pytest.approx(0.06)

    def test_singular_sec_also_matches(self):
        assert rm.parse_graph_gib("Graph capturing finished in 1 sec, took 0.01 GiB") is not None

    def test_absent_returns_none(self):
        """No capture line while not --eager means the tok/s number is silently an eager
        number; the caller turns this None into a hard failure."""
        assert rm.parse_graph_gib("INFO model loaded\n") is None


class TestDegeneracy:
    def test_healthy_sample_passes(self):
        """Verbatim generation from the real L4 run — repetitive but legitimate for a 360M."""
        text = ("1. Primary Colors: Red, Blue, Yellow 2. Primary Colors: Red, Blue, Yellow "
                "3. One Fact: The primary colors are the three primary hues that can be "
                "combined to create all other colors in a mix of colors.")
        assert not rm.is_degenerate(text)

    def test_single_token_loop_is_degenerate(self):
        assert rm.is_degenerate("the " * 40)

    def test_short_phrase_loop_is_degenerate(self):
        """The classic broken-quantization failure: a repeating n-gram, not a repeating word.
        A naive unique-count check misses this; a unique *ratio* catches it."""
        assert rm.is_degenerate("alpha beta gamma " * 12)

    def test_empty_and_stub_are_degenerate(self):
        assert rm.is_degenerate("")
        assert rm.is_degenerate("   \n ")
        assert rm.is_degenerate("Sure!")

    def test_short_but_complete_answer_passes(self):
        assert not rm.is_degenerate("Paris is the capital city of France.")


class TestFootprintGate:
    def test_within_tolerance_passes(self):
        assert rm.footprint_ok(0.24, 0.25, tol=0.15)

    def test_bf16_sized_model_fails_a_4bpw_expectation(self):
        """The silently-dense trap: HF ignores quantization_config, the model generates fine,
        and only the footprint reveals it. 360M at bf16 ~0.72 GiB vs 4bpw ~0.24."""
        assert not rm.footprint_ok(0.72, 0.24, tol=0.15)

    def test_missing_measurement_is_not_a_pass(self):
        assert not rm.footprint_ok(None, 0.24)


class TestCaptureSizes:
    def test_default_covers_requested_batches(self):
        """vLLM's default derives from max_num_seqs*2 (often [1,2]), so an uncaptured B=32
        silently benches off-graph."""
        sizes = rm.resolve_capture_sizes(None, [1, 32])
        assert 1 in sizes and 32 in sizes
        assert sizes == sorted(sizes)

    def test_unusual_batch_is_added(self):
        assert 48 in rm.resolve_capture_sizes(None, [1, 48])

    def test_capped_at_moe_limit(self):
        """GLQ MoE above 256 routes to a host-syncing Python loop that cannot be
        stream-captured; capturing past it raises cudaErrorStreamCaptureUnsupported."""
        assert max(rm.resolve_capture_sizes(None, [1, 512])) <= 256

    def test_explicit_spec_wins(self):
        assert rm.resolve_capture_sizes("1,2,4", [1]) == [1, 2, 4]


class TestBatches:
    def test_parses_comma_list(self):
        assert rm.parse_batches("1,8,32") == [1, 8, 32]

    def test_tolerates_spaces_and_trailing_comma(self):
        assert rm.parse_batches(" 1, 32 ,") == [1, 32]

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            rm.parse_batches("0,4")


class TestMultimodalDetection:
    def test_conditional_generation_arch_needs_mm0(self):
        """Passing only image+audio leaves <|video|> placeholders and still crashes, so the
        harness sets all three keys or none."""
        assert rm.needs_mm0(["Gemma4UnifiedForConditionalGeneration"])

    def test_plain_causal_lm_does_not(self):
        assert not rm.needs_mm0(["LlamaForCausalLM"])
