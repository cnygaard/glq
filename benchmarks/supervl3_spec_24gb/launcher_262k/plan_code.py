"""Print what glq-code would serve, without starting vLLM: main() runs as usual, and the
supervisor it builds is printed instead of started."""
import sys
import glq.code as code
from glq.supervisor import VllmSupervisor


class _Print(VllmSupervisor):
    def __init__(self, **kw):
        super().__init__(**kw)
        env = {k: v for k, v in __import__("glq.supervisor", fromlist=["child_env"]).child_env(
            device=self.device, expert_arena_gib=self.expert_arena_gib,
            max_num_seqs=self.step_tokens, min_step_tokens=self.min_step_tokens).items()
            if k.startswith("GLQ_MOE")}
        print(f"PLAN window={self.max_model_len} offload={self.expert_offload_gib} "
              f"arena={self.expert_arena_gib} k={self.spec_tokens} util={self.gpu_memory_utilization:.4f} "
              f"resident={self.resident_bytes / 2**30:.2f} vram={self._vram_bytes / 2**30:.2f} "
              f"ram={self._ram_bytes / 2**30:.1f} note={self._spec_note!r}")
        print("ENV", env)
        print("ARGV", " ".join(self.argv()))
        raise SystemExit(0)


code.VllmSupervisor = _Print
sys.exit(code.main(sys.argv[1:]))
