"""Print what glq-code would serve on a card of PLAN_VRAM_GIB, without starting vLLM.

main() runs as usual against the real Hub, host RAM and flags; only the VRAM the planner sees
is replaced. The printed `util_here` is the plan's utilization rescaled to this card, so
`vllm serve` with it leaves the same absolute budget the smaller card would have.
"""
import os
import subprocess
import sys

import glq.code as code
from glq.supervisor import VllmSupervisor, child_env

GIB = 2 ** 30
PLAN_GIB = float(os.environ["PLAN_VRAM_GIB"])
_real = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    text=True).split()[0]) * 2 ** 20
code._vram_bytes = lambda: int(PLAN_GIB * GIB)


class _Print(VllmSupervisor):
    def __init__(self, **kw):
        super().__init__(**kw)
        env = {k: v for k, v in child_env(
            device=self.device, expert_arena_gib=self.expert_arena_gib,
            max_num_seqs=self.step_tokens, min_step_tokens=self.min_step_tokens).items()
            if k.startswith("GLQ_MOE")}
        util = self.gpu_memory_utilization
        print(f"PLAN vram={PLAN_GIB} window={self.max_model_len} offload={self.expert_offload_gib} "
              f"arena={self.expert_arena_gib} k={self.spec_tokens} util={util:.4f} "
              f"util_here={util * PLAN_GIB * GIB / _real:.4f} "
              f"resident={self.resident_bytes / GIB:.2f} ram={self._ram_bytes / GIB:.1f} "
              f"note={self._spec_note!r}")
        print("ENV", " ".join(f"{k}={v}" for k, v in env.items()))
        print("ARGV", " ".join(self.argv()))
        raise SystemExit(0)


code.VllmSupervisor = _Print
sys.exit(code.main(sys.argv[1:]))
