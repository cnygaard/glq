"""The trellis rate table, apart from the codebook.

Its own module so the serving-side sizing (`glq_vllm.moe_shapes`) can read it without
importing `glq.trellis`, whose module-level `torch.compile` pulls in inductor: 2.75 s added to
every `glq-chat` / `glq-code` launch that prices an expert arena. `glq.trellis` re-exports it,
so the quantizer reads the same table.
"""
from __future__ import annotations


def trellis_rvq_recipe(bpw):
    """Per-stage trellis rate K for integer ``bpw`` 2-8.

    2-4 is a single native-rate trellis. 5-8 stacks a K=4 primary with a K=(bpw-4)
    residual: 5=4+1, 6=4+2, 7=4+3, 8=4+4. Two reasons it is not one native K=5..8 code:

    * Quality — a native rate loses the trellis coding gain as K grows, because a weight's
      context is only ``ceil(L/K)`` symbols (2 at K=8 with L=16). Measured on 6 real
      SmolLM3-3B layers, native ties stacked at 5 bpw (+0.09 dB) then trails by 0.32 /
      0.89 / 2.37 dB at 6 / 7 / 8.
    * Kernels — ``glq_trellis.cu`` packs each decode chunk into a uint32 of width ``8*R``,
      so R>=5 has no kernel at all. Every stage here stays at R<=4.

    6 bpw is 4+2 rather than the marginally better 3+3 (+0.22 dB): ``tr_ld_cs`` vectorizes
    uint2/uint4 but emits three scalar loads for uint3, so 3+3 costs 6 load instructions
    against 4+2's 2 for the same bits.
    """
    bpw = int(bpw)
    if not 2 <= bpw <= 8:
        raise ValueError(f"trellis supports integer bpw 2-8, got {bpw}")
    return [bpw] if bpw <= 4 else [4, bpw - 4]
