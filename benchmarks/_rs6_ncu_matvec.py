"""RS6 ncu target: the fused 3inst matvec (FUSE_IN + RS4a shuffle) on the two shipped
shapes — n=2048 single-block and n=11008 block-diag. Random packed int16 is fine for a
perf/register measurement (any bits decode). Gate: regs <= 64, local-mem traffic 0."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from glq import inference_kernel as ik  # noqa: E402
from glq.hadamard import _block_decompose  # noqa: E402
from glq.quantized_linear import _pack_block_meta  # noqa: E402


def main():
    assert ik._try_load_cuda_ext()
    ext = ik._glq_cuda
    torch.manual_seed(0)
    K, m = 4, 2048

    n = 2048
    packed = torch.randint(-32768, 32767, (m // 16 * (n // 16), 16 * K),
                           dtype=torch.int16, device="cuda")
    x = (torch.randn(n, device="cuda") * 0.5).half()
    sv = torch.where(torch.rand(n, device="cuda") < 0.5, -1.0, 1.0).half()
    for _ in range(3):
        ext.glq_decode_matvec_trellis_3inst_fusein_cuda(x, sv, packed, m, n, n, 1.0)
    torch.cuda.synchronize()
    print("RESULT ncu_shape n=2048 done")

    n2 = 11008
    blocks = _block_decompose(n2)
    bnm = _pack_block_meta(blocks).cuda()
    packed2 = torch.randint(-32768, 32767, (m // 16 * (n2 // 16), 16 * K),
                            dtype=torch.int16, device="cuda")
    x2 = (torch.randn(n2, device="cuda") * 0.5).half()
    sv2 = torch.where(torch.rand(n2, device="cuda") < 0.5, -1.0, 1.0).half()
    for _ in range(3):
        ext.glq_decode_matvec_trellis_3inst_fusein_cuda(
            x2, sv2, packed2, m, n2, n2, 1.0,
            blocks_n_meta=bnm, num_blocks=len(blocks), max_bs=max(blocks))
    torch.cuda.synchronize()
    print(f"RESULT ncu_shape n=11008 blocks={blocks} done")


if __name__ == "__main__":
    main()
