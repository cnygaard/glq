"""Step 0 of PLE CPU offload: is a UVA view over PINNED INT16 usable as a gather source?

Everything in the offload design rests on one assumption: that vLLM's
``get_accelerator_view_from_cpu_tensor`` (-> ``torch.ops._C.get_cuda_view_from_cpu_tensor``)
accepts an int16 tensor, and that ``index_select`` over the resulting CUDA view returns the
same rows the resident tensor would. vLLM only ever applies it to bf16/fp8 tables, so the
int16 path is unexercised and the C++ op may well constrain dtype.

Falsify it here, before building on it. Also times the gather so the later
"is offload slow?" question starts from a measured number rather than an argument about
PCIe bandwidth (the plan's risk #2: at ~80 B/row the cost is latency, not bandwidth).

Shapes mirror the real checkpoint's PLE table: row width 40 int16 (= ceil(160*4/16)), i.e.
80 B/row, against a dense bf16 row of 320 B.
"""
import sys
import time

import torch


ROW_W = 40          # int16 per row, as in trellis_packed [vocab, 40]
EMB_DIM = 160       # the decoded row width


def _uva(t: torch.Tensor):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    return get_accelerator_view_from_cpu_tensor(t)


def probe_dtype(dtype, rows=1 << 20):
    """Can we even build a view, and does a gather match the CPU-side truth?"""
    host = torch.empty(rows, ROW_W, dtype=dtype, pin_memory=True)
    if dtype.is_floating_point:
        host.copy_(torch.randn(rows, ROW_W).to(dtype))
    else:
        host.copy_(torch.randint(-32768, 32767, (rows, ROW_W), dtype=torch.int32).to(dtype))
    try:
        view = _uva(host)
    except Exception as e:                                   # noqa: BLE001
        return f"VIEW FAILED: {type(e).__name__}: {e}", None, None
    info = (f"view ok: dtype={view.dtype} device={view.device} "
            f"shape={tuple(view.shape)} pinned_src={host.is_pinned()}")

    ids = torch.randint(0, rows, (4096,), device="cuda")
    got = view.index_select(0, ids)                           # the gather the op performs
    want = host.index_select(0, ids.cpu())
    exact = torch.equal(got.cpu(), want)
    return info, exact, (view, host)


def time_gather(view, n_ids, iters=50):
    rows = view.shape[0]
    ids = torch.randint(0, rows, (n_ids,), device="cuda")
    for _ in range(5):
        view.index_select(0, ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        view.index_select(0, ids)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    gib = n_ids * ROW_W * 2 / 2 ** 30
    return us, gib / (us / 1e6) if us else 0.0


def main():
    if not torch.cuda.is_available():
        print("PROBE_SKIP: no CUDA")
        return 1
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")

    for dtype in (torch.bfloat16, torch.int16):
        info, exact, pair = probe_dtype(dtype)
        tag = str(dtype).replace("torch.", "")
        print(f"\n--- {tag} ---")
        print(f"  {info}")
        if exact is None:
            continue
        print(f"  gather bit-exact vs host index_select: {exact}")
        view, _host = pair
        # bf16 is the control: vLLM already ships this dtype, so if int16 behaves the same
        # the difference cannot be blamed on dtype.
        for n in (8, 24, 512):
            us, bw = time_gather(view, n)
            print(f"  index_select {n:4d} rows: {us:8.1f} us  ({bw:6.2f} GiB/s)")

    print("\nVERDICT: int16 UVA is usable if the int16 block above shows "
          "'view ok' AND 'bit-exact ... True'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
