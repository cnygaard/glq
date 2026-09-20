"""Does the fused GatedDeltaNet decode kernel help MORE on weaker hardware?

The reference rule is a chain of aten ops over the K x V recurrent state: bandwidth- and
dispatch-bound, and it leans on thread count to hide that. The kernel is two passes with no
temporaries. If the reference degrades faster than the kernel as cores are removed, then the
kernel matters most exactly where GLQ's goal points -- modest machines running big models.

Isolated from the model on purpose: 36 layers' worth of state and routing, no weights, no
checkpoint. Runs anywhere in seconds.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from glq.inference_kernel_cpu import _try_load_cpu_ext  # noqa: E402

#: Qwen3.8-Flash-Next: 48 value heads, 128 key dim, 128 value dim, 36 GDN layers.
H, K, V, LAYERS = 48, 128, 128, 36


def oracle_step(S, q, k, v, g, beta):
    """torch_recurrent_gated_delta_rule's inner loop, transcribed (modeling_qwen4_exp:440)."""
    S = S * g.exp()[..., None, None]
    kv = (S * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv) * beta.unsqueeze(-1)
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    return (S * q.unsqueeze(-1)).sum(dim=-2), S


def _inputs(b=1):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    S = r(b, H, K, V) * 0.1
    q = torch.nn.functional.normalize(r(b, H, K), dim=-1) / (K ** 0.5)
    k = torch.nn.functional.normalize(r(b, H, K), dim=-1)
    return S, q, k, r(b, H, V), -torch.rand(b, H, generator=g) * 0.5, \
        torch.rand(b, H, generator=g)


def _bench(fn, n=20):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", default="1,2,4,8,16")
    ap.add_argument("--tiers", default="")
    args = ap.parse_args()

    if not _try_load_cpu_ext():
        print("CPU extension unavailable")
        return 1
    from glq import inference_kernel_cpu as ikc
    ext = ikc._glq_cpu
    if not hasattr(ext, "glq_gdn_recurrent_step_cpu"):
        print("this build has no glq_gdn_recurrent_step_cpu")
        return 1

    S, q, k, v, g, beta = _inputs()
    print(f"state {S.numel() * 4 / 2**20:.1f} MiB/layer, {H} heads {K}x{V}, "
          f"{LAYERS} GDN layers, isa={ext.glq_cpu_active_isa()}")
    print(f"{'threads':>8} {'reference':>12} {'kernel':>10} {'speedup':>9} "
          f"{'ref 36L':>10} {'ker 36L':>10}")

    prev = torch.get_num_threads()
    try:
        for t in [int(x) for x in args.threads.split(",")]:
            if t > os.cpu_count():
                continue
            torch.set_num_threads(t)
            S1, S2 = S.clone(), S.clone()
            ref = _bench(lambda: oracle_step(S1, q, k, v, g, beta))
            ker = _bench(lambda: ext.glq_gdn_recurrent_step_cpu(S2, q, k, v, g, beta))
            print(f"{t:>8} {ref:>10.3f}ms {ker:>8.3f}ms {ref / ker:>8.1f}x "
                  f"{ref * LAYERS:>8.1f}ms {ker * LAYERS:>8.1f}ms", flush=True)
    finally:
        torch.set_num_threads(prev)

    if args.tiers:
        print("\nkernel by ISA tier (threads=1, so this is pure per-core width):")
        torch.set_num_threads(1)
        try:
            for name in args.tiers.split(","):
                if not ext.glq_cpu_isa_available(name):
                    print(f"  {name:>12}: unavailable")
                    continue
                ext.glq_cpu_set_isa(name)
                S3 = S.clone()
                ker = _bench(lambda: ext.glq_gdn_recurrent_step_cpu(S3, q, k, v, g, beta))
                print(f"  {name:>12}: {ker:>7.3f}ms/layer  {ker * LAYERS:>7.1f}ms/token")
        finally:
            ext.glq_cpu_set_isa("auto")
            torch.set_num_threads(prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
