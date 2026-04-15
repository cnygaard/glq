"""Per-layer sensitivity profiling and bit-width allocation for mixed-precision GLQ."""

from __future__ import annotations


def allocate_bpw(
    sensitivities: dict[str, float],
    layer_sizes: dict[str, int],
    target_avg_bpw: float,
    min_bpw: int = 2,
    max_bpw: int = 4,
) -> dict[str, int]:
    """Allocate per-layer bpw to minimize total proxy loss within a bit budget.

    Uses greedy marginal-gain: layers with highest proxy_loss per weight
    get upgraded first (they benefit most from extra bits).

    Args:
        sensitivities: {layer_name: proxy_loss_at_min_bpw}
        layer_sizes: {layer_name: m * n (number of weights)}
        target_avg_bpw: target average bits per weight (e.g. 2.5)
        min_bpw: minimum per-layer bpw (default 2)
        max_bpw: maximum per-layer bpw (default 4)

    Returns:
        {layer_name: bpw} assignment
    """
    allowed = sorted({b for b in (2, 3, 4, 5, 6, 7, 8) if min_bpw <= b <= max_bpw})
    if not allowed:
        raise ValueError(f"No valid bpw in [{min_bpw}, {max_bpw}]")

    total_weights = sum(layer_sizes.values())
    budget = target_avg_bpw * total_weights

    # Start everything at the minimum
    base = allowed[0]
    allocation = {name: base for name in sensitivities}
    current_bits = base * total_weights

    # Greedy: repeatedly pick the upgrade (layer, target_bpw) with best
    # sensitivity-per-bit, considering all possible jumps (2→3, 2→4, 3→4).
    while True:
        best_gain = -1
        best_name = None
        best_target = None
        best_cost = 0

        for name in allocation:
            cur = allocation[name]
            for target_bpw in allowed:
                if target_bpw <= cur:
                    continue
                extra = (target_bpw - cur) * layer_sizes[name]
                if current_bits + extra > budget:
                    continue
                gain = sensitivities[name] / extra  # proxy_loss per bit
                if gain > best_gain:
                    best_gain = gain
                    best_name = name
                    best_target = target_bpw
                    best_cost = extra

        if best_name is None:
            break
        allocation[best_name] = best_target
        current_bits += best_cost

    return allocation


def print_allocation_summary(
    allocation: dict[str, int],
    layer_sizes: dict[str, int],
    sensitivities: dict[str, float] | None = None,
):
    """Print a summary of the bpw allocation."""
    total_weights = sum(layer_sizes.values())
    total_bits = sum(allocation[n] * layer_sizes[n] for n in allocation)
    avg_bpw = total_bits / total_weights

    by_bpw = {}
    for name, bpw in allocation.items():
        by_bpw.setdefault(bpw, []).append(name)

    print(f"\nBit allocation summary:")
    print(f"  Average bpw: {avg_bpw:.3f}")
    print(f"  Total weights: {total_weights:,}")
    for bpw in sorted(by_bpw):
        n_layers = len(by_bpw[bpw])
        n_weights = sum(layer_sizes[n] for n in by_bpw[bpw])
        print(f"  {bpw}bpw: {n_layers} sublayers ({n_weights:,} weights, {n_weights/total_weights:.1%})")

    if sensitivities:
        print(f"\n  Top 10 most sensitive (highest proxy_loss density):")
        ranked = sorted(sensitivities.items(), key=lambda x: x[1] / layer_sizes[x[0]], reverse=True)
        for name, loss in ranked[:10]:
            print(f"    {allocation[name]}bpw  {loss:.4e}  {name}")
