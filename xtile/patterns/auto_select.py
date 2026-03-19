"""
xtile.patterns.auto_select - Automatic pattern selection engine.

Selects the best compute-communication overlap pattern for a given problem
shape and hardware configuration.  The decision logic is based on the
empirical observations from the Iris paper (Section 5).

Also provides :func:`benchmark_all_patterns` to run every pattern on a
given problem and return a comparison table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from xtile.patterns import Pattern


def auto_select(
    op: str,
    M: int,
    N: int,
    K: int,
    world_size: int,
    hw_info: Optional[Any] = None,
    ctx: Optional[Any] = None,
) -> "Pattern":
    """Select the best overlap pattern based on problem shape and hardware.

    The heuristic is derived from the Iris paper (Section 5) experimental
    observations on H100 / MI300X clusters.

    Decision logic:
        1. ``N / world_size < 1024`` and ``K > 16384``
           --> :class:`FusedSequentialPattern`
           (small output shard, large reduction -- scatter cost is low
           relative to GEMM, simple fusion suffices)

        2. ``N / world_size < 2048`` and ``K > 8192``
           --> :class:`ProducerConsumerPattern`
           (moderate output shard, needs more overlap than fused-sequential
           but WG specialization overhead is not justified)

        3. ``N > 4096`` and ``K > 8192``
           --> :class:`WGSpecializedPattern`
           (large output, many tiles -- enough work to keep both compute
           and comm worker pools saturated)

        4. Default --> :class:`BulkSyncPattern`
           (small problem or unknown shape -- safest baseline)

    Args:
        op: Operation type string, e.g. ``"gemm_allscatter"``,
            ``"gemm_allgather"``.  Currently only ``"gemm_allscatter"``
            is fully supported; others fall back to bulk_sync.
        M: Number of rows in the output matrix.
        N: Number of columns in the full (un-sharded) output matrix.
        K: Shared (reduction) dimension.
        world_size: Number of GPUs in the process group.
        hw_info: Optional hardware info object (e.g.
            :class:`~xtile.backends.base.DeviceProperties`).  Reserved
            for future hardware-aware selection.
        ctx: Optional distributed context.  If provided, the returned
            pattern is pre-initialized with this context.  If ``None``,
            the caller must initialize the pattern manually.

    Returns:
        An instance of the selected :class:`Pattern` subclass (if *ctx* is
        provided) or the pattern **class** itself (if *ctx* is ``None``).

    Raises:
        ValueError: If *op* is not a recognized operation string.

    Examples::

        # Get a pre-initialized pattern
        pattern = auto_select("gemm_allscatter", M=4096, N=8192, K=4096,
                              world_size=8, ctx=my_ctx)
        pattern.execute(A, B, C)

        # Get pattern class for deferred initialization
        PatternCls = auto_select("gemm_allscatter", M=4096, N=8192, K=4096,
                                 world_size=8)
        pattern = PatternCls(my_ctx)

    TODO: Incorporate hw_info (SM count, bandwidth) into the decision.
    TODO: Add learned auto-tuning based on historical benchmark data.
    TODO: Support "gemm_allgather" and other fused operations.
    """
    from xtile.patterns.bulk_sync import BulkSyncPattern
    from xtile.patterns.fused_sequential import FusedSequentialPattern
    from xtile.patterns.producer_consumer import ProducerConsumerPattern
    from xtile.patterns.wg_specialized import WGSpecializedPattern

    _SUPPORTED_OPS = {"gemm_allscatter", "gemm_allgather", "gemm_reducescatter"}

    if op not in _SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported operation {op!r}.  "
            f"Supported operations: {sorted(_SUPPORTED_OPS)}"
        )

    # -- Heuristic selection (Iris Section 5) --

    n_per_rank = N // max(world_size, 1)

    if n_per_rank < 1024 and K > 16384:
        # Small output shard, large K: fused sequential is sufficient.
        # The scatter overhead per tile is small relative to the GEMM,
        # so the simple "compute then immediately scatter" approach
        # achieves good overlap without the complexity of dual streams.
        selected_cls = FusedSequentialPattern

    elif n_per_rank < 2048 and K > 8192:
        # Moderate output shard: producer-consumer gives better overlap
        # than fused sequential by decoupling compute and comm onto
        # separate streams, but WG specialization's single-kernel
        # overhead is not yet justified.
        selected_cls = ProducerConsumerPattern

    elif N > 4096 and K > 8192:
        # Large output with large K: WG specialization achieves the best
        # overlap because there are enough tiles to keep both compute
        # and comm SM pools fully saturated within a single kernel.
        selected_cls = WGSpecializedPattern

    else:
        # Small problem or shape does not clearly favor any overlap
        # strategy.  Fall back to bulk-sync (safest, no overlap).
        selected_cls = BulkSyncPattern

    # TODO: Refine thresholds based on hw_info (SM count, NVLink/xGMI
    #       bandwidth, L2 cache size) when available.
    # TODO: For very small M (< 256), consider a non-persistent kernel
    #       variant that avoids the overhead of persistent tile scheduling.

    if ctx is not None:
        return selected_cls(ctx)
    return selected_cls  # type: ignore[return-value]


def benchmark_all_patterns(
    A: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
    ctx: Any,
    warmup: int = 10,
    iters: int = 100,
) -> Dict[str, Any]:
    """Run all overlap patterns and return a comparison table.

    Instantiates every pattern with the given context, benchmarks each one,
    and returns the collected results sorted by mean latency.

    Args:
        A: Input tensor ``(M, K)`` on the local device.
        B: Input tensor ``(K, N)`` on the local device.
        C: Output tensor ``(M, N_local)`` on the local device.
        ctx: Distributed context.
        warmup: Warm-up iterations per pattern.
        iters: Timed iterations per pattern.

    Returns:
        A dict with:
            ``"results"``: list of per-pattern dicts (from
            :meth:`Pattern.benchmark`), sorted by ``mean_ms`` ascending.
            ``"best"``: name of the fastest pattern.
            ``"M"``, ``"N"``, ``"K"``: problem dimensions.
            ``"world_size"``: number of GPUs.

    Example::

        results = benchmark_all_patterns(A, B, C, ctx)
        for r in results["results"]:
            print(f"{r['pattern']:25s}  {r['mean_ms']:.3f} ms")
        print(f"Best: {results['best']}")
    """
    from xtile.patterns.bulk_sync import BulkSyncPattern
    from xtile.patterns.fused_sequential import FusedSequentialPattern
    from xtile.patterns.producer_consumer import ProducerConsumerPattern
    from xtile.patterns.wg_specialized import WGSpecializedPattern

    M, K = A.shape
    _, N = B.shape

    all_pattern_classes = [
        BulkSyncPattern,
        FusedSequentialPattern,
        ProducerConsumerPattern,
        WGSpecializedPattern,
    ]

    results: List[Dict[str, Any]] = []
    for cls in all_pattern_classes:
        pattern = cls(ctx)
        try:
            result = pattern.benchmark(A, B, C, warmup=warmup, iters=iters)
            results.append(result)
        except Exception as e:
            # Record the failure but continue with other patterns.
            # TODO: Log the exception via xtile.utils.logging when available.
            results.append({
                "pattern": pattern.name,
                "mean_ms": float("inf"),
                "min_ms": float("inf"),
                "max_ms": float("inf"),
                "iters": 0,
                "error": str(e),
            })

    # Sort by mean latency (fastest first)
    results.sort(key=lambda r: r["mean_ms"])

    best_name = results[0]["pattern"] if results else "none"

    return {
        "results": results,
        "best": best_name,
        "M": M,
        "N": N,
        "K": K,
        "world_size": ctx.world_size,
    }
