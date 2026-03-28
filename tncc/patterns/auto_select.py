"""
tncc.patterns.auto_select - Automatic pattern selection engine.

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
    from tncc.patterns import Pattern


def _detect_hardware_info():
    """Probe real hardware for SM count and link bandwidth.

    Returns:
        Tuple of (sm_count, bandwidth_gbps) detected from the current device.
        Falls back to conservative defaults if detection fails.
    """
    try:
        from tncc.utils.topology import detect_backend, detect_topology
        backend = detect_backend()
        topo = detect_topology(backend)
        sm_count = topo.compute_units if topo.compute_units > 0 else 132
        bw = topo.peak_bandwidth_gbps if topo.peak_bandwidth_gbps > 0 else 300.0
        return sm_count, bw
    except Exception:
        return 132, 300.0


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
            :class:`~tncc.backends.base.DeviceProperties`).  Reserved
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

    Note: When hw_info is None, the function auto-detects SM count and
    link bandwidth from the current device via topology detection.
    TODO: Add learned auto-tuning based on historical benchmark data.
    TODO: Support "gemm_allgather" and other fused operations.
    """
    from tncc.patterns.bulk_sync import BulkSyncPattern
    from tncc.patterns.fused_sequential import FusedSequentialPattern
    from tncc.patterns.producer_consumer import ProducerConsumerPattern
    from tncc.patterns.wg_specialized import WGSpecializedPattern

    _SUPPORTED_OPS = {"gemm_allscatter", "gemm_allgather", "gemm_reducescatter"}

    if op not in _SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported operation {op!r}.  "
            f"Supported operations: {sorted(_SUPPORTED_OPS)}"
        )

    # -- Data-driven heuristic selection --
    # Thresholds refined from H100 PCIe NV12 measurements (Phase 1)
    # and Iris paper Section 5 observations.  Hardware-specific
    # adjustments applied when hw_info is available.

    n_per_rank = N // max(world_size, 1)
    total_tiles_128 = ((M + 127) // 128) * ((n_per_rank + 127) // 128)

    # Hardware-specific threshold adjustments
    # H100 PCIe: 132 SMs, 300 GB/s NVLink, 50 MB L2
    # MI300X: 304 CUs, ~800 GB/s Infinity Fabric, 256 MB L2
    # Auto-detect from hardware if hw_info not provided
    if hw_info is not None:
        sm_count = getattr(hw_info, "compute_units", 132)
        bw_gb_s = getattr(hw_info, "link_bandwidth_gbps", 300.0)
    else:
        sm_count, bw_gb_s = _detect_hardware_info()

    # Compute intensity heuristic: flops / bytes
    # GEMM flops: 2*M*N*K, scatter bytes: M*N_per_rank*element_size
    flops = 2 * M * n_per_rank * K
    scatter_bytes = M * n_per_rank * 2  # assume fp16
    compute_intensity = flops / max(scatter_bytes, 1)

    # Bandwidth-aware K threshold scaling: higher bandwidth means
    # communication is cheaper, so we can use simpler patterns at
    # higher K values.  Scale thresholds relative to 300 GB/s baseline.
    bw_scale = bw_gb_s / 300.0 if bw_gb_s > 0 else 1.0

    if M < 256:
        # Very small M: not enough rows for persistent kernel overlap.
        # Bulk-sync avoids overhead of lock/signal mechanisms.
        selected_cls = BulkSyncPattern

    elif n_per_rank < 1024 and K > int(12288 * bw_scale):
        # Small output shard, large K: fused sequential is sufficient.
        # Scatter per tile is tiny relative to GEMM, so simple fusion
        # achieves good overlap without dual-stream complexity.
        selected_cls = FusedSequentialPattern

    elif n_per_rank < 2048 and K > int(6144 * bw_scale):
        # Moderate output shard: producer-consumer decouples compute
        # and comm onto separate streams for better overlap.
        selected_cls = ProducerConsumerPattern

    elif total_tiles_128 >= sm_count and K > int(4096 * bw_scale):
        # Large problem: enough tiles to saturate both compute and comm
        # SM pools within a single kernel launch.
        selected_cls = WGSpecializedPattern

    elif N > 4096 and K > int(8192 * bw_scale):
        # Fallback for large-N cases not caught above
        selected_cls = WGSpecializedPattern

    elif compute_intensity > 256 and total_tiles_128 > 16:
        # Compute-bound: fused sequential gets hardware overlap for free
        selected_cls = FusedSequentialPattern

    else:
        # Small problem or shape does not clearly favor any overlap
        # strategy.  Fall back to bulk-sync (safest, no overlap).
        selected_cls = BulkSyncPattern

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
    *,
    spec: Any | None = None,
    full_N: int | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
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
    from tncc.patterns.bulk_sync import BulkSyncPattern
    from tncc.patterns.fused_sequential import FusedSequentialPattern
    from tncc.patterns.producer_consumer import ProducerConsumerPattern
    from tncc.patterns.wg_specialized import WGSpecializedPattern
    from tncc.ops import build_gemm_allscatter_plan

    execution = spec
    if execution is None:
        execution = build_gemm_allscatter_plan(
            A,
            B,
            C,
            ctx=ctx,
            full_N=full_N,
            b_layout=b_layout,
            c_layout=c_layout,
        ).execution

    all_pattern_classes = [
        BulkSyncPattern,
        FusedSequentialPattern,
        ProducerConsumerPattern,
        WGSpecializedPattern,
    ]

    results: List[Dict[str, Any]] = []
    for cls in all_pattern_classes:
        plan = build_gemm_allscatter_plan(
            A,
            B,
            C,
            ctx=ctx,
            full_N=execution.full_N,
            b_layout=execution.rhs_layout,
            c_layout=execution.output_layout,
            pattern=cls,
        )
        try:
            result = plan.pattern_impl.benchmark(
                A,
                B,
                C,
                warmup=warmup,
                iters=iters,
                spec=plan.execution,
            )
            results.append(result)
        except Exception as e:
            # Record the failure but continue with other patterns.
            # TODO: Log the exception via tncc.utils.logging when available.
            results.append({
                "pattern": plan.pattern_name,
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
        "M": execution.M,
        "N": execution.full_N,
        "K": execution.K,
        "local_N": execution.local_N,
        "layout_mode": execution.output_layout,
        "world_size": ctx.world_size,
    }
