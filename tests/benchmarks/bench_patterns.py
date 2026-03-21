#!/usr/bin/env python3
"""Pattern overlap efficiency benchmark.

Measures the performance of all 4 overlap patterns across Iris-style
problem sizes.  Reports speedup vs bulk_sync and overlap efficiency.

Usage:
    python3 tests/benchmarks/bench_patterns.py
    python3 tests/benchmarks/bench_patterns.py --quick

Requires >= 2 GPUs with peer access.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import xtile
from xtile.patterns.contracts import resolve_pattern_execution
from xtile.utils.benchmark_results import default_pattern_benchmark_path, write_json


# ---------------------------------------------------------------------------
# Problem sizes (Iris paper Section 5 + extensions)
# ---------------------------------------------------------------------------

# (M, N, K) tuples -- Iris paper typical sizes
IRIS_SIZES = [
    (4096, 4096, 4096),     # Small square
    (8192, 4608, 36864),    # Iris config 1
    (8192, 3584, 14336),    # Iris config 2
    (8192, 8192, 30720),    # Iris config 3
    (4096, 8192, 8192),     # Wide output
    (2048, 16384, 8192),    # Very wide output
]

QUICK_SIZES = [
    (4096, 4096, 4096),
    (8192, 4608, 14336),
]

_HEAP_ALIGNMENT = 256
_HEAP_GRANULARITY = 64 * 1024 * 1024
_HEAP_SAFETY_MARGIN = 64 * 1024 * 1024


@dataclass
class PatternResult:
    pattern: str
    M: int
    N: int
    K: int
    mean_ms: float
    min_ms: float
    max_ms: float
    speedup_vs_bulk: float
    overlap_efficiency: float


def _round_up(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    return (value + alignment - 1) & ~(alignment - 1)


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Return aligned bytes required for one heap-backed tensor."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    return _round_up(math.prod(shape) * element_size, _HEAP_ALIGNMENT)


def _required_heap_size(
    M: int,
    N: int,
    K: int,
    world_size: int,
    dtype: torch.dtype,
) -> int:
    """Estimate per-rank heap bytes required for one pattern benchmark."""
    n_per_rank = N // world_size
    required = 0
    required += _tensor_nbytes((M, K), dtype)
    required += _tensor_nbytes((K, n_per_rank), dtype)
    required += _tensor_nbytes((M, n_per_rank), dtype)
    required += _HEAP_SAFETY_MARGIN
    return _round_up(required, _HEAP_GRANULARITY)


def _cleanup_contexts(contexts: list[xtile.XTileContext]) -> None:
    """Release heaps attached to benchmark contexts."""
    for ctx in contexts:
        if ctx.heap is not None:
            ctx.heap.cleanup()


def benchmark_size(
    M: int, N: int, K: int,
    ctx: xtile.XTileContext,
    warmup: int = 5,
    iters: int = 20,
) -> tuple[list[PatternResult], dict[str, object]]:
    """Benchmark all 4 patterns on a given problem size."""
    from xtile.patterns.bulk_sync import BulkSyncPattern
    from xtile.patterns.fused_sequential import FusedSequentialPattern
    from xtile.patterns.producer_consumer import ProducerConsumerPattern
    from xtile.patterns.wg_specialized import WGSpecializedPattern

    world_size = ctx.world_size
    rank = ctx.rank
    torch.cuda.set_device(rank)

    # Allocate matrices
    N_per_rank = N // world_size
    A = ctx.randn(M, K, dtype=torch.float16)
    B = ctx.randn(K, N_per_rank, dtype=torch.float16)
    C = ctx.zeros(M, N_per_rank, dtype=torch.float16)
    ctx.barrier()
    execution = resolve_pattern_execution(
        A,
        B,
        C,
        rank=ctx.rank,
        world_size=ctx.world_size,
        full_N=N,
        b_layout="shard",
        c_layout="shard",
    )

    pattern_classes = [
        BulkSyncPattern,
        FusedSequentialPattern,
        ProducerConsumerPattern,
        WGSpecializedPattern,
    ]

    results = []
    bulk_sync_min = None

    for cls in pattern_classes:
        pattern = cls(ctx)
        try:
            # Warmup
            for _ in range(warmup):
                C.zero_()
                pattern.execute(A, B, C, spec=execution)
            ctx.barrier()

            # Timed iterations
            times: list[float] = []
            for _ in range(iters):
                C.zero_()
                start = time.perf_counter()
                pattern.execute(A, B, C, spec=execution)
                ctx.barrier()
                end = time.perf_counter()
                times.append((end - start) * 1e3)

            mean_ms = sum(times) / len(times)
            min_ms = min(times)
            max_ms = max(times)

            if pattern.name == "bulk_sync":
                bulk_sync_min = min_ms

            results.append(PatternResult(
                pattern=pattern.name,
                M=M, N=N, K=K,
                mean_ms=mean_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                speedup_vs_bulk=0.0,
                overlap_efficiency=0.0,
            ))
        except Exception as e:
            results.append(PatternResult(
                pattern=pattern.name,
                M=M, N=N, K=K,
                mean_ms=float("inf"),
                min_ms=float("inf"),
                max_ms=float("inf"),
                speedup_vs_bulk=0.0,
                overlap_efficiency=0.0,
            ))
            print(f"    {pattern.name}: ERROR - {e}")

    # Compute speedup vs bulk_sync
    if bulk_sync_min and bulk_sync_min > 0:
        for r in results:
            if r.min_ms < float("inf"):
                r.speedup_vs_bulk = bulk_sync_min / r.min_ms

    # Overlap efficiency: 1 - (fused_time / bulk_time)
    # Where bulk_time represents gemm_time + scatter_time
    if bulk_sync_min and bulk_sync_min > 0:
        for r in results:
            if r.min_ms < float("inf") and r.pattern != "bulk_sync":
                r.overlap_efficiency = 1.0 - (r.min_ms / bulk_sync_min)

    metadata = {
        "spec": execution.to_dict(),
        "dtype": str(A.dtype).replace("torch.", ""),
        "device": str(A.device),
    }
    return results, metadata


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Benchmark only 2 representative sizes")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per pattern")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations per pattern")
    parser.add_argument(
        "--heap-size-mb",
        type=int,
        default=None,
        help="Per-rank symmetric heap size override in MiB. Must cover the estimated requirement.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(default_pattern_benchmark_path()),
        help="Structured JSON output path for the latest pattern benchmark results.",
    )
    return parser.parse_args(argv)


def main():
    assert torch.cuda.device_count() >= 2, "Need >= 2 GPUs"

    args = _parse_args(sys.argv[1:])
    sizes = QUICK_SIZES if args.quick else IRIS_SIZES

    print("=== XTile Pattern Overlap Efficiency Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"Sizes: {len(sizes)} configurations")
    print()

    world_size = 2

    all_results: list[PatternResult] = []
    size_payloads: list[dict[str, object]] = []

    for M, N, K in sizes:
        required_heap = _required_heap_size(M, N, K, world_size, torch.float16)
        heap_size = required_heap
        if args.heap_size_mb is not None:
            heap_size = args.heap_size_mb * 1024 * 1024
            if heap_size < required_heap:
                raise ValueError(
                    "Requested --heap-size-mb is too small for this problem: "
                    f"need at least {required_heap / 1024 / 1024:.0f} MiB, "
                    f"got {args.heap_size_mb} MiB"
                )

        contexts = xtile.init_local(world_size=world_size, heap_size=heap_size)
        try:
            print(f"--- M={M}, N={N}, K={K} ---")
            print(
                "  heap per rank: "
                f"{heap_size / 1024 / 1024:.0f} MiB "
                f"(required {required_heap / 1024 / 1024:.0f} MiB)"
            )
            results, size_metadata = benchmark_size(
                M,
                N,
                K,
                contexts[0],
                warmup=args.warmup,
                iters=args.iters,
            )
            all_results.extend(results)
            best = min(results, key=lambda item: item.min_ms)
            size_payloads.append({
                "M": M,
                "N": N,
                "K": K,
                "local_N": N // world_size,
                "required_heap_bytes": required_heap,
                "heap_size_bytes": heap_size,
                "metadata": size_metadata,
                "results": [
                    {
                        "pattern": r.pattern,
                        "mean_ms": r.mean_ms,
                        "min_ms": r.min_ms,
                        "max_ms": r.max_ms,
                        "speedup_vs_bulk": r.speedup_vs_bulk,
                        "overlap_efficiency": r.overlap_efficiency,
                    }
                    for r in results
                ],
                "best_pattern": best.pattern,
                "best_speedup_vs_bulk": best.speedup_vs_bulk,
            })

            for r in results:
                speedup_str = f"{r.speedup_vs_bulk:.3f}x" if r.speedup_vs_bulk > 0 else "N/A"
                eff_str = f"{r.overlap_efficiency*100:.1f}%" if r.overlap_efficiency != 0 else "N/A"
                print(
                    f"  {r.pattern:25s} | "
                    f"mean={r.mean_ms:8.3f} ms | "
                    f"min={r.min_ms:8.3f} ms | "
                    f"speedup={speedup_str:>7s} | "
                    f"overlap_eff={eff_str:>6s}"
                )
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            _cleanup_contexts(contexts)

    # Summary table
    print()
    print("=" * 100)
    print("=== Summary: Best overlap pattern per size ===")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s} | {'Best Pattern':>25s} | {'Speedup':>8s} | {'Overlap Eff':>11s}")
    print("-" * 100)

    for M, N, K in sizes:
        size_results = [r for r in all_results if r.M == M and r.N == N and r.K == K]
        if size_results:
            best = min(size_results, key=lambda r: r.min_ms)
            speedup_str = f"{best.speedup_vs_bulk:.3f}x"
            eff_str = f"{best.overlap_efficiency*100:.1f}%"
            print(
                f"{M:>6d} {N:>6d} {K:>6d} | "
                f"{best.pattern:>25s} | "
                f"{speedup_str:>8s} | "
                f"{eff_str:>11s}"
            )

    # Check target: at least one pattern achieves >= 1.3x vs bulk_sync
    best_speedup = max((r.speedup_vs_bulk for r in all_results), default=0.0)
    print(f"\nBest speedup vs bulk_sync: {best_speedup:.3f}x")
    print(f"Target (>=1.3x): {'MET' if best_speedup >= 1.3 else 'NOT MET'}")

    if size_payloads:
        sample_ctx = contexts[0] if "contexts" in locals() else None
        sample_heap = sample_ctx.heap if sample_ctx is not None else None
        payload = {
            "schema_version": 1,
            "benchmark": "pattern_overlap",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "environment": {
                "gpu_name": torch.cuda.get_device_name(0),
                "visible_gpus": torch.cuda.device_count(),
                "world_size": world_size,
                "backend": sample_ctx.backend_name if sample_ctx is not None else "unknown",
                "allocator_backend": "torch_bump",
                "heap_mode": sample_heap.mode if sample_heap is not None else "unknown",
                "transport_strategy": sample_heap.transport_strategy if sample_heap is not None else "unknown",
                "layout_mode": "shard",
                "b_layout": "shard",
                "c_layout": "shard",
                "dtype": "float16",
                "quick_mode": args.quick,
                "warmup": args.warmup,
                "iters": args.iters,
            },
            "sizes": size_payloads,
            "summary": {
                "best_speedup_vs_bulk": best_speedup,
                "size_count": len(size_payloads),
            },
        }
        output_path = write_json(Path(args.output_json), payload)
        print(f"Structured results written to: {output_path}")


if __name__ == "__main__":
    main()
