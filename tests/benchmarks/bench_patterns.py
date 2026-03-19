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

import sys
import time
from dataclasses import dataclass

import torch

from xtile.memory.symmetric_heap import SymmetricHeap
from xtile.backends import get_backend, detect_hardware


# ---------------------------------------------------------------------------
# Distributed context (matches pattern API expectations)
# ---------------------------------------------------------------------------

class DistCtx:
    """Minimal distributed context for pattern execution."""

    def __init__(self, rank: int, world_size: int, heap_bases, backend):
        self.rank = rank
        self.world_size = world_size
        self.heap_bases = heap_bases
        self.backend = backend


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


def benchmark_size(
    M: int, N: int, K: int,
    heaps: list[SymmetricHeap],
    warmup: int = 5,
    iters: int = 20,
) -> list[PatternResult]:
    """Benchmark all 4 patterns on a given problem size."""
    from xtile.patterns.bulk_sync import BulkSyncPattern
    from xtile.patterns.fused_sequential import FusedSequentialPattern
    from xtile.patterns.producer_consumer import ProducerConsumerPattern
    from xtile.patterns.wg_specialized import WGSpecializedPattern

    world_size = len(heaps)
    rank = 0
    torch.cuda.set_device(rank)

    # Resolve backend
    hw = detect_hardware()
    backend = get_backend(hw)

    bases = heaps[rank].get_heap_bases()
    ctx = DistCtx(rank=rank, world_size=world_size, heap_bases=bases, backend=backend)

    # Allocate matrices
    N_per_rank = N // world_size
    A = heaps[rank].allocate_tensor((M, K), torch.float16)
    B = heaps[rank].allocate_tensor((K, N_per_rank), torch.float16)
    C = heaps[rank].allocate_tensor((M, N_per_rank), torch.float16)

    A.normal_()
    B.normal_()
    C.zero_()
    torch.cuda.synchronize(rank)

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
                pattern.execute(A, B, C)
            torch.cuda.synchronize(rank)

            # Timed iterations
            times: list[float] = []
            for _ in range(iters):
                C.zero_()
                start = time.perf_counter()
                pattern.execute(A, B, C)
                torch.cuda.synchronize(rank)
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

    return results


def main():
    assert torch.cuda.device_count() >= 2, "Need >= 2 GPUs"

    quick = "--quick" in sys.argv
    sizes = QUICK_SIZES if quick else IRIS_SIZES

    print("=== XTile Pattern Overlap Efficiency Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"Sizes: {len(sizes)} configurations")
    print()

    # Create heaps large enough for all sizes
    heap_size = 512 * 1024 * 1024  # 512 MB
    world_size = 2

    all_results: list[PatternResult] = []

    for M, N, K in sizes:
        heaps = SymmetricHeap.create_all(size=heap_size, world_size=world_size)
        try:
            print(f"--- M={M}, N={N}, K={K} ---")
            results = benchmark_size(M, N, K, heaps, warmup=3, iters=10)
            all_results.extend(results)

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
            for h in heaps:
                h.cleanup()

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


if __name__ == "__main__":
    main()
