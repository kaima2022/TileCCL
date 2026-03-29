# SPDX-License-Identifier: Apache-2.0
"""Compare all four overlap patterns on a single problem size.

Runs each pattern (BulkSync, FusedSequential, ProducerConsumer,
WG-Specialized) on a GEMM + AllScatter workload and prints timing.

Requirements: 2x NVIDIA GPUs with NVLink.

Usage:
    python examples/benchmark_patterns.py
"""

from __future__ import annotations

import time

import torch

import tncc
from tncc.ops import build_gemm_allscatter_plan
from tncc.patterns import (
    BulkSyncPattern,
    FusedSequentialPattern,
    ProducerConsumerPattern,
    WGSpecializedPattern,
)

PATTERNS = [
    ("BulkSync", BulkSyncPattern),
    ("FusedSequential", FusedSequentialPattern),
    ("ProducerConsumer", ProducerConsumerPattern),
    ("WG-Specialized", WGSpecializedPattern),
]


def main() -> None:
    ctxs = tncc.init_local(world_size=2, heap_size=1 << 30)
    ctx = ctxs[0]

    M, K, N = 8192, 36864, 4608 * ctx.world_size
    warmup, iters = 10, 50

    A = ctx.randn(M, K, dtype=torch.float16)
    B = ctx.randn(K, N, dtype=torch.float16)
    C = ctx.zeros(M, N, dtype=torch.float16)

    print(f"Problem: M={M}, K={K}, N={N}, world_size={ctx.world_size}")
    print(f"Warmup: {warmup}, Iterations: {iters}")
    print("-" * 52)

    results: list[tuple[str, float]] = []

    for name, pattern_cls in PATTERNS:
        plan = build_gemm_allscatter_plan(
            A, B, C, ctx=ctx, pattern=pattern_cls,
        )

        # Warmup.
        for _ in range(warmup):
            plan.execute(A, B, C)
        torch.cuda.synchronize()

        # Timed iterations.
        t0 = time.perf_counter()
        for _ in range(iters):
            plan.execute(A, B, C)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / iters * 1000

        results.append((name, elapsed_ms))
        print(f"  {name:<22s}  {elapsed_ms:8.2f} ms")

    # Summary.
    baseline = results[0][1]
    print("-" * 52)
    for name, ms in results:
        speedup = baseline / ms
        print(f"  {name:<22s}  {speedup:.3f}x vs BulkSync")


if __name__ == "__main__":
    main()
