"""
xtile.patterns - Pre-built compute-communication overlap patterns.

This module provides a library of overlap patterns inspired by the Iris
taxonomy (ASPLOS 2024).  Each pattern implements a different strategy for
overlapping GEMM computation with all-scatter (or all-gather) communication,
ranging from simple bulk-synchronous execution to fine-grained workgroup
specialization.

Pattern hierarchy (increasing overlap complexity):
    1. BulkSyncPattern        -- no overlap, baseline reference
    2. FusedSequentialPattern  -- tile-level sequential overlap
    3. ProducerConsumerPattern -- tile-level parallel overlap (dual stream)
    4. WGSpecializedPattern    -- CU/SM-level parallel overlap (single kernel)

Typical usage::

    from xtile.patterns import auto_select

    pattern = auto_select("gemm_allscatter", M=4096, N=8192, K=4096,
                          world_size=8)
    pattern.execute(A, B, C)

Or benchmark all patterns on a given problem::

    from xtile.patterns import benchmark_all_patterns
    results = benchmark_all_patterns(A, B, C, ctx)
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Pattern(abc.ABC):
    """Abstract base class for all overlap patterns.

    Every concrete pattern must:
    * Set :attr:`name` to a human-readable identifier.
    * Accept a *ctx* object in ``__init__`` that carries distributed context
      (rank, world_size, remote pointers, backend, etc.).
    * Implement :meth:`execute` which performs the fused GEMM + communication.

    The optional :meth:`benchmark` helper runs the pattern with a
    warm-up phase and returns timing statistics.
    """

    name: str = "base"

    def __init__(self, ctx: Any) -> None:
        """Initialize the pattern with a distributed context.

        Args:
            ctx: Distributed context object carrying at minimum
                ``rank``, ``world_size``, ``heap_bases`` (from
                :meth:`~xtile.memory.symmetric_heap.SymmetricHeap.get_heap_bases`),
                and a ``backend`` implementing
                :class:`~xtile.backends.base.BackendInterface`.
        """
        self.ctx = ctx

    @abc.abstractmethod
    def execute(self, A: "torch.Tensor", B: "torch.Tensor",
                C: "torch.Tensor", **kwargs: Any) -> None:
        """Run the fused GEMM + communication pattern.

        Computes ``C = A @ B`` (distributed) and scatters/gathers the
        result tiles to peer GPUs according to the pattern's strategy.

        Args:
            A: Input tensor of shape ``(M, K)`` on the local device.
            B: Input tensor of shape ``(K, N)`` on the local device.
            C: Output tensor of shape ``(M, N)`` (or the local shard)
               on the local device.  Written in-place.
            **kwargs: Pattern-specific overrides (e.g. block sizes).
        """

    def benchmark(
        self,
        A: "torch.Tensor",
        B: "torch.Tensor",
        C: "torch.Tensor",
        warmup: int = 10,
        iters: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark this pattern and return timing statistics.

        Args:
            A: Input tensor of shape ``(M, K)``.
            B: Input tensor of shape ``(K, N)``.
            C: Output tensor (written in-place each iteration).
            warmup: Number of warm-up iterations (not timed).
            iters: Number of timed iterations.

        Returns:
            A dict with keys ``"pattern"``, ``"mean_ms"``, ``"min_ms"``,
            ``"max_ms"``, and ``"iters"``.
        """
        import torch  # runtime import -- only needed when actually benchmarking

        # Warm-up
        for _ in range(warmup):
            self.execute(A, B, C)
        torch.cuda.synchronize()

        # Timed iterations
        times: list[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            self.execute(A, B, C)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1e3)  # ms

        return {
            "pattern": self.name,
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "iters": iters,
        }


# ---------------------------------------------------------------------------
# Public imports -- concrete patterns and utilities
# ---------------------------------------------------------------------------

from xtile.patterns.bulk_sync import BulkSyncPattern
from xtile.patterns.fused_sequential import FusedSequentialPattern
from xtile.patterns.producer_consumer import ProducerConsumerPattern
from xtile.patterns.wg_specialized import WGSpecializedPattern
from xtile.patterns.auto_select import auto_select, benchmark_all_patterns

__all__ = [
    "Pattern",
    "BulkSyncPattern",
    "FusedSequentialPattern",
    "ProducerConsumerPattern",
    "WGSpecializedPattern",
    "auto_select",
    "benchmark_all_patterns",
]
