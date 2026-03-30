# SPDX-License-Identifier: Apache-2.0
"""
tncc.patterns - Pre-built compute-communication overlap patterns.

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

    from tncc.patterns import auto_select

    pattern = auto_select("gemm_allscatter", M=4096, N=8192, K=4096,
                          world_size=8)
    pattern.execute(A, B, C)

Or benchmark all patterns on a given problem::

    from tncc.patterns import benchmark_all_patterns
    results = benchmark_all_patterns(A, B, C, ctx)
"""

from __future__ import annotations

import abc
import time
from typing import TYPE_CHECKING, Any, Dict

from tncc.utils.feature_gates import (
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_transport_supported,
)

if TYPE_CHECKING:
    import torch

    from tncc.patterns.contracts import PatternExecutionSpec


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
                :meth:`~tncc.memory.symmetric_heap.SymmetricHeap.get_heap_bases`),
                and a ``backend`` implementing
                :class:`~tncc.backends.base.BackendInterface`.
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
            B: Input tensor of shape ``(K, N)`` or ``(K, N_per_rank)``
               depending on the explicit execution contract.
            C: Output tensor of shape ``(M, N)`` or ``(M, N_per_rank)``.
            **kwargs: Contract metadata such as ``spec``, ``full_N``,
               ``b_layout`` and ``c_layout``.
        """

    def benchmark(
        self,
        A: "torch.Tensor",
        B: "torch.Tensor",
        C: "torch.Tensor",
        warmup: int = 10,
        iters: int = 100,
        **execute_kwargs: Any,
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
            self.execute(A, B, C, **execute_kwargs)
        torch.cuda.synchronize()

        # Timed iterations
        times: list[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            self.execute(A, B, C, **execute_kwargs)
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

    def resolve_execution(
        self,
        A: "torch.Tensor",
        B: "torch.Tensor",
        C: "torch.Tensor",
        *,
        spec: "PatternExecutionSpec | None" = None,
        full_N: int | None = None,
        b_layout: str | None = None,
        c_layout: str | None = None,
        storage_kind: str = "symmetric",
    ) -> "PatternExecutionSpec":
        """Resolve or validate a canonical execution contract."""
        if spec is not None:
            return spec
        from tncc.patterns.contracts import resolve_pattern_execution

        return resolve_pattern_execution(
            A,
            B,
            C,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            full_N=full_N,
            b_layout=b_layout,
            c_layout=c_layout,
            storage_kind=storage_kind,
        )

    def require_device_remote_access_runtime(self, *, operation: str) -> None:
        """Fail fast when the attached runtime cannot safely run remote Triton access."""
        heap = self.ctx.require_heap()
        if heap.mode != "multiprocess":
            return
        if multiprocess_device_remote_access_transport_supported(
            heap.transport_strategy
        ):
            return
        raise ValueError(
            multiprocess_device_remote_access_detail(
                transport_strategy=heap.transport_strategy,
                operation=operation,
            )
        )


# ---------------------------------------------------------------------------
# Public imports -- concrete patterns and utilities
# ---------------------------------------------------------------------------

from tncc.patterns.auto_select import auto_select, benchmark_all_patterns
from tncc.patterns.bulk_sync import BulkSyncPattern
from tncc.patterns.contracts import (
    PatternExecutionSpec,
    PatternTensorSpec,
    resolve_pattern_execution,
)
from tncc.patterns.fused_sequential import FusedSequentialPattern
from tncc.patterns.producer_consumer import ProducerConsumerPattern
from tncc.patterns.wg_specialized import WGSpecializedPattern

__all__ = [
    "Pattern",
    "BulkSyncPattern",
    "FusedSequentialPattern",
    "ProducerConsumerPattern",
    "WGSpecializedPattern",
    "PatternExecutionSpec",
    "PatternTensorSpec",
    "resolve_pattern_execution",
    "auto_select",
    "benchmark_all_patterns",
]
