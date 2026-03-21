"""High-level user-facing operations built on top of patterns."""

from __future__ import annotations

from typing import Any

import xtile
from xtile.patterns import (
    BulkSyncPattern,
    FusedSequentialPattern,
    Pattern,
    ProducerConsumerPattern,
    WGSpecializedPattern,
)
from xtile.patterns.auto_select import auto_select
from xtile.patterns.contracts import PatternExecutionSpec, resolve_pattern_execution

_PATTERN_ALIASES = {
    "auto": None,
    "bulk_sync": BulkSyncPattern,
    "fused_sequential": FusedSequentialPattern,
    "fused_seq": FusedSequentialPattern,
    "producer_consumer": ProducerConsumerPattern,
    "pc": ProducerConsumerPattern,
    "wg_specialized": WGSpecializedPattern,
    "wg_spec": WGSpecializedPattern,
}


def gemm_allscatter(
    A,
    B,
    C,
    *,
    ctx: xtile.XTileContext | None = None,
    full_N: int | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
    pattern: str | type[Pattern] | Pattern | None = "auto",
    hw_info: object | None = None,
    storage_kind: str = "symmetric",
) -> Any:
    """Run GEMM + all-scatter through a stable high-level entrypoint."""
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    execution = resolve_pattern_execution(
        A,
        B,
        C,
        rank=resolved_ctx.rank,
        world_size=resolved_ctx.world_size,
        full_N=full_N,
        b_layout=b_layout,
        c_layout=c_layout,
        storage_kind=storage_kind,
    )
    pattern_impl = _resolve_pattern_impl(
        pattern=pattern,
        ctx=resolved_ctx,
        execution=execution,
        hw_info=hw_info,
    )
    pattern_impl.execute(A, B, C, spec=execution)
    return C


def gemm_reducescatter(*args: Any, **kwargs: Any) -> Any:
    """Reserved high-level API placeholder."""
    raise NotImplementedError(
        "xtile.ops.gemm_reducescatter(...) is not wired yet. "
        "The shape/layout contract and high-level entrypoint landed first."
    )


def allgather(*args: Any, **kwargs: Any) -> Any:
    """Reserved high-level API placeholder."""
    raise NotImplementedError(
        "xtile.ops.allgather(...) is not wired yet. "
        "The shape/layout contract and high-level entrypoint landed first."
    )


def _resolve_pattern_impl(
    *,
    pattern: str | type[Pattern] | Pattern | None,
    ctx: xtile.XTileContext,
    execution: PatternExecutionSpec,
    hw_info: object | None,
) -> Pattern:
    if isinstance(pattern, Pattern):
        return pattern
    if pattern is None or pattern == "auto":
        return auto_select(
            "gemm_allscatter",
            M=execution.M,
            N=execution.full_N,
            K=execution.K,
            world_size=execution.world_size,
            hw_info=hw_info,
            ctx=ctx,
        )
    if isinstance(pattern, str):
        pattern_cls = _PATTERN_ALIASES.get(pattern)
        if pattern_cls is None:
            raise ValueError(
                f"Unknown pattern {pattern!r}. Supported: {sorted(_PATTERN_ALIASES)}"
            )
        return pattern_cls(ctx)
    if isinstance(pattern, type) and issubclass(pattern, Pattern):
        return pattern(ctx)
    raise TypeError(
        "pattern must be a pattern name, Pattern subclass, Pattern instance, or 'auto'"
    )
