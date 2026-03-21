"""High-level user-facing operations built on top of patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
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

PatternLike = str | type[Pattern] | Pattern | None

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


@dataclass(frozen=True, slots=True)
class GemmAllScatterPlan:
    """Executable plan for one GEMM + all-scatter invocation contract.

    The public API should resolve shape/layout semantics once at plan-build
    time, then reuse the same validated plan for execution. This keeps the
    user-facing contract narrow while allowing internal execution to stay
    pattern- and layout-aware.
    """

    ctx: xtile.XTileContext
    execution: PatternExecutionSpec
    pattern_name: str
    pattern_impl: Pattern = field(repr=False)
    storage_kind: str = "symmetric"

    def validate_tensors(self, A: Any, B: Any, C: Any) -> None:
        """Re-validate that tensors still match the plan's execution contract."""
        execution = resolve_pattern_execution(
            A,
            B,
            C,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            full_N=self.execution.full_N,
            b_layout=self.execution.rhs_layout,
            c_layout=self.execution.output_layout,
            storage_kind=self.storage_kind,
        )
        if execution != self.execution:
            raise ValueError(
                "Tensor contract no longer matches this GemmAllScatterPlan. "
                "Build a new plan for tensors with different logical or physical layout."
            )

    def execute(
        self,
        A: Any,
        B: Any,
        C: Any,
        *,
        validate: bool = True,
    ) -> Any:
        """Execute the pre-resolved plan.

        Validation is enabled by default for safety. Callers that repeatedly
        execute the exact same tensor contract may disable it to avoid
        redundant host-side validation overhead.
        """
        if validate:
            self.validate_tensors(A, B, C)
        self.pattern_impl.execute(A, B, C, spec=self.execution)
        return C

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for logs, debug output, or benchmark metadata."""
        return {
            "op": "gemm_allscatter",
            "pattern": self.pattern_name,
            "ctx": {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "backend": self.ctx.backend_name,
                "device": self.ctx.device,
            },
            "storage_kind": self.storage_kind,
            "execution": self.execution.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AllGatherPlan:
    """Executable plan for the high-level allgather collective."""

    ctx: xtile.XTileContext
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    block_size: int
    storage_kind: str = "symmetric"

    def validate_tensors(self, src: Any, output: Any) -> None:
        """Validate that tensors still match the plan."""
        _validate_allgather_contract(
            src,
            output,
            ctx=self.ctx,
            storage_kind=self.storage_kind,
            expected_input_shape=self.input_shape,
            expected_output_shape=self.output_shape,
            expected_block_size=self.block_size,
        )

    def execute(self, src: Any, output: Any, *, validate: bool = True) -> Any:
        """Execute the pre-resolved allgather plan."""
        if validate:
            self.validate_tensors(src, output)
        from xtile.primitives.collectives import allgather as collective_allgather

        collective_allgather(src, output, self.ctx.require_heap())
        return output

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for debug and benchmark metadata."""
        return {
            "op": "allgather",
            "ctx": {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "backend": self.ctx.backend_name,
                "device": self.ctx.device,
            },
            "storage_kind": self.storage_kind,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "block_size": self.block_size,
        }


def build_gemm_allscatter_plan(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    full_N: int | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
    pattern: PatternLike = "auto",
    hw_info: object | None = None,
    storage_kind: str = "symmetric",
) -> GemmAllScatterPlan:
    """Resolve a reusable GEMM + all-scatter execution plan.

    Public contract policy:
    - If no layout hints are supplied, treat the call as the stable
      full/full user contract.
    - Sharded expert usage must be explicit via layout hints or the
      dedicated ``gemm_allscatter_sharded(...)`` wrapper.
    """
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    resolved_b_layout, resolved_c_layout = _resolve_public_layout_contract(
        b_layout=b_layout,
        c_layout=c_layout,
    )
    execution = resolve_pattern_execution(
        A,
        B,
        C,
        rank=resolved_ctx.rank,
        world_size=resolved_ctx.world_size,
        full_N=full_N,
        b_layout=resolved_b_layout,
        c_layout=resolved_c_layout,
        storage_kind=storage_kind,
    )
    pattern_impl = _resolve_pattern_impl(
        pattern=pattern,
        ctx=resolved_ctx,
        execution=execution,
        hw_info=hw_info,
    )
    return GemmAllScatterPlan(
        ctx=resolved_ctx,
        execution=execution,
        pattern_name=pattern_impl.name,
        pattern_impl=pattern_impl,
        storage_kind=storage_kind,
    )


def gemm_allscatter(
    A,
    B,
    C,
    *,
    ctx: xtile.XTileContext | None = None,
    full_N: int | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
    pattern: PatternLike = "auto",
    hw_info: object | None = None,
    storage_kind: str = "symmetric",
) -> Any:
    """Run GEMM + all-scatter through the stable public full-buffer API.

    If ``b_layout`` / ``c_layout`` are omitted, the call is interpreted as
    the canonical public ``full/full`` contract. Expert sharded usage should
    prefer :func:`gemm_allscatter_sharded`.
    """
    plan = build_gemm_allscatter_plan(
        A,
        B,
        C,
        ctx=ctx,
        full_N=full_N,
        b_layout=b_layout,
        c_layout=c_layout,
        pattern=pattern,
        hw_info=hw_info,
        storage_kind=storage_kind,
    )
    return plan.execute(A, B, C, validate=False)


def gemm_allscatter_sharded(
    A: Any,
    B_shard: Any,
    C_shard: Any,
    *,
    full_N: int,
    ctx: xtile.XTileContext | None = None,
    pattern: PatternLike = "auto",
    hw_info: object | None = None,
    storage_kind: str = "symmetric",
) -> Any:
    """Expert API for shard/shard GEMM + all-scatter execution."""
    plan = build_gemm_allscatter_plan(
        A,
        B_shard,
        C_shard,
        ctx=ctx,
        full_N=full_N,
        b_layout="shard",
        c_layout="shard",
        pattern=pattern,
        hw_info=hw_info,
        storage_kind=storage_kind,
    )
    return plan.execute(A, B_shard, C_shard, validate=False)


def gemm_reducescatter(*args: Any, **kwargs: Any) -> Any:
    """Reserved high-level API placeholder."""
    raise NotImplementedError(
        "xtile.ops.gemm_reducescatter(...) is not wired yet. "
        "The shape/layout contract and high-level entrypoint landed first."
    )


def allgather(
    src: Any,
    output: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    storage_kind: str = "symmetric",
) -> Any:
    """Run the stable high-level allgather collective."""
    plan = build_allgather_plan(
        src,
        output,
        ctx=ctx,
        storage_kind=storage_kind,
    )
    return plan.execute(src, output, validate=False)


def build_allgather_plan(
    src: Any,
    output: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    storage_kind: str = "symmetric",
) -> AllGatherPlan:
    """Resolve a reusable plan for the high-level allgather collective."""
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    block_size = _validate_allgather_contract(
        src,
        output,
        ctx=resolved_ctx,
        storage_kind=storage_kind,
    )
    return AllGatherPlan(
        ctx=resolved_ctx,
        input_shape=tuple(int(dim) for dim in src.shape),
        output_shape=tuple(int(dim) for dim in output.shape),
        block_size=block_size,
        storage_kind=storage_kind,
    )


def _resolve_pattern_impl(
    *,
    pattern: PatternLike,
    ctx: xtile.XTileContext,
    execution: PatternExecutionSpec,
    hw_info: object | None,
) -> Pattern:
    if isinstance(pattern, Pattern):
        pattern_ctx = getattr(pattern, "ctx", None)
        if pattern_ctx is not None:
            if getattr(pattern_ctx, "rank", ctx.rank) != ctx.rank:
                raise ValueError(
                    "Pattern instance rank does not match the provided context: "
                    f"{getattr(pattern_ctx, 'rank', None)} != {ctx.rank}"
                )
            if getattr(pattern_ctx, "world_size", ctx.world_size) != ctx.world_size:
                raise ValueError(
                    "Pattern instance world_size does not match the provided context: "
                    f"{getattr(pattern_ctx, 'world_size', None)} != {ctx.world_size}"
                )
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


def _resolve_public_layout_contract(
    *,
    b_layout: str | None,
    c_layout: str | None,
) -> tuple[str, str]:
    """Resolve the public API layout contract.

    The stable user-facing API defaults to ``full/full`` when no layout
    hints are provided. Sharded execution remains available, but must be
    explicit so the call site stays readable and auditable.
    """
    if b_layout is None and c_layout is None:
        return "full", "full"
    if (b_layout is None) != (c_layout is None):
        raise ValueError(
            "b_layout and c_layout must be specified together. "
            "Use gemm_allscatter(...) for the default full/full public contract, "
            "or gemm_allscatter_sharded(...) for the shard/shard expert contract."
        )
    return b_layout, c_layout


def _validate_allgather_contract(
    src: Any,
    output: Any,
    *,
    ctx: xtile.XTileContext,
    storage_kind: str,
    expected_input_shape: tuple[int, ...] | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
    expected_block_size: int | None = None,
) -> int:
    """Validate an allgather contract and return the logical block size."""
    del storage_kind  # Reserved for future non-symmetric storage backends.

    heap = ctx.require_heap()
    if src.device != output.device:
        raise ValueError(
            f"allgather expects src/output on the same device, got {src.device} vs {output.device}"
        )
    if str(src.device) != ctx.device:
        raise ValueError(
            f"allgather tensors must reside on the attached heap device {ctx.device}, got {src.device}"
        )
    if not src.is_contiguous():
        raise ValueError("allgather currently requires src to be contiguous")
    if not output.is_contiguous():
        raise ValueError("allgather currently requires output to be contiguous")

    _require_tensor_on_heap(src, ctx=ctx, name="src")
    _require_tensor_on_heap(output, ctx=ctx, name="output")

    block_size = int(src.numel())
    expected_output_numel = block_size * ctx.world_size
    if int(output.numel()) != expected_output_numel:
        raise ValueError(
            "allgather output.numel must equal src.numel * world_size: "
            f"{output.numel()} != {block_size} * {ctx.world_size}"
        )

    input_shape = tuple(int(dim) for dim in src.shape)
    output_shape = tuple(int(dim) for dim in output.shape)
    if expected_input_shape is not None and input_shape != expected_input_shape:
        raise ValueError(
            f"allgather src shape no longer matches the plan: {input_shape} != {expected_input_shape}"
        )
    if expected_output_shape is not None and output_shape != expected_output_shape:
        raise ValueError(
            "allgather output shape no longer matches the plan: "
            f"{output_shape} != {expected_output_shape}"
        )
    if expected_block_size is not None and block_size != expected_block_size:
        raise ValueError(
            f"allgather src.numel no longer matches the plan: {block_size} != {expected_block_size}"
        )

    return block_size


def _require_tensor_on_heap(tensor: Any, *, ctx: xtile.XTileContext, name: str) -> None:
    """Ensure the tensor resides in the attached symmetric heap."""
    heap = ctx.require_heap()
    try:
        heap.get_offset(int(tensor.data_ptr()))
    except Exception as exc:
        raise ValueError(
            f"{name} must reside in the attached symmetric heap for ctx rank {ctx.rank}"
        ) from exc
