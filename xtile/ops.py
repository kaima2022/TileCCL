"""High-level user-facing operations built on top of patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
from xtile.utils.feature_gates import (
    multiprocess_device_collectives_detail,
    multiprocess_device_collectives_enabled,
    multiprocess_device_collectives_transport_supported,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_transport_supported,
)

PatternLike = str | type[Pattern] | Pattern | None
LayoutKind = Literal["full", "shard"]

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

_PENDING_GEMM_REDUCESCATTER_SINGLE_PROCESS: dict[
    tuple[object, ...],
    dict[int, tuple["GemmReduceScatterPlan", Any, Any]],
] = {}
_PENDING_GEMM_ALLGATHER_SINGLE_PROCESS: dict[
    tuple[object, ...],
    dict[int, tuple["GemmAllGatherPlan", Any, Any, Any]],
] = {}


@dataclass(frozen=True, slots=True)
class GemmAllScatterContract:
    """Resolved public contract for one high-level GEMM + all-scatter call."""

    M: int
    K: int
    full_N: int
    rhs_cols: int
    output_cols: int
    rank: int
    world_size: int
    rhs_layout: LayoutKind
    output_layout: LayoutKind
    storage_kind: str = "symmetric"

    def to_dict(self) -> dict[str, object]:
        """Serialize the contract for logs, docs, and debug output."""
        return {
            "M": self.M,
            "K": self.K,
            "full_N": self.full_N,
            "rhs_cols": self.rhs_cols,
            "output_cols": self.output_cols,
            "rank": self.rank,
            "world_size": self.world_size,
            "rhs_layout": self.rhs_layout,
            "output_layout": self.output_layout,
            "storage_kind": self.storage_kind,
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
class GemmAllScatterMixedLayoutPlan:
    """Host wrapper for mixed public GEMM + all-scatter layout contracts."""

    ctx: xtile.XTileContext
    contract: GemmAllScatterContract
    pattern_name: str
    direct_plan: GemmAllScatterPlan = field(repr=False)
    materialization: str = "full_to_shard"
    storage_kind: str = "symmetric"

    def validate_tensors(self, A: Any, B: Any, C: Any) -> None:
        """Re-validate that tensors still match the public mixed-layout contract."""
        contract = _resolve_gemm_allscatter_contract(
            A,
            B,
            C,
            ctx=self.ctx,
            full_N=self.contract.full_N,
            b_layout=self.contract.rhs_layout,
            c_layout=self.contract.output_layout,
            storage_kind=self.storage_kind,
        )
        if contract != self.contract:
            raise ValueError(
                "Tensor contract no longer matches this mixed-layout GemmAllScatter plan. "
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
        """Execute the mixed-layout wrapper via a heap-backed internal plan."""
        if validate:
            self.validate_tensors(A, B, C)
        if self.materialization == "full_to_shard":
            return self._execute_full_to_shard(A, B, C)
        raise RuntimeError(
            f"Unsupported mixed-layout materialization {self.materialization!r}"
        )

    def _execute_full_to_shard(self, A: Any, B: Any, C: Any) -> Any:
        """Materialize a full symmetric output, then expose the local shard."""
        if self.contract.world_size == 1:
            self.direct_plan.execute(A, B, C, validate=False)
            return C

        full_output = self.ctx.workspace(
            "gemm_allscatter.full_output",
            self.contract.M,
            self.contract.full_N,
            dtype=C.dtype,
        )
        self.direct_plan.execute(A, B, full_output, validate=False)

        shard_cols = self.contract.output_cols
        shard_offset = 0 if self.contract.world_size == 1 else self.contract.rank * shard_cols
        C.copy_(full_output[:, shard_offset:shard_offset + shard_cols])
        return C

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for debug output and benchmark metadata."""
        return {
            "op": "gemm_allscatter",
            "materialization": self.materialization,
            "pattern": self.pattern_name,
            "ctx": {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "backend": self.ctx.backend_name,
                "device": self.ctx.device,
            },
            "storage_kind": self.storage_kind,
            "public_contract": self.contract.to_dict(),
            "direct_plan": self.direct_plan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GemmAllGatherContract:
    """Resolved public contract for one high-level GEMM + allgather call."""

    M: int
    K: int
    full_N: int
    shard_cols: int
    rank: int
    world_size: int
    storage_kind: str = "symmetric"

    def to_dict(self) -> dict[str, object]:
        """Serialize the contract for docs, logs, and debug output."""
        return {
            "M": self.M,
            "K": self.K,
            "full_N": self.full_N,
            "shard_cols": self.shard_cols,
            "rank": self.rank,
            "world_size": self.world_size,
            "storage_kind": self.storage_kind,
        }


@dataclass(frozen=True, slots=True)
class GemmAllGatherPlan:
    """Executable host-side plan for GEMM + allgather."""

    ctx: xtile.XTileContext
    contract: GemmAllGatherContract
    allgather_plan: "AllGatherPlan" = field(repr=False)
    materialization: str = "local_gemm_plus_allgather"
    pack_layout: str = "rank_major_column_shards"
    storage_kind: str = "symmetric"

    def validate_tensors(self, A: Any, B_shard: Any, C: Any) -> None:
        """Re-validate that tensors still match the resolved public contract."""
        contract = _resolve_gemm_allgather_contract(
            A,
            B_shard,
            C,
            ctx=self.ctx,
            storage_kind=self.storage_kind,
        )
        if contract != self.contract:
            raise ValueError(
                "Tensor contract no longer matches this GemmAllGatherPlan. "
                "Build a new plan for tensors with different logical shapes or dtypes."
            )

    def execute(
        self,
        A: Any,
        B_shard: Any,
        C: Any,
        *,
        validate: bool = True,
    ) -> Any:
        """Execute the pre-resolved host GEMM + allgather plan."""
        if validate:
            self.validate_tensors(A, B_shard, C)

        local_shard = self.ctx.workspace(
            "gemm_allgather.local_output_shard",
            self.contract.M,
            self.contract.shard_cols,
            dtype=C.dtype,
        )
        _run_local_gemm(A, B_shard, out=local_shard)
        gather_output = C
        if self.contract.world_size > 1:
            gather_output = self.ctx.workspace(
                "gemm_allgather.gathered_output_shards",
                self.contract.world_size,
                self.contract.M,
                self.contract.shard_cols,
                dtype=C.dtype,
            )
        if (
            self.contract.world_size > 1
            and self.ctx.require_heap().mode == "single_process"
        ):
            return _execute_gemm_allgather_single_process(
                self,
                local_shard=local_shard,
                gather_output=gather_output,
                output=C,
            )

        self.allgather_plan.execute(local_shard, gather_output, validate=False)
        if self.contract.world_size > 1:
            self.ctx.barrier()
            C.copy_(
                gather_output.permute(1, 0, 2).reshape(
                    self.contract.M,
                    self.contract.full_N,
                )
            )
        return C

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for docs, debug output, and benchmark metadata."""
        return {
            "op": "gemm_allgather",
            "ctx": {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "backend": self.ctx.backend_name,
                "device": self.ctx.device,
            },
            "storage_kind": self.storage_kind,
            "materialization": self.materialization,
            "pack_layout": self.pack_layout,
            "contract": self.contract.to_dict(),
            "allgather_plan": self.allgather_plan.to_dict(),
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


@dataclass(frozen=True, slots=True)
class ReduceScatterPlan:
    """Executable plan for the high-level reduce_scatter collective."""

    ctx: xtile.XTileContext
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    block_size: int
    implementation: str = "auto"
    storage_kind: str = "symmetric"

    def validate_tensors(self, src: Any, output: Any) -> None:
        """Validate that tensors still match the plan."""
        _validate_reduce_scatter_contract(
            src,
            output,
            ctx=self.ctx,
            storage_kind=self.storage_kind,
            expected_input_shape=self.input_shape,
            expected_output_shape=self.output_shape,
            expected_block_size=self.block_size,
        )

    def execute(self, src: Any, output: Any, *, validate: bool = True) -> Any:
        """Execute the pre-resolved reduce_scatter plan."""
        if validate:
            self.validate_tensors(src, output)
        from xtile.primitives.collectives import reduce_scatter as collective_reduce_scatter

        collective_reduce_scatter(
            src,
            output,
            self.ctx.require_heap(),
            implementation=self.implementation,
        )
        return output

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for debug and benchmark metadata."""
        return {
            "op": "reduce_scatter",
            "ctx": {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "backend": self.ctx.backend_name,
                "device": self.ctx.device,
            },
            "storage_kind": self.storage_kind,
            "implementation": self.implementation,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "block_size": self.block_size,
        }


@dataclass(frozen=True, slots=True)
class GemmReduceScatterContract:
    """Resolved public contract for one high-level GEMM + reduce-scatter call."""

    M: int
    K: int
    full_N: int
    output_cols: int
    rank: int
    world_size: int
    storage_kind: str = "symmetric"

    def to_dict(self) -> dict[str, object]:
        """Serialize the contract for logs, docs, and debug output."""
        return {
            "M": self.M,
            "K": self.K,
            "full_N": self.full_N,
            "output_cols": self.output_cols,
            "rank": self.rank,
            "world_size": self.world_size,
            "storage_kind": self.storage_kind,
        }


@dataclass(frozen=True, slots=True)
class GemmReduceScatterPlan:
    """Executable host-side plan for GEMM + reduce-scatter."""

    ctx: xtile.XTileContext
    contract: GemmReduceScatterContract
    reduce_scatter_plan: ReduceScatterPlan = field(repr=False)
    implementation: str = "auto"
    pack_layout: str = "rank_major_column_shards"
    storage_kind: str = "symmetric"

    def validate_tensors(self, A: Any, B: Any, C: Any) -> None:
        """Re-validate that tensors still match the resolved public contract."""
        contract = _resolve_gemm_reducescatter_contract(
            A,
            B,
            C,
            ctx=self.ctx,
            storage_kind=self.storage_kind,
        )
        if contract != self.contract:
            raise ValueError(
                "Tensor contract no longer matches this GemmReduceScatterPlan. "
                "Build a new plan for tensors with different logical shapes or dtypes."
            )

    def execute(
        self,
        A: Any,
        B: Any,
        C: Any,
        *,
        validate: bool = True,
    ) -> Any:
        """Execute the pre-resolved host GEMM + reduce-scatter plan."""
        if validate:
            self.validate_tensors(A, B, C)

        local_full = self.ctx.workspace(
            "gemm_reducescatter.local_full_output",
            self.contract.M,
            self.contract.full_N,
            dtype=C.dtype,
        )
        _run_local_gemm(A, B, out=local_full)

        reduce_src = local_full
        if self.contract.world_size > 1:
            packed = self.ctx.workspace(
                "gemm_reducescatter.packed_input",
                self.contract.world_size,
                self.contract.M,
                self.contract.output_cols,
                dtype=C.dtype,
            )
            packed.copy_(
                local_full.view(
                    self.contract.M,
                    self.contract.world_size,
                    self.contract.output_cols,
                ).permute(1, 0, 2)
            )
            reduce_src = packed

        if (
            self.contract.world_size > 1
            and self.ctx.require_heap().mode == "single_process"
        ):
            return _execute_gemm_reducescatter_single_process(
                self,
                reduce_src=reduce_src,
                output=C,
            )

        self.reduce_scatter_plan.execute(reduce_src, C, validate=False)
        return C

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for debug output and benchmark metadata."""
        return {
            "op": "gemm_reducescatter",
            "ctx": {
                "rank": self.ctx.rank,
                "world_size": self.ctx.world_size,
                "backend": self.ctx.backend_name,
                "device": self.ctx.device,
            },
            "storage_kind": self.storage_kind,
            "implementation": self.implementation,
            "pack_layout": self.pack_layout,
            "contract": self.contract.to_dict(),
            "reduce_scatter_plan": self.reduce_scatter_plan.to_dict(),
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
) -> GemmAllScatterPlan | GemmAllScatterMixedLayoutPlan:
    """Resolve a reusable GEMM + all-scatter execution plan.

    Public contract policy:
    - If no layout hints are supplied, treat the call as the stable
      full/full user contract.
    - Sharded expert usage must be explicit via layout hints or the
      dedicated ``gemm_allscatter_sharded(...)`` wrapper.
    """
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    _require_gemm_allscatter_runtime(resolved_ctx)
    resolved_b_layout, resolved_c_layout = _resolve_public_layout_contract(
        b_layout=b_layout,
        c_layout=c_layout,
    )
    contract = _resolve_gemm_allscatter_contract(
        A,
        B,
        C,
        ctx=resolved_ctx,
        full_N=full_N,
        b_layout=resolved_b_layout,
        c_layout=resolved_c_layout,
        storage_kind=storage_kind,
    )

    if contract.rhs_layout == "full" and contract.output_layout == "shard":
        full_output = _shape_only_tensor((contract.M, contract.full_N))
        direct_execution = resolve_pattern_execution(
            A,
            B,
            full_output,
            rank=resolved_ctx.rank,
            world_size=resolved_ctx.world_size,
            full_N=contract.full_N,
            b_layout="full",
            c_layout="full",
            storage_kind=storage_kind,
        )
        pattern_impl = _resolve_pattern_impl(
            pattern=pattern,
            ctx=resolved_ctx,
            execution=direct_execution,
            hw_info=hw_info,
        )
        return GemmAllScatterMixedLayoutPlan(
            ctx=resolved_ctx,
            contract=contract,
            pattern_name=pattern_impl.name,
            direct_plan=GemmAllScatterPlan(
                ctx=resolved_ctx,
                execution=direct_execution,
                pattern_name=pattern_impl.name,
                pattern_impl=pattern_impl,
                storage_kind=storage_kind,
            ),
            materialization="full_to_shard",
            storage_kind=storage_kind,
        )
    if contract.rhs_layout != contract.output_layout:
        raise ValueError(
            "Mixed multi-rank layout wrapper is only implemented for b_layout='full', "
            "c_layout='shard' right now. "
            "The inverse shard/full case is intentionally still rejected because "
            "current gemm_allscatter shard/shard execution does not define a stable "
            "local-shard ownership contract for assembling a full output; that path "
            "belongs to a future gemm_allgather-style API. "
            f"Got b_layout={contract.rhs_layout!r}, c_layout={contract.output_layout!r}."
        )

    execution = resolve_pattern_execution(
        A,
        B,
        C,
        rank=resolved_ctx.rank,
        world_size=resolved_ctx.world_size,
        full_N=contract.full_N,
        b_layout=contract.rhs_layout,
        c_layout=contract.output_layout,
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


def build_gemm_allgather_plan(
    A: Any,
    B_shard: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    storage_kind: str = "symmetric",
) -> GemmAllGatherPlan:
    """Resolve a reusable host-side plan for GEMM + allgather.

    Public contract:
    - ``A``: full LHS matrix of shape ``(M, K)``
    - ``B_shard``: rank-local RHS column shard of shape ``(K, N / world_size)``
    - ``C``: full output matrix of shape ``(M, N)``

    The current implementation is intentionally conservative. It performs one
    local GEMM into a heap-backed shard workspace and then reuses the validated
    high-level :func:`allgather` contract to assemble the full output.
    """
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    _require_device_remote_access_runtime(
        resolved_ctx,
        operation="xtile.ops.gemm_allgather(...)",
    )
    contract = _resolve_gemm_allgather_contract(
        A,
        B_shard,
        C,
        ctx=resolved_ctx,
        storage_kind=storage_kind,
    )
    allgather_plan = AllGatherPlan(
        ctx=resolved_ctx,
        input_shape=(contract.M, contract.shard_cols),
        output_shape=(
            tuple(int(dim) for dim in C.shape)
            if contract.world_size == 1
            else (contract.world_size, contract.M, contract.shard_cols)
        ),
        block_size=contract.M * contract.shard_cols,
        storage_kind=storage_kind,
    )
    return GemmAllGatherPlan(
        ctx=resolved_ctx,
        contract=contract,
        allgather_plan=allgather_plan,
        storage_kind=storage_kind,
    )


def gemm_allgather(
    A: Any,
    B_shard: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    storage_kind: str = "symmetric",
) -> Any:
    """Run GEMM + allgather through the stable high-level host contract."""
    plan = build_gemm_allgather_plan(
        A,
        B_shard,
        C,
        ctx=ctx,
        storage_kind=storage_kind,
    )
    return plan.execute(A, B_shard, C, validate=False)


def build_gemm_reducescatter_plan(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    implementation: str = "auto",
    storage_kind: str = "symmetric",
) -> GemmReduceScatterPlan:
    """Resolve a reusable host-side plan for GEMM + reduce-scatter.

    Public contract:
    - ``A``: local rank contribution of shape ``(M, K)``
    - ``B``: full RHS matrix of shape ``(K, N)``
    - ``C``: rank-local output shard of shape ``(M, N / world_size)``

    The current implementation is intentionally conservative. It performs
    local GEMM into a heap-backed workspace, repacks the full result into a
    rank-major contiguous reduce-scatter input, then reuses the validated
    high-level :func:`reduce_scatter` contract.
    """
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    contract = _resolve_gemm_reducescatter_contract(
        A,
        B,
        C,
        ctx=resolved_ctx,
        storage_kind=storage_kind,
    )
    resolved_implementation = _resolve_reduce_scatter_implementation(
        resolved_ctx,
        implementation=implementation,
    )
    reduce_input_shape = (
        (contract.M, contract.full_N)
        if contract.world_size == 1
        else (contract.world_size, contract.M, contract.output_cols)
    )
    reduce_scatter_plan = ReduceScatterPlan(
        ctx=resolved_ctx,
        input_shape=reduce_input_shape,
        output_shape=tuple(int(dim) for dim in C.shape),
        block_size=int(C.numel()),
        implementation=resolved_implementation,
        storage_kind=storage_kind,
    )
    return GemmReduceScatterPlan(
        ctx=resolved_ctx,
        contract=contract,
        reduce_scatter_plan=reduce_scatter_plan,
        implementation=resolved_implementation,
        storage_kind=storage_kind,
    )


def gemm_reducescatter(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    implementation: str = "auto",
    storage_kind: str = "symmetric",
) -> Any:
    """Run GEMM + reduce-scatter through the stable high-level host contract."""
    plan = build_gemm_reducescatter_plan(
        A,
        B,
        C,
        ctx=ctx,
        implementation=implementation,
        storage_kind=storage_kind,
    )
    return plan.execute(A, B, C, validate=False)


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


def reduce_scatter(
    src: Any,
    output: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    implementation: str = "auto",
    storage_kind: str = "symmetric",
) -> Any:
    """Run the stable high-level reduce_scatter collective."""
    plan = build_reduce_scatter_plan(
        src,
        output,
        ctx=ctx,
        implementation=implementation,
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
    _require_device_remote_access_runtime(
        resolved_ctx,
        operation="xtile.ops.allgather(...)",
    )
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


def build_reduce_scatter_plan(
    src: Any,
    output: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    implementation: str = "auto",
    storage_kind: str = "symmetric",
) -> ReduceScatterPlan:
    """Resolve a reusable plan for the high-level reduce_scatter collective."""
    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    block_size = _validate_reduce_scatter_contract(
        src,
        output,
        ctx=resolved_ctx,
        storage_kind=storage_kind,
    )
    implementation = _resolve_reduce_scatter_implementation(
        resolved_ctx,
        implementation=implementation,
    )
    return ReduceScatterPlan(
        ctx=resolved_ctx,
        input_shape=tuple(int(dim) for dim in src.shape),
        output_shape=tuple(int(dim) for dim in output.shape),
        block_size=block_size,
        implementation=implementation,
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


def _resolve_gemm_allscatter_contract(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext,
    full_N: int | None,
    b_layout: LayoutKind,
    c_layout: LayoutKind,
    storage_kind: str,
) -> GemmAllScatterContract:
    """Resolve and validate the public high-level GEMM + all-scatter contract."""
    ctx.require_heap()
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError(
            "gemm_allscatter expects 2-D tensors: "
            f"got A.ndim={A.ndim}, B.ndim={B.ndim}, C.ndim={C.ndim}"
        )
    if str(A.device) != ctx.device or str(B.device) != ctx.device or str(C.device) != ctx.device:
        raise ValueError(
            "gemm_allscatter tensors must all reside on the context device "
            f"{ctx.device}, got A={A.device}, B={B.device}, C={C.device}"
        )

    M, K = int(A.shape[0]), int(A.shape[1])
    b_k, b_cols = int(B.shape[0]), int(B.shape[1])
    c_m, c_cols = int(C.shape[0]), int(C.shape[1])
    if b_k != K:
        raise ValueError(f"B.shape[0] must equal A.shape[1]: got {b_k} vs {K}")
    if c_m != M:
        raise ValueError(f"C.shape[0] must equal A.shape[0]: got {c_m} vs {M}")

    if full_N is None:
        full_candidates = {
            _full_N_from_layout(cols=b_cols, layout=b_layout, world_size=ctx.world_size),
            _full_N_from_layout(cols=c_cols, layout=c_layout, world_size=ctx.world_size),
        }
        if len(full_candidates) != 1:
            raise ValueError(
                "Provided layout hints disagree on full_N: "
                f"resolved candidates={sorted(full_candidates)}"
            )
        resolved_full_N = next(iter(full_candidates))
    else:
        resolved_full_N = int(full_N)

    if resolved_full_N <= 0:
        raise ValueError(f"full_N must be positive, got {resolved_full_N}")
    if resolved_full_N % ctx.world_size != 0:
        raise ValueError(
            f"full_N must be divisible by world_size: {resolved_full_N} % {ctx.world_size} != 0"
        )

    shard_cols = resolved_full_N if ctx.world_size == 1 else resolved_full_N // ctx.world_size
    expected_b_cols = resolved_full_N if b_layout == "full" else shard_cols
    expected_c_cols = resolved_full_N if c_layout == "full" else shard_cols
    if b_cols != expected_b_cols:
        raise ValueError(
            f"B layout={b_layout!r} expects {expected_b_cols} columns, got {b_cols}"
        )
    if c_cols != expected_c_cols:
        raise ValueError(
            f"C layout={c_layout!r} expects {expected_c_cols} columns, got {c_cols}"
        )

    return GemmAllScatterContract(
        M=M,
        K=K,
        full_N=resolved_full_N,
        rhs_cols=b_cols,
        output_cols=c_cols,
        rank=ctx.rank,
        world_size=ctx.world_size,
        rhs_layout=b_layout,
        output_layout=c_layout,
        storage_kind=storage_kind,
    )


def _resolve_gemm_reducescatter_contract(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext,
    storage_kind: str,
) -> GemmReduceScatterContract:
    """Resolve and validate the public high-level GEMM + reduce-scatter contract."""
    ctx.require_heap()
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError(
            "gemm_reducescatter expects 2-D tensors: "
            f"got A.ndim={A.ndim}, B.ndim={B.ndim}, C.ndim={C.ndim}"
        )
    if str(A.device) != ctx.device or str(B.device) != ctx.device or str(C.device) != ctx.device:
        raise ValueError(
            "gemm_reducescatter tensors must all reside on the context device "
            f"{ctx.device}, got A={A.device}, B={B.device}, C={C.device}"
        )
    if not C.is_contiguous():
        raise ValueError(
            "gemm_reducescatter currently requires C to be contiguous so the "
            "reduce_scatter output contract stays well-defined."
        )
    _require_tensor_on_heap(C, ctx=ctx, name="C")

    if A.dtype != B.dtype or B.dtype != C.dtype:
        raise ValueError(
            "gemm_reducescatter currently requires A, B, and C to share one dtype: "
            f"got A={A.dtype}, B={B.dtype}, C={C.dtype}"
        )
    if not A.is_floating_point() or not B.is_floating_point() or not C.is_floating_point():
        raise ValueError(
            "gemm_reducescatter currently requires floating-point tensors."
        )

    M, K = int(A.shape[0]), int(A.shape[1])
    b_k, full_N = int(B.shape[0]), int(B.shape[1])
    c_m, c_cols = int(C.shape[0]), int(C.shape[1])
    if b_k != K:
        raise ValueError(f"B.shape[0] must equal A.shape[1]: got {b_k} vs {K}")
    if c_m != M:
        raise ValueError(f"C.shape[0] must equal A.shape[0]: got {c_m} vs {M}")
    if full_N <= 0:
        raise ValueError(f"B.shape[1] must be positive, got {full_N}")
    if full_N % ctx.world_size != 0:
        raise ValueError(
            "gemm_reducescatter requires B.shape[1] to be divisible by world_size: "
            f"{full_N} % {ctx.world_size} != 0"
        )

    expected_output_cols = full_N if ctx.world_size == 1 else full_N // ctx.world_size
    if c_cols != expected_output_cols:
        raise ValueError(
            "gemm_reducescatter output columns must match the rank-local shard width: "
            f"expected {expected_output_cols}, got {c_cols}"
        )

    return GemmReduceScatterContract(
        M=M,
        K=K,
        full_N=full_N,
        output_cols=c_cols,
        rank=ctx.rank,
        world_size=ctx.world_size,
        storage_kind=storage_kind,
    )


def _resolve_gemm_allgather_contract(
    A: Any,
    B_shard: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext,
    storage_kind: str,
) -> GemmAllGatherContract:
    """Resolve and validate the public high-level GEMM + allgather contract."""
    ctx.require_heap()
    if A.ndim != 2 or B_shard.ndim != 2 or C.ndim != 2:
        raise ValueError(
            "gemm_allgather expects 2-D tensors: "
            f"got A.ndim={A.ndim}, B_shard.ndim={B_shard.ndim}, C.ndim={C.ndim}"
        )
    if (
        str(A.device) != ctx.device
        or str(B_shard.device) != ctx.device
        or str(C.device) != ctx.device
    ):
        raise ValueError(
            "gemm_allgather tensors must all reside on the context device "
            f"{ctx.device}, got A={A.device}, B_shard={B_shard.device}, C={C.device}"
        )
    if not C.is_contiguous():
        raise ValueError(
            "gemm_allgather currently requires C to be contiguous so the "
            "allgather output contract stays well-defined."
        )
    _require_tensor_on_heap(C, ctx=ctx, name="C")

    if A.dtype != B_shard.dtype or B_shard.dtype != C.dtype:
        raise ValueError(
            "gemm_allgather currently requires A, B_shard, and C to share one dtype: "
            f"got A={A.dtype}, B_shard={B_shard.dtype}, C={C.dtype}"
        )
    if (
        not A.is_floating_point()
        or not B_shard.is_floating_point()
        or not C.is_floating_point()
    ):
        raise ValueError("gemm_allgather currently requires floating-point tensors.")

    M, K = int(A.shape[0]), int(A.shape[1])
    b_k, shard_cols = int(B_shard.shape[0]), int(B_shard.shape[1])
    c_m, full_N = int(C.shape[0]), int(C.shape[1])
    if b_k != K:
        raise ValueError(f"B_shard.shape[0] must equal A.shape[1]: got {b_k} vs {K}")
    if c_m != M:
        raise ValueError(f"C.shape[0] must equal A.shape[0]: got {c_m} vs {M}")
    if full_N <= 0:
        raise ValueError(f"C.shape[1] must be positive, got {full_N}")
    if full_N % ctx.world_size != 0:
        raise ValueError(
            "gemm_allgather requires C.shape[1] to be divisible by world_size: "
            f"{full_N} % {ctx.world_size} != 0"
        )
    expected_shard_cols = full_N if ctx.world_size == 1 else full_N // ctx.world_size
    if shard_cols != expected_shard_cols:
        raise ValueError(
            "gemm_allgather RHS shard columns must match the rank-local shard width: "
            f"expected {expected_shard_cols}, got {shard_cols}"
        )

    return GemmAllGatherContract(
        M=M,
        K=K,
        full_N=full_N,
        shard_cols=shard_cols,
        rank=ctx.rank,
        world_size=ctx.world_size,
        storage_kind=storage_kind,
    )


def _full_N_from_layout(*, cols: int, layout: LayoutKind, world_size: int) -> int:
    """Resolve logical full_N from one tensor shape + explicit layout."""
    if layout == "full":
        return cols
    return cols if world_size == 1 else cols * world_size


class _ShapeOnlyTensor:
    """Minimal tensor-like object used for shape-only contract resolution."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.ndim = 2


def _shape_only_tensor(shape: tuple[int, int]) -> _ShapeOnlyTensor:
    """Return a tensor-like shape stub for contract resolution."""
    return _ShapeOnlyTensor(shape)


def _execute_gemm_reducescatter_single_process(
    plan: GemmReduceScatterPlan,
    *,
    reduce_src: Any,
    output: Any,
) -> Any:
    """Finalize GEMM + reduce-scatter once all local ranks have staged inputs.

    In ``single_process`` mode, rank-local calls often happen sequentially in one
    Python process. The first rank cannot immediately enter ``reduce_scatter``
    because peer ranks have not populated their symmetric workspaces yet. This
    helper stages the rank-local packed input and completes the collective only
    after every local rank in the heap group has arrived.
    """
    key = _single_process_gemm_reducescatter_key(plan, output)
    pending = _PENDING_GEMM_REDUCESCATTER_SINGLE_PROCESS.setdefault(key, {})
    pending[plan.ctx.rank] = (plan, reduce_src, output)
    if len(pending) < plan.contract.world_size:
        return output

    try:
        for rank in range(plan.contract.world_size):
            rank_plan, rank_src, rank_output = pending[rank]
            rank_plan.reduce_scatter_plan.execute(
                rank_src,
                rank_output,
                validate=False,
            )
    finally:
        _PENDING_GEMM_REDUCESCATTER_SINGLE_PROCESS.pop(key, None)
    return output


def _single_process_gemm_reducescatter_key(
    plan: GemmReduceScatterPlan,
    output: Any,
) -> tuple[object, ...]:
    """Return a process-local coordination key for staged single-process runs."""
    heap_bases = tuple(int(base) for base in plan.ctx.heap_bases.tolist())
    return (
        "gemm_reducescatter",
        heap_bases,
        plan.contract.M,
        plan.contract.K,
        plan.contract.full_N,
        plan.contract.output_cols,
        str(output.dtype),
        plan.implementation,
        plan.storage_kind,
    )


def _execute_gemm_allgather_single_process(
    plan: GemmAllGatherPlan,
    *,
    local_shard: Any,
    gather_output: Any,
    output: Any,
) -> Any:
    """Finalize GEMM + allgather once every local rank has staged its shard."""
    key = _single_process_gemm_allgather_key(plan, output)
    pending = _PENDING_GEMM_ALLGATHER_SINGLE_PROCESS.setdefault(key, {})
    pending[plan.ctx.rank] = (plan, local_shard, gather_output, output)
    if len(pending) < plan.contract.world_size:
        return output

    try:
        for rank in range(plan.contract.world_size):
            rank_plan, rank_local_shard, rank_gather_output, _ = pending[rank]
            rank_plan.allgather_plan.execute(
                rank_local_shard,
                rank_gather_output,
                validate=False,
            )
        for rank in range(plan.contract.world_size):
            pending[rank][0].ctx.backend.synchronize()
        for rank in range(plan.contract.world_size):
            rank_plan, _, rank_gather_output, rank_output = pending[rank]
            rank_output.copy_(
                rank_gather_output.permute(1, 0, 2).reshape(
                    rank_plan.contract.M,
                    rank_plan.contract.full_N,
                )
            )
    finally:
        _PENDING_GEMM_ALLGATHER_SINGLE_PROCESS.pop(key, None)
    return output


def _single_process_gemm_allgather_key(
    plan: GemmAllGatherPlan,
    output: Any,
) -> tuple[object, ...]:
    """Return a process-local coordination key for staged single-process runs."""
    heap_bases = tuple(int(base) for base in plan.ctx.heap_bases.tolist())
    return (
        "gemm_allgather",
        heap_bases,
        plan.contract.M,
        plan.contract.K,
        plan.contract.full_N,
        plan.contract.shard_cols,
        str(output.dtype),
        plan.storage_kind,
    )


def _run_local_gemm(A: Any, B: Any, *, out: Any) -> None:
    """Execute one local GEMM into ``out`` with a conservative fallback."""
    import torch

    try:
        torch.mm(A, B, out=out)
    except (RuntimeError, TypeError):
        out.copy_(torch.matmul(A, B))


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


def _validate_reduce_scatter_contract(
    src: Any,
    output: Any,
    *,
    ctx: xtile.XTileContext,
    storage_kind: str,
    expected_input_shape: tuple[int, ...] | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
    expected_block_size: int | None = None,
) -> int:
    """Validate a reduce_scatter contract and return the logical block size."""
    del storage_kind  # Reserved for future non-symmetric storage backends.

    if src.device != output.device:
        raise ValueError(
            f"reduce_scatter expects src/output on the same device, got {src.device} vs {output.device}"
        )
    if str(src.device) != ctx.device:
        raise ValueError(
            f"reduce_scatter tensors must reside on the attached heap device {ctx.device}, got {src.device}"
        )
    if not src.is_contiguous():
        raise ValueError("reduce_scatter currently requires src to be contiguous")
    if not output.is_contiguous():
        raise ValueError("reduce_scatter currently requires output to be contiguous")

    _require_tensor_on_heap(src, ctx=ctx, name="src")
    _require_tensor_on_heap(output, ctx=ctx, name="output")

    block_size = int(output.numel())
    expected_input_numel = block_size * ctx.world_size
    if int(src.numel()) != expected_input_numel:
        raise ValueError(
            "reduce_scatter src.numel must equal output.numel * world_size: "
            f"{src.numel()} != {block_size} * {ctx.world_size}"
        )

    input_shape = tuple(int(dim) for dim in src.shape)
    output_shape = tuple(int(dim) for dim in output.shape)
    if expected_input_shape is not None and input_shape != expected_input_shape:
        raise ValueError(
            f"reduce_scatter src shape no longer matches the plan: {input_shape} != {expected_input_shape}"
        )
    if expected_output_shape is not None and output_shape != expected_output_shape:
        raise ValueError(
            "reduce_scatter output shape no longer matches the plan: "
            f"{output_shape} != {expected_output_shape}"
        )
    if expected_block_size is not None and block_size != expected_block_size:
        raise ValueError(
            f"reduce_scatter output.numel no longer matches the plan: {block_size} != {expected_block_size}"
        )

    return block_size


def _resolve_reduce_scatter_implementation(
    ctx: xtile.XTileContext,
    *,
    implementation: str = "auto",
) -> str:
    """Resolve the default reduce_scatter implementation for this context."""
    heap = ctx.require_heap()
    if implementation not in {"auto", "reference", "device"}:
        raise ValueError(
            "implementation must be one of {'auto', 'reference', 'device'}, "
            f"got {implementation!r}"
        )

    if implementation == "auto":
        if heap.mode == "single_process":
            return "reference"
        if not multiprocess_device_collectives_enabled():
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                )
            )
        if not multiprocess_device_collectives_transport_supported(
            heap.transport_strategy
        ):
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                )
            )
        return "device"
    if implementation == "reference" and heap.mode != "single_process":
        raise ValueError(
            "implementation='reference' is only available for single-process symmetric heaps."
        )
    if implementation == "device" and heap.mode == "single_process":
        raise ValueError(
            "implementation='device' is not validated for single-process symmetric heaps. "
            "Use implementation='reference' (or 'auto') until the device path is proven correct."
        )
    if implementation == "device" and heap.mode != "single_process":
        if not multiprocess_device_collectives_enabled():
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                )
            )
        if not multiprocess_device_collectives_transport_supported(
            heap.transport_strategy
        ):
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                )
            )
    return implementation


def _require_device_remote_access_runtime(
    ctx: xtile.XTileContext,
    *,
    operation: str,
) -> None:
    """Fail fast when Triton device-side remote access is not transport-safe."""
    heap = ctx.require_heap()
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


def _require_gemm_allscatter_runtime(ctx: xtile.XTileContext) -> None:
    """Validate the runtime prerequisites for GEMM + all-scatter execution."""
    _require_device_remote_access_runtime(
        ctx,
        operation="xtile.ops.gemm_allscatter(...)",
    )


def _require_tensor_on_heap(tensor: Any, *, ctx: xtile.XTileContext, name: str) -> None:
    """Ensure the tensor resides in the attached symmetric heap."""
    heap = ctx.require_heap()
    if not heap.owns_tensor(tensor):
        raise ValueError(
            f"{name} must reside in the attached symmetric heap for ctx rank {ctx.rank}"
        )
