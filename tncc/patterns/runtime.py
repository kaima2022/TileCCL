# SPDX-License-Identifier: Apache-2.0
"""Shared runtime helpers for tile-collective patterns and host plans.

This module centralises three pieces of policy that were previously duplicated:

1. Dual-role SM partitioning for compute/communication worker pools.
2. Stage-role SM partitioning for gather/compute/scatter chains.
3. Stable runtime metadata for staged GEMM+collective plans.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DualRoleScheduler:
    """Resolved compute/communication worker split for one launch."""

    total_sms: int
    compute_sms: int
    comm_sms: int
    policy: str = "fractional_split"

    @property
    def grid_sms(self) -> int:
        """Return the combined worker grid for single-kernel launches."""
        return self.compute_sms + self.comm_sms

    def to_dict(self) -> dict[str, object]:
        """Serialize the scheduler for logs, docs, and benchmark payloads."""
        return {
            "total_sms": self.total_sms,
            "compute_sms": self.compute_sms,
            "comm_sms": self.comm_sms,
            "grid_sms": self.grid_sms,
            "policy": self.policy,
        }


@dataclass(frozen=True, slots=True)
class StageRoleScheduler:
    """Resolved gather/compute/scatter worker split for one launch."""

    total_sms: int
    gather_sms: int
    compute_sms: int
    scatter_sms: int
    policy: str = "fractional_split"

    @property
    def collective_sms(self) -> int:
        """Return the combined worker budget for collective stages."""
        return self.gather_sms + self.scatter_sms

    @property
    def grid_sms(self) -> int:
        """Return the combined worker grid for one chained launch."""
        return self.gather_sms + self.compute_sms + self.scatter_sms

    def to_dict(self) -> dict[str, object]:
        """Serialize the scheduler for logs, docs, and benchmark payloads."""
        return {
            "total_sms": self.total_sms,
            "gather_sms": self.gather_sms,
            "compute_sms": self.compute_sms,
            "scatter_sms": self.scatter_sms,
            "collective_sms": self.collective_sms,
            "grid_sms": self.grid_sms,
            "policy": self.policy,
        }


@dataclass(frozen=True, slots=True)
class TileCollectiveRuntime:
    """Stable runtime metadata shared by staged GEMM+collective plans."""

    op: str
    scheduler: str
    communication: str
    signal_mode: str
    wait_mode: str
    flow_control: str
    workspace_names: tuple[str, ...]
    role_order: tuple[str, ...] = ()

    @property
    def stage_count(self) -> int:
        """Return how many execution roles participate in the runtime."""
        return len(self.role_order)

    def to_dict(self) -> dict[str, object]:
        """Serialize the runtime contract for plan metadata and debugging."""
        return {
            "op": self.op,
            "scheduler": self.scheduler,
            "communication": self.communication,
            "signal_mode": self.signal_mode,
            "wait_mode": self.wait_mode,
            "flow_control": self.flow_control,
            "workspace_names": list(self.workspace_names),
            "role_order": list(self.role_order),
            "stage_count": self.stage_count,
        }


def resolve_dual_role_scheduler(
    total_sms: int,
    *,
    compute_sms: int = 0,
    comm_sms: int = 0,
    comm_sm_fraction: float = 0.2,
) -> DualRoleScheduler:
    """Resolve a stable compute/communication worker split.

    Explicit user overrides take precedence. Otherwise a single shared
    fractional policy is applied so multi-role patterns use the same split.
    """
    if total_sms < 2:
        raise ValueError(
            "Dual-role scheduling requires at least 2 compute units, "
            f"got total_sms={total_sms}."
        )

    if compute_sms > 0 or comm_sms > 0:
        if compute_sms <= 0 or comm_sms <= 0:
            raise ValueError(
                "Explicit dual-role scheduling requires both compute_sms and "
                f"comm_sms to be > 0, got compute_sms={compute_sms}, comm_sms={comm_sms}."
            )
        if compute_sms + comm_sms > total_sms:
            raise ValueError(
                "Explicit dual-role scheduling cannot exceed the device SM count: "
                f"{compute_sms} + {comm_sms} > {total_sms}."
            )
        return DualRoleScheduler(
            total_sms=total_sms,
            compute_sms=compute_sms,
            comm_sms=comm_sms,
            policy="explicit",
        )

    comm = max(1, int(total_sms * comm_sm_fraction))
    compute = total_sms - comm
    if compute < 1:
        compute = 1
        comm = total_sms - compute
    if comm < 1:
        comm = 1
        compute = total_sms - comm

    return DualRoleScheduler(
        total_sms=total_sms,
        compute_sms=compute,
        comm_sms=comm,
        policy="fractional_split",
    )


def resolve_stage_role_scheduler(
    total_sms: int,
    *,
    gather_sms: int = 0,
    compute_sms: int = 0,
    scatter_sms: int = 0,
    collective_sm_fraction: float = 0.2,
) -> StageRoleScheduler:
    """Resolve a stable gather/compute/scatter worker split.

    Explicit user overrides take precedence. Otherwise a single shared
    fractional policy is applied so future chained kernels inherit one
    stable stage-role contract.
    """
    if total_sms < 3:
        raise ValueError(
            "Stage-role scheduling requires at least 3 compute units, "
            f"got total_sms={total_sms}."
        )

    if gather_sms > 0 or compute_sms > 0 or scatter_sms > 0:
        if gather_sms <= 0 or compute_sms <= 0 or scatter_sms <= 0:
            raise ValueError(
                "Explicit stage-role scheduling requires gather_sms, compute_sms, "
                "and scatter_sms to be > 0, got "
                f"gather_sms={gather_sms}, compute_sms={compute_sms}, scatter_sms={scatter_sms}."
            )
        if gather_sms + compute_sms + scatter_sms > total_sms:
            raise ValueError(
                "Explicit stage-role scheduling cannot exceed the device SM count: "
                f"{gather_sms} + {compute_sms} + {scatter_sms} > {total_sms}."
            )
        return StageRoleScheduler(
            total_sms=total_sms,
            gather_sms=gather_sms,
            compute_sms=compute_sms,
            scatter_sms=scatter_sms,
            policy="explicit",
        )

    collective_budget = max(2, int(total_sms * collective_sm_fraction))
    collective_budget = min(collective_budget, total_sms - 1)
    gather = max(1, collective_budget // 2)
    scatter = max(1, collective_budget - gather)
    compute = total_sms - gather - scatter

    if compute < 1:
        compute = 1
        scatter = max(1, total_sms - gather - compute)
    if gather + compute + scatter > total_sms:
        scatter = total_sms - gather - compute

    return StageRoleScheduler(
        total_sms=total_sms,
        gather_sms=gather,
        compute_sms=compute,
        scatter_sms=scatter,
        policy="fractional_split",
    )


def resolve_tile_collective_runtime(op: str) -> TileCollectiveRuntime:
    """Return the shared runtime contract for staged GEMM+collective plans."""
    if op == "gemm_allgather":
        return TileCollectiveRuntime(
            op=op,
            scheduler="tile_scheduler_v1",
            communication="software_multicast",
            signal_mode="binary",
            wait_mode="try_wait_capable",
            flow_control="credit_optional",
            workspace_names=(
                "gemm_allgather.local_output_shard",
                "gemm_allgather.gathered_output_shards",
            ),
            role_order=("compute", "gather"),
        )
    if op == "gemm_reducescatter":
        return TileCollectiveRuntime(
            op=op,
            scheduler="tile_scheduler_v1",
            communication="reduce_scatter",
            signal_mode="counting",
            wait_mode="wait_ge",
            flow_control="credit_optional",
            workspace_names=(
                "gemm_reducescatter.local_full_output",
                "gemm_reducescatter.packed_input",
            ),
            role_order=("compute", "scatter"),
        )
    if op == "allgather_gemm_reducescatter":
        return TileCollectiveRuntime(
            op=op,
            scheduler="stage_role_scheduler_v1",
            communication="software_multicast_then_reduce_scatter",
            signal_mode="binary_then_counting",
            wait_mode="try_wait_then_wait_ge",
            flow_control="credit_optional",
            workspace_names=(
                "allgather_gemm_reducescatter.stage_gather_input",
                "allgather_gemm_reducescatter.stage_gather_tiles",
                "allgather_gemm_reducescatter.stage_accumulators",
                "allgather_gemm_reducescatter.stage_scatter_output",
            ),
            role_order=("gather", "compute", "scatter"),
        )
    raise ValueError(f"Unsupported tile collective runtime op {op!r}")
