# SPDX-License-Identifier: Apache-2.0
"""Tests for shared pattern/runtime scheduling helpers."""

from __future__ import annotations

import pytest

from tncc.patterns.runtime import (
    resolve_dual_role_scheduler,
    resolve_stage_role_scheduler,
    resolve_tile_collective_runtime,
)


def test_resolve_dual_role_scheduler_uses_fractional_policy() -> None:
    """Default scheduling should share one stable fraction-based split."""
    scheduler = resolve_dual_role_scheduler(20, comm_sm_fraction=0.2)

    assert scheduler.total_sms == 20
    assert scheduler.compute_sms == 16
    assert scheduler.comm_sms == 4
    assert scheduler.grid_sms == 20
    assert scheduler.policy == "fractional_split"


def test_resolve_dual_role_scheduler_respects_explicit_override() -> None:
    """Explicit compute/comm splits should round-trip unchanged."""
    scheduler = resolve_dual_role_scheduler(20, compute_sms=12, comm_sms=4)

    assert scheduler.compute_sms == 12
    assert scheduler.comm_sms == 4
    assert scheduler.grid_sms == 16
    assert scheduler.policy == "explicit"


def test_resolve_dual_role_scheduler_rejects_partial_override() -> None:
    """A single explicit worker count is ambiguous and should fail."""
    with pytest.raises(ValueError, match="both compute_sms and comm_sms"):
        resolve_dual_role_scheduler(20, compute_sms=8)


def test_resolve_dual_role_scheduler_rejects_oversubscription() -> None:
    """The shared scheduler should fail fast on impossible explicit splits."""
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_dual_role_scheduler(20, compute_sms=12, comm_sms=12)


def test_resolve_stage_role_scheduler_uses_fractional_policy() -> None:
    """Default chained scheduling should reserve collective workers symmetrically."""
    scheduler = resolve_stage_role_scheduler(20, collective_sm_fraction=0.2)

    assert scheduler.total_sms == 20
    assert scheduler.gather_sms == 2
    assert scheduler.compute_sms == 16
    assert scheduler.scatter_sms == 2
    assert scheduler.collective_sms == 4
    assert scheduler.grid_sms == 20
    assert scheduler.policy == "fractional_split"


def test_resolve_stage_role_scheduler_respects_explicit_override() -> None:
    """Explicit gather/compute/scatter splits should round-trip unchanged."""
    scheduler = resolve_stage_role_scheduler(
        20,
        gather_sms=2,
        compute_sms=12,
        scatter_sms=2,
    )

    assert scheduler.gather_sms == 2
    assert scheduler.compute_sms == 12
    assert scheduler.scatter_sms == 2
    assert scheduler.grid_sms == 16
    assert scheduler.policy == "explicit"


def test_resolve_stage_role_scheduler_rejects_partial_override() -> None:
    """A partial stage-role override is ambiguous and should fail."""
    with pytest.raises(ValueError, match="gather_sms, compute_sms, and scatter_sms"):
        resolve_stage_role_scheduler(20, compute_sms=8)


def test_resolve_stage_role_scheduler_rejects_oversubscription() -> None:
    """Stage-role scheduling should fail fast on impossible explicit splits."""
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_stage_role_scheduler(
            20,
            gather_sms=4,
            compute_sms=12,
            scatter_sms=8,
        )


def test_resolve_tile_collective_runtime_exposes_allgather_contract() -> None:
    """The staged allgather runtime metadata should be stable and explicit."""
    runtime = resolve_tile_collective_runtime("gemm_allgather")

    assert runtime.scheduler == "tile_scheduler_v1"
    assert runtime.communication == "software_multicast"
    assert runtime.signal_mode == "binary"
    assert runtime.wait_mode == "try_wait_capable"
    assert runtime.role_order == ("compute", "gather")
    assert runtime.stage_count == 2
    assert runtime.workspace_names == (
        "gemm_allgather.local_output_shard",
        "gemm_allgather.gathered_output_shards",
    )


def test_resolve_tile_collective_runtime_exposes_reducescatter_contract() -> None:
    """The staged reduce-scatter runtime metadata should surface counting semantics."""
    runtime = resolve_tile_collective_runtime("gemm_reducescatter")

    assert runtime.scheduler == "tile_scheduler_v1"
    assert runtime.communication == "reduce_scatter"
    assert runtime.signal_mode == "counting"
    assert runtime.wait_mode == "wait_ge"
    assert runtime.role_order == ("compute", "scatter")
    assert runtime.stage_count == 2
    assert runtime.workspace_names == (
        "gemm_reducescatter.local_full_output",
        "gemm_reducescatter.packed_input",
    )


def test_resolve_tile_collective_runtime_exposes_chained_contract() -> None:
    """The chained runtime metadata should define the future stage-role contract."""
    runtime = resolve_tile_collective_runtime("allgather_gemm_reducescatter")

    assert runtime.scheduler == "stage_role_scheduler_v1"
    assert runtime.communication == "software_multicast_then_reduce_scatter"
    assert runtime.signal_mode == "binary_then_counting"
    assert runtime.wait_mode == "try_wait_then_wait_ge"
    assert runtime.flow_control == "credit_optional"
    assert runtime.role_order == ("gather", "compute", "scatter")
    assert runtime.stage_count == 3
    assert runtime.workspace_names == (
        "allgather_gemm_reducescatter.stage_gather_input",
        "allgather_gemm_reducescatter.stage_gather_tiles",
        "allgather_gemm_reducescatter.stage_accumulators",
        "allgather_gemm_reducescatter.stage_scatter_output",
    )
