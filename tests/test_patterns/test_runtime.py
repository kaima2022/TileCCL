# SPDX-License-Identifier: Apache-2.0
"""Tests for shared pattern/runtime scheduling helpers."""

from __future__ import annotations

import pytest

from tncc.patterns.runtime import (
    execute_staged_tile_collective,
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
    assert runtime.execution_model == "shared_staged_runtime_v1"
    assert runtime.workspace_protocol == "ctx_workspace_cache_v1"
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
    assert runtime.execution_model == "shared_staged_runtime_v1"
    assert runtime.workspace_protocol == "ctx_workspace_cache_v1"
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
    assert runtime.execution_model == "shared_staged_runtime_v1"
    assert runtime.workspace_protocol == "ctx_workspace_cache_v1"
    assert runtime.workspace_names == (
        "allgather_gemm_reducescatter.stage_gather_input",
        "allgather_gemm_reducescatter.stage_gather_tiles",
        "allgather_gemm_reducescatter.stage_accumulators",
        "allgather_gemm_reducescatter.stage_scatter_output",
    )


def test_execute_staged_tile_collective_runs_shared_stage_flow() -> None:
    """The shared executor should standardize workspace and stage ordering."""

    class _DummyHeap:
        mode = "multiprocess"

    class _DummyCtx:
        world_size = 2

        def __init__(self) -> None:
            self.heap = _DummyHeap()
            self.workspace_calls: list[tuple[str, tuple[int, ...], str]] = []

        def workspace(self, name: str, *size: int, dtype, zero: bool = False):
            del zero
            self.workspace_calls.append((name, size, dtype))
            return {"name": name, "size": size, "dtype": dtype}

        def require_heap(self):
            return self.heap

    runtime = resolve_tile_collective_runtime("gemm_allgather")
    ctx = _DummyCtx()
    log: list[tuple[str, object]] = []

    def _run_local_stage(local_buffer):
        log.append(("local", local_buffer["name"]))

    def _prepare_collective(local_buffer):
        log.append(("prepare", local_buffer["name"]))
        return local_buffer, {"name": runtime.workspace_names[1]}

    def _execute_collective(local_buffer, collective_output):
        log.append(("collective", (local_buffer["name"], collective_output["name"])))
        return collective_output

    def _finalize_output(collective_output, final_output):
        log.append(("finalize", (collective_output["name"], final_output)))
        return "done"

    result = execute_staged_tile_collective(
        ctx=ctx,
        runtime=runtime,
        local_shape=(32, 64),
        dtype="float16",
        run_local_stage=_run_local_stage,
        prepare_collective=_prepare_collective,
        execute_collective=_execute_collective,
        final_output="output",
        finalize_output=_finalize_output,
    )

    assert result == "done"
    assert ctx.workspace_calls == [
        ("gemm_allgather.local_output_shard", (32, 64), "float16")
    ]
    assert log == [
        ("local", "gemm_allgather.local_output_shard"),
        ("prepare", "gemm_allgather.local_output_shard"),
        (
            "collective",
            (
                "gemm_allgather.local_output_shard",
                "gemm_allgather.gathered_output_shards",
            ),
        ),
        ("finalize", ("gemm_allgather.gathered_output_shards", "output")),
    ]


def test_execute_staged_tile_collective_uses_single_process_coordinator() -> None:
    """Single-process local multi-GPU runs should bypass immediate collective launch."""

    class _DummyHeap:
        mode = "single_process"

    class _DummyCtx:
        world_size = 2

        def __init__(self) -> None:
            self.heap = _DummyHeap()

        def workspace(self, name: str, *size: int, dtype, zero: bool = False):
            del zero
            return {"name": name, "size": size, "dtype": dtype}

        def require_heap(self):
            return self.heap

    runtime = resolve_tile_collective_runtime("gemm_reducescatter")
    ctx = _DummyCtx()
    log: list[str] = []

    def _unexpected_collective(_src, _output):
        raise AssertionError("single-process execution should use the coordinator path")

    def _single_process(src, collective_output, final_output):
        log.append(src["name"])
        assert collective_output == "output"
        assert final_output == "output"
        return "coordinated"

    result = execute_staged_tile_collective(
        ctx=ctx,
        runtime=runtime,
        local_shape=(32, 64),
        dtype="float16",
        run_local_stage=lambda _local: None,
        prepare_collective=lambda local: (local, "output"),
        execute_collective=_unexpected_collective,
        final_output="output",
        single_process_execute=_single_process,
    )

    assert result == "coordinated"
    assert log == ["gemm_reducescatter.local_full_output"]
