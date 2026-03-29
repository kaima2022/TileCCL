# SPDX-License-Identifier: Apache-2.0
"""Tests for structured benchmark-result helpers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import importlib.util
import io

import tncc
from tncc.utils.benchmark_results import (
    benchmark_environment_health,
    canonical_benchmark_run,
    default_collective_bulk_sync_benchmark_path,
    default_collective_comm_only_benchmark_path,
    default_gemm_benchmark_path,
    describe_runtime_metadata_snapshot,
    describe_runtime_support_snapshot,
    emit_benchmark_environment_warnings,
    is_canonical_benchmark_output,
    project_root,
    runtime_metadata_snapshot,
    runtime_support_snapshot,
)


def test_runtime_support_snapshot_from_context(skip_no_gpu, device_info) -> None:
    """Existing contexts should serialize into a stable support payload."""
    ctx = tncc.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )

    payload = runtime_support_snapshot(ctx)

    assert payload["context"]["backend"] == device_info.backend
    assert payload["context"]["has_heap"] is False
    assert payload["ops"]["gemm_allscatter"]["state"] == "partial"
    assert payload["ops"]["gemm_allgather"]["state"] == "partial"
    assert payload["ops"]["allreduce"]["state"] == "partial"
    assert payload["ops"]["gemm_reducescatter"]["state"] == "partial"


def test_describe_runtime_support_snapshot_with_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """Temporary support snapshots should preserve heap-backed capabilities."""
    from tncc.memory.symmetric_heap import SymmetricHeap

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        payload = describe_runtime_support_snapshot(
            backend=device_info.backend,
            rank=0,
            world_size=1,
            heap=heaps[0],
            force_backend=True,
        )

        assert payload["context"]["has_heap"] is True
        assert payload["context"]["heap_mode"] == "single_process"
        assert payload["context"]["transport_strategy"] == "peer_access"
        assert payload["ops"]["allreduce"]["state"] == "supported"
        assert payload["ops"]["reduce_scatter"]["state"] == "supported"
        assert payload["collectives"]["collectives.reduce_scatter_launcher"]["state"] == "supported"
        assert payload["collectives"]["collectives.allreduce_launcher"]["state"] == "supported"
    finally:
        for heap in heaps:
            heap.cleanup()


def test_runtime_metadata_snapshot_from_context(
    skip_no_gpu,
    device_info,
) -> None:
    """Existing contexts should also expose unified runtime metadata."""
    from tncc.memory.symmetric_heap import SymmetricHeap

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        ctx = tncc.init(
            backend=device_info.backend,
            rank=0,
            world_size=1,
            heap=heaps[0],
            force_backend=True,
        )
        payload = runtime_metadata_snapshot(ctx)

        assert payload["backend"] == device_info.backend
        assert payload["has_heap"] is True
        assert payload["heap"]["allocator"]["name"] == "torch_bump"
        assert payload["heap"]["allocator"]["capabilities"]["external_mapping"] is False
        assert payload["heap"]["allocator"]["external_tensor_import_mode"] == "copy"
        assert payload["heap"]["allocator"]["external_mapping_mode"] == "none"
        assert payload["heap"]["external_memory_interface"]["mapping_mode"] == "none"
        assert payload["heap"]["external_memory_interface"]["copy_import_supported"] is True
        assert payload["heap"]["segment_layout"]["primary_segment_id"] == "heap"
        assert payload["heap"]["segment_layout"]["exportable_segment_ids"] == ["heap"]
        assert payload["heap"]["exportable_segments"][0]["segment_id"] == "heap"
        assert payload["heap"]["exportable_segments"][0]["is_primary_segment"] is True
        assert payload["heap"]["allocator"]["peer_transport_modes"] == [
            "ctypes_ipc",
            "pytorch_ipc",
            "peer_access_pointer_exchange",
        ]
        assert payload["heap"]["allocator"]["peer_import_access_kinds"] == [
            "local",
            "peer_direct",
            "mapped_remote",
            "remote_pointer",
        ]
        assert payload["heap"]["allocator"]["memory_model"]["peer_mapping_model"] == (
            "rank_ordered_import_table"
        )
        assert payload["heap"]["allocator"]["memory_model"]["external_mapping_mode"] == "none"
        assert payload["heap"]["segments"][0]["segment_id"] == "heap"
        assert payload["heap"]["peer_exports"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_exports"][0]["segment_id"] == "heap"
        assert payload["heap"]["peer_export_catalog"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_export_catalog"][0]["segment_ids"] == ["heap"]
        assert payload["heap"]["peer_imports"][0]["segment_id"] == "heap"
        assert payload["heap"]["peer_imports"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_imports"][0]["access_kind"] == "local"
        assert payload["heap"]["peer_import_catalog"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_import_catalog"][0]["segment_ids"] == ["heap"]
        assert len(payload["heap"]["peer_memory_map"]) == 1
        assert payload["heap"]["peer_memory_map"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_memory_map"][0]["access_kind"] == "local"
    finally:
        for heap in heaps:
            heap.cleanup()


def test_describe_runtime_metadata_snapshot_without_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """Temporary runtime metadata snapshots should work without a heap."""
    payload = describe_runtime_metadata_snapshot(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )

    assert payload["backend"] == device_info.backend
    assert payload["has_heap"] is False
    assert payload["heap"] is None


def test_benchmark_environment_health_flags_foreign_gpu_activity(monkeypatch) -> None:
    """Benchmark health snapshot should flag pre-run foreign compute activity."""

    def _fake_run(command, capture_output, text, check):
        assert capture_output is True
        assert text is True
        assert check is True
        query_arg = next(item for item in command if item.startswith("--query-"))
        if query_arg.startswith("--query-gpu="):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "0, GPU-AAA, NVIDIA H100 PCIe, 34197, 81559, 0, 0\n"
                    "1, GPU-BBB, NVIDIA H100 PCIe, 1055, 81559, 100, 0\n"
                ),
                stderr="",
            )
        if query_arg.startswith("--query-compute-apps="):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "GPU-AAA, 3286824, llama-box, 34197\n"
                    "GPU-BBB, 1054772, python, 1055\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("tncc.utils.benchmark_results.subprocess.run", _fake_run)

    snapshot = benchmark_environment_health(visible_gpu_count=2)

    assert snapshot["status"] == "contaminated"
    assert snapshot["target_gpu_indices"] == [0, 1]
    assert len(snapshot["gpus"]) == 2
    assert snapshot["gpus"][0]["contamination_reasons"] == ["resident_compute_processes"]
    assert snapshot["gpus"][1]["contamination_reasons"] == [
        "resident_compute_processes",
        "active_gpu_utilization",
    ]
    assert any("GPU 0 is not isolated" in warning for warning in snapshot["warnings"])
    assert any("GPU 1 is not isolated" in warning for warning in snapshot["warnings"])


def test_emit_benchmark_environment_warnings_formats_warning_block() -> None:
    """Benchmark warning emitter should produce one readable warning block."""
    stream = io.StringIO()
    emit_benchmark_environment_warnings(
        {
            "status": "contaminated",
            "warnings": [
                "GPU 1 is not isolated before benchmark: util=100%, resident_compute_processes=[python (pid=1)]."
            ],
        },
        stream=stream,
    )

    rendered = stream.getvalue()
    assert "benchmark environment is contaminated before the run" in rendered
    assert "GPU 1 is not isolated before benchmark" in rendered


def test_is_canonical_benchmark_output_matches_figures_data() -> None:
    """Canonical benchmark outputs should be recognized by directory."""
    assert is_canonical_benchmark_output(default_collective_bulk_sync_benchmark_path())
    assert is_canonical_benchmark_output(default_gemm_benchmark_path())
    assert is_canonical_benchmark_output(default_collective_comm_only_benchmark_path())
    assert not is_canonical_benchmark_output(Path("/tmp/not_tncc_benchmark.json"))


def test_canonical_benchmark_run_rejects_parallel_nonblocking_probe() -> None:
    """A second process should not be able to enter the canonical lock."""
    output_path = default_gemm_benchmark_path()
    repo_root = project_root()
    script = """
from pathlib import Path
from tncc.utils.benchmark_results import canonical_benchmark_run

try:
    with canonical_benchmark_run(Path(%r), blocking=False):
        raise SystemExit(2)
except RuntimeError:
    raise SystemExit(0)
""" % str(output_path)

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not pythonpath
        else f"{repo_root}{os.pathsep}{pythonpath}"
    )

    with canonical_benchmark_run(output_path):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert proc.returncode == 0, proc.stderr


def _load_comm_only_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "tests" / "benchmarks" / "bench_collective_comm_only.py"
    spec = importlib.util.spec_from_file_location(
        "_bench_collective_comm_only_test",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault("_bench_collective_comm_only_test", module)
    spec.loader.exec_module(module)
    return module


def test_collective_comm_only_aggregate_preserves_allreduce_execution_metadata() -> None:
    """Comm-only aggregation should keep rank-invariant allreduce execution metadata."""
    bench = _load_comm_only_benchmark_module()

    rank_payloads = [
        {
            "results": [
                {
                    "collective": "allreduce",
                    "size_bytes": 65536,
                    "tncc": {
                        "times_ms": [1.0, 1.1],
                        "correct": True,
                        "implementation": "device_staged_pipeline",
                        "protocol": "slot_epoch_pipeline",
                        "kernel_family": "ws2_specialized",
                        "reuse_handshake": "ws2_epoch_ack",
                        "message_bytes": 65536,
                        "message_regime": "throughput",
                        "cta_policy": "multi_cta_pipeline",
                        "epoch_policy": "per_chunk_slot_epoch",
                        "chunk_elems": 4096,
                        "num_chunks": 4,
                        "pipeline_slots": 4,
                        "grid_size": 4,
                        "num_warps": 4,
                        "workspace_bytes": 65568,
                    },
                    "nccl": {
                        "times_ms": [0.5, 0.6],
                        "correct": True,
                    },
                }
            ]
        },
        {
            "results": [
                {
                    "collective": "allreduce",
                    "size_bytes": 65536,
                    "tncc": {
                        "times_ms": [0.9, 1.0],
                        "correct": True,
                        "implementation": "device_staged_pipeline",
                        "protocol": "slot_epoch_pipeline",
                        "kernel_family": "ws2_specialized",
                        "reuse_handshake": "ws2_epoch_ack",
                        "message_bytes": 65536,
                        "message_regime": "throughput",
                        "cta_policy": "multi_cta_pipeline",
                        "epoch_policy": "per_chunk_slot_epoch",
                        "chunk_elems": 4096,
                        "num_chunks": 4,
                        "pipeline_slots": 4,
                        "grid_size": 4,
                        "num_warps": 4,
                        "workspace_bytes": 65568,
                    },
                    "nccl": {
                        "times_ms": [0.45, 0.55],
                        "correct": True,
                    },
                }
            ]
        },
    ]

    cases, summary = bench._aggregate_rank_results(rank_payloads, world_size=2)

    assert len(cases) == 1
    case = cases[0]
    assert case["tncc"]["implementation"] == "device_staged_pipeline"
    assert case["tncc"]["protocol"] == "slot_epoch_pipeline"
    assert case["tncc"]["kernel_family"] == "ws2_specialized"
    assert case["tncc"]["reuse_handshake"] == "ws2_epoch_ack"
    assert case["tncc"]["message_bytes"] == 65536
    assert case["tncc"]["message_regime"] == "throughput"
    assert case["tncc"]["cta_policy"] == "multi_cta_pipeline"
    assert case["tncc"]["epoch_policy"] == "per_chunk_slot_epoch"
    assert case["tncc"]["chunk_elems"] == 4096
    assert summary["peak_by_collective"]["allreduce"]["peak_tncc_bandwidth_gbps"] > 0.0
