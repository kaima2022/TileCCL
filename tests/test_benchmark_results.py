"""Tests for structured benchmark-result helpers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import xtile
from xtile.utils.benchmark_results import (
    canonical_benchmark_run,
    default_gemm_benchmark_path,
    describe_runtime_metadata_snapshot,
    describe_runtime_support_snapshot,
    is_canonical_benchmark_output,
    project_root,
    runtime_metadata_snapshot,
    runtime_support_snapshot,
)


def test_runtime_support_snapshot_from_context(skip_no_gpu, device_info) -> None:
    """Existing contexts should serialize into a stable support payload."""
    ctx = xtile.init(
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
    assert payload["ops"]["gemm_reducescatter"]["state"] == "partial"


def test_describe_runtime_support_snapshot_with_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """Temporary support snapshots should preserve heap-backed capabilities."""
    from xtile.memory.symmetric_heap import SymmetricHeap

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
        assert payload["ops"]["reduce_scatter"]["state"] == "supported"
        assert payload["collectives"]["collectives.reduce_scatter_launcher"]["state"] == "supported"
    finally:
        for heap in heaps:
            heap.cleanup()


def test_runtime_metadata_snapshot_from_context(
    skip_no_gpu,
    device_info,
) -> None:
    """Existing contexts should also expose unified runtime metadata."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        ctx = xtile.init(
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
        assert payload["heap"]["segments"][0]["segment_id"] == "heap"
        assert payload["heap"]["peer_exports"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_exports"][0]["segment_id"] == "heap"
        assert payload["heap"]["peer_imports"][0]["segment_id"] == "heap"
        assert payload["heap"]["peer_imports"][0]["peer_rank"] == 0
        assert payload["heap"]["peer_imports"][0]["access_kind"] == "local"
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


def test_is_canonical_benchmark_output_matches_figures_data() -> None:
    """Canonical benchmark outputs should be recognized by directory."""
    assert is_canonical_benchmark_output(default_gemm_benchmark_path())
    assert not is_canonical_benchmark_output(Path("/tmp/not_xtile_benchmark.json"))


def test_canonical_benchmark_run_rejects_parallel_nonblocking_probe() -> None:
    """A second process should not be able to enter the canonical lock."""
    output_path = default_gemm_benchmark_path()
    repo_root = project_root()
    script = """
from pathlib import Path
from xtile.utils.benchmark_results import canonical_benchmark_run

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
