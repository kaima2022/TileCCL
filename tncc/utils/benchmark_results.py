# SPDX-License-Identifier: Apache-2.0
"""Helpers for benchmark result artifacts shared by scripts and docs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def figures_data_dir() -> Path:
    """Return the structured data directory used by plotting scripts."""
    path = project_root() / "figures" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_pattern_benchmark_path() -> Path:
    """Return the canonical latest pattern-benchmark JSON path."""
    return figures_data_dir() / "pattern_overlap_latest.json"


def default_gemm_benchmark_path() -> Path:
    """Return the canonical latest GEMM benchmark JSON path."""
    return figures_data_dir() / "gemm_latest.json"


def default_p2p_benchmark_path() -> Path:
    """Return the canonical latest P2P benchmark JSON path."""
    return figures_data_dir() / "p2p_latest.json"


def default_collective_comm_only_benchmark_path() -> Path:
    """Return the canonical latest comm-only collective benchmark JSON path."""
    return figures_data_dir() / "collective_comm_only_latest.json"


def default_collective_bulk_sync_benchmark_path() -> Path:
    """Return the canonical latest collective-vs-bulk-sync benchmark JSON path."""
    return figures_data_dir() / "collective_bulk_sync_latest.json"


def canonical_benchmark_lock_path() -> Path:
    """Return the repo-global lock used for canonical benchmark runs."""
    return figures_data_dir() / ".canonical_benchmark.lock"


def is_canonical_benchmark_output(path: str | Path) -> bool:
    """Return whether *path* targets the canonical structured-data directory."""
    output_path = Path(path).expanduser()
    try:
        resolved_parent = output_path.resolve(strict=False).parent
    except OSError:
        return False
    return resolved_parent == figures_data_dir().resolve()


def acquire_canonical_benchmark_lock(
    output_path: str | Path,
    *,
    blocking: bool = True,
):
    """Acquire the repo-global canonical benchmark lock when needed.

    Benchmarks that write to ``figures/data`` are treated as canonical runs and
    must not overlap on the same GPUs, otherwise the generated artifacts become
    contaminated by cross-run interference.
    """
    if not is_canonical_benchmark_output(output_path):
        return None

    lock_path = canonical_benchmark_lock_path()
    handle = lock_path.open("a+", encoding="utf-8")
    lock_flags = fcntl.LOCK_EX
    if not blocking:
        lock_flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), lock_flags)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "canonical benchmark lock is already held; rerun after the current "
            "figures/data benchmark finishes or use a non-canonical output path."
        ) from exc

    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "pid": os.getpid(),
            "output_path": str(Path(output_path).expanduser().resolve(strict=False)),
        },
        handle,
    )
    handle.flush()
    return handle


@contextmanager
def canonical_benchmark_run(
    output_path: str | Path,
    *,
    blocking: bool = True,
):
    """Serialize canonical benchmark runs that target ``figures/data``."""
    handle = acquire_canonical_benchmark_lock(output_path, blocking=blocking)
    try:
        yield
    finally:
        if handle is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write benchmark JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a structured benchmark JSON payload."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nvidia_smi_csv_query(
    *,
    target: str,
    fields: tuple[str, ...],
) -> list[dict[str, str]] | None:
    """Run one ``nvidia-smi`` CSV query and return parsed rows."""
    command = [
        "nvidia-smi",
        f"--query-{target}={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    rows: list[dict[str, str]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    return rows


def _parse_optional_int(value: str | None) -> int | None:
    """Parse one optional numeric field from ``nvidia-smi``."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized in {"N/A", "[Not Supported]"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def _numeric_visible_gpu_indices() -> list[int] | None:
    """Return numeric ``CUDA_VISIBLE_DEVICES`` indices when directly available."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None

    indices: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if not item.isdigit():
            return None
        indices.append(int(item))
    return indices or []


def benchmark_environment_health(
    *,
    visible_gpu_count: int | None = None,
) -> dict[str, Any]:
    """Collect one pre-run GPU contamination snapshot for benchmark guardrails."""
    snapshot: dict[str, Any] = {
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_visible_gpu_count": visible_gpu_count,
        "status": "unavailable",
        "target_gpu_indices": [],
        "gpus": [],
        "warnings": [],
    }

    gpu_rows = _nvidia_smi_csv_query(
        target="gpu",
        fields=(
            "index",
            "uuid",
            "name",
            "memory.used",
            "memory.total",
            "utilization.gpu",
            "utilization.memory",
        ),
    )
    if gpu_rows is None:
        snapshot["detail"] = "nvidia-smi is unavailable or the GPU query failed."
        return snapshot

    process_rows = _nvidia_smi_csv_query(
        target="compute-apps",
        fields=("gpu_uuid", "pid", "process_name", "used_gpu_memory"),
    )
    processes_by_uuid: dict[str, list[dict[str, Any]]] = {}
    for row in process_rows or []:
        gpu_uuid = row.get("gpu_uuid")
        if not gpu_uuid:
            continue
        processes_by_uuid.setdefault(gpu_uuid, []).append(
            {
                "pid": _parse_optional_int(row.get("pid")),
                "process_name": row.get("process_name"),
                "used_gpu_memory_mib": _parse_optional_int(row.get("used_gpu_memory")),
            }
        )

    visible_indices = _numeric_visible_gpu_indices()
    if visible_indices is None:
        target_gpu_indices = [
            int(row["index"])
            for row in gpu_rows[: visible_gpu_count or len(gpu_rows)]
        ]
    else:
        target_gpu_indices = visible_indices[: visible_gpu_count or len(visible_indices)]
    snapshot["target_gpu_indices"] = target_gpu_indices

    warnings: list[str] = []
    contaminated = False
    for row in gpu_rows:
        gpu_index = _parse_optional_int(row.get("index"))
        if gpu_index is None or gpu_index not in target_gpu_indices:
            continue

        gpu_uuid = row.get("uuid", "")
        utilization_gpu_pct = _parse_optional_int(row.get("utilization.gpu"))
        utilization_memory_pct = _parse_optional_int(row.get("utilization.memory"))
        memory_used_mib = _parse_optional_int(row.get("memory.used"))
        memory_total_mib = _parse_optional_int(row.get("memory.total"))
        compute_processes = processes_by_uuid.get(gpu_uuid, [])

        reasons: list[str] = []
        if compute_processes:
            reasons.append("resident_compute_processes")
        if utilization_gpu_pct is not None and utilization_gpu_pct > 0:
            reasons.append("active_gpu_utilization")

        gpu_snapshot = {
            "index": gpu_index,
            "uuid": gpu_uuid,
            "name": row.get("name"),
            "memory_used_mib": memory_used_mib,
            "memory_total_mib": memory_total_mib,
            "utilization_gpu_pct": utilization_gpu_pct,
            "utilization_memory_pct": utilization_memory_pct,
            "compute_processes": compute_processes,
            "contamination_reasons": reasons,
        }
        snapshot["gpus"].append(gpu_snapshot)

        if reasons:
            contaminated = True
            process_brief = ", ".join(
                f"{proc.get('process_name')} (pid={proc.get('pid')})"
                for proc in compute_processes
            )
            if process_brief:
                warnings.append(
                    "GPU "
                    f"{gpu_index} is not isolated before benchmark: "
                    f"util={utilization_gpu_pct or 0}%, "
                    f"resident_compute_processes=[{process_brief}], "
                    f"memory_used={memory_used_mib or 0} MiB."
                )
            else:
                warnings.append(
                    "GPU "
                    f"{gpu_index} reports pre-run utilization={utilization_gpu_pct or 0}% "
                    "without an enumerated compute process; benchmark isolation is not guaranteed."
                )

    snapshot["warnings"] = warnings
    snapshot["status"] = "contaminated" if contaminated else "clean"
    return snapshot


def emit_benchmark_environment_warnings(
    snapshot: dict[str, Any],
    *,
    stream=None,
) -> None:
    """Emit one concise benchmark-environment warning block when needed."""
    if snapshot.get("status") != "contaminated":
        return

    output = stream if stream is not None else sys.stderr
    print(
        "WARNING: benchmark environment is contaminated before the run; "
        "treat performance numbers as noisy and non-canonical.",
        file=output,
    )
    for warning in snapshot.get("warnings", []):
        print(f"WARNING: {warning}", file=output)


def runtime_support_snapshot(ctx: Any) -> dict[str, Any]:
    """Serialize the runtime support matrix for an existing context."""
    if hasattr(ctx, "support_matrix"):
        return ctx.support_matrix().to_dict()

    import tncc

    return tncc.describe_runtime_support(ctx).to_dict()


def runtime_metadata_snapshot(ctx: Any) -> dict[str, Any]:
    """Serialize the structured runtime metadata for an existing context."""
    if hasattr(ctx, "runtime_metadata"):
        return ctx.runtime_metadata()

    return {
        "rank": getattr(ctx, "rank", None),
        "world_size": getattr(ctx, "world_size", None),
        "device": getattr(ctx, "device", None),
        "backend": getattr(ctx, "backend_name", None),
        "has_heap": getattr(ctx, "heap", None) is not None,
        "heap": getattr(ctx, "heap", None).metadata()
        if getattr(ctx, "heap", None) is not None
        and hasattr(getattr(ctx, "heap", None), "metadata")
        else None,
    }


def describe_runtime_support_snapshot(
    *,
    backend: str = "auto",
    rank: int = 0,
    world_size: int = 1,
    heap: Any | None = None,
    force_backend: bool = False,
) -> dict[str, Any]:
    """Build a temporary context and serialize its runtime support matrix."""
    import tncc

    ctx = tncc.init(
        backend=backend,
        rank=rank,
        world_size=world_size,
        heap=heap,
        force_backend=force_backend,
    )
    return runtime_support_snapshot(ctx)


def describe_runtime_metadata_snapshot(
    *,
    backend: str = "auto",
    rank: int = 0,
    world_size: int = 1,
    heap: Any | None = None,
    force_backend: bool = False,
) -> dict[str, Any]:
    """Build a temporary context and serialize its runtime metadata."""
    import tncc

    ctx = tncc.init(
        backend=backend,
        rank=rank,
        world_size=world_size,
        heap=heap,
        force_backend=force_backend,
    )
    return runtime_metadata_snapshot(ctx)
