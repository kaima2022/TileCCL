"""Helpers for benchmark result artifacts shared by scripts and docs."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
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


def runtime_support_snapshot(ctx: Any) -> dict[str, Any]:
    """Serialize the runtime support matrix for an existing context."""
    if hasattr(ctx, "support_matrix"):
        return ctx.support_matrix().to_dict()

    import xtile

    return xtile.describe_runtime_support(ctx).to_dict()


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
    import xtile

    ctx = xtile.init(
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
    import xtile

    ctx = xtile.init(
        backend=backend,
        rank=rank,
        world_size=world_size,
        heap=heap,
        force_backend=force_backend,
    )
    return runtime_metadata_snapshot(ctx)
