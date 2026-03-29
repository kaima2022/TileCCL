# SPDX-License-Identifier: Apache-2.0
"""tncc.utils.profiling - Profiling and benchmarking utilities.

Provides a context-manager profiler for tile kernels, bandwidth normalisation
helpers, and result formatting / persistence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# TileProfiler
# ---------------------------------------------------------------------------

class TileProfiler:
    """Context manager that records kernel execution times.

    Usage::

        profiler = TileProfiler("my_kernel")
        with profiler:
            my_kernel[grid](...)
            torch.cuda.synchronize()

        print(f"Elapsed: {profiler.elapsed_ms:.2f} ms")

    Multiple invocations accumulate into :attr:`history`, and summary
    statistics are available via :meth:`summary`.
    """

    def __init__(self, name: str = "kernel") -> None:
        self.name = name
        self.history: list[float] = []
        self._start: float = 0.0
        self._elapsed: float = 0.0

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "TileProfiler":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._elapsed = (time.perf_counter() - self._start) * 1e3  # ms
        self.history.append(self._elapsed)

    # -- properties ---------------------------------------------------------

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock time of the most recent context-manager block (ms)."""
        return self._elapsed

    # -- summary statistics -------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return summary statistics over all recorded invocations.

        Returns:
            Dict with keys ``name``, ``count``, ``mean_ms``, ``min_ms``,
            ``max_ms``, ``total_ms``.
        """
        if not self.history:
            return {
                "name": self.name,
                "count": 0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "total_ms": 0.0,
            }
        return {
            "name": self.name,
            "count": len(self.history),
            "mean_ms": sum(self.history) / len(self.history),
            "min_ms": min(self.history),
            "max_ms": max(self.history),
            "total_ms": sum(self.history),
        }

    def reset(self) -> None:
        """Clear all recorded history."""
        self.history.clear()
        self._elapsed = 0.0


# ---------------------------------------------------------------------------
# Bandwidth helpers
# ---------------------------------------------------------------------------

def bandwidth_to_normalized(
    bytes_transferred: int | float,
    time_s: float,
    peak_bw_gbps: float,
) -> float:
    """Compute achieved bandwidth as a fraction of peak.

    Args:
        bytes_transferred: Total bytes moved during the transfer.
        time_s: Wall-clock time in seconds.
        peak_bw_gbps: Theoretical peak bandwidth in GB/s (e.g. 450.0 for
            NVLink on H100).

    Returns:
        Normalised bandwidth in the range ``[0.0, 1.0]`` (or above 1.0 if
        the measurement exceeds the theoretical peak, which can happen due
        to caching effects).
    """
    if time_s <= 0.0 or peak_bw_gbps <= 0.0:
        return 0.0
    achieved_gbps = (bytes_transferred / time_s) / 1e9
    return achieved_gbps / peak_bw_gbps


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_benchmark_table(results: list[dict[str, Any]]) -> str:
    """Format a list of benchmark result dicts as an ASCII table.

    Each dict should contain at least ``"name"`` or ``"pattern"`` and numeric
    fields to display.

    Args:
        results: List of result dicts from :meth:`TileProfiler.summary` or
            :meth:`Pattern.benchmark`.

    Returns:
        A human-readable ASCII table string.
    """
    if not results:
        return "(no results)"

    # Determine columns: take union of all keys, name/pattern first
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Prioritise 'name' or 'pattern' as the first column
    priority = {"name", "pattern"}
    ordered: list[str] = [k for k in all_keys if k in priority]
    ordered += [k for k in all_keys if k not in priority]

    # Build header
    col_widths: dict[str, int] = {}
    for key in ordered:
        max_w = len(key)
        for r in results:
            val = r.get(key, "")
            if isinstance(val, float):
                cell = f"{val:.3f}"
            else:
                cell = str(val)
            max_w = max(max_w, len(cell))
        col_widths[key] = max_w

    sep = "  "
    header = sep.join(k.ljust(col_widths[k]) for k in ordered)
    divider = sep.join("-" * col_widths[k] for k in ordered)

    lines: list[str] = [header, divider]
    for r in results:
        cells: list[str] = []
        for k in ordered:
            val = r.get(k, "")
            if isinstance(val, float):
                cell = f"{val:.3f}"
            else:
                cell = str(val)
            cells.append(cell.ljust(col_widths[k]))
        lines.append(sep.join(cells))

    return "\n".join(lines)


def save_benchmark_results(
    results: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Save benchmark results to a JSON file.

    Args:
        results: List of result dicts.
        path: Output file path (will be created or overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Triton Proton integration
# ---------------------------------------------------------------------------

class ProtonProfiler:
    """Wrapper for Triton's built-in Proton profiler.

    Captures kernel-level timing and memory access patterns.
    Falls back gracefully if Proton is not available.

    Usage::

        profiler = ProtonProfiler("my_session")
        with profiler.session():
            my_kernel[grid](...)
            torch.cuda.synchronize()

        profiler.export("output.json")
    """

    def __init__(self, name: str = "tncc_profile") -> None:
        self.name = name
        self._proton = None
        self._session_id = None
        self._available = False
        try:
            import triton.profiler as proton
            self._proton = proton
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Whether Proton is available in this Triton installation."""
        return self._available

    class _NoOpContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def session(self) -> Any:
        """Start a profiling session (context manager).

        Returns a no-op context if Proton is unavailable.
        """
        if not self._available:
            return self._NoOpContext()
        return self._ProtonSession(self._proton, self.name)

    class _ProtonSession:
        def __init__(self, proton, name):
            self._proton = proton
            self._name = name

        def __enter__(self):
            try:
                self._proton.start(self._name)
            except Exception:
                pass
            return self

        def __exit__(self, *args):
            try:
                self._proton.finalize()
            except Exception:
                pass

    def export(self, path: str | Path) -> None:
        """Export profiling data to a file.

        Args:
            path: Output file path (JSON format).
        """
        if not self._available:
            print(f"[ProtonProfiler] Proton not available, skipping export to {path}")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Proton writes its own output files; this is a placeholder
        # for future integration with proton's export API.
        print(f"[ProtonProfiler] Session '{self.name}' data available via Proton CLI")


# ---------------------------------------------------------------------------
# Overlap timeline
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single event in a compute/comm overlap timeline."""
    name: str
    start_ms: float
    end_ms: float
    stream: str  # "compute" or "comm"
    metadata: dict[str, Any] = field(default_factory=dict)


class OverlapTimeline:
    """Records and visualizes compute/communication overlap.

    Captures events on compute and communication streams, then
    computes overlap metrics and optionally exports timeline data.

    Usage::

        timeline = OverlapTimeline()
        timeline.record("gemm_tile_0", 0.0, 1.5, "compute")
        timeline.record("scatter_tile_0", 0.5, 1.2, "comm")
        metrics = timeline.compute_overlap()
        print(f"Overlap: {metrics['overlap_ratio']:.1%}")
    """

    def __init__(self) -> None:
        self.events: list[TimelineEvent] = []

    def record(
        self,
        name: str,
        start_ms: float,
        end_ms: float,
        stream: str,
        **metadata: Any,
    ) -> None:
        """Record a timeline event.

        Args:
            name: Event name (e.g. "gemm_tile_3").
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds.
            stream: "compute" or "comm".
            **metadata: Additional key-value pairs.
        """
        self.events.append(TimelineEvent(
            name=name,
            start_ms=start_ms,
            end_ms=end_ms,
            stream=stream,
            metadata=metadata,
        ))

    def compute_overlap(self) -> dict[str, Any]:
        """Compute overlap metrics between compute and comm streams.

        Returns:
            Dict with keys:
                total_compute_ms: Total compute time.
                total_comm_ms: Total communication time.
                overlap_ms: Time where both streams are active.
                overlap_ratio: overlap_ms / max(compute, comm).
                total_wall_ms: Wall clock from first event start to
                    last event end.
                efficiency: 1 - (wall / (compute + comm)), measures
                    how much overlap was achieved.
        """
        compute_events = [e for e in self.events if e.stream == "compute"]
        comm_events = [e for e in self.events if e.stream == "comm"]

        if not compute_events or not comm_events:
            return {
                "total_compute_ms": sum(e.end_ms - e.start_ms for e in compute_events),
                "total_comm_ms": sum(e.end_ms - e.start_ms for e in comm_events),
                "overlap_ms": 0.0,
                "overlap_ratio": 0.0,
                "total_wall_ms": 0.0,
                "efficiency": 0.0,
            }

        total_compute = sum(e.end_ms - e.start_ms for e in compute_events)
        total_comm = sum(e.end_ms - e.start_ms for e in comm_events)

        # Compute overlap using interval intersection
        overlap = 0.0
        for c in compute_events:
            for m in comm_events:
                start = max(c.start_ms, m.start_ms)
                end = min(c.end_ms, m.end_ms)
                if end > start:
                    overlap += end - start

        all_events = compute_events + comm_events
        wall_start = min(e.start_ms for e in all_events)
        wall_end = max(e.end_ms for e in all_events)
        total_wall = wall_end - wall_start

        max_stream = max(total_compute, total_comm)
        overlap_ratio = overlap / max_stream if max_stream > 0 else 0.0

        total_work = total_compute + total_comm
        efficiency = 1.0 - (total_wall / total_work) if total_work > 0 else 0.0

        return {
            "total_compute_ms": round(total_compute, 4),
            "total_comm_ms": round(total_comm, 4),
            "overlap_ms": round(overlap, 4),
            "overlap_ratio": round(overlap_ratio, 4),
            "total_wall_ms": round(total_wall, 4),
            "efficiency": round(efficiency, 4),
        }

    def export_json(self, path: str | Path) -> None:
        """Export timeline events to JSON for visualization.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "events": [
                {
                    "name": e.name,
                    "start_ms": e.start_ms,
                    "end_ms": e.end_ms,
                    "stream": e.stream,
                    "duration_ms": e.end_ms - e.start_ms,
                    **e.metadata,
                }
                for e in self.events
            ],
            "metrics": self.compute_overlap(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print a human-readable overlap summary."""
        metrics = self.compute_overlap()
        print(f"  Compute:  {metrics['total_compute_ms']:.3f} ms")
        print(f"  Comm:     {metrics['total_comm_ms']:.3f} ms")
        print(f"  Overlap:  {metrics['overlap_ms']:.3f} ms ({metrics['overlap_ratio']:.1%})")
        print(f"  Wall:     {metrics['total_wall_ms']:.3f} ms")
        print(f"  Efficiency: {metrics['efficiency']:.1%}")


# ---------------------------------------------------------------------------
# Communication heatmap
# ---------------------------------------------------------------------------

class CommHeatmap:
    """Records and visualizes tile-level communication patterns.

    Tracks which ranks send data to which other ranks, enabling
    diagnosis of NVLink congestion and load imbalance.

    Usage::

        heatmap = CommHeatmap(world_size=8)
        heatmap.record(src=0, dst=1, bytes_transferred=1024*1024)
        heatmap.record(src=0, dst=2, bytes_transferred=512*1024)
        heatmap.print_matrix()
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        # Traffic matrix: [src][dst] -> total bytes
        self._traffic: list[list[int]] = [
            [0] * world_size for _ in range(world_size)
        ]
        self._count: list[list[int]] = [
            [0] * world_size for _ in range(world_size)
        ]

    def record(self, src: int, dst: int, bytes_transferred: int) -> None:
        """Record a communication event.

        Args:
            src: Source rank.
            dst: Destination rank.
            bytes_transferred: Number of bytes sent.
        """
        self._traffic[src][dst] += bytes_transferred
        self._count[src][dst] += 1

    def get_traffic_matrix(self) -> list[list[int]]:
        """Return the raw traffic matrix (bytes)."""
        return [row[:] for row in self._traffic]

    def get_count_matrix(self) -> list[list[int]]:
        """Return the message count matrix."""
        return [row[:] for row in self._count]

    def print_matrix(self, unit: str = "MB") -> None:
        """Print the traffic matrix as an ASCII table.

        Args:
            unit: Display unit: "B", "KB", "MB", "GB".
        """
        divisors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        div = divisors.get(unit, 1024**2)

        ws = self.world_size
        # Header
        header = "       " + "".join(f"  GPU{j:d}  " for j in range(ws))
        print(header)
        print("  " + "-" * (len(header) - 2))

        for i in range(ws):
            cells = []
            for j in range(ws):
                val = self._traffic[i][j] / div
                if val == 0:
                    cells.append("    -   ")
                else:
                    cells.append(f"{val:7.1f} ")
            print(f"  GPU{i}  " + "".join(cells))

        # Summary
        total = sum(sum(row) for row in self._traffic)
        print(f"\n  Total traffic: {total / div:.1f} {unit}")

        # Check balance
        row_totals = [sum(row) for row in self._traffic]
        if row_totals and max(row_totals) > 0:
            imbalance = max(row_totals) / max(min(r for r in row_totals if r > 0), 1)
            print(f"  Max/min send ratio: {imbalance:.2f}x")

    def export_json(self, path: str | Path) -> None:
        """Export heatmap data to JSON.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "world_size": self.world_size,
            "traffic_bytes": self._traffic,
            "message_count": self._count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
