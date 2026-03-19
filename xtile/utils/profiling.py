"""xtile.utils.profiling - Profiling and benchmarking utilities.

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
