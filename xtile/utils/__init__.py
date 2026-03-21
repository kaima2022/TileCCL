"""xtile.utils - Utility modules for topology detection and profiling."""

from xtile.utils.topology import TopologyDetector
from xtile.utils.profiling import TileProfiler
from xtile.utils.benchmark_results import (
    default_pattern_benchmark_path,
    figures_data_dir,
    project_root,
    read_json,
    write_json,
)

__all__ = [
    "TopologyDetector",
    "TileProfiler",
    "project_root",
    "figures_data_dir",
    "default_pattern_benchmark_path",
    "write_json",
    "read_json",
]
