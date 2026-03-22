"""xtile.utils - Utility modules for topology detection and profiling."""

from xtile.utils.topology import TopologyDetector
from xtile.utils.profiling import TileProfiler
from xtile.utils.benchmark_results import (
    canonical_benchmark_run,
    default_gemm_benchmark_path,
    default_p2p_benchmark_path,
    default_pattern_benchmark_path,
    describe_runtime_metadata_snapshot,
    describe_runtime_support_snapshot,
    figures_data_dir,
    is_canonical_benchmark_output,
    project_root,
    read_json,
    runtime_metadata_snapshot,
    runtime_support_snapshot,
    write_json,
)
from xtile.utils.feature_gates import (
    FORCE_MULTIPROCESS_TRANSPORT_ENV,
    MULTIPROCESS_DEVICE_COLLECTIVES_ENV,
    forced_multiprocess_transport,
    multiprocess_device_collectives_detail,
    multiprocess_device_collectives_enabled,
    multiprocess_device_collectives_transport_supported,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_transport_supported,
)

__all__ = [
    "TopologyDetector",
    "TileProfiler",
    "project_root",
    "figures_data_dir",
    "default_gemm_benchmark_path",
    "default_p2p_benchmark_path",
    "default_pattern_benchmark_path",
    "is_canonical_benchmark_output",
    "canonical_benchmark_run",
    "runtime_metadata_snapshot",
    "runtime_support_snapshot",
    "describe_runtime_metadata_snapshot",
    "describe_runtime_support_snapshot",
    "FORCE_MULTIPROCESS_TRANSPORT_ENV",
    "MULTIPROCESS_DEVICE_COLLECTIVES_ENV",
    "forced_multiprocess_transport",
    "multiprocess_device_collectives_enabled",
    "multiprocess_device_collectives_detail",
    "multiprocess_device_collectives_transport_supported",
    "multiprocess_device_remote_access_detail",
    "multiprocess_device_remote_access_transport_supported",
    "write_json",
    "read_json",
]
