# SPDX-License-Identifier: Apache-2.0
"""tncc.utils.topology - Hardware topology detection utilities.

Provides functions to detect the GPU backend, query device properties, and
build a topology description that the rest of TNCC uses to select optimal
communication patterns and kernel launch parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TopologyInfo (standalone -- mirrors backends.base.TopologyInfo but is
# usable without importing the backend layer)
# ---------------------------------------------------------------------------

@dataclass
class TopologyInfo:
    """Hardware topology snapshot.

    Attributes:
        backend: ``"cuda"`` or ``"hip"``.
        num_devices: Number of visible GPU devices.
        device_name: Human-readable name of the current device.
        compute_units: SM count (NVIDIA) or CU count (AMD).
        warp_size: 32 for NVIDIA, 64 for AMD.
        global_memory_bytes: VRAM in bytes.
        link_type: Primary interconnect (e.g. ``"NVLink"``, ``"InfinityFabric"``).
        peak_bandwidth_gbps: Estimated peak per-device bandwidth in GB/s.
        peer_access_matrix: ``num_devices x num_devices`` boolean reachability.
    """

    backend: str
    num_devices: int = 1
    device_name: str = "Unknown"
    compute_units: int = 0
    warp_size: int = 32
    global_memory_bytes: int = 0
    link_type: str = "PCIe"
    peak_bandwidth_gbps: float = 0.0
    peer_access_matrix: list[list[bool]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TopologyDetector
# ---------------------------------------------------------------------------

class TopologyDetector:
    """Stateless helper that probes GPU hardware and returns a :class:`TopologyInfo`.

    Usage::

        detector = TopologyDetector()
        info = detector.detect()
        detector.print_info(info)
    """

    def detect(self, backend: str = "auto") -> TopologyInfo:
        """Run full topology detection.

        Args:
            backend: ``"cuda"``, ``"hip"``, or ``"auto"``.

        Returns:
            Populated :class:`TopologyInfo`.
        """
        if backend == "auto":
            backend = detect_backend()
        return detect_topology(backend)

    @staticmethod
    def print_info(info: TopologyInfo) -> None:
        """Pretty-print topology to stdout."""
        print_topology_info(info)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """Detect ``"cuda"`` or ``"hip"`` from the installed PyTorch build.

    Returns:
        ``"cuda"`` or ``"hip"``.

    Raises:
        RuntimeError: If no supported GPU backend is found.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is required for backend detection.")

    hip_version = getattr(torch.version, "hip", None)
    if hip_version is not None:
        return "hip"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "No GPU backend detected. Install PyTorch with CUDA or ROCm support."
    )


def detect_topology(backend: str) -> TopologyInfo:
    """Full topology detection for the given backend.

    Args:
        backend: ``"cuda"`` or ``"hip"``.

    Returns:
        A fully-populated :class:`TopologyInfo`.
    """
    import torch

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_devices == 0:
        return TopologyInfo(backend=backend, num_devices=0)

    # Current device properties
    current_device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device)

    warp_size = get_warp_size(backend)
    compute_units = get_num_compute_units(backend)

    # Build peer-access matrix
    peer_matrix: list[list[bool]] = []
    for i in range(num_devices):
        row: list[bool] = []
        for j in range(num_devices):
            if i == j:
                row.append(True)
            else:
                row.append(torch.cuda.can_device_access_peer(i, j))
        peer_matrix.append(row)

    # Determine link type
    all_connected = all(
        peer_matrix[i][j]
        for i in range(num_devices)
        for j in range(num_devices)
    )

    if backend == "hip":
        link_type = "InfinityFabric" if all_connected else "PCIe"
        # MI250X ~ 200 GB/s, MI300X ~ 896 GB/s aggregate
        peak_bw = 200.0
    else:
        link_type = "NVLink" if all_connected and num_devices > 1 else "PCIe"
        # H100 NVLink ~ 450 GB/s per direction
        peak_bw = 450.0 if link_type == "NVLink" else 32.0

    return TopologyInfo(
        backend=backend,
        num_devices=num_devices,
        device_name=props.name,
        compute_units=compute_units,
        warp_size=warp_size,
        global_memory_bytes=props.total_mem,
        link_type=link_type,
        peak_bandwidth_gbps=peak_bw,
        peer_access_matrix=peer_matrix,
    )


def get_num_compute_units(backend: str) -> int:
    """Return the SM count (NVIDIA) or CU count (AMD) for the current device.

    Args:
        backend: ``"cuda"`` or ``"hip"``.

    Returns:
        Number of compute units, or 0 if unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.multi_processor_count
    except Exception:
        logger.warning("Could not query compute unit count.")
        return 0


def get_warp_size(backend: str) -> int:
    """Return the warp/wavefront size for the given backend.

    Args:
        backend: ``"cuda"`` or ``"hip"``.

    Returns:
        32 for NVIDIA, 64 for AMD.
    """
    if backend == "hip":
        return 64
    return 32


def get_optimal_num_sms(backend: str) -> int:
    """Return the total SM/CU count suitable for persistent kernel launches.

    For persistent-style kernels the grid is typically launched with exactly
    ``NUM_SMS`` blocks so that every SM gets one block.

    Args:
        backend: ``"cuda"`` or ``"hip"``.

    Returns:
        SM/CU count for the current device.
    """
    return get_num_compute_units(backend)


def print_topology_info(info: TopologyInfo) -> None:
    """Pretty-print a :class:`TopologyInfo` to stdout.

    Args:
        info: Topology to display.
    """
    print("=" * 60)
    print("  TNCC Hardware Topology")
    print("=" * 60)
    print(f"  Backend         : {info.backend}")
    print(f"  Device          : {info.device_name}")
    print(f"  Num devices     : {info.num_devices}")
    print(f"  Compute units   : {info.compute_units}")
    print(f"  Warp size       : {info.warp_size}")
    print(f"  VRAM            : {info.global_memory_bytes / (1024**3):.1f} GB")
    print(f"  Interconnect    : {info.link_type}")
    print(f"  Peak BW (est.)  : {info.peak_bandwidth_gbps:.1f} GB/s")

    if info.peer_access_matrix:
        print(f"\n  Peer access matrix ({info.num_devices}x{info.num_devices}):")
        header = "       " + " ".join(f"GPU{j}" for j in range(info.num_devices))
        print(header)
        for i, row in enumerate(info.peer_access_matrix):
            cells = " ".join(f" {'Y' if v else 'N'}  " for v in row)
            print(f"  GPU{i}  {cells}")
    print("=" * 60)
