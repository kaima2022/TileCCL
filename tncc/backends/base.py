"""
tncc.backends.base - Abstract base class for hardware backends.

Defines the BackendInterface that all GPU backends (CUDA, HIP) must implement,
along with dataclasses describing hardware topology and device properties.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class TopologyInfo:
    """Hardware topology information for a multi-GPU system.

    Describes the interconnect topology between GPUs, including link types,
    per-link bandwidth, and the peer-access reachability matrix.
    """

    num_devices: int
    """Total number of GPU devices visible in this system."""

    link_type: str
    """Primary interconnect type, e.g. ``"NVLink"``, ``"NVSwitch"``,
    ``"InfinityFabric"``, ``"PCIe"``."""

    link_bandwidth_gbps: list[float]
    """Per-device peak bidirectional bandwidth in GB/s.
    Length equals *num_devices*; entry *i* is the bandwidth of the link
    connecting device *i* to the fabric."""

    peer_access_matrix: list[list[bool]]
    """``num_devices x num_devices`` boolean matrix.
    ``peer_access_matrix[i][j]`` is ``True`` when device *i* can directly
    access device *j*'s memory without staging through the host."""


@dataclass
class DeviceProperties:
    """Vendor-neutral description of a single GPU device."""

    name: str
    """Human-readable device name, e.g. ``"NVIDIA H100"``."""

    compute_units: int
    """Number of compute units (SMs on NVIDIA, CUs on AMD)."""

    warp_size: int
    """Warp width -- 32 for NVIDIA, 64 for AMD (wavefront)."""

    global_memory_bytes: int
    """Total global (VRAM) memory in bytes."""

    l2_cache_bytes: int
    """L2 cache size in bytes."""

    compute_capability: tuple[int, int]
    """Major/minor compute capability.
    For AMD this maps to the GFX version, e.g. ``(9, 4)`` for gfx942."""

    backend_type: str
    """Backend identifier -- ``"cuda"`` or ``"hip"``."""


class BackendInterface(abc.ABC):
    """Abstract base class that every TNCC hardware backend must implement.

    The interface covers four responsibilities:

    1. **IPC** -- setting up inter-process communication so that each GPU
       process can access every other GPU's memory.
    2. **Memory management** -- allocating / freeing device memory and
       exchanging IPC handles so remote pointers are usable locally.
    3. **Topology detection** -- querying the hardware to learn link types
       and bandwidths, which the upper layers use to pick optimal
       communication patterns.
    4. **Synchronization** -- device-level barriers and fences.
    """

    # ------------------------------------------------------------------
    # IPC
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def init_ipc(self) -> None:
        """Initialize IPC between all GPUs in the current process group.

        After this call, every rank must be able to open IPC handles
        exported by every other rank.  Implementations typically call
        ``torch.distributed.barrier()`` internally.
        """

    @abc.abstractmethod
    def get_ipc_handle(self, ptr: int) -> bytes:
        """Return a serialisable IPC handle for *ptr*.

        Args:
            ptr: Device pointer (as an integer) previously returned by
                :meth:`allocate`.

        Returns:
            An opaque bytes object (64 bytes for both CUDA and HIP) that
            can be transmitted to a peer process and opened via
            :meth:`open_ipc_handle`.
        """

    @abc.abstractmethod
    def open_ipc_handle(self, handle: bytes) -> int:
        """Open an IPC handle received from a peer process.

        Args:
            handle: Opaque handle previously produced by
                :meth:`get_ipc_handle` on a *different* process.

        Returns:
            A device pointer (as an integer) that is valid in **this**
            process's address space and maps the same physical memory.
        """

    def close_ipc_handle(self, ptr: int) -> None:
        """Close an IPC mapping previously opened by :meth:`open_ipc_handle`.

        The pointer becomes invalid after this call.  Implementations
        should call the runtime's close/unmap function (e.g.
        ``cudaIpcCloseMemHandle``).  The default implementation is a no-op
        (process-exit cleanup is sufficient for most runtimes).

        Args:
            ptr: Device pointer obtained from :meth:`open_ipc_handle`.
        """

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def allocate(self, size: int) -> int:
        """Allocate *size* bytes of device memory.

        Args:
            size: Allocation size in bytes.  Must be > 0.

        Returns:
            Device pointer as a Python int.

        Raises:
            RuntimeError: If the underlying allocation call fails.
        """

    @abc.abstractmethod
    def free(self, ptr: int) -> None:
        """Free device memory previously returned by :meth:`allocate`.

        Args:
            ptr: Device pointer to free.

        Raises:
            RuntimeError: If the underlying free call fails.
        """

    @abc.abstractmethod
    def get_heap_bases(self, local_ptr: int, world_size: int) -> "torch.Tensor":
        """Exchange local pointers across all ranks via all-gather.

        Each rank contributes *local_ptr*; after the collective every rank
        holds a tensor of shape ``(world_size,)`` containing all pointers.

        Args:
            local_ptr: This rank's base pointer for the symmetric heap.
            world_size: Total number of ranks in the process group.

        Returns:
            A ``torch.int64`` tensor of length *world_size* on CPU.
        """

    @abc.abstractmethod
    def memcpy_d2d(self, dst: int, src: int, size: int) -> None:
        """Copy *size* bytes from device address *src* to device address *dst*.

        Both pointers must reside in device memory (same device or
        peer-accessible).

        Args:
            dst: Destination device pointer.
            src: Source device pointer.
            size: Number of bytes to copy.

        Raises:
            RuntimeError: If the copy fails.
        """

    # ------------------------------------------------------------------
    # Topology & device info
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def detect_topology(self) -> TopologyInfo:
        """Probe the hardware and return topology information.

        Returns:
            A :class:`TopologyInfo` describing the interconnect.
        """

    @abc.abstractmethod
    def get_device_properties(self) -> DeviceProperties:
        """Return properties of the *current* device.

        Returns:
            A :class:`DeviceProperties` populated from the runtime.
        """

    def enable_peer_access(self, peer_device: int) -> None:
        """Enable direct memory access to *peer_device* from the current device.

        Safe to call multiple times (already-enabled is silently ignored).

        Args:
            peer_device: Device ordinal to enable access to.
        """

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def synchronize(self) -> None:
        """Block the host thread until all pending device work completes."""
