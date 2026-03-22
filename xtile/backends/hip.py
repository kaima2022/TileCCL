"""
xtile.backends.hip - AMD HIP backend for XTile.

Implements :class:`~xtile.backends.base.BackendInterface` by wrapping the
HIP runtime through ctypes.  The module is structured so that it can be
imported on any machine (missing ``libamdhip64.so`` is handled gracefully),
but actual GPU calls will only succeed when ROCm is installed and an AMD
GPU is present.

Key AMD-specific values
-----------------------
* HIP IPC handle size: **64 bytes**
* Wavefront (warp) size: **64**
* Primary interconnect: **Infinity Fabric**
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct
from typing import Optional

import torch
import torch.distributed as dist

from xtile.backends.base import BackendInterface, DeviceProperties, TopologyInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIP_IPC_HANDLE_SIZE: int = 64
"""Size in bytes of ``hipIpcMemHandle_t``."""

HIP_MEMCPY_DEVICE_TO_DEVICE: int = 3
"""hipMemcpyDeviceToDevice enum value."""

HIP_SUCCESS: int = 0
"""hipError_t success code."""

AMD_WARP_SIZE: int = 64
"""AMD wavefront width."""


class HipIpcMemHandle(ctypes.Structure):
    """ctypes Structure matching ``hipIpcMemHandle_t``.

    Must be a Structure (not a ctypes Array) so that ctypes passes the
    64-byte payload **by value** on the C calling convention, matching the
    HIP API signature::

        hipError_t hipIpcOpenMemHandle(void **devPtr,
                                       hipIpcMemHandle_t handle,  // by value
                                       unsigned int flags);
    """
    _fields_ = [("reserved", ctypes.c_char * HIP_IPC_HANDLE_SIZE)]


# ---------------------------------------------------------------------------
# ctypes wrapper around libamdhip64
# ---------------------------------------------------------------------------

class _HIPRuntime:
    """Thin ctypes wrapper around the HIP runtime shared library.

    All methods raise :class:`RuntimeError` when the underlying HIP call
    returns a non-zero error code.
    """

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._load_library()

    # -- library loading ---------------------------------------------------

    def _load_library(self) -> None:
        """Try to load ``libamdhip64.so`` from standard ROCm locations."""
        search_paths = [
            os.environ.get("HIP_PATH", "") + "/lib/libamdhip64.so",
            "/opt/rocm/lib/libamdhip64.so",
            "libamdhip64.so",
        ]
        for path in search_paths:
            if not path or path == "/lib/libamdhip64.so":
                continue
            try:
                self._lib = ctypes.CDLL(path)
                logger.info("Loaded HIP runtime from %s", path)
                self._setup_signatures()
                return
            except OSError:
                continue

        # Last-resort: let the dynamic linker search
        resolved = ctypes.util.find_library("amdhip64")
        if resolved:
            try:
                self._lib = ctypes.CDLL(resolved)
                logger.info("Loaded HIP runtime via find_library: %s", resolved)
                self._setup_signatures()
                return
            except OSError:
                pass

        logger.warning(
            "libamdhip64.so not found -- HIP backend will not be functional. "
            "Install ROCm to enable AMD GPU support."
        )

    def _setup_signatures(self) -> None:
        """Declare ctypes argtypes / restypes for every HIP symbol we use."""
        lib = self._lib
        if lib is None:
            raise RuntimeError("HIP runtime library not loaded")

        # hipMalloc(void** ptr, size_t size) -> hipError_t
        lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.hipMalloc.restype = ctypes.c_int

        # hipFree(void* ptr) -> hipError_t
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipFree.restype = ctypes.c_int

        # hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind)
        lib.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.hipMemcpy.restype = ctypes.c_int

        # hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr)
        lib.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(HipIpcMemHandle),
            ctypes.c_void_p,
        ]
        lib.hipIpcGetMemHandle.restype = ctypes.c_int

        # hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags)
        # NOTE: handle is passed BY VALUE (64-byte struct on the stack).
        # Using ctypes.Structure ensures by-value semantics; a ctypes Array
        # would decay to a pointer, causing hipErrorInvalidValue.
        lib.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            HipIpcMemHandle,
            ctypes.c_uint,
        ]
        lib.hipIpcOpenMemHandle.restype = ctypes.c_int

        # hipIpcCloseMemHandle(void* devPtr)
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.hipIpcCloseMemHandle.restype = ctypes.c_int

        # hipDeviceSynchronize()
        lib.hipDeviceSynchronize.argtypes = []
        lib.hipDeviceSynchronize.restype = ctypes.c_int

        # hipGetDeviceCount(int* count)
        lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDeviceCount.restype = ctypes.c_int

        # hipDeviceCanAccessPeer(int* canAccess, int deviceId, int peerDeviceId)
        lib.hipDeviceCanAccessPeer.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.hipDeviceCanAccessPeer.restype = ctypes.c_int

        # hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
        lib.hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
        lib.hipDeviceEnablePeerAccess.restype = ctypes.c_int

        # hipGetDevice(int* deviceId)
        lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDevice.restype = ctypes.c_int

        # hipSetDevice(int deviceId)
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipSetDevice.restype = ctypes.c_int

    # -- helpers -----------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return ``True`` if the HIP shared library was loaded."""
        return self._lib is not None

    def _check(self, err: int, func_name: str) -> None:
        """Raise on non-zero HIP error code."""
        if err != HIP_SUCCESS:
            raise RuntimeError(f"HIP error in {func_name}: error code {err}")

    # -- wrapped APIs ------------------------------------------------------

    def malloc(self, size: int) -> int:
        """``hipMalloc`` -- allocate *size* bytes, return device pointer."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        ptr = ctypes.c_void_p()
        err = self._lib.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
        self._check(err, "hipMalloc")
        return ptr.value or 0

    def free(self, ptr: int) -> None:
        """``hipFree``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        err = self._lib.hipFree(ctypes.c_void_p(ptr))
        self._check(err, "hipFree")

    def memcpy_d2d(self, dst: int, src: int, size: int) -> None:
        """``hipMemcpy`` with ``hipMemcpyDeviceToDevice``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        err = self._lib.hipMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(size),
            ctypes.c_int(HIP_MEMCPY_DEVICE_TO_DEVICE),
        )
        self._check(err, "hipMemcpy(D2D)")

    def ipc_get_handle(self, ptr: int) -> bytes:
        """``hipIpcGetMemHandle`` -- return 64-byte IPC handle."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        handle = HipIpcMemHandle()
        err = self._lib.hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
        self._check(err, "hipIpcGetMemHandle")
        return ctypes.string_at(ctypes.byref(handle), HIP_IPC_HANDLE_SIZE)

    def ipc_open_handle(self, handle: bytes) -> int:
        """``hipIpcOpenMemHandle`` -- open peer handle, return local pointer."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        h = HipIpcMemHandle.from_buffer_copy(handle)
        ptr = ctypes.c_void_p()
        # flags = 0 : default (hipIpcMemLazyEnablePeerAccess)
        err = self._lib.hipIpcOpenMemHandle(ctypes.byref(ptr), h, ctypes.c_uint(0))
        self._check(err, "hipIpcOpenMemHandle")
        return ptr.value or 0

    def ipc_close_handle(self, ptr: int) -> None:
        """``hipIpcCloseMemHandle``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        err = self._lib.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
        self._check(err, "hipIpcCloseMemHandle")

    def device_synchronize(self) -> None:
        """``hipDeviceSynchronize``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        err = self._lib.hipDeviceSynchronize()
        self._check(err, "hipDeviceSynchronize")

    def get_device_count(self) -> int:
        """``hipGetDeviceCount``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        count = ctypes.c_int(0)
        err = self._lib.hipGetDeviceCount(ctypes.byref(count))
        self._check(err, "hipGetDeviceCount")
        return count.value

    def get_device(self) -> int:
        """``hipGetDevice`` -- return currently active device ordinal."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        dev = ctypes.c_int(0)
        err = self._lib.hipGetDevice(ctypes.byref(dev))
        self._check(err, "hipGetDevice")
        return dev.value

    def set_device(self, device_id: int) -> None:
        """``hipSetDevice``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        err = self._lib.hipSetDevice(ctypes.c_int(device_id))
        self._check(err, "hipSetDevice")

    def device_can_access_peer(self, device: int, peer: int) -> bool:
        """``hipDeviceCanAccessPeer``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        can_access = ctypes.c_int(0)
        err = self._lib.hipDeviceCanAccessPeer(
            ctypes.byref(can_access), ctypes.c_int(device), ctypes.c_int(peer),
        )
        self._check(err, "hipDeviceCanAccessPeer")
        return can_access.value != 0

    def enable_peer_access(self, peer_device: int) -> None:
        """``hipDeviceEnablePeerAccess``."""
        if self._lib is None:
            raise RuntimeError("HIP runtime library not loaded")
        err = self._lib.hipDeviceEnablePeerAccess(ctypes.c_int(peer_device), ctypes.c_uint(0))
        # Error 704 = hipErrorPeerAccessAlreadyEnabled -- safe to ignore
        if err != HIP_SUCCESS and err != 704:
            self._check(err, "hipDeviceEnablePeerAccess")


# Singleton: instantiated once per process import.
_hip = _HIPRuntime()


# ---------------------------------------------------------------------------
# HIPBackend -- BackendInterface implementation
# ---------------------------------------------------------------------------

class HIPBackend(BackendInterface):
    """AMD HIP implementation of :class:`~xtile.backends.base.BackendInterface`.

    Uses ctypes to call the HIP runtime directly (no Python bindings
    required beyond ``libamdhip64.so``).  Handle exchange is performed via
    ``torch.distributed.all_gather`` so there is **no MPI dependency**.
    """

    def __init__(self) -> None:
        if not _hip.available:
            raise RuntimeError(
                "HIP runtime library not found. "
                "Ensure ROCm is installed and libamdhip64.so is on the library path."
            )
        self._device_id: int = _hip.get_device()

    # -- IPC ---------------------------------------------------------------

    def init_ipc(self) -> None:
        """Enable peer access between all AMD GPUs and synchronize ranks."""
        num_devices = _hip.get_device_count()
        current = self._device_id

        for peer in range(num_devices):
            if peer == current:
                continue
            if _hip.device_can_access_peer(current, peer):
                _hip.enable_peer_access(peer)
                logger.debug("Enabled peer access %d -> %d", current, peer)
            else:
                logger.warning("Peer access %d -> %d not available", current, peer)

        # Barrier so every rank has finished enabling peer access before any
        # IPC handle exchange begins.
        if dist.is_initialized():
            dist.barrier()
        logger.info("HIP IPC initialised on device %d (%d devices)", current, num_devices)

    # -- Memory management -------------------------------------------------

    def allocate(self, size: int) -> int:
        """Allocate *size* bytes of HIP device memory."""
        if size <= 0:
            raise ValueError(f"Allocation size must be > 0, got {size}")
        return _hip.malloc(size)

    def free(self, ptr: int) -> None:
        """Free HIP device memory."""
        _hip.free(ptr)

    def get_ipc_handle(self, ptr: int) -> bytes:
        """Return 64-byte HIP IPC handle for *ptr*."""
        return _hip.ipc_get_handle(ptr)

    def open_ipc_handle(self, handle: bytes) -> int:
        """Open a remote HIP IPC handle and return a local device pointer."""
        if len(handle) != HIP_IPC_HANDLE_SIZE:
            raise ValueError(
                f"Expected {HIP_IPC_HANDLE_SIZE}-byte IPC handle, got {len(handle)} bytes"
            )
        return _hip.ipc_open_handle(handle)

    def close_ipc_handle(self, ptr: int) -> None:
        """Close a remote HIP IPC mapping."""
        _hip.ipc_close_handle(ptr)

    def get_heap_bases(self, local_ptr: int, world_size: int) -> torch.Tensor:
        """Exchange base pointers across all ranks via torch.distributed.

        Uses ``all_gather`` on a 1-element int64 tensor so that every rank
        ends up with the full set of base pointers.
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialised before get_heap_bases()")

        local_tensor = torch.tensor([local_ptr], dtype=torch.int64)
        gathered: list[torch.Tensor] = [
            torch.zeros(1, dtype=torch.int64) for _ in range(world_size)
        ]
        dist.all_gather(gathered, local_tensor)
        return torch.cat(gathered)

    def memcpy_d2d(self, dst: int, src: int, size: int) -> None:
        """Device-to-device copy using ``hipMemcpy``."""
        _hip.memcpy_d2d(dst, src, size)

    # -- Topology & device info --------------------------------------------

    def detect_topology(self) -> TopologyInfo:
        """Detect AMD GPU topology.

        Queries peer-access capabilities and assumes Infinity Fabric
        interconnect for peer-accessible device pairs.  Bandwidth values
        are set to nominal Infinity Fabric rates where peer access is
        available, and zero otherwise.

        Note:
            Precise per-link bandwidth detection on AMD requires parsing
            ``/sys/class/drm/card*/device/`` sysfs entries or using
            ``rocm-smi``.  This implementation provides a conservative
            default that upper layers can override.
        """
        num_devices = _hip.get_device_count()

        # Build peer-access matrix
        peer_matrix: list[list[bool]] = []
        for i in range(num_devices):
            row: list[bool] = []
            for j in range(num_devices):
                if i == j:
                    row.append(True)
                else:
                    row.append(_hip.device_can_access_peer(i, j))
            peer_matrix.append(row)

        # Determine primary link type.  If all peers are accessible we
        # assume a fully-connected Infinity Fabric topology.
        all_connected = all(
            peer_matrix[i][j] for i in range(num_devices) for j in range(num_devices)
        )
        link_type = "InfinityFabric" if all_connected else "PCIe"

        # Nominal bandwidth estimate: AMD MI250X Infinity Fabric ~ 200 GB/s
        # per direction, MI300X ~ 896 GB/s aggregate.  We use a conservative
        # placeholder that callers can refine.
        _NOMINAL_IF_BW_GBPS = 200.0
        bandwidth: list[float] = []
        for i in range(num_devices):
            has_fabric_peer = any(peer_matrix[i][j] for j in range(num_devices) if j != i)
            bandwidth.append(_NOMINAL_IF_BW_GBPS if has_fabric_peer else 0.0)

        return TopologyInfo(
            num_devices=num_devices,
            link_type=link_type,
            link_bandwidth_gbps=bandwidth,
            peer_access_matrix=peer_matrix,
        )

    def get_device_properties(self) -> DeviceProperties:
        """Return device properties for the currently active AMD GPU.

        This queries the device through PyTorch's ROCm integration
        (``torch.cuda.get_device_properties``) which works on HIP builds
        of PyTorch.  We fall back to conservative defaults if the call
        is unavailable.
        """
        device_id = self._device_id

        try:
            # On ROCm builds of PyTorch, torch.cuda.* functions map to HIP
            props = torch.cuda.get_device_properties(device_id)
            return DeviceProperties(
                name=props.name,
                compute_units=props.multi_processor_count,
                warp_size=AMD_WARP_SIZE,
                global_memory_bytes=props.total_mem,
                l2_cache_bytes=getattr(props, "l2_cache_size", 0),
                compute_capability=(props.major, props.minor),
                backend_type="hip",
            )
        except Exception:
            logger.warning(
                "torch.cuda.get_device_properties() unavailable; "
                "returning placeholder device properties."
            )
            return DeviceProperties(
                name=f"AMD GPU {device_id}",
                compute_units=0,
                warp_size=AMD_WARP_SIZE,
                global_memory_bytes=0,
                l2_cache_bytes=0,
                compute_capability=(0, 0),
                backend_type="hip",
            )

    # -- Synchronization ---------------------------------------------------

    def synchronize(self) -> None:
        """Block until all HIP work on the current device completes."""
        _hip.device_synchronize()
