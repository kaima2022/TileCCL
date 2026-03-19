"""
xtile.backends.cuda - NVIDIA CUDA backend for XTile.

Implements :class:`~xtile.backends.base.BackendInterface` by wrapping the
CUDA runtime through ctypes.  On machines without an NVIDIA GPU the module
can still be imported; actual GPU calls will fail with a clear error.

Key NVIDIA-specific values
--------------------------
* CUDA IPC handle size: **64 bytes** (``cudaIpcMemHandle_t``)
* Warp size: **32**
* Primary interconnect: **NVLink / NVSwitch** (falls back to PCIe)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import subprocess
from typing import Optional

import torch
import torch.distributed as dist

from xtile.backends.base import BackendInterface, DeviceProperties, TopologyInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUDA_IPC_HANDLE_SIZE: int = 64
"""Size in bytes of ``cudaIpcMemHandle_t``."""

CUDA_MEMCPY_DEVICE_TO_DEVICE: int = 3
"""cudaMemcpyKind enum value for device-to-device."""

CUDA_SUCCESS: int = 0
"""cudaError_t success code (``cudaSuccess``)."""

NVIDIA_WARP_SIZE: int = 32
"""NVIDIA warp width."""


# ---------------------------------------------------------------------------
# ctypes wrapper around libcudart
# ---------------------------------------------------------------------------

class _CUDARuntime:
    """Thin ctypes wrapper around the CUDA runtime shared library.

    All methods raise :class:`RuntimeError` when the underlying CUDA call
    returns a non-zero error code.
    """

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._load_library()

    # -- library loading ---------------------------------------------------

    def _load_library(self) -> None:
        """Try to load ``libcudart.so`` from standard CUDA locations."""
        cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
        search_paths = [
            os.path.join(cuda_home, "lib64", "libcudart.so") if cuda_home else "",
            os.path.join(cuda_home, "lib", "libcudart.so") if cuda_home else "",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "libcudart.so",
        ]
        for path in search_paths:
            if not path:
                continue
            try:
                self._lib = ctypes.CDLL(path)
                logger.info("Loaded CUDA runtime from %s", path)
                self._setup_signatures()
                return
            except OSError:
                continue

        # Last-resort: let the dynamic linker search
        resolved = ctypes.util.find_library("cudart")
        if resolved:
            try:
                self._lib = ctypes.CDLL(resolved)
                logger.info("Loaded CUDA runtime via find_library: %s", resolved)
                self._setup_signatures()
                return
            except OSError:
                pass

        logger.warning(
            "libcudart.so not found -- CUDA backend will not be functional. "
            "Install the CUDA Toolkit to enable NVIDIA GPU support."
        )

    def _setup_signatures(self) -> None:
        """Declare ctypes argtypes / restypes for every CUDA symbol we use."""
        lib = self._lib
        if lib is None:
            raise RuntimeError("CUDA runtime library not loaded")

        # cudaMalloc(void** devPtr, size_t size) -> cudaError_t
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int

        # cudaFree(void* devPtr) -> cudaError_t
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int

        # cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
        lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.cudaMemcpy.restype = ctypes.c_int

        # cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)
        lib.cudaIpcGetMemHandle.argtypes = [
            ctypes.c_char * CUDA_IPC_HANDLE_SIZE,
            ctypes.c_void_p,
        ]
        lib.cudaIpcGetMemHandle.restype = ctypes.c_int

        # cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
        lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char * CUDA_IPC_HANDLE_SIZE,
            ctypes.c_uint,
        ]
        lib.cudaIpcOpenMemHandle.restype = ctypes.c_int

        # cudaIpcCloseMemHandle(void* devPtr)
        lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

        # cudaDeviceSynchronize()
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int

        # cudaGetDeviceCount(int* count)
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int

        # cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
        lib.cudaDeviceCanAccessPeer.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cudaDeviceCanAccessPeer.restype = ctypes.c_int

        # cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
        lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
        lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int

        # cudaGetDevice(int* device)
        lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDevice.restype = ctypes.c_int

        # cudaSetDevice(int device)
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int

    # -- helpers -----------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return ``True`` if the CUDA shared library was loaded."""
        return self._lib is not None

    def _check(self, err: int, func_name: str) -> None:
        """Raise on non-zero CUDA error code."""
        if err != CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error in {func_name}: error code {err}")

    # -- wrapped APIs ------------------------------------------------------

    def malloc(self, size: int) -> int:
        """``cudaMalloc`` -- allocate *size* bytes, return device pointer."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        ptr = ctypes.c_void_p()
        err = self._lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
        self._check(err, "cudaMalloc")
        return ptr.value or 0

    def free(self, ptr: int) -> None:
        """``cudaFree``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        err = self._lib.cudaFree(ctypes.c_void_p(ptr))
        self._check(err, "cudaFree")

    def memcpy_d2d(self, dst: int, src: int, size: int) -> None:
        """``cudaMemcpy`` with ``cudaMemcpyDeviceToDevice``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        err = self._lib.cudaMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(size),
            ctypes.c_int(CUDA_MEMCPY_DEVICE_TO_DEVICE),
        )
        self._check(err, "cudaMemcpy(D2D)")

    def ipc_get_handle(self, ptr: int) -> bytes:
        """``cudaIpcGetMemHandle`` -- return 64-byte IPC handle."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        handle = (ctypes.c_char * CUDA_IPC_HANDLE_SIZE)()
        err = self._lib.cudaIpcGetMemHandle(handle, ctypes.c_void_p(ptr))
        self._check(err, "cudaIpcGetMemHandle")
        return bytes(handle)

    def ipc_open_handle(self, handle: bytes) -> int:
        """``cudaIpcOpenMemHandle`` -- open peer handle, return local pointer."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        buf = (ctypes.c_char * CUDA_IPC_HANDLE_SIZE).from_buffer_copy(handle)
        ptr = ctypes.c_void_p()
        # flags = 1 : cudaIpcMemLazyEnablePeerAccess
        err = self._lib.cudaIpcOpenMemHandle(ctypes.byref(ptr), buf, ctypes.c_uint(1))
        self._check(err, "cudaIpcOpenMemHandle")
        return ptr.value or 0

    def ipc_close_handle(self, ptr: int) -> None:
        """``cudaIpcCloseMemHandle``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        err = self._lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
        self._check(err, "cudaIpcCloseMemHandle")

    def device_synchronize(self) -> None:
        """``cudaDeviceSynchronize``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        err = self._lib.cudaDeviceSynchronize()
        self._check(err, "cudaDeviceSynchronize")

    def get_device_count(self) -> int:
        """``cudaGetDeviceCount``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        count = ctypes.c_int(0)
        err = self._lib.cudaGetDeviceCount(ctypes.byref(count))
        self._check(err, "cudaGetDeviceCount")
        return count.value

    def get_device(self) -> int:
        """``cudaGetDevice`` -- return currently active device ordinal."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        dev = ctypes.c_int(0)
        err = self._lib.cudaGetDevice(ctypes.byref(dev))
        self._check(err, "cudaGetDevice")
        return dev.value

    def set_device(self, device_id: int) -> None:
        """``cudaSetDevice``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        err = self._lib.cudaSetDevice(ctypes.c_int(device_id))
        self._check(err, "cudaSetDevice")

    def device_can_access_peer(self, device: int, peer: int) -> bool:
        """``cudaDeviceCanAccessPeer``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        can_access = ctypes.c_int(0)
        err = self._lib.cudaDeviceCanAccessPeer(
            ctypes.byref(can_access), ctypes.c_int(device), ctypes.c_int(peer),
        )
        self._check(err, "cudaDeviceCanAccessPeer")
        return can_access.value != 0

    def enable_peer_access(self, peer_device: int) -> None:
        """``cudaDeviceEnablePeerAccess``."""
        if self._lib is None:
            raise RuntimeError("CUDA runtime library not loaded")
        err = self._lib.cudaDeviceEnablePeerAccess(ctypes.c_int(peer_device), ctypes.c_uint(0))
        # Error 50704 / 704 = cudaErrorPeerAccessAlreadyEnabled -- safe to ignore
        if err != CUDA_SUCCESS and err != 704:
            self._check(err, "cudaDeviceEnablePeerAccess")


# Singleton: instantiated once per process import.
_cuda = _CUDARuntime()


# ---------------------------------------------------------------------------
# NVLink / NVSwitch topology helper
# ---------------------------------------------------------------------------

def _detect_nvlink_topology(num_devices: int) -> str:
    """Try to detect NVLink / NVSwitch topology via ``nvidia-smi``.

    Returns:
        ``"NVSwitch"`` if any device uses NVSwitch, ``"NVLink"`` if NVLink
        is detected but not NVSwitch, or ``"PCIe"`` as fallback.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.lower()
        if "nvswitch" in output or "nv12" in output:
            return "NVSwitch"
        if "nvlink" in output or "nv" in output:
            return "NVLink"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.debug("nvidia-smi not available for topology detection")

    # Fallback: try NVML through nvidia-ml-py if installed
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        for i in range(num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for link in range(6):  # NVLink 0-5
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                    if state:
                        pynvml.nvmlShutdown()
                        return "NVLink"
                except pynvml.NVMLError:
                    continue
        pynvml.nvmlShutdown()
    except Exception:
        logger.debug("NVML not available for topology detection")

    return "PCIe"


# ---------------------------------------------------------------------------
# CUDABackend -- BackendInterface implementation
# ---------------------------------------------------------------------------

class CUDABackend(BackendInterface):
    """NVIDIA CUDA implementation of :class:`~xtile.backends.base.BackendInterface`.

    Uses ctypes to call the CUDA runtime directly.  Handle exchange is
    performed via ``torch.distributed.all_gather`` -- no MPI dependency.
    """

    def __init__(self) -> None:
        if not _cuda.available:
            raise RuntimeError(
                "CUDA runtime library not found. "
                "Ensure the CUDA Toolkit is installed and libcudart.so is on the "
                "library path."
            )
        self._device_id: int = _cuda.get_device()

    # -- IPC ---------------------------------------------------------------

    def init_ipc(self) -> None:
        """Enable peer access between all NVIDIA GPUs and synchronize ranks."""
        num_devices = _cuda.get_device_count()
        current = self._device_id

        for peer in range(num_devices):
            if peer == current:
                continue
            if _cuda.device_can_access_peer(current, peer):
                _cuda.enable_peer_access(peer)
                logger.debug("Enabled peer access %d -> %d", current, peer)
            else:
                logger.warning("Peer access %d -> %d not available", current, peer)

        # Barrier so every rank has finished enabling peer access before any
        # IPC handle exchange begins.
        if dist.is_initialized():
            dist.barrier()
        logger.info("CUDA IPC initialised on device %d (%d devices)", current, num_devices)

    # -- Memory management -------------------------------------------------

    def allocate(self, size: int) -> int:
        """Allocate *size* bytes of CUDA device memory."""
        if size <= 0:
            raise ValueError(f"Allocation size must be > 0, got {size}")
        return _cuda.malloc(size)

    def free(self, ptr: int) -> None:
        """Free CUDA device memory."""
        _cuda.free(ptr)

    def get_ipc_handle(self, ptr: int) -> bytes:
        """Return 64-byte CUDA IPC handle for *ptr*."""
        return _cuda.ipc_get_handle(ptr)

    def open_ipc_handle(self, handle: bytes) -> int:
        """Open a remote CUDA IPC handle and return a local device pointer."""
        if len(handle) != CUDA_IPC_HANDLE_SIZE:
            raise ValueError(
                f"Expected {CUDA_IPC_HANDLE_SIZE}-byte IPC handle, got {len(handle)} bytes"
            )
        return _cuda.ipc_open_handle(handle)

    def close_ipc_handle(self, ptr: int) -> None:
        """Close a remote CUDA IPC mapping."""
        _cuda.ipc_close_handle(ptr)

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
        """Device-to-device copy using ``cudaMemcpy``."""
        _cuda.memcpy_d2d(dst, src, size)

    # -- Topology & device info --------------------------------------------

    def detect_topology(self) -> TopologyInfo:
        """Detect NVIDIA GPU topology.

        Queries peer-access capabilities and probes for NVLink / NVSwitch
        via ``nvidia-smi`` or NVML.  Bandwidth values are set to nominal
        NVLink rates where peer access is available, and zero otherwise.
        """
        num_devices = _cuda.get_device_count()

        # Build peer-access matrix
        peer_matrix: list[list[bool]] = []
        for i in range(num_devices):
            row: list[bool] = []
            for j in range(num_devices):
                if i == j:
                    row.append(True)
                else:
                    row.append(_cuda.device_can_access_peer(i, j))
            peer_matrix.append(row)

        # Detect link type
        link_type = _detect_nvlink_topology(num_devices)

        # Nominal bandwidth estimates:
        #   NVSwitch (H100): ~900 GB/s bidirectional
        #   NVLink  (A100):  ~600 GB/s bidirectional
        #   PCIe 4.0 x16:    ~32 GB/s bidirectional
        bw_map = {"NVSwitch": 900.0, "NVLink": 600.0, "PCIe": 32.0}
        nominal_bw = bw_map.get(link_type, 32.0)

        bandwidth: list[float] = []
        for i in range(num_devices):
            has_peer = any(peer_matrix[i][j] for j in range(num_devices) if j != i)
            bandwidth.append(nominal_bw if has_peer else 0.0)

        return TopologyInfo(
            num_devices=num_devices,
            link_type=link_type,
            link_bandwidth_gbps=bandwidth,
            peer_access_matrix=peer_matrix,
        )

    def get_device_properties(self) -> DeviceProperties:
        """Return device properties for the currently active NVIDIA GPU.

        Uses PyTorch's ``torch.cuda.get_device_properties`` which works
        on CUDA builds.
        """
        device_id = self._device_id

        try:
            props = torch.cuda.get_device_properties(device_id)
            return DeviceProperties(
                name=props.name,
                compute_units=props.multi_processor_count,
                warp_size=NVIDIA_WARP_SIZE,
                global_memory_bytes=props.total_mem,
                l2_cache_bytes=getattr(props, "l2_cache_size", 0),
                compute_capability=(props.major, props.minor),
                backend_type="cuda",
            )
        except Exception:
            logger.warning(
                "torch.cuda.get_device_properties() unavailable; "
                "returning placeholder device properties."
            )
            return DeviceProperties(
                name=f"NVIDIA GPU {device_id}",
                compute_units=0,
                warp_size=NVIDIA_WARP_SIZE,
                global_memory_bytes=0,
                l2_cache_bytes=0,
                compute_capability=(0, 0),
                backend_type="cuda",
            )

    def enable_peer_access(self, peer_device: int) -> None:
        """Enable peer access to *peer_device* from the current device."""
        _cuda.enable_peer_access(peer_device)

    # -- Synchronization ---------------------------------------------------

    def synchronize(self) -> None:
        """Block until all CUDA work on the current device completes."""
        _cuda.device_synchronize()
