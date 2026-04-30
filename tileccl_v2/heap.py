# SPDX-License-Identifier: Apache-2.0
"""
tileccl_v2.heap — Symmetric memory heap for multi-GPU communication.

Multi-process IPC from Day 1 (architecture decision #6, modified by Codex).
Uses torch.distributed for IPC handle exchange.

Single-process convenience via create_all() for testing.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_ALIGN = 256  # Byte alignment (GPU cache line)
_IPC_HANDLE_SIZE = 64  # cudaIpcMemHandle_t is 64 bytes

# CUDA runtime for peer access and IPC (not exposed by PyTorch 2.6.0)
# TODO(ROCm): add libamdhip64.so fallback with hipDeviceEnablePeerAccess for AMD GPU support
try:
    _libcudart = ctypes.CDLL("libcudart.so")
except OSError:
    _libcudart = None


class _cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * _IPC_HANDLE_SIZE)]


def _enable_peer_access(device: int, peer: int) -> bool:
    """Enable P2P access from device to peer via CUDA runtime. Returns True on success."""
    if _libcudart is None:
        return False
    ret = _libcudart.cudaDeviceEnablePeerAccess(ctypes.c_int(peer), ctypes.c_int(0))
    # 0 = success, 704 = cudaErrorPeerAccessAlreadyEnabled
    return ret in (0, 704)


def _ipc_get_mem_handle(dev_ptr: int) -> bytes:
    """Get CUDA IPC memory handle for a device pointer."""
    if _libcudart is None:
        raise RuntimeError("libcudart not available for IPC")
    handle = _cudaIpcMemHandle_t()
    ret = _libcudart.cudaIpcGetMemHandle(
        ctypes.byref(handle), ctypes.c_void_p(dev_ptr)
    )
    if ret != 0:
        raise RuntimeError(f"cudaIpcGetMemHandle failed with error {ret}")
    return bytes(handle.reserved)


def _ipc_open_mem_handle(handle_bytes: bytes) -> int:
    """Open a CUDA IPC memory handle and return the mapped device pointer."""
    if _libcudart is None:
        raise RuntimeError("libcudart not available for IPC")
    handle = _cudaIpcMemHandle_t()
    for i, b in enumerate(handle_bytes):
        handle.reserved[i] = b
    dev_ptr = ctypes.c_void_p()
    # cudaIpcMemLazyEnablePeerAccess = 1
    ret = _libcudart.cudaIpcOpenMemHandle(
        ctypes.byref(dev_ptr), handle, ctypes.c_uint(1)
    )
    if ret != 0:
        raise RuntimeError(f"cudaIpcOpenMemHandle failed with error {ret}")
    return dev_ptr.value


def _ipc_close_mem_handle(dev_ptr: int) -> None:
    """Close a CUDA IPC memory handle mapping."""
    if _libcudart is None:
        return
    _libcudart.cudaIpcCloseMemHandle(ctypes.c_void_p(dev_ptr))


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


class SymmetricHeap:
    """Symmetric memory heap for cross-GPU IPC.

    Each rank allocates an identically-sized heap. Heap base addresses are
    exchanged so that translate_ptr can compute remote pointers.

    Multi-process mode (default):
        Requires torch.distributed to be initialized. IPC handles are
        exchanged via all_gather so each rank can access all other heaps.

    Single-process mode (create_all):
        All GPUs managed from one process with cudaEnablePeerAccess.
        Used for testing and single-node development.

    Parameters
    ----------
    size : int
        Heap size in bytes per GPU.
    rank : int
        This GPU's rank (0-indexed).
    world_size : int
        Total number of GPUs.
    """

    def __init__(
        self,
        size: int,
        rank: int,
        world_size: int,
        *,
        _peer_bases: Optional[list[int]] = None,
        group=None,
    ) -> None:
        self._cleaned_up = False  # Must be first for safe __del__

        if size <= 0:
            raise ValueError(f"Heap size must be positive, got {size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank={rank} out of range for world_size={world_size}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")

        self._size = size
        self._rank = rank
        self._world_size = world_size
        self._offset = 0  # bump allocator offset

        self._device = torch.device("cuda", rank)
        with torch.cuda.device(self._device):
            self._buffer = torch.empty(size, dtype=torch.uint8, device=self._device)
        self._local_ptr = self._buffer.data_ptr()

        if _peer_bases is not None:
            # Single-process mode: bases provided by create_all()
            self._mode = "single_process"
            self._heap_bases = torch.tensor(
                _peer_bases, dtype=torch.int64, device=self._device
            )
            self._ipc_handles: list = []
            self._ipc_tensors: list = []
            self._ipc_mapped_ptrs: list = []
        elif world_size == 1:
            self._mode = "single_process"
            self._heap_bases = torch.tensor(
                [self._local_ptr], dtype=torch.int64, device=self._device
            )
            self._ipc_handles = []
            self._ipc_tensors = []
            self._ipc_mapped_ptrs = []
        else:
            # Multi-process mode: exchange IPC handles via torch.distributed
            self._mode = "multiprocess"
            self._ipc_handles = []
            self._ipc_tensors = []
            self._setup_multiprocess(group=group)

        logger.info(
            "Rank %d: SymmetricHeap initialized (%s, %d bytes, mode=%s)",
            rank, self._device, size, self._mode,
        )

    def _setup_multiprocess(self, group=None) -> None:
        """Exchange CUDA IPC handles via torch.distributed all_gather.

        Each rank exports its heap buffer as a CUDA IPC handle, exchanges
        handles with all peers, and opens remote handles to get locally-valid
        mapped pointers. This works across processes on the same machine.

        Parameters
        ----------
        group : Optional[dist.ProcessGroup]
            Process group for all_gather. Should be a gloo group since
            IPC handles are CPU tensors. If None, uses the default group
            (requires gloo or CPU-capable backend).
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized for multi-process "
                "SymmetricHeap. Call dist.init_process_group() first."
            )

        # Get IPC handle for local heap buffer
        local_handle = _ipc_get_mem_handle(self._local_ptr)
        local_handle_tensor = torch.tensor(
            list(local_handle), dtype=torch.uint8
        )  # CPU tensor for gloo

        # Exchange IPC handles via all_gather (gloo works on CPU tensors)
        gathered = [
            torch.zeros(_IPC_HANDLE_SIZE, dtype=torch.uint8)
            for _ in range(self._world_size)
        ]
        dist.all_gather(gathered, local_handle_tensor, group=group)

        # Open remote IPC handles to get locally-valid mapped pointers
        bases = [0] * self._world_size
        bases[self._rank] = self._local_ptr

        self._ipc_mapped_ptrs = []
        for peer in range(self._world_size):
            if peer == self._rank:
                continue
            handle_bytes = bytes(gathered[peer].tolist())
            with torch.cuda.device(self._device):
                mapped_ptr = _ipc_open_mem_handle(handle_bytes)
            bases[peer] = mapped_ptr
            self._ipc_mapped_ptrs.append(mapped_ptr)

        self._heap_bases = torch.tensor(
            bases, dtype=torch.int64, device=self._device
        )
        logger.info(
            "Rank %d: IPC handles exchanged, %d remote heaps mapped",
            self._rank, len(self._ipc_mapped_ptrs),
        )

    @classmethod
    def create_all(
        cls,
        size: int,
        world_size: int,
    ) -> list[SymmetricHeap]:
        """Create heaps for all GPUs from a single process.

        Enables peer access between all GPU pairs. Recommended for testing.
        """
        n_gpus = torch.cuda.device_count()
        if world_size > n_gpus:
            raise RuntimeError(
                f"Requested world_size={world_size} but only {n_gpus} GPUs available"
            )

        # Enable peer access between all pairs
        for i in range(world_size):
            for j in range(world_size):
                if i != j:
                    with torch.cuda.device(i):
                        _enable_peer_access(i, j)

        # Allocate buffers and collect base pointers
        buffers = []
        for rank in range(world_size):
            device = torch.device("cuda", rank)
            with torch.cuda.device(device):
                buf = torch.empty(size, dtype=torch.uint8, device=device)
            buffers.append(buf)

        bases = [buf.data_ptr() for buf in buffers]

        # Create heap objects with pre-computed bases
        heaps = []
        for rank in range(world_size):
            heap = cls(size=size, rank=rank, world_size=world_size, _peer_bases=bases)
            heap._buffer = buffers[rank]
            heap._local_ptr = buffers[rank].data_ptr()
            heaps.append(heap)
        return heaps

    # ------------------------------------------------------------------ query

    @property
    def size(self) -> int:
        return self._size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def local_ptr(self) -> int:
        return self._local_ptr

    def get_heap_bases(self) -> torch.Tensor:
        """Return [world_size] int64 tensor of heap base addresses.

        This tensor lives on this rank's GPU and is passed to Triton
        kernels for translate_ptr.
        """
        return self._heap_bases

    # ------------------------------------------------------------------ alloc

    def allocate_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Bump-allocate a tensor from this rank's heap.

        The returned tensor's data pointer lies within [local_ptr, local_ptr + size).
        """
        if not shape:
            raise ValueError("shape must be non-empty")
        if not all(isinstance(s, int) and s > 0 for s in shape):
            raise ValueError(f"All shape dimensions must be positive integers, got {shape}")
        elem_size = torch.tensor([], dtype=dtype).element_size()
        n_elements = 1
        for s in shape:
            n_elements *= s
        n_bytes = _round_up(n_elements * elem_size, _ALIGN)

        if self._offset + n_bytes > self._size:
            raise RuntimeError(
                f"SymmetricHeap OOM: need {n_bytes} bytes but only "
                f"{self._size - self._offset} remaining "
                f"(total={self._size}, used={self._offset})"
            )

        ptr = self._local_ptr + self._offset
        self._offset += n_bytes

        # Create a tensor view into the heap buffer
        byte_view = self._buffer[self._offset - n_bytes : self._offset - n_bytes + n_elements * elem_size]
        tensor = byte_view.view(dtype).reshape(shape)
        return tensor

    def allocate_signal_buffer(self, n_tiles: int) -> torch.Tensor:
        """Allocate a zero-initialized int32 signal buffer for n_tiles.

        Used as the locks/signals tensor for tile_signal/tile_wait.
        """
        buf = self.allocate_tensor((n_tiles,), dtype=torch.int32)
        buf.zero_()
        return buf

    def reset_allocator(self) -> None:
        """Reset the bump allocator (all previous allocations become invalid)."""
        self._offset = 0

    @property
    def bytes_used(self) -> int:
        return self._offset

    @property
    def bytes_free(self) -> int:
        return self._size - self._offset

    # ------------------------------------------------------------------ cleanup

    def cleanup(self) -> None:
        """Release resources including IPC handle mappings."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        # Close IPC mapped pointers (multi-process mode)
        for ptr in getattr(self, '_ipc_mapped_ptrs', []):
            try:
                _ipc_close_mem_handle(ptr)
            except Exception:
                pass
        self._ipc_mapped_ptrs = []
        self._buffer = None
        if hasattr(self, '_heap_bases'):
            self._heap_bases = None
        if hasattr(self, '_rank'):
            logger.debug("Rank %d: SymmetricHeap cleaned up", self._rank)

    def __enter__(self) -> SymmetricHeap:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"SymmetricHeap(size={self._size}, rank={self._rank}, "
            f"world_size={self._world_size}, mode={self._mode!r})"
        )
