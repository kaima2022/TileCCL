"""
xtile.memory.symmetric_heap - Symmetric memory heap for multi-GPU communication.

Supports two modes of operation:

**Single-process mode** (recommended for single-node):
    All GPUs are managed from one process.  Peer access is enabled between
    GPUs so that kernels on any GPU can directly dereference pointers from
    any other GPU's heap over NVLink.  No IPC handles needed.

    Usage::

        heaps = SymmetricHeap.create_all(size=1 << 30, world_size=2)
        A = heaps[0].allocate_tensor((M, K), dtype=torch.float16)
        bases = heaps[0].get_heap_bases()
        # Launch Triton kernels on GPU 0 with bases containing GPU 1's ptr

**Multi-process mode** (for multi-node or when required):
    Each process owns one GPU.  IPC handles are exchanged via
    ``torch.distributed`` so each process can map remote heaps into
    its address space.  Requires ``torch.distributed`` init first.

    Usage::

        with SymmetricHeap(size=1 << 30, rank=rank, world_size=ws) as heap:
            A = heap.allocate_tensor((M, K), dtype=torch.float16)
            bases = heap.get_heap_bases()
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.distributed as dist

from xtile.backends import get_backend, detect_hardware
from xtile.backends.base import BackendInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALIGN = 256  # Byte alignment for bump allocator (matches GPU cache lines)


def _round_up(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    return (value + alignment - 1) & ~(alignment - 1)


def _resolve_backend(backend: str) -> tuple[str, BackendInterface]:
    """Resolve the backend name and return ``(name, instance)``."""
    if backend == "auto":
        hw = detect_hardware()
        if hw == "none":
            raise RuntimeError(
                "No GPU backend detected. XTile requires CUDA or HIP (ROCm)."
            )
        backend = hw
    return backend, get_backend(backend)


# ---------------------------------------------------------------------------
# Allocation record kept by the bump allocator
# ---------------------------------------------------------------------------

class _AllocRecord:
    """Bookkeeping for a single tensor allocated within the heap."""

    __slots__ = ("offset", "size", "shape", "dtype")

    def __init__(self, offset: int, size: int, shape: tuple[int, ...], dtype: torch.dtype):
        self.offset = offset
        self.size = size
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return (
            f"_AllocRecord(offset=0x{self.offset:x}, size={self.size}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )


# ---------------------------------------------------------------------------
# SymmetricHeap
# ---------------------------------------------------------------------------

class SymmetricHeap:
    """Symmetric memory heap for multi-GPU communication.

    Each GPU allocates the same size heap.  Depending on the mode, either
    peer access (single-process) or IPC handles (multi-process) enable
    cross-GPU memory access via pointer translation.

    Parameters
    ----------
    size : int
        Heap size in bytes **per GPU**.
    rank : int
        This GPU's rank (0-indexed).
    world_size : int
        Total number of GPUs.
    backend : str
        ``"hip"``, ``"cuda"``, or ``"auto"``.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        size: int,
        rank: int,
        world_size: int,
        backend: str = "auto",
        *,
        _peer_bases: Optional[list[int]] = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"Heap size must be positive, got {size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank={rank} out of range for world_size={world_size}"
            )
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")

        self._size: int = size
        self._rank: int = rank
        self._world_size: int = world_size

        # Resolve backend ---------------------------------------------------
        backend_name, backend_impl = _resolve_backend(backend)
        self._backend_name: str = backend_name
        self._backend: BackendInterface = backend_impl

        # Allocate the local heap as one contiguous torch buffer -------------
        logger.info(
            "Rank %d: allocating symmetric heap of %s",
            rank, _human_bytes(size),
        )
        self._device = torch.device("cuda", rank)
        self._buffer: Optional[torch.Tensor] = torch.empty(
            size, dtype=torch.uint8, device=self._device,
        )
        self._local_ptr: int = self._buffer.data_ptr()
        logger.debug("Rank %d: local heap base = 0x%x", rank, self._local_ptr)

        # Build heap_bases ---------------------------------------------------
        self._remote_ptrs: list[int] = []
        self._heap_bases: Optional[torch.Tensor] = None
        self._ipc_opened: list[Optional[int]] = []

        if _peer_bases is not None:
            # Single-process mode: bases provided by create_all()
            self._remote_ptrs = list(_peer_bases)
            self._heap_bases = torch.tensor(
                _peer_bases, dtype=torch.int64, device=self._device,
            )
        elif world_size == 1:
            # Trivial single-GPU case
            self._remote_ptrs = [self._local_ptr]
            self._heap_bases = torch.tensor(
                [self._local_ptr], dtype=torch.int64, device=self._device,
            )
        else:
            # Multi-process mode: try IPC, fall back to all_gather of pointers
            self._setup_multiprocess()

        # Bump allocator state -----------------------------------------------
        self._bump_offset: int = 0
        self._alloc_records: list[_AllocRecord] = []

        # Cleanup flag -------------------------------------------------------
        self._cleaned_up: bool = False

    # ---------------------------------------------------------- class methods

    @classmethod
    def create_all(
        cls,
        size: int,
        world_size: int,
        backend: str = "auto",
    ) -> list["SymmetricHeap"]:
        """Create symmetric heaps for ALL GPUs from a single process.

        This is the recommended entry point for single-node multi-GPU.
        Peer access is enabled between all GPU pairs, and each heap's
        ``heap_bases`` contains direct device pointers (no IPC).

        Parameters
        ----------
        size : int
            Heap size in bytes per GPU.
        world_size : int
            Number of GPUs to use (must be <= ``torch.cuda.device_count()``).
        backend : str
            ``"cuda"``, ``"hip"``, or ``"auto"``.

        Returns
        -------
        list[SymmetricHeap]
            One heap per GPU rank.  ``heaps[i]`` is on ``cuda:i``.
        """
        num_gpus = torch.cuda.device_count()
        if world_size > num_gpus:
            raise RuntimeError(
                f"Requested world_size={world_size} but only {num_gpus} "
                f"GPUs available"
            )

        # Phase 1: Enable peer access between all GPU pairs via backend.
        backend_name, backend_impl = _resolve_backend(backend)
        for i in range(world_size):
            torch.cuda.set_device(i)
            for j in range(world_size):
                if i != j and torch.cuda.can_device_access_peer(i, j):
                    try:
                        backend_impl.enable_peer_access(j)
                    except RuntimeError:
                        pass  # already enabled

        # Phase 2: Allocate heap buffers on each GPU.
        buffers: list[torch.Tensor] = []
        bases: list[int] = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            buf = torch.empty(size, dtype=torch.uint8, device=f"cuda:{rank}")
            buffers.append(buf)
            bases.append(buf.data_ptr())

        # Phase 3: Create SymmetricHeap objects with peer bases.
        heaps: list[SymmetricHeap] = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            heap = cls.__new__(cls)
            heap._size = size
            heap._rank = rank
            heap._world_size = world_size
            backend_name, backend_impl = _resolve_backend(backend)
            heap._backend_name = backend_name
            heap._backend = backend_impl
            heap._device = torch.device("cuda", rank)
            heap._buffer = buffers[rank]
            heap._local_ptr = bases[rank]
            heap._remote_ptrs = list(bases)
            heap._ipc_opened = []
            heap._heap_bases = torch.tensor(
                bases, dtype=torch.int64, device=f"cuda:{rank}",
            )
            heap._bump_offset = 0
            heap._alloc_records = []
            heap._cleaned_up = False
            heaps.append(heap)

        logger.info(
            "Created %d symmetric heaps (%s each) with peer access",
            world_size, _human_bytes(size),
        )
        return heaps

    # ---------------------------------------------------------- multi-process

    def _setup_multiprocess(self) -> None:
        """Exchange heap pointers across ranks for multi-process mode.

        First tries CUDA IPC handles.  If IPC is unavailable (e.g. some
        container environments), falls back to exchanging raw pointers
        with peer access enabled -- this requires all processes to be on
        the same node with NVLink/NVSwitch.
        """
        self._backend.init_ipc()

        # Try IPC handle exchange
        try:
            local_handle: bytes = self._backend.get_ipc_handle(self._local_ptr)
            handle_len = len(local_handle)

            # Exchange handles as GPU uint8 tensors (NCCL-compatible)
            local_t = torch.tensor(
                list(local_handle), dtype=torch.uint8, device=self._device,
            )
            gathered = [
                torch.zeros(handle_len, dtype=torch.uint8, device=self._device)
                for _ in range(self._world_size)
            ]
            dist.all_gather(gathered, local_t)

            # Open each remote handle
            remote_ptrs: list[int] = []
            ipc_opened: list[Optional[int]] = []
            for r in range(self._world_size):
                if r == self._rank:
                    remote_ptrs.append(self._local_ptr)
                    ipc_opened.append(None)
                else:
                    handle_bytes = bytes(gathered[r].cpu().tolist())
                    remote_ptr = self._backend.open_ipc_handle(handle_bytes)
                    remote_ptrs.append(remote_ptr)
                    ipc_opened.append(remote_ptr)

            self._remote_ptrs = remote_ptrs
            self._ipc_opened = ipc_opened
            self._heap_bases = torch.tensor(
                remote_ptrs, dtype=torch.int64, device=self._device,
            )
            dist.barrier()
            logger.info("Rank %d: IPC setup complete", self._rank)
            return

        except RuntimeError as e:
            logger.warning(
                "Rank %d: IPC handle exchange failed (%s), "
                "falling back to peer-access pointer exchange",
                self._rank, e,
            )

        # Fallback: exchange raw base pointers via all_gather.
        # This works when all ranks are on the same node with peer access.
        local_base = torch.tensor(
            [self._local_ptr], dtype=torch.int64, device=self._device,
        )
        all_bases = [
            torch.zeros(1, dtype=torch.int64, device=self._device)
            for _ in range(self._world_size)
        ]
        dist.all_gather(all_bases, local_base)

        self._remote_ptrs = [int(b.item()) for b in all_bases]
        self._heap_bases = torch.cat(all_bases).to(device=self._device)
        self._ipc_opened = []
        dist.barrier()
        logger.info("Rank %d: peer-access pointer exchange complete", self._rank)

    # ---------------------------------------------------------- allocation

    def allocate_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate a tensor within the symmetric heap.

        The returned tensor is a view of the pre-allocated heap buffer.
        Its ``data_ptr()`` lies within ``[heap_base, heap_base + size)``.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired tensor shape.
        dtype : torch.dtype
            Element data type.

        Returns
        -------
        torch.Tensor

        Raises
        ------
        RuntimeError
            If the heap does not have enough free space.
        """
        if self._buffer is None:
            raise RuntimeError("SymmetricHeap has been cleaned up")

        numel = math.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        byte_size = numel * element_size

        aligned_offset = _round_up(self._bump_offset, _ALIGN)

        if aligned_offset + byte_size > self._size:
            raise RuntimeError(
                f"Symmetric heap exhausted on rank {self._rank}: "
                f"requested {byte_size} bytes at offset 0x{aligned_offset:x}, "
                f"but heap size is {self._size} bytes "
                f"({self._size - aligned_offset} bytes remaining)"
            )

        byte_slice = self._buffer.narrow(0, aligned_offset, byte_size)
        tensor = byte_slice.view(dtype).reshape(shape)

        record = _AllocRecord(
            offset=aligned_offset, size=byte_size, shape=shape, dtype=dtype,
        )
        self._alloc_records.append(record)
        self._bump_offset = aligned_offset + byte_size

        logger.debug(
            "Rank %d: allocated tensor %s %s at heap+0x%x (ptr 0x%x)",
            self._rank, shape, dtype, aligned_offset,
            self._local_ptr + aligned_offset,
        )
        return tensor

    # ---------------------------------------------------------- query

    def get_heap_bases(self) -> torch.Tensor:
        """Return a ``(world_size,)`` int64 tensor of all ranks' heap bases.

        Entry ``i`` is the device pointer to rank *i*'s heap, directly
        dereferenceable by kernels on this GPU (via peer access or IPC).
        Pass this tensor to ``@triton.jit`` kernels for :func:`translate_ptr`.
        """
        if self._heap_bases is None:
            raise RuntimeError("Heap bases not initialized.")
        return self._heap_bases

    @property
    def local_base(self) -> int:
        """Device pointer to this rank's heap base."""
        return self._local_ptr

    @property
    def size(self) -> int:
        """Heap size in bytes."""
        return self._size

    @property
    def rank(self) -> int:
        """This GPU's rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of GPUs."""
        return self._world_size

    @property
    def bytes_allocated(self) -> int:
        """Number of bytes currently allocated in this heap."""
        return self._bump_offset

    @property
    def bytes_free(self) -> int:
        """Remaining free bytes in this heap."""
        return self._size - self._bump_offset

    # ---------------------------------------------------------- translation

    def translate(self, local_ptr: int, to_rank: int) -> int:
        """Host-side pointer translation (primarily for debugging).

        Converts a pointer within **this** rank's heap to the equivalent
        pointer in *to_rank*'s heap.
        """
        if to_rank < 0 or to_rank >= self._world_size:
            raise ValueError(
                f"to_rank={to_rank} out of range [0, {self._world_size})"
            )
        offset = self.get_offset(local_ptr)
        return self._remote_ptrs[to_rank] + offset

    def get_offset(self, ptr: int) -> int:
        """Return the byte offset of *ptr* within this rank's heap."""
        offset = ptr - self._local_ptr
        if offset < 0 or offset >= self._size:
            raise ValueError(
                f"Pointer 0x{ptr:x} is outside the heap "
                f"[0x{self._local_ptr:x}, 0x{self._local_ptr + self._size:x})"
            )
        return offset

    # ---------------------------------------------------------- sync

    def barrier(self) -> None:
        """Synchronize the current GPU stream.

        For multi-process mode, also calls ``dist.barrier()``.
        """
        self._backend.synchronize()
        if self._world_size > 1 and dist.is_initialized():
            dist.barrier()

    # ---------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Release resources.  Safe to call multiple times."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        logger.info("Rank %d: cleaning up symmetric heap", self._rank)

        # Close IPC handles if any
        for ptr in self._ipc_opened:
            if ptr is not None:
                try:
                    self._backend.close_ipc_handle(ptr)
                except Exception:
                    logger.debug(
                        "Rank %d: failed to close IPC handle 0x%x",
                        self._rank, ptr, exc_info=True,
                    )

        self._buffer = None
        self._local_ptr = 0
        self._heap_bases = None
        self._remote_ptrs = []
        self._ipc_opened = []
        self._alloc_records.clear()
        self._bump_offset = 0

    # ---------------------------------------------------------- dunder

    def __enter__(self) -> "SymmetricHeap":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SymmetricHeap(size={_human_bytes(self._size)}, "
            f"rank={self._rank}/{self._world_size}, "
            f"backend={self._backend_name!r}, "
            f"allocated={_human_bytes(self._bump_offset)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    """Return a human-readable byte string (e.g. ``'1.00 GiB'``)."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0  # type: ignore[assignment]
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"
