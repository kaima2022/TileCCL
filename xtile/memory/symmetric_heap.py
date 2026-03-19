"""
xtile.memory.symmetric_heap - Symmetric memory heap for multi-GPU communication.

Each GPU allocates the same size heap, exchanges IPC handles via
torch.distributed.all_gather, and builds a heap_bases tensor so that every
GPU can translate local pointers to remote pointers.

Inspired by Iris's symmetric memory model and OpenSHMEM concepts.
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
    """Resolve the backend name and return ``(name, instance)``.

    Uses the central :func:`xtile.backends.get_backend` factory so that
    backend construction logic is not duplicated.
    """
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

    Each GPU allocates the same size heap.  IPC handles are exchanged so
    every GPU can access every other GPU's heap via pointer translation.

    Inspired by Iris's symmetric memory model and OpenSHMEM concepts.

    Usage::

        with SymmetricHeap(size=1 << 30, rank=rank, world_size=ws) as heap:
            A = heap.allocate_tensor((M, K), dtype=torch.float16)
            bases = heap.get_heap_bases()
            # ... launch Triton kernels with *bases* as a parameter ...
            heap.barrier()

    Parameters
    ----------
    size : int
        Heap size in bytes **per GPU**.  Every rank allocates the same amount.
    rank : int
        This process's rank inside the ``torch.distributed`` process group.
    world_size : int
        Total number of ranks (GPUs) in the group.
    backend : str
        ``"hip"``, ``"cuda"``, or ``"auto"`` (auto-detect from PyTorch).
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        size: int,
        rank: int,
        world_size: int,
        backend: str = "auto",
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

        # Allocate the local heap on this GPU --------------------------------
        logger.info(
            "Rank %d: allocating symmetric heap of %s",
            rank,
            _human_bytes(size),
        )
        self._local_ptr: int = self._backend.allocate(size)
        logger.debug("Rank %d: local heap base = 0x%x", rank, self._local_ptr)

        # Exchange IPC handles and build heap_bases --------------------------
        self._remote_ptrs: list[int] = []  # remote mappings (index = rank)
        self._heap_bases: Optional[torch.Tensor] = None
        self._ipc_setup_done: bool = False
        self._setup_ipc()

        # Bump allocator state -----------------------------------------------
        self._bump_offset: int = 0  # next free byte (relative to heap start)
        self._alloc_records: list[_AllocRecord] = []

        # Cleanup flag -------------------------------------------------------
        self._cleaned_up: bool = False

    # ---------------------------------------------------------- IPC helpers

    def _setup_ipc(self) -> None:
        """Exchange IPC handles across all ranks and populate *_heap_bases*."""
        if self._world_size == 1:
            # Single-GPU fast path -- no IPC needed.
            self._remote_ptrs = [self._local_ptr]
            self._heap_bases = torch.tensor(
                [self._local_ptr], dtype=torch.int64, device=f"cuda:{self._rank}"
            )
            self._ipc_setup_done = True
            return

        # 1. Initialise IPC on the backend (may be a no-op).
        self._backend.init_ipc()

        # 2. Get this rank's IPC handle for the heap allocation.
        local_handle: bytes = self._backend.get_ipc_handle(self._local_ptr)

        # 3. All-gather handles across ranks.
        #    We wrap the handle bytes into a uint8 tensor for transport.
        handle_len = len(local_handle)
        local_tensor = torch.frombuffer(bytearray(local_handle), dtype=torch.uint8).clone()

        gathered: list[torch.Tensor] = [
            torch.zeros(handle_len, dtype=torch.uint8) for _ in range(self._world_size)
        ]
        dist.all_gather(gathered, local_tensor)

        # 4. Open each remote handle to obtain a local-address-space pointer.
        remote_ptrs: list[int] = []
        for r in range(self._world_size):
            if r == self._rank:
                remote_ptrs.append(self._local_ptr)
            else:
                handle_bytes = bytes(gathered[r].tolist())
                remote_ptr = self._backend.open_ipc_handle(handle_bytes)
                remote_ptrs.append(remote_ptr)
                logger.debug(
                    "Rank %d: opened IPC handle from rank %d -> 0x%x",
                    self._rank, r, remote_ptr,
                )

        self._remote_ptrs = remote_ptrs

        # 5. Build heap_bases tensor on the current GPU.
        self._heap_bases = torch.tensor(
            remote_ptrs, dtype=torch.int64, device=f"cuda:{self._rank}"
        )
        self._ipc_setup_done = True

        # 6. Barrier to make sure every rank has finished opening handles
        #    before anyone starts issuing remote accesses.
        dist.barrier()
        logger.info("Rank %d: IPC setup complete", self._rank)

    # ---------------------------------------------------------- allocation

    def allocate_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate a tensor within the symmetric heap.

        The returned :class:`torch.Tensor` is backed by a slice of the
        pre-allocated heap memory.  Its data pointer lies within
        ``[heap_base, heap_base + heap_size)``.

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired tensor shape.
        dtype : torch.dtype
            Element data type (e.g. ``torch.float16``, ``torch.float32``).

        Returns
        -------
        torch.Tensor
            A tensor whose storage lives inside this symmetric heap.

        Raises
        ------
        RuntimeError
            If the heap does not have enough free space.
        """
        numel = math.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        byte_size = numel * element_size

        # Align the current bump offset.
        aligned_offset = _round_up(self._bump_offset, _ALIGN)

        if aligned_offset + byte_size > self._size:
            raise RuntimeError(
                f"Symmetric heap exhausted on rank {self._rank}: "
                f"requested {byte_size} bytes at offset 0x{aligned_offset:x}, "
                f"but heap size is {self._size} bytes "
                f"({self._size - aligned_offset} bytes remaining)"
            )

        # Compute the device pointer for this allocation.
        tensor_ptr = self._local_ptr + aligned_offset

        # Create a torch.Tensor backed by our heap memory.
        # torch.from_blob (available since PyTorch 2.4, a project dependency)
        # accepts a raw data pointer and produces a tensor without copying.
        tensor = _tensor_from_device_ptr(
            ptr=tensor_ptr,
            shape=shape,
            dtype=dtype,
            device_index=self._rank,
        )

        # Bookkeeping.
        record = _AllocRecord(
            offset=aligned_offset, size=byte_size, shape=shape, dtype=dtype,
        )
        self._alloc_records.append(record)
        self._bump_offset = aligned_offset + byte_size

        logger.debug(
            "Rank %d: allocated tensor %s %s at heap offset 0x%x",
            self._rank, shape, dtype, aligned_offset,
        )
        return tensor

    # ---------------------------------------------------------- query

    def get_heap_bases(self) -> torch.Tensor:
        """Return a tensor of all ranks' heap base addresses.

        Returns
        -------
        torch.Tensor
            A ``torch.int64`` tensor of shape ``(world_size,)`` residing on
            the current GPU.  Entry ``i`` is the device pointer to rank *i*'s
            heap, **as mapped into this rank's address space** (via IPC).

        This tensor is intended to be passed into ``@triton.jit`` kernels
        for device-side pointer translation.
        """
        if self._heap_bases is None:
            raise RuntimeError("IPC setup has not been completed.")
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
        """This process's rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of ranks."""
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

        Converts a device pointer that lies within **this** rank's heap to
        the equivalent pointer inside *to_rank*'s heap (as mapped into this
        rank's address space via IPC).

        Parameters
        ----------
        local_ptr : int
            A pointer within this rank's heap.
        to_rank : int
            Target rank whose address space to translate into.

        Returns
        -------
        int
            The translated pointer.

        Raises
        ------
        ValueError
            If *local_ptr* is outside this rank's heap or *to_rank* is invalid.
        """
        if to_rank < 0 or to_rank >= self._world_size:
            raise ValueError(
                f"to_rank={to_rank} out of range [0, {self._world_size})"
            )
        offset = self.get_offset(local_ptr)
        return self._remote_ptrs[to_rank] + offset

    def get_offset(self, ptr: int) -> int:
        """Return the byte offset of *ptr* within this rank's heap.

        Parameters
        ----------
        ptr : int
            Device pointer that must lie within ``[local_base, local_base + size)``.

        Returns
        -------
        int
            Offset in bytes from the heap base.

        Raises
        ------
        ValueError
            If *ptr* is outside the heap bounds.
        """
        offset = ptr - self._local_ptr
        if offset < 0 or offset >= self._size:
            raise ValueError(
                f"Pointer 0x{ptr:x} is outside the heap "
                f"[0x{self._local_ptr:x}, 0x{self._local_ptr + self._size:x})"
            )
        return offset

    # ---------------------------------------------------------- sync

    def barrier(self) -> None:
        """Global barrier across all ranks.

        Ensures all ranks have reached this point before any proceeds.
        Also synchronises the current GPU stream so that all preceding
        device operations are visible.
        """
        self._backend.synchronize()
        if self._world_size > 1:
            dist.barrier()

    # ---------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Release all IPC resources and free the heap.

        Safe to call multiple times.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True

        logger.info("Rank %d: cleaning up symmetric heap", self._rank)

        # Release remote IPC mappings.
        # NOTE: The backend's ``open_ipc_handle`` returns pointers that must
        # *not* be freed with ``free``; they are unmapped by closing the IPC
        # handle.  Concrete backends should expose a ``close_ipc_handle`` if
        # the runtime requires it.  For now, we rely on process-exit cleanup
        # because both CUDA and HIP automatically unmap IPC handles when the
        # opening process terminates.

        # Free our local heap allocation.
        if self._local_ptr:
            try:
                self._backend.free(self._local_ptr)
            except Exception:
                logger.warning(
                    "Rank %d: failed to free heap allocation", self._rank,
                    exc_info=True,
                )
            self._local_ptr = 0

        self._heap_bases = None
        self._remote_ptrs = []
        self._alloc_records.clear()
        self._bump_offset = 0

    # ---------------------------------------------------------- dunder

    def __enter__(self) -> "SymmetricHeap":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        # Best-effort cleanup when the object is garbage-collected.
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
# Internal helpers
# ---------------------------------------------------------------------------

def _tensor_from_device_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_index: int,
) -> torch.Tensor:
    """Create a :class:`torch.Tensor` that directly references device memory
    at *ptr*, without copying.

    This uses :func:`torch.from_blob` (available since PyTorch 2.4) which
    accepts a raw data pointer and produces a tensor backed by that memory.
    """
    import ctypes

    numel = math.prod(shape)
    # torch.from_blob expects a ctypes pointer (or int) and the target device.
    device = torch.device("cuda", device_index)
    tensor = torch.from_blob(
        ctypes.c_void_p(ptr),
        size=shape,
        dtype=dtype,
        device=device,
    )
    return tensor


def _human_bytes(n: int) -> str:
    """Return a human-readable byte string (e.g. ``'1.00 GiB'``)."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0  # type: ignore[assignment]
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"
