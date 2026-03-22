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

from dataclasses import dataclass
import logging
from typing import Optional

import torch
import torch.distributed as dist

from xtile.backends import get_backend, detect_hardware
from xtile.backends.base import BackendInterface
from xtile.memory.allocators import (
    BaseSymmetricAllocator,
    PeerMemoryExportDescriptor,
    create_allocator,
)
from xtile.utils.feature_gates import (
    FORCE_MULTIPROCESS_TRANSPORT_ENV,
    forced_multiprocess_transport,
)

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


@dataclass(frozen=True, slots=True)
class PeerMemoryMapEntry:
    """One rank-visible peer-memory mapping entry."""

    peer_rank: int
    allocator_name: str
    transport: str
    mapped_ptr: int
    exported_base_ptr: int
    size_bytes: int
    device: str
    is_local_rank: bool
    cleanup_kind: str

    def to_dict(self) -> dict[str, object]:
        """Serialize mapping metadata for diagnostics and docs."""
        return {
            "peer_rank": self.peer_rank,
            "allocator_name": self.allocator_name,
            "transport": self.transport,
            "mapped_ptr": self.mapped_ptr,
            "exported_base_ptr": self.exported_base_ptr,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "is_local_rank": self.is_local_rank,
            "cleanup_kind": self.cleanup_kind,
        }


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
        allocator_type: str = "torch_bump",
        _peer_bases: Optional[list[int]] = None,
        _existing_buffer: Optional[torch.Tensor] = None,
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
        self._allocator_type: str = allocator_type
        self._allocator: BaseSymmetricAllocator = create_allocator(
            allocator_type=allocator_type,
            size=size,
            device=self._device,
            existing_buffer=_existing_buffer,
        )
        self._buffer: Optional[torch.Tensor] = self._allocator.buffer
        self._local_ptr: int = self._allocator.base_ptr
        logger.debug("Rank %d: local heap base = 0x%x", rank, self._local_ptr)

        # Build heap_bases ---------------------------------------------------
        self._remote_ptrs: list[int] = []
        self._heap_bases: Optional[torch.Tensor] = None
        self._ipc_opened: list[Optional[int]] = []
        self._ipc_storages: list[Optional[torch.UntypedStorage]] = []
        self._peer_exports: list[PeerMemoryExportDescriptor] = []
        self._peer_map: list[PeerMemoryMapEntry] = []
        self._peer_buffers: Optional[list[torch.Tensor]] = None
        self._mode: str = "single_process" if _peer_bases is not None or world_size == 1 else "multiprocess"
        self._transport_strategy: str = "local_only" if world_size == 1 else "unknown"

        if _peer_bases is not None:
            # Single-process mode: bases provided by create_all()
            self._remote_ptrs = list(_peer_bases)
            self._heap_bases = torch.tensor(
                _peer_bases, dtype=torch.int64, device=self._device,
            )
            self._transport_strategy = "peer_access"
            self._peer_exports = self._build_local_peer_exports(
                transport="peer_access",
                remote_ptrs=self._remote_ptrs,
            )
            self._peer_map = self._build_peer_map(
                peer_exports=self._peer_exports,
                remote_ptrs=self._remote_ptrs,
                ipc_opened=self._ipc_opened,
                ipc_storages=self._ipc_storages,
            )
        elif world_size == 1:
            # Trivial single-GPU case
            self._remote_ptrs = [self._local_ptr]
            self._heap_bases = torch.tensor(
                [self._local_ptr], dtype=torch.int64, device=self._device,
            )
            self._peer_buffers = [self._buffer]
            self._peer_exports = self._build_local_peer_exports(
                transport="local_only",
                remote_ptrs=self._remote_ptrs,
            )
            self._peer_map = self._build_peer_map(
                peer_exports=self._peer_exports,
                remote_ptrs=self._remote_ptrs,
                ipc_opened=self._ipc_opened,
                ipc_storages=self._ipc_storages,
            )
        else:
            # Multi-process mode: try IPC, fall back to all_gather of pointers
            self._setup_multiprocess()

        # Cleanup flag -------------------------------------------------------
        self._cleaned_up: bool = False

    # ---------------------------------------------------------- class methods

    @classmethod
    def create_all(
        cls,
        size: int,
        world_size: int,
        backend: str = "auto",
        *,
        allocator_type: str = "torch_bump",
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
            heap._allocator_type = allocator_type
            heap._allocator = create_allocator(
                allocator_type=allocator_type,
                size=size,
                device=heap._device,
                existing_buffer=buffers[rank],
            )
            heap._buffer = heap._allocator.buffer
            heap._local_ptr = heap._allocator.base_ptr
            heap._remote_ptrs = list(bases)
            heap._ipc_opened = []
            heap._ipc_storages = []
            heap._peer_exports = heap._build_local_peer_exports(
                transport="peer_access",
                remote_ptrs=heap._remote_ptrs,
            )
            heap._peer_map = heap._build_peer_map(
                peer_exports=heap._peer_exports,
                remote_ptrs=heap._remote_ptrs,
                ipc_opened=heap._ipc_opened,
                ipc_storages=heap._ipc_storages,
            )
            heap._heap_bases = torch.tensor(
                bases, dtype=torch.int64, device=f"cuda:{rank}",
            )
            heap._peer_buffers = list(buffers)
            heap._mode = "single_process"
            heap._transport_strategy = "peer_access"
            heap._cleaned_up = False
            heaps.append(heap)

        logger.info(
            "Created %d symmetric heaps (%s each) with peer access via allocator=%s",
            world_size, _human_bytes(size), allocator_type,
        )
        return heaps

    # ---------------------------------------------------------- multi-process

    def _gather_peer_exports(
        self,
        local_export: PeerMemoryExportDescriptor,
    ) -> list[PeerMemoryExportDescriptor]:
        """Exchange allocator export descriptors across ranks."""
        exports: list[PeerMemoryExportDescriptor | None] = [None] * self._world_size
        dist.all_gather_object(exports, local_export)
        resolved: list[PeerMemoryExportDescriptor] = []
        for rank, export in enumerate(exports):
            if export is None:
                raise RuntimeError(
                    f"Rank {self._rank}: missing export descriptor from rank {rank}"
                )
            resolved.append(export)
        return resolved

    def _build_local_peer_exports(
        self,
        *,
        transport: str,
        remote_ptrs: list[int],
    ) -> list[PeerMemoryExportDescriptor]:
        """Build synthetic peer-export descriptors for local/single-process modes."""
        return [
            PeerMemoryExportDescriptor(
                allocator_name=self.allocator_name,
                transport=transport,
                size_bytes=self._size,
                base_ptr=int(ptr),
                device=f"cuda:{rank}",
                payload=int(ptr),
            )
            for rank, ptr in enumerate(remote_ptrs)
        ]

    def _build_peer_map(
        self,
        *,
        peer_exports: list[PeerMemoryExportDescriptor],
        remote_ptrs: list[int],
        ipc_opened: list[Optional[int]],
        ipc_storages: list[Optional[torch.UntypedStorage]],
    ) -> list[PeerMemoryMapEntry]:
        """Build the structured peer-mapping view for diagnostics."""
        entries: list[PeerMemoryMapEntry] = []
        for rank, export in enumerate(peer_exports):
            cleanup_kind = "none"
            if rank < len(ipc_opened) and ipc_opened[rank] is not None:
                cleanup_kind = "ipc_handle"
            elif rank < len(ipc_storages) and ipc_storages[rank] is not None:
                cleanup_kind = "storage"
            entries.append(
                PeerMemoryMapEntry(
                    peer_rank=rank,
                    allocator_name=export.allocator_name,
                    transport=export.transport,
                    mapped_ptr=int(remote_ptrs[rank]),
                    exported_base_ptr=export.base_ptr,
                    size_bytes=export.size_bytes,
                    device=export.device,
                    is_local_rank=(rank == self._rank),
                    cleanup_kind=cleanup_kind,
                )
            )
        return entries

    def _resolve_imported_peers(
        self,
        *,
        transport: str,
        gathered_exports: list[PeerMemoryExportDescriptor],
    ) -> tuple[list[int], list[Optional[int]], list[Optional[torch.UntypedStorage]]]:
        """Import peer descriptors through the active allocator."""
        remote_ptrs: list[int] = []
        ipc_opened: list[Optional[int]] = []
        ipc_storages: list[Optional[torch.UntypedStorage]] = []
        for peer_rank, export in enumerate(gathered_exports):
            if peer_rank == self._rank:
                remote_ptrs.append(self._local_ptr)
                ipc_opened.append(None)
                ipc_storages.append(None)
                continue
            imported = self._allocator.import_peer_memory(
                export,
                backend=self._backend,
            )
            remote_ptrs.append(imported.mapped_ptr)
            if transport == "ctypes_ipc":
                ipc_opened.append(int(imported.cleanup_resource))
                ipc_storages.append(None)
            elif transport == "pytorch_ipc":
                ipc_opened.append(None)
                ipc_storages.append(imported.cleanup_resource)  # type: ignore[arg-type]
            else:
                ipc_opened.append(None)
                ipc_storages.append(None)
        return remote_ptrs, ipc_opened, ipc_storages

    def _apply_peer_mapping_state(
        self,
        *,
        peer_exports: list[PeerMemoryExportDescriptor],
        remote_ptrs: list[int],
        ipc_opened: list[Optional[int]],
        ipc_storages: list[Optional[torch.UntypedStorage]],
    ) -> None:
        """Commit one resolved peer-mapping state to the heap."""
        self._peer_exports = list(peer_exports)
        self._remote_ptrs = list(remote_ptrs)
        self._ipc_opened = list(ipc_opened)
        self._ipc_storages = list(ipc_storages)
        self._peer_map = self._build_peer_map(
            peer_exports=self._peer_exports,
            remote_ptrs=self._remote_ptrs,
            ipc_opened=self._ipc_opened,
            ipc_storages=self._ipc_storages,
        )

    def _setup_multiprocess(self) -> None:
        """Exchange heap pointers across ranks for multi-process mode.

        Tries the device-safe strategies in order:

        1. **Raw ctypes IPC** — ``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``
           (historically the most direct path, but some Linux/driver
           environments may still reject it, so fallback remains required).

        Additional transports such as PyTorch fd-passing CUDA storage IPC or
        raw cross-process pointer exchange remain available only through the
        explicit force-transport diagnostic path. Real multiprocess Triton
        remote-access diagnostics currently validate only ``ctypes_ipc`` as a
        device-dereferenceable transport.
        """
        self._backend.init_ipc()
        forced_strategy = forced_multiprocess_transport()
        strategies = (
            [forced_strategy]
            if forced_strategy is not None
            else [
                "ctypes_ipc",
            ]
        )
        errors: list[tuple[str, Exception]] = []
        for strategy in strategies:
            try:
                if strategy == "ctypes_ipc":
                    self._setup_multiprocess_ctypes_ipc()
                elif strategy == "pytorch_ipc":
                    self._setup_multiprocess_pytorch_ipc()
                elif strategy == "peer_access_pointer_exchange":
                    self._setup_multiprocess_peer_access_pointer_exchange()
                else:
                    raise AssertionError(f"Unhandled transport strategy {strategy!r}")
                return
            except Exception as exc:
                errors.append((strategy, exc))
                if forced_strategy is not None:
                    break
                logger.warning(
                    "Rank %d: multiprocess transport %s failed (%s), trying next fallback",
                    self._rank,
                    strategy,
                    exc,
                )

        detail = "; ".join(
            f"{strategy}: {type(exc).__name__}({exc})"
            for strategy, exc in errors
        )
        if forced_strategy is not None:
            raise RuntimeError(
                f"Forced multiprocess transport {forced_strategy!r} failed on rank "
                f"{self._rank}. Set {FORCE_MULTIPROCESS_TRANSPORT_ENV}=auto to "
                f"restore the normal fallback chain. Details: {detail}"
            ) from errors[-1][1]
        raise RuntimeError(
            f"All multiprocess transport strategies failed on rank {self._rank}: "
            f"{detail}"
        ) from errors[-1][1]

    def _setup_multiprocess_ctypes_ipc(self) -> None:
        """Map peer heaps via raw CUDA/HIP IPC handles."""
        local_export = self._allocator.export_peer_memory(
            transport="ctypes_ipc",
            backend=self._backend,
        )
        gathered_exports = self._gather_peer_exports(local_export)
        remote_ptrs, ipc_opened, ipc_storages = self._resolve_imported_peers(
            transport="ctypes_ipc",
            gathered_exports=gathered_exports,
        )
        self._apply_peer_mapping_state(
            peer_exports=gathered_exports,
            remote_ptrs=remote_ptrs,
            ipc_opened=ipc_opened,
            ipc_storages=ipc_storages,
        )
        self._heap_bases = torch.tensor(
            remote_ptrs, dtype=torch.int64, device=self._device,
        )
        self._transport_strategy = "ctypes_ipc"
        dist.barrier()
        logger.info("Rank %d: ctypes IPC setup complete", self._rank)

    def _setup_multiprocess_pytorch_ipc(self) -> None:
        """Map peer heaps via PyTorch's fd-passing CUDA storage IPC."""
        for peer_rank in range(self._world_size):
            if peer_rank == self._rank:
                continue
            if torch.cuda.can_device_access_peer(self._rank, peer_rank):
                try:
                    self._backend.enable_peer_access(peer_rank)
                except RuntimeError:
                    pass

        local_export = self._allocator.export_peer_memory(
            transport="pytorch_ipc",
            backend=self._backend,
        )
        gathered_exports = self._gather_peer_exports(local_export)
        remote_ptrs, ipc_opened, ipc_storages = self._resolve_imported_peers(
            transport="pytorch_ipc",
            gathered_exports=gathered_exports,
        )
        self._apply_peer_mapping_state(
            peer_exports=gathered_exports,
            remote_ptrs=remote_ptrs,
            ipc_opened=ipc_opened,
            ipc_storages=ipc_storages,
        )
        self._heap_bases = torch.tensor(
            remote_ptrs, dtype=torch.int64, device=self._device,
        )
        self._transport_strategy = "pytorch_ipc"
        dist.barrier()
        logger.info("Rank %d: PyTorch IPC setup complete", self._rank)

    def _setup_multiprocess_peer_access_pointer_exchange(self) -> None:
        """Exchange raw heap pointers after enabling peer access where possible.

        This path is kept only for forced diagnostics. Raw virtual addresses
        from another process are not a validated public transport contract.
        """
        for peer_rank in range(self._world_size):
            if peer_rank == self._rank:
                continue
            if torch.cuda.can_device_access_peer(self._rank, peer_rank):
                try:
                    self._backend.enable_peer_access(peer_rank)
                except RuntimeError:
                    pass

        local_export = self._allocator.export_peer_memory(
            transport="peer_access_pointer_exchange",
            backend=self._backend,
        )
        gathered_exports = self._gather_peer_exports(local_export)
        remote_ptrs, ipc_opened, ipc_storages = self._resolve_imported_peers(
            transport="peer_access_pointer_exchange",
            gathered_exports=gathered_exports,
        )
        self._apply_peer_mapping_state(
            peer_exports=gathered_exports,
            remote_ptrs=remote_ptrs,
            ipc_opened=ipc_opened,
            ipc_storages=ipc_storages,
        )
        self._heap_bases = torch.tensor(
            remote_ptrs, dtype=torch.int64, device=self._device,
        )
        self._transport_strategy = "peer_access_pointer_exchange"
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
        tensor = self._allocator.allocate_tensor(shape, dtype)
        logger.debug(
            "Rank %d: allocated tensor %s %s via allocator=%s (ptr 0x%x)",
            self._rank,
            shape,
            dtype,
            self._allocator.name,
            int(tensor.data_ptr()),
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
        return self._allocator.bytes_allocated

    @property
    def bytes_free(self) -> int:
        """Remaining free bytes in this heap."""
        return self._allocator.bytes_free

    @property
    def mode(self) -> str:
        """Heap establishment mode: single_process or multiprocess."""
        return self._mode

    @property
    def transport_strategy(self) -> str:
        """Concrete heap-establishment strategy used by this heap."""
        return self._transport_strategy

    def get_peer_buffer(self, rank: int) -> torch.Tensor:
        """Return the local-process tensor backing *rank*'s heap buffer.

        This is only available in single-process mode created via
        :meth:`create_all`. It is intended for host-side reference paths
        and diagnostics, not hot-path kernels.
        """
        if self._peer_buffers is None:
            raise RuntimeError(
                "Peer heap buffers are only available in single-process mode."
            )
        if rank < 0 or rank >= self._world_size:
            raise ValueError(
                f"rank={rank} out of range [0, {self._world_size})"
            )
        return self._peer_buffers[rank]

    @property
    def allocator_name(self) -> str:
        """Return the active allocator backend identifier."""
        return self._allocator.name

    def allocator_metadata(self) -> dict[str, object]:
        """Return structured allocator metadata for docs and benchmarks."""
        return self._allocator.describe()

    def peer_export_descriptors(self) -> tuple[PeerMemoryExportDescriptor, ...]:
        """Return the rank-visible peer export descriptors for this heap."""
        return tuple(self._peer_exports)

    def peer_memory_map(self) -> tuple[PeerMemoryMapEntry, ...]:
        """Return the structured peer-memory mapping table."""
        return tuple(self._peer_map)

    def peer_memory_map_metadata(self) -> list[dict[str, object]]:
        """Return peer-memory mapping metadata in JSON-friendly form."""
        return [entry.to_dict() for entry in self._peer_map]

    def metadata(self) -> dict[str, object]:
        """Return the structured heap metadata for docs and diagnostics."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self._backend_name,
            "device": str(self._device),
            "size_bytes": self._size,
            "local_base": self._local_ptr,
            "mode": self._mode,
            "transport_strategy": self._transport_strategy,
            "allocator": self.allocator_metadata(),
            "peer_exports": [export.to_dict() for export in self._peer_exports],
            "peer_memory_map": self.peer_memory_map_metadata(),
        }

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """Return ``True`` when *tensor* resides in this symmetric heap."""
        return self._allocator.owns_tensor(tensor)

    def is_symmetric(self, tensor: torch.Tensor) -> bool:
        """Public alias for :meth:`owns_tensor`."""
        return self.owns_tensor(tensor)

    def import_external_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Materialize *tensor* inside the attached heap via the allocator."""
        return self._allocator.import_external_tensor(tensor)

    def as_symmetric(self, tensor: torch.Tensor) -> torch.Tensor:
        """Public alias for :meth:`import_external_tensor`."""
        return self.import_external_tensor(tensor)

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

        try:
            self._allocator.cleanup()
        except Exception:
            logger.debug(
                "Rank %d: allocator cleanup raised", self._rank, exc_info=True,
            )
        self._buffer = None
        self._local_ptr = 0
        self._heap_bases = None
        self._remote_ptrs = []
        self._ipc_opened = []
        self._ipc_storages = []
        self._peer_exports = []
        self._peer_map = []
        self._peer_buffers = None
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
            f"allocator={self.allocator_name!r}, "
            f"allocated={_human_bytes(self.bytes_allocated)})"
        )

    @property
    def _bump_offset(self) -> int:
        """Backward-compatible alias for allocator bump offset."""
        return self._allocator.bytes_allocated

    @_bump_offset.setter
    def _bump_offset(self, value: int) -> None:
        self._allocator.bytes_allocated = value

    @property
    def _alloc_records(self):
        """Backward-compatible alias for allocator allocation records."""
        return self._allocator.alloc_records

    @_alloc_records.setter
    def _alloc_records(self, value) -> None:
        self._allocator.alloc_records = value


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
