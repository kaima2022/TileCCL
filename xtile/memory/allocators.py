"""Allocator backends for :mod:`xtile.memory.symmetric_heap`.

The current repository still uses a single contiguous torch buffer as the
actual heap storage, but the heap runtime should not hard-code allocation and
ownership logic forever. This module introduces an allocator-first boundary so
future import/map backends can plug into the same public heap surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from xtile.backends.base import BackendInterface

_ALIGN = 256


def _round_up(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass(slots=True)
class AllocationRecord:
    """Bookkeeping for one tensor sub-allocation inside an allocator."""

    offset: int
    size: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class MemorySegmentDescriptor:
    """Structured description of one allocator-owned memory segment."""

    segment_id: str
    segment_kind: str
    allocator_name: str
    base_ptr: int
    size_bytes: int
    device: str

    def to_dict(self) -> dict[str, object]:
        """Return structured segment metadata for docs and diagnostics."""
        return {
            "segment_id": self.segment_id,
            "segment_kind": self.segment_kind,
            "allocator_name": self.allocator_name,
            "base_ptr": self.base_ptr,
            "size_bytes": self.size_bytes,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class AllocatorMemoryModelDescriptor:
    """Structured description of one allocator's current memory model."""

    allocator_name: str
    local_segment_layout: str
    peer_import_model: str
    peer_mapping_model: str
    external_tensor_import_mode: str
    external_mapping_mode: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-friendly allocator memory-model metadata."""
        return {
            "allocator_name": self.allocator_name,
            "local_segment_layout": self.local_segment_layout,
            "peer_import_model": self.peer_import_model,
            "peer_mapping_model": self.peer_mapping_model,
            "external_tensor_import_mode": self.external_tensor_import_mode,
            "external_mapping_mode": self.external_mapping_mode,
        }


@dataclass(frozen=True, slots=True)
class AllocatorSegmentLayoutDescriptor:
    """Structured description of the allocator's exportable segment layout."""

    allocator_name: str
    layout_kind: str
    segment_count: int
    exportable_segment_count: int
    primary_segment_id: str
    exportable_segment_ids: tuple[str, ...]
    multi_segment: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly segment-layout metadata."""
        return {
            "allocator_name": self.allocator_name,
            "layout_kind": self.layout_kind,
            "segment_count": self.segment_count,
            "exportable_segment_count": self.exportable_segment_count,
            "primary_segment_id": self.primary_segment_id,
            "exportable_segment_ids": list(self.exportable_segment_ids),
            "multi_segment": self.multi_segment,
        }


@dataclass(frozen=True, slots=True)
class ExternalMemoryInterfaceDescriptor:
    """Structured description of one allocator's external-memory interface."""

    allocator_name: str
    import_mode: str
    mapping_mode: str
    copy_import_supported: bool
    zero_copy_mapping_supported: bool
    fd_passing: bool
    dmabuf_mapping: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly external-memory interface metadata."""
        return {
            "allocator_name": self.allocator_name,
            "import_mode": self.import_mode,
            "mapping_mode": self.mapping_mode,
            "copy_import_supported": self.copy_import_supported,
            "zero_copy_mapping_supported": self.zero_copy_mapping_supported,
            "fd_passing": self.fd_passing,
            "dmabuf_mapping": self.dmabuf_mapping,
        }


@dataclass(frozen=True, slots=True)
class PeerMemoryExportDescriptor:
    """Structured description of one exportable peer-memory region."""

    peer_rank: int
    segment_id: str
    segment_kind: str
    allocator_name: str
    transport: str
    size_bytes: int
    base_ptr: int
    device: str
    payload: object

    def to_dict(self) -> dict[str, object]:
        """Return structured metadata for docs, diagnostics, and tests."""
        return {
            "peer_rank": self.peer_rank,
            "segment_id": self.segment_id,
            "segment_kind": self.segment_kind,
            "allocator_name": self.allocator_name,
            "transport": self.transport,
            "size_bytes": self.size_bytes,
            "base_ptr": self.base_ptr,
            "device": self.device,
            "payload_type": type(self.payload).__name__,
        }


@dataclass(frozen=True, slots=True)
class ImportedPeerMemory:
    """Result of importing one peer-memory descriptor into the local process."""

    peer_rank: int
    segment_id: str
    segment_kind: str
    allocator_name: str
    transport: str
    access_kind: str
    mapped_ptr: int
    exported_base_ptr: int
    size_bytes: int
    device: str
    cleanup_kind: str
    cleanup_resource: object | None = None

    def to_dict(self) -> dict[str, object]:
        """Return structured imported-peer metadata for diagnostics."""
        return {
            "peer_rank": self.peer_rank,
            "segment_id": self.segment_id,
            "segment_kind": self.segment_kind,
            "allocator_name": self.allocator_name,
            "transport": self.transport,
            "access_kind": self.access_kind,
            "mapped_ptr": self.mapped_ptr,
            "exported_base_ptr": self.exported_base_ptr,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "cleanup_kind": self.cleanup_kind,
        }


class BaseSymmetricAllocator(ABC):
    """Allocator interface used by :class:`xtile.memory.SymmetricHeap`."""

    def __init__(
        self,
        *,
        size: int,
        device: torch.device,
    ) -> None:
        self._size = int(size)
        self._device = device

    @property
    def size(self) -> int:
        """Return the managed heap size in bytes."""
        return self._size

    @property
    def device(self) -> torch.device:
        """Return the device that owns this allocator."""
        return self._device

    @property
    @abstractmethod
    def name(self) -> str:
        """Short allocator identifier used in docs and metadata."""

    @property
    @abstractmethod
    def buffer(self) -> torch.Tensor:
        """Return the raw byte buffer backing the heap."""

    @property
    @abstractmethod
    def base_ptr(self) -> int:
        """Return the base device pointer of the managed heap."""

    @property
    @abstractmethod
    def bytes_allocated(self) -> int:
        """Return the number of bytes consumed by sub-allocations."""

    @bytes_allocated.setter
    @abstractmethod
    def bytes_allocated(self, value: int) -> None:
        """Override the current bump offset.

        This remains writable to preserve compatibility with benchmark helpers
        that reset heaps between runs.
        """

    @property
    def bytes_free(self) -> int:
        """Return remaining allocator capacity in bytes."""
        return self.size - self.bytes_allocated

    @property
    @abstractmethod
    def alloc_records(self) -> list[AllocationRecord]:
        """Return the mutable allocation record list."""

    @alloc_records.setter
    @abstractmethod
    def alloc_records(self, value: list[AllocationRecord]) -> None:
        """Replace the mutable allocation record list."""

    @abstractmethod
    def segment_descriptors(self) -> tuple[MemorySegmentDescriptor, ...]:
        """Return the allocator-owned memory segments."""

    def exportable_segment_descriptors(self) -> tuple[MemorySegmentDescriptor, ...]:
        """Return the allocator segments exportable through the current runtime."""
        return self.segment_descriptors()

    def primary_segment(self) -> MemorySegmentDescriptor:
        """Return the single exportable segment supported by the current runtime."""
        segments = self.exportable_segment_descriptors()
        if len(segments) != 1:
            raise RuntimeError(
                "The current SymmetricHeap runtime only supports one exportable "
                f"segment per allocator, but allocator {self.name!r} exposes "
                f"{len(segments)} segments."
            )
        return segments[0]

    @abstractmethod
    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate one tensor view from the managed heap."""

    @abstractmethod
    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """Return ``True`` when *tensor* resides inside this heap."""

    @abstractmethod
    def import_external_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Materialize an external tensor inside this allocator's heap."""

    @abstractmethod
    def export_peer_memory(
        self,
        *,
        peer_rank: int,
        transport: str,
        backend: "BackendInterface",
    ) -> PeerMemoryExportDescriptor:
        """Export allocator-managed peer memory for one transport strategy."""

    @abstractmethod
    def import_peer_memory(
        self,
        export: PeerMemoryExportDescriptor,
        *,
        backend: "BackendInterface",
    ) -> ImportedPeerMemory:
        """Import one peer-memory export produced by another rank."""

    def cleanup(self) -> None:
        """Release allocator-owned resources."""
        # The current torch-backed allocator only owns one torch buffer, so
        # dropping references is enough. More advanced allocators can override.
        return None

    def capabilities(self) -> dict[str, bool]:
        """Return structured allocator capability flags."""
        return {
            "multi_segment": len(self.segment_descriptors()) > 1,
            "peer_export": True,
            "peer_import": True,
            "external_import_copy": True,
            "external_mapping": False,
            "fd_passing": False,
            "dmabuf_mapping": False,
        }

    def external_tensor_import_mode(self) -> str:
        """Return the current external-tensor import mode."""
        return "copy"

    def external_mapping_mode(self) -> str:
        """Return the current external-mapping mode."""
        return "none"

    def local_segment_layout(self) -> str:
        """Return how allocator-owned local memory segments are laid out."""
        return "single_contiguous_device_heap"

    def segment_layout_kind(self) -> str:
        """Return the current exportable segment-layout model."""
        return "single_exportable_segment"

    def peer_import_model(self) -> str:
        """Return how peer imports are represented after transport setup."""
        return "per_rank_transport_resolved_imports"

    def peer_mapping_model(self) -> str:
        """Return how imported peer mappings are indexed by the runtime."""
        return "rank_ordered_import_table"

    def peer_transport_modes(self) -> tuple[str, ...]:
        """Return the allocator-supported peer transport modes."""
        return (
            "ctypes_ipc",
            "pytorch_ipc",
            "peer_access_pointer_exchange",
        )

    def peer_import_access_kind(
        self,
        *,
        transport: str,
        is_local_rank: bool,
    ) -> str:
        """Return how imported peer memory is accessed for one transport."""
        if is_local_rank:
            return "local"
        if transport == "peer_access":
            return "peer_direct"
        if transport in {"ctypes_ipc", "pytorch_ipc"}:
            return "mapped_remote"
        if transport == "peer_access_pointer_exchange":
            return "remote_pointer"
        if transport == "local_only":
            return "local"
        raise ValueError(
            f"Unsupported transport {transport!r} for allocator {self.name!r}"
        )

    def peer_import_access_kinds(self) -> tuple[str, ...]:
        """Return the access semantics currently expressible by this allocator."""
        return (
            "local",
            "peer_direct",
            "mapped_remote",
            "remote_pointer",
        )

    def memory_model_descriptor(self) -> AllocatorMemoryModelDescriptor:
        """Return the allocator's current memory-model descriptor."""
        return AllocatorMemoryModelDescriptor(
            allocator_name=self.name,
            local_segment_layout=self.local_segment_layout(),
            peer_import_model=self.peer_import_model(),
            peer_mapping_model=self.peer_mapping_model(),
            external_tensor_import_mode=self.external_tensor_import_mode(),
            external_mapping_mode=self.external_mapping_mode(),
        )

    def memory_model(self) -> dict[str, str]:
        """Return JSON-friendly allocator memory-model metadata."""
        return self.memory_model_descriptor().to_dict()

    def segment_layout_descriptor(self) -> AllocatorSegmentLayoutDescriptor:
        """Return the allocator's structured exportable segment-layout descriptor."""
        segments = self.segment_descriptors()
        exportable_segments = self.exportable_segment_descriptors()
        return AllocatorSegmentLayoutDescriptor(
            allocator_name=self.name,
            layout_kind=self.segment_layout_kind(),
            segment_count=len(segments),
            exportable_segment_count=len(exportable_segments),
            primary_segment_id=self.primary_segment().segment_id,
            exportable_segment_ids=tuple(
                segment.segment_id for segment in exportable_segments
            ),
            multi_segment=len(segments) > 1,
        )

    def segment_layout(self) -> dict[str, object]:
        """Return JSON-friendly exportable segment-layout metadata."""
        return self.segment_layout_descriptor().to_dict()

    def external_memory_interface_descriptor(self) -> ExternalMemoryInterfaceDescriptor:
        """Return the allocator's structured external-memory interface descriptor."""
        capabilities = self.capabilities()
        return ExternalMemoryInterfaceDescriptor(
            allocator_name=self.name,
            import_mode=self.external_tensor_import_mode(),
            mapping_mode=self.external_mapping_mode(),
            copy_import_supported=bool(capabilities["external_import_copy"]),
            zero_copy_mapping_supported=bool(capabilities["external_mapping"]),
            fd_passing=bool(capabilities["fd_passing"]),
            dmabuf_mapping=bool(capabilities["dmabuf_mapping"]),
        )

    def external_memory_interface(self) -> dict[str, object]:
        """Return JSON-friendly external-memory interface metadata."""
        return self.external_memory_interface_descriptor().to_dict()

    def describe(self) -> dict[str, object]:
        """Return structured allocator metadata for docs and diagnostics."""
        segments = self.segment_descriptors()
        exportable_segments = self.exportable_segment_descriptors()
        return {
            "name": self.name,
            "device": str(self.device),
            "size_bytes": self.size,
            "bytes_allocated": self.bytes_allocated,
            "bytes_free": self.bytes_free,
            "segment_count": len(segments),
            "capabilities": self.capabilities(),
            "external_tensor_import_mode": self.external_tensor_import_mode(),
            "external_mapping_mode": self.external_mapping_mode(),
            "external_memory_interface": self.external_memory_interface(),
            "segment_layout": self.segment_layout(),
            "peer_transport_modes": list(self.peer_transport_modes()),
            "peer_import_access_kinds": list(self.peer_import_access_kinds()),
            "memory_model": self.memory_model(),
            "segments": [segment.to_dict() for segment in segments],
            "exportable_segments": [
                segment.to_dict() for segment in exportable_segments
            ],
        }


class TorchBumpAllocator(BaseSymmetricAllocator):
    """Torch-buffer allocator with aligned bump allocation.

    This is intentionally conservative: it keeps XTile's current runtime model
    intact while moving allocation, ownership checks, and external import
    materialization behind an allocator boundary.
    """

    def __init__(
        self,
        *,
        size: int,
        device: torch.device,
        existing_buffer: Optional[torch.Tensor] = None,
        alignment: int = _ALIGN,
    ) -> None:
        super().__init__(size=size, device=device)
        self._alignment = int(alignment)
        if existing_buffer is None:
            self._buffer: Optional[torch.Tensor] = torch.empty(
                size,
                dtype=torch.uint8,
                device=device,
            )
        else:
            if existing_buffer.dtype != torch.uint8:
                raise ValueError(
                    f"existing_buffer must use dtype=torch.uint8, got {existing_buffer.dtype}"
                )
            if tuple(existing_buffer.shape) != (size,):
                raise ValueError(
                    "existing_buffer shape must match heap size exactly: "
                    f"{tuple(existing_buffer.shape)} != ({size},)"
                )
            if existing_buffer.device != device:
                raise ValueError(
                    f"existing_buffer device {existing_buffer.device} does not match {device}"
                )
            self._buffer = existing_buffer
        self._bump_offset = 0
        self._alloc_records: list[AllocationRecord] = []

    @property
    def name(self) -> str:
        """Return the allocator identifier used across benchmarks/docs."""
        return "torch_bump"

    @property
    def buffer(self) -> torch.Tensor:
        """Return the raw heap buffer."""
        if self._buffer is None:
            raise RuntimeError("Allocator has been cleaned up.")
        return self._buffer

    @property
    def base_ptr(self) -> int:
        """Return the base pointer of the managed heap."""
        return int(self.buffer.data_ptr())

    @property
    def bytes_allocated(self) -> int:
        """Return the current bump offset."""
        return self._bump_offset

    @bytes_allocated.setter
    def bytes_allocated(self, value: int) -> None:
        """Override the current bump offset."""
        offset = int(value)
        if offset < 0 or offset > self.size:
            raise ValueError(
                f"bytes_allocated must stay within [0, {self.size}], got {offset}"
            )
        self._bump_offset = offset

    @property
    def alloc_records(self) -> list[AllocationRecord]:
        """Return the mutable allocation record list."""
        return self._alloc_records

    @alloc_records.setter
    def alloc_records(self, value: list[AllocationRecord]) -> None:
        """Replace the mutable allocation record list."""
        self._alloc_records = value

    def segment_descriptors(self) -> tuple[MemorySegmentDescriptor, ...]:
        """Return the single contiguous heap segment owned by this allocator."""
        return (
            MemorySegmentDescriptor(
                segment_id="heap",
                segment_kind="device_heap",
                allocator_name=self.name,
                base_ptr=self.base_ptr,
                size_bytes=self.size,
                device=str(self.device),
            ),
        )

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate one tensor view from the heap buffer."""
        if self._buffer is None:
            raise RuntimeError("Allocator has been cleaned up.")

        numel = math.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        byte_size = numel * element_size
        aligned_offset = _round_up(self._bump_offset, self._alignment)

        if aligned_offset + byte_size > self.size:
            raise RuntimeError(
                "Symmetric heap exhausted via allocator "
                f"{self.name}: requested {byte_size} bytes at offset "
                f"0x{aligned_offset:x}, heap size is {self.size} bytes "
                f"({self.size - aligned_offset} bytes remaining)"
            )

        byte_slice = self._buffer.narrow(0, aligned_offset, byte_size)
        tensor = byte_slice.view(dtype).reshape(shape)
        self._alloc_records.append(
            AllocationRecord(
                offset=aligned_offset,
                size=byte_size,
                shape=shape,
                dtype=dtype,
            )
        )
        self._bump_offset = aligned_offset + byte_size
        return tensor

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """Return ``True`` when *tensor* lies within the managed heap."""
        if self._buffer is None:
            return False
        if tensor.device != self.device:
            return False
        if tensor.numel() == 0:
            return True
        ptr = int(tensor.data_ptr())
        base = self.base_ptr
        return base <= ptr < base + self.size

    def import_external_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy one external device tensor onto the symmetric heap."""
        if self._buffer is None:
            raise RuntimeError("Allocator has been cleaned up.")
        if tensor.device != self.device:
            raise ValueError(
                "import_external_tensor requires the tensor to already be on the "
                f"allocator device {self.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                "import_external_tensor requires a contiguous tensor. "
                "Call .contiguous() before importing."
            )
        imported = self.allocate_tensor(tuple(int(dim) for dim in tensor.shape), tensor.dtype)
        imported.copy_(tensor)
        return imported

    def export_peer_memory(
        self,
        *,
        peer_rank: int,
        transport: str,
        backend: "BackendInterface",
    ) -> PeerMemoryExportDescriptor:
        """Export the allocator's single backing region for peer import."""
        segment = self.primary_segment()
        if transport == "ctypes_ipc":
            payload: object = backend.get_ipc_handle(self.base_ptr)
        elif transport == "pytorch_ipc":
            payload = self.buffer.untyped_storage()._share_cuda_()
        elif transport == "peer_access_pointer_exchange":
            payload = self.base_ptr
        else:
            raise ValueError(
                f"Unsupported transport {transport!r} for allocator {self.name!r}"
            )
        return PeerMemoryExportDescriptor(
            peer_rank=peer_rank,
            segment_id=segment.segment_id,
            segment_kind=segment.segment_kind,
            allocator_name=self.name,
            transport=transport,
            size_bytes=self.size,
            base_ptr=self.base_ptr,
            device=str(self.device),
            payload=payload,
        )

    def import_peer_memory(
        self,
        export: PeerMemoryExportDescriptor,
        *,
        backend: "BackendInterface",
    ) -> ImportedPeerMemory:
        """Import one peer region described by *export* into the local process."""
        if export.allocator_name != self.name:
            raise ValueError(
                "Allocator import mismatch: "
                f"local allocator={self.name!r}, export allocator={export.allocator_name!r}"
            )
        if export.transport == "ctypes_ipc":
            if not isinstance(export.payload, bytes):
                raise TypeError(
                    "ctypes_ipc exports must carry a raw bytes payload."
                )
            mapped_ptr = backend.open_ipc_handle(export.payload)
            return ImportedPeerMemory(
                peer_rank=export.peer_rank,
                segment_id=export.segment_id,
                segment_kind=export.segment_kind,
                allocator_name=export.allocator_name,
                transport=export.transport,
                access_kind=self.peer_import_access_kind(
                    transport=export.transport,
                    is_local_rank=False,
                ),
                mapped_ptr=mapped_ptr,
                exported_base_ptr=export.base_ptr,
                size_bytes=export.size_bytes,
                device=export.device,
                cleanup_kind="ipc_handle",
                cleanup_resource=mapped_ptr,
            )
        if export.transport == "pytorch_ipc":
            if not isinstance(export.payload, tuple):
                raise TypeError(
                    "pytorch_ipc exports must carry a tuple payload from _share_cuda_()."
                )
            storage = torch.UntypedStorage._new_shared_cuda(*export.payload)
            return ImportedPeerMemory(
                peer_rank=export.peer_rank,
                segment_id=export.segment_id,
                segment_kind=export.segment_kind,
                allocator_name=export.allocator_name,
                transport=export.transport,
                access_kind=self.peer_import_access_kind(
                    transport=export.transport,
                    is_local_rank=False,
                ),
                mapped_ptr=storage.data_ptr(),
                exported_base_ptr=export.base_ptr,
                size_bytes=export.size_bytes,
                device=export.device,
                cleanup_kind="storage",
                cleanup_resource=storage,
            )
        if export.transport == "peer_access_pointer_exchange":
            if not isinstance(export.payload, int):
                raise TypeError(
                    "peer_access_pointer_exchange exports must carry an integer pointer."
                )
            return ImportedPeerMemory(
                peer_rank=export.peer_rank,
                segment_id=export.segment_id,
                segment_kind=export.segment_kind,
                allocator_name=export.allocator_name,
                transport=export.transport,
                access_kind=self.peer_import_access_kind(
                    transport=export.transport,
                    is_local_rank=False,
                ),
                mapped_ptr=export.payload,
                exported_base_ptr=export.base_ptr,
                size_bytes=export.size_bytes,
                device=export.device,
                cleanup_kind="none",
            )
        raise ValueError(
            f"Unsupported transport {export.transport!r} for allocator {self.name!r}"
        )

    def cleanup(self) -> None:
        """Release the backing buffer and local bookkeeping."""
        self._buffer = None
        self._bump_offset = 0
        self._alloc_records.clear()


def create_allocator(
    *,
    allocator_type: str,
    size: int,
    device: torch.device,
    existing_buffer: Optional[torch.Tensor] = None,
) -> BaseSymmetricAllocator:
    """Return the allocator implementation requested by *allocator_type*."""
    normalized = allocator_type.lower()
    if normalized in {"torch", "torch_bump"}:
        return TorchBumpAllocator(
            size=size,
            device=device,
            existing_buffer=existing_buffer,
        )
    raise ValueError(
        f"Unknown allocator_type {allocator_type!r}. Supported: 'torch', 'torch_bump'."
    )
