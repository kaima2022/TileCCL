"""Tests for xtile.memory.symmetric_heap.SymmetricHeap.

Comprehensive tests covering:
- Constructor argument validation (no GPU needed for some)
- Bump allocator alignment, exhaustion, and overlap guarantees
- Byte tracking (bytes_allocated / bytes_free)
- Context manager cleanup
- Multi-GPU IPC handle exchange and cross-GPU pointer translation

Single-GPU tests require one GPU (world_size=1, no IPC).
Multi-GPU tests are marked ``@pytest.mark.multigpu`` and need >= 2 GPUs.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Constants used by the SymmetricHeap implementation
# ---------------------------------------------------------------------------

_ALIGN = 256  # must match xtile.memory.symmetric_heap._ALIGN


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


class _FakeValidationAllocator:
    """Minimal allocator stub for peer-state validation unit tests."""

    def __init__(self, *, base_ptr: int, size: int, device: str) -> None:
        from xtile.memory.allocators import MemorySegmentDescriptor

        self._name = "torch_bump"
        self._segment = MemorySegmentDescriptor(
            segment_id="heap",
            segment_kind="device_heap",
            allocator_name=self._name,
            base_ptr=base_ptr,
            size_bytes=size,
            device=device,
        )

    @property
    def name(self) -> str:
        return self._name

    def primary_segment(self):
        return self._segment


def _make_peer_export(
    *,
    rank: int,
    base_ptr: int,
    size: int,
    transport: str = "ctypes_ipc",
    segment_id: str = "heap",
    segment_kind: str = "device_heap",
    allocator_name: str = "torch_bump",
):
    from xtile.memory.allocators import PeerMemoryExportDescriptor

    return PeerMemoryExportDescriptor(
        peer_rank=rank,
        segment_id=segment_id,
        segment_kind=segment_kind,
        allocator_name=allocator_name,
        transport=transport,
        size_bytes=size,
        base_ptr=base_ptr,
        device=f"cuda:{rank}",
        payload=f"export:{rank}".encode(),
    )


def _make_peer_import(
    *,
    rank: int,
    mapped_ptr: int,
    exported_base_ptr: int,
    size: int,
    transport: str = "ctypes_ipc",
    segment_id: str = "heap",
    segment_kind: str = "device_heap",
    allocator_name: str = "torch_bump",
    cleanup_kind: str = "ipc_handle",
):
    from xtile.memory.allocators import ImportedPeerMemory

    return ImportedPeerMemory(
        peer_rank=rank,
        segment_id=segment_id,
        segment_kind=segment_kind,
        allocator_name=allocator_name,
        transport=transport,
        mapped_ptr=mapped_ptr,
        exported_base_ptr=exported_base_ptr,
        size_bytes=size,
        device=f"cuda:{rank}",
        cleanup_kind=cleanup_kind,
    )


def _make_validation_heap():
    from xtile.memory.symmetric_heap import SymmetricHeap

    heap = SymmetricHeap.__new__(SymmetricHeap)
    heap._rank = 0
    heap._world_size = 2
    heap._size = 4096
    heap._device = torch.device("cuda", 0)
    heap._local_ptr = 0x1000
    heap._allocator = _FakeValidationAllocator(
        base_ptr=heap._local_ptr,
        size=heap._size,
        device=str(heap._device),
    )
    heap._peer_exports = []
    heap._peer_imports = []
    return heap


# ---------------------------------------------------------------------------
# Unit tests (single GPU or mock -- no IPC needed)
# ---------------------------------------------------------------------------


class TestSymmetricHeapUnit:
    """Unit tests that work with a single GPU or mock."""

    # ---- argument validation (no GPU required) ---------------------------

    def test_creation_rejects_zero_size(self) -> None:
        """size=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            from xtile.memory.symmetric_heap import SymmetricHeap
            SymmetricHeap(size=0, rank=0, world_size=1)

    def test_creation_rejects_negative_size(self) -> None:
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            from xtile.memory.symmetric_heap import SymmetricHeap
            SymmetricHeap(size=-1024, rank=0, world_size=1)

    def test_creation_rejects_rank_out_of_range(self) -> None:
        """rank >= world_size raises ValueError."""
        with pytest.raises(ValueError, match="rank="):
            from xtile.memory.symmetric_heap import SymmetricHeap
            SymmetricHeap(size=1024, rank=2, world_size=2)

    def test_creation_rejects_negative_rank(self) -> None:
        """Negative rank raises ValueError."""
        with pytest.raises(ValueError, match="rank="):
            from xtile.memory.symmetric_heap import SymmetricHeap
            SymmetricHeap(size=1024, rank=-1, world_size=1)

    def test_creation_rejects_zero_world_size(self) -> None:
        """world_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="world_size"):
            from xtile.memory.symmetric_heap import SymmetricHeap
            SymmetricHeap(size=1024, rank=0, world_size=0)

    def test_creation_validates_args(self) -> None:
        """Various bad size/rank/world_size combinations raise ValueError."""
        from xtile.memory.symmetric_heap import SymmetricHeap
        bad_combos = [
            dict(size=0, rank=0, world_size=1),
            dict(size=-100, rank=0, world_size=1),
            dict(size=1024, rank=1, world_size=1),
            dict(size=1024, rank=-1, world_size=1),
            dict(size=1024, rank=0, world_size=0),
            dict(size=1024, rank=0, world_size=-1),
        ]
        for kwargs in bad_combos:
            with pytest.raises(ValueError):
                SymmetricHeap(**kwargs)

    def test_multiprocess_auto_transport_only_tries_device_safe_chain(
        self,
        monkeypatch,
    ) -> None:
        """Default multiprocess setup should no longer auto-fallback to unsafe transports."""
        from xtile.memory.symmetric_heap import SymmetricHeap

        heap = SymmetricHeap.__new__(SymmetricHeap)
        heap._backend = MagicMock()
        heap._backend.init_ipc = MagicMock()
        heap._rank = 0
        heap._world_size = 2

        calls: list[str] = []
        heap._setup_multiprocess_ctypes_ipc = MagicMock(
            side_effect=lambda: calls.append("ctypes_ipc")
        )
        heap._setup_multiprocess_pytorch_ipc = MagicMock(
            side_effect=lambda: calls.append("pytorch_ipc")
        )
        heap._setup_multiprocess_peer_access_pointer_exchange = MagicMock(
            side_effect=lambda: calls.append("peer_access_pointer_exchange")
        )

        monkeypatch.delenv("XTILE_FORCE_MULTIPROCESS_TRANSPORT", raising=False)
        heap._setup_multiprocess()

        assert calls == ["ctypes_ipc"]

    def test_multiprocess_auto_transport_no_longer_falls_back_to_pytorch_ipc(
        self,
        monkeypatch,
    ) -> None:
        """If the device-safe path fails, auto mode should fail closed instead of downgrading silently."""
        from xtile.memory.symmetric_heap import SymmetricHeap

        heap = SymmetricHeap.__new__(SymmetricHeap)
        heap._backend = MagicMock()
        heap._backend.init_ipc = MagicMock()
        heap._rank = 0
        heap._world_size = 2

        heap._setup_multiprocess_ctypes_ipc = MagicMock(
            side_effect=RuntimeError("ctypes failed")
        )
        heap._setup_multiprocess_pytorch_ipc = MagicMock()
        heap._setup_multiprocess_peer_access_pointer_exchange = MagicMock()

        monkeypatch.delenv("XTILE_FORCE_MULTIPROCESS_TRANSPORT", raising=False)
        with pytest.raises(RuntimeError, match="All multiprocess transport strategies failed"):
            heap._setup_multiprocess()

        heap._setup_multiprocess_pytorch_ipc.assert_not_called()
        heap._setup_multiprocess_peer_access_pointer_exchange.assert_not_called()

    def test_multiprocess_forced_transport_still_dispatches_diagnostic_path(
        self,
        monkeypatch,
    ) -> None:
        """Forced diagnostics should still be able to target non-default transports."""
        from xtile.memory.symmetric_heap import SymmetricHeap

        heap = SymmetricHeap.__new__(SymmetricHeap)
        heap._backend = MagicMock()
        heap._backend.init_ipc = MagicMock()
        heap._rank = 0
        heap._world_size = 2

        calls: list[str] = []
        heap._setup_multiprocess_ctypes_ipc = MagicMock(
            side_effect=lambda: calls.append("ctypes_ipc")
        )
        heap._setup_multiprocess_pytorch_ipc = MagicMock(
            side_effect=lambda: calls.append("pytorch_ipc")
        )
        heap._setup_multiprocess_peer_access_pointer_exchange = MagicMock(
            side_effect=lambda: calls.append("peer_access_pointer_exchange")
        )

        monkeypatch.setenv("XTILE_FORCE_MULTIPROCESS_TRANSPORT", "pytorch_ipc")
        heap._setup_multiprocess()

        assert calls == ["pytorch_ipc"]

    def test_validate_peer_mapping_state_rejects_export_world_size_mismatch(
        self,
    ) -> None:
        """peer_exports length must always match world_size."""
        heap = _make_validation_heap()

        peer_exports = [
            _make_peer_export(rank=0, base_ptr=heap._local_ptr, size=heap._size),
        ]
        peer_imports = [
            _make_peer_import(
                rank=0,
                mapped_ptr=heap._local_ptr,
                exported_base_ptr=heap._local_ptr,
                size=heap._size,
                cleanup_kind="none",
            ),
            _make_peer_import(
                rank=1,
                mapped_ptr=0x2000,
                exported_base_ptr=0x2000,
                size=heap._size,
            ),
        ]

        with pytest.raises(RuntimeError, match="peer_exports has 1 entries"):
            heap._validate_peer_mapping_state(
                peer_exports=peer_exports,
                peer_imports=peer_imports,
            )

    def test_validate_peer_mapping_state_rejects_local_import_pointer_drift(
        self,
    ) -> None:
        """Local-rank import must still point at the local heap base."""
        heap = _make_validation_heap()

        peer_exports = [
            _make_peer_export(rank=0, base_ptr=heap._local_ptr, size=heap._size),
            _make_peer_export(rank=1, base_ptr=0x2000, size=heap._size),
        ]
        peer_imports = [
            _make_peer_import(
                rank=0,
                mapped_ptr=heap._local_ptr + 0x80,
                exported_base_ptr=heap._local_ptr,
                size=heap._size,
                cleanup_kind="none",
            ),
            _make_peer_import(
                rank=1,
                mapped_ptr=0x2000,
                exported_base_ptr=0x2000,
                size=heap._size,
            ),
        ]

        with pytest.raises(RuntimeError, match="local import mapped_ptr"):
            heap._validate_peer_mapping_state(
                peer_exports=peer_exports,
                peer_imports=peer_imports,
            )

    def test_validate_peer_mapping_state_rejects_export_import_metadata_mismatch(
        self,
    ) -> None:
        """Each peer import must align with its paired export metadata."""
        heap = _make_validation_heap()

        peer_exports = [
            _make_peer_export(rank=0, base_ptr=heap._local_ptr, size=heap._size),
            _make_peer_export(rank=1, base_ptr=0x2000, size=heap._size),
        ]
        peer_imports = [
            _make_peer_import(
                rank=0,
                mapped_ptr=heap._local_ptr,
                exported_base_ptr=heap._local_ptr,
                size=heap._size,
                cleanup_kind="none",
            ),
            _make_peer_import(
                rank=1,
                mapped_ptr=0x2000,
                exported_base_ptr=0x2000,
                size=heap._size,
                segment_kind="dma_buf_segment",
            ),
        ]

        with pytest.raises(RuntimeError, match="segment_kind"):
            heap._validate_peer_mapping_state(
                peer_exports=peer_exports,
                peer_imports=peer_imports,
            )

    def test_validate_peer_mapping_state_rejects_embedded_peer_rank_mismatch(
        self,
    ) -> None:
        """Structured peer records must not disagree with their list position."""
        heap = _make_validation_heap()

        peer_exports = [
            _make_peer_export(rank=0, base_ptr=heap._local_ptr, size=heap._size),
            _make_peer_export(rank=0, base_ptr=0x2000, size=heap._size),
        ]
        peer_imports = [
            _make_peer_import(
                rank=0,
                mapped_ptr=heap._local_ptr,
                exported_base_ptr=heap._local_ptr,
                size=heap._size,
                cleanup_kind="none",
            ),
            _make_peer_import(
                rank=1,
                mapped_ptr=0x2000,
                exported_base_ptr=0x2000,
                size=heap._size,
            ),
        ]

        with pytest.raises(RuntimeError, match="export peer_rank 0"):
            heap._validate_peer_mapping_state(
                peer_exports=peer_exports,
                peer_imports=peer_imports,
            )

    def test_apply_peer_mapping_state_fails_closed_on_invalid_local_mapping(
        self,
        symmetric_heap,
    ) -> None:
        """Invalid peer state should fail before mutating heap_bases."""
        heap = symmetric_heap
        old_bases = heap.get_heap_bases().clone()
        segment = heap.segment_descriptors()[0]
        export = _make_peer_export(
            rank=0,
            base_ptr=heap.local_base,
            size=heap.size,
            transport="local_only",
            segment_id=segment.segment_id,
            segment_kind=segment.segment_kind,
            allocator_name=heap.allocator_name,
        )
        invalid_import = _make_peer_import(
            rank=0,
            mapped_ptr=heap.local_base + 256,
            exported_base_ptr=heap.local_base,
            size=heap.size,
            transport="local_only",
            segment_id=segment.segment_id,
            segment_kind=segment.segment_kind,
            allocator_name=heap.allocator_name,
            cleanup_kind="none",
        )

        with pytest.raises(RuntimeError, match="local import mapped_ptr"):
            heap._apply_peer_mapping_state(
                peer_exports=[export],
                peer_imports=[invalid_import],
            )

        assert torch.equal(heap.get_heap_bases(), old_bases)

    def test_apply_peer_mapping_state_canonicalizes_records_by_peer_rank(
        self,
    ) -> None:
        """Incoming peer records may be self-describing but unordered."""
        heap = _make_validation_heap()
        heap._refresh_heap_bases = MagicMock()

        peer_exports = [
            _make_peer_export(rank=1, base_ptr=0x2000, size=heap._size),
            _make_peer_export(rank=0, base_ptr=heap._local_ptr, size=heap._size),
        ]
        peer_imports = [
            _make_peer_import(
                rank=1,
                mapped_ptr=0x2000,
                exported_base_ptr=0x2000,
                size=heap._size,
            ),
            _make_peer_import(
                rank=0,
                mapped_ptr=heap._local_ptr,
                exported_base_ptr=heap._local_ptr,
                size=heap._size,
                cleanup_kind="none",
            ),
        ]

        heap._apply_peer_mapping_state(
            peer_exports=peer_exports,
            peer_imports=peer_imports,
        )

        assert [export.peer_rank for export in heap._peer_exports] == [0, 1]
        assert [imported.peer_rank for imported in heap._peer_imports] == [0, 1]
        heap._refresh_heap_bases.assert_called_once()

    def test_apply_peer_mapping_state_rejects_duplicate_peer_rank(
        self,
    ) -> None:
        """Canonicalization should fail closed on duplicate peer ranks."""
        heap = _make_validation_heap()
        heap._refresh_heap_bases = MagicMock()

        peer_exports = [
            _make_peer_export(rank=0, base_ptr=heap._local_ptr, size=heap._size),
            _make_peer_export(rank=0, base_ptr=0x2000, size=heap._size),
        ]
        peer_imports = [
            _make_peer_import(
                rank=0,
                mapped_ptr=heap._local_ptr,
                exported_base_ptr=heap._local_ptr,
                size=heap._size,
                cleanup_kind="none",
            ),
            _make_peer_import(
                rank=1,
                mapped_ptr=0x2000,
                exported_base_ptr=0x2000,
                size=heap._size,
            ),
        ]

        with pytest.raises(RuntimeError, match="duplicate peer_rank=0"):
            heap._apply_peer_mapping_state(
                peer_exports=peer_exports,
                peer_imports=peer_imports,
            )

    # ---- bump allocator (requires 1 GPU via symmetric_heap fixture) ------

    def test_bump_allocator_alignment(self, symmetric_heap) -> None:
        """Allocations are 256-byte aligned."""
        heap = symmetric_heap
        # First allocation: 100 bytes (not a multiple of 256)
        t1 = heap.allocate_tensor(shape=(25,), dtype=torch.float32)  # 100 bytes
        ptr1 = t1.data_ptr()
        offset1 = ptr1 - heap.local_base
        assert offset1 % _ALIGN == 0, f"First allocation offset {offset1} not {_ALIGN}-aligned"

        # Second allocation should also be aligned
        t2 = heap.allocate_tensor(shape=(10,), dtype=torch.float32)  # 40 bytes
        ptr2 = t2.data_ptr()
        offset2 = ptr2 - heap.local_base
        assert offset2 % _ALIGN == 0, f"Second allocation offset {offset2} not {_ALIGN}-aligned"

    def test_bump_allocator_exhaustion(self, device_info) -> None:
        """Allocating more than heap size raises RuntimeError."""
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

        from xtile.memory.symmetric_heap import SymmetricHeap

        # Create a tiny heap (4096 bytes)
        heap = SymmetricHeap(size=4096, rank=0, world_size=1)
        try:
            # This should succeed: 1024 floats = 4096 bytes, but alignment
            # may push total above 4096 for a second allocation.
            heap.allocate_tensor(shape=(512,), dtype=torch.float32)  # 2048 bytes

            # This should fail: another 2048 bytes + alignment overhead
            # exceeds the 4096-byte heap.
            with pytest.raises(RuntimeError, match="exhausted"):
                heap.allocate_tensor(shape=(1024,), dtype=torch.float32)  # 4096 bytes
        finally:
            heap.cleanup()

    def test_multiple_allocations_no_overlap(self, symmetric_heap) -> None:
        """Multiple tensors don't overlap in memory."""
        heap = symmetric_heap
        tensors = []
        for _ in range(5):
            t = heap.allocate_tensor(shape=(64,), dtype=torch.float32)
            tensors.append(t)

        # Check pairwise non-overlap
        regions = []
        for t in tensors:
            start = t.data_ptr()
            end = start + t.nelement() * t.element_size()
            regions.append((start, end))

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                s_i, e_i = regions[i]
                s_j, e_j = regions[j]
                overlaps = not (e_i <= s_j or e_j <= s_i)
                assert not overlaps, (
                    f"Tensors {i} [{s_i:#x}, {e_i:#x}) and "
                    f"{j} [{s_j:#x}, {e_j:#x}) overlap"
                )

    # ---- byte tracking ---------------------------------------------------

    def test_bytes_allocated_tracking(self, symmetric_heap) -> None:
        """bytes_allocated and bytes_free are correct after allocations."""
        heap = symmetric_heap
        initial_free = heap.bytes_free
        initial_allocated = heap.bytes_allocated
        assert initial_allocated == 0
        assert initial_free == heap.size

        # Allocate 256 floats = 1024 bytes
        heap.allocate_tensor(shape=(256,), dtype=torch.float32)

        assert heap.bytes_allocated > 0
        assert heap.bytes_free < initial_free
        assert heap.bytes_allocated + heap.bytes_free == heap.size

    def test_bytes_allocated_after_multiple(self, symmetric_heap) -> None:
        """Bytes tracking remains consistent after multiple allocations."""
        heap = symmetric_heap
        heap.allocate_tensor(shape=(128,), dtype=torch.float32)  # 512 bytes
        heap.allocate_tensor(shape=(64,), dtype=torch.float16)   # 128 bytes

        # The total should reflect both allocations plus alignment padding
        assert heap.bytes_allocated > 0
        assert heap.bytes_allocated + heap.bytes_free == heap.size

    # ---- repr ------------------------------------------------------------

    def test_repr(self, symmetric_heap) -> None:
        """__repr__ is informative."""
        r = repr(symmetric_heap)
        assert "SymmetricHeap" in r
        assert "rank=" in r
        assert "backend=" in r

    def test_repr_includes_size(self, symmetric_heap) -> None:
        """__repr__ mentions the heap size."""
        r = repr(symmetric_heap)
        # The repr uses _human_bytes, which produces e.g. "1.00 MiB"
        assert "MiB" in r or "KiB" in r or "B" in r

    # ---- context manager -------------------------------------------------

    def test_context_manager_cleanup(self, device_info) -> None:
        """Exiting context manager calls cleanup."""
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

        from xtile.memory.symmetric_heap import SymmetricHeap

        with SymmetricHeap(size=4096, rank=0, world_size=1) as heap:
            assert not heap._cleaned_up
            heap.allocate_tensor(shape=(4,), dtype=torch.float32)

        assert heap._cleaned_up

    def test_context_manager_cleanup_on_exception(self, device_info) -> None:
        """Cleanup runs even if an exception occurs inside the with block."""
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

        from xtile.memory.symmetric_heap import SymmetricHeap

        heap = None
        with pytest.raises(RuntimeError, match="deliberate"):
            with SymmetricHeap(size=4096, rank=0, world_size=1) as h:
                heap = h
                raise RuntimeError("deliberate error")

        assert heap is not None
        assert heap._cleaned_up

    def test_double_cleanup_is_safe(self, device_info) -> None:
        """Calling cleanup() multiple times does not raise."""
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

        from xtile.memory.symmetric_heap import SymmetricHeap

        heap = SymmetricHeap(size=4096, rank=0, world_size=1)
        heap.cleanup()
        heap.cleanup()  # should not raise
        assert heap._cleaned_up

    # ---- basic properties ------------------------------------------------

    def test_basic_properties(self, symmetric_heap) -> None:
        """size, rank, world_size properties are correct."""
        heap = symmetric_heap
        assert heap.size == 1024 * 1024
        assert heap.rank == 0
        assert heap.world_size == 1

    def test_local_base_nonzero(self, symmetric_heap) -> None:
        """local_base is a non-zero device pointer."""
        assert symmetric_heap.local_base != 0

    def test_heap_bases_shape(self, symmetric_heap) -> None:
        """get_heap_bases() returns a tensor of shape (world_size,)."""
        bases = symmetric_heap.get_heap_bases()
        assert bases.shape == (1,)
        assert bases.dtype == torch.int64

    def test_heap_bases_contains_local_base(self, symmetric_heap) -> None:
        """For world_size=1, heap_bases[0] equals local_base."""
        bases = symmetric_heap.get_heap_bases()
        assert bases[0].item() == symmetric_heap.local_base

    # ---- allocate_tensor details -----------------------------------------

    def test_allocate_tensor_shape_dtype(self, symmetric_heap) -> None:
        """Allocated tensor has the requested shape and dtype."""
        t = symmetric_heap.allocate_tensor(shape=(32, 16), dtype=torch.float16)
        assert t.shape == (32, 16)
        assert t.dtype == torch.float16

    def test_allocate_tensor_on_cuda(self, symmetric_heap) -> None:
        """Allocated tensor resides on a CUDA device."""
        t = symmetric_heap.allocate_tensor(shape=(8,), dtype=torch.float32)
        assert t.device.type == "cuda"

    def test_allocate_tensor_ptr_within_heap(self, symmetric_heap) -> None:
        """Tensor data_ptr lies within the heap bounds."""
        heap = symmetric_heap
        t = heap.allocate_tensor(shape=(64,), dtype=torch.float32)
        ptr = t.data_ptr()
        base = heap.local_base
        assert base <= ptr < base + heap.size

    def test_allocator_metadata_reports_active_backend(self, symmetric_heap) -> None:
        """Allocator metadata should expose the canonical backend name."""
        metadata = symmetric_heap.allocator_metadata()
        assert metadata["name"] == "torch_bump"
        assert metadata["size_bytes"] == symmetric_heap.size
        assert metadata["bytes_allocated"] == symmetric_heap.bytes_allocated
        assert metadata["segment_count"] == 1
        assert metadata["capabilities"]["peer_export"] is True
        assert metadata["capabilities"]["peer_import"] is True
        assert metadata["capabilities"]["external_import_copy"] is True
        assert metadata["capabilities"]["external_mapping"] is False
        assert metadata["capabilities"]["fd_passing"] is False
        assert metadata["capabilities"]["dmabuf_mapping"] is False
        assert len(metadata["segments"]) == 1
        assert metadata["segments"][0]["segment_id"] == "heap"
        assert metadata["segments"][0]["segment_kind"] == "device_heap"
        assert metadata["segments"][0]["base_ptr"] == symmetric_heap.local_base

    def test_segment_metadata_reports_local_heap_segment(self, symmetric_heap) -> None:
        """Heap metadata should expose allocator-owned local segment metadata."""
        segments = symmetric_heap.segment_metadata()

        assert len(segments) == 1
        assert segments[0]["segment_id"] == "heap"
        assert segments[0]["segment_kind"] == "device_heap"
        assert segments[0]["allocator_name"] == symmetric_heap.allocator_name
        assert segments[0]["base_ptr"] == symmetric_heap.local_base
        assert segments[0]["size_bytes"] == symmetric_heap.size
        assert segments[0]["owner_rank"] == symmetric_heap.rank
        assert segments[0]["is_local_rank"] is True

    def test_peer_memory_map_metadata_reports_local_segment(self, symmetric_heap) -> None:
        """Single-rank heaps should still expose one structured mapping entry."""
        metadata = symmetric_heap.peer_memory_map_metadata()
        exports = symmetric_heap.peer_export_descriptors()
        export_metadata = symmetric_heap.peer_export_metadata()
        export = symmetric_heap.peer_export_descriptor(0)
        imported = symmetric_heap.peer_import(0)
        imports = symmetric_heap.peer_import_metadata()

        assert len(metadata) == 1
        assert len(exports) == 1
        assert len(export_metadata) == 1
        assert len(imports) == 1
        assert export.peer_rank == 0
        assert imported.peer_rank == 0
        assert metadata[0]["peer_rank"] == 0
        assert metadata[0]["segment_id"] == "heap"
        assert metadata[0]["segment_kind"] == "device_heap"
        assert metadata[0]["transport"] == "local_only"
        assert metadata[0]["mapped_ptr"] == symmetric_heap.local_base
        assert metadata[0]["exported_base_ptr"] == symmetric_heap.local_base
        assert metadata[0]["size_bytes"] == symmetric_heap.size
        assert metadata[0]["is_local_rank"] is True
        assert metadata[0]["cleanup_kind"] == "none"
        assert exports[0].segment_id == "heap"
        assert exports[0].peer_rank == 0
        assert exports[0].segment_kind == "device_heap"
        assert exports[0].transport == "local_only"
        assert exports[0].base_ptr == symmetric_heap.local_base
        assert export.base_ptr == symmetric_heap.local_base
        assert export_metadata[0]["peer_rank"] == 0
        assert export_metadata[0]["segment_id"] == "heap"
        assert export_metadata[0]["segment_kind"] == "device_heap"
        assert export_metadata[0]["transport"] == "local_only"
        assert export_metadata[0]["base_ptr"] == symmetric_heap.local_base
        assert imports[0]["peer_rank"] == 0
        assert imports[0]["segment_id"] == "heap"
        assert imports[0]["transport"] == "local_only"
        assert imports[0]["mapped_ptr"] == symmetric_heap.local_base
        assert imports[0]["exported_base_ptr"] == symmetric_heap.local_base
        assert imports[0]["cleanup_kind"] == "none"

    def test_import_external_tensor_materializes_heap_copy(self, symmetric_heap) -> None:
        """import_external_tensor should copy data onto the symmetric heap."""
        external = torch.arange(32, device=symmetric_heap.get_heap_bases().device, dtype=torch.float32)
        imported = symmetric_heap.import_external_tensor(external)

        assert symmetric_heap.owns_tensor(imported)
        assert symmetric_heap.is_symmetric(imported)
        assert imported.data_ptr() != external.data_ptr()
        assert torch.allclose(imported, external)

        external.zero_()
        assert not torch.allclose(imported, external)

    def test_peer_accessors_validate_rank(self, symmetric_heap) -> None:
        """Rank-addressed peer accessors should fail on invalid ranks."""
        with pytest.raises(ValueError, match="rank=1"):
            symmetric_heap.peer_export_descriptor(1)
        with pytest.raises(ValueError, match="rank=1"):
            symmetric_heap.peer_import(1)

    def test_allocator_exports_structured_ctypes_peer_descriptor(
        self,
        symmetric_heap,
    ) -> None:
        """The allocator should own the structured export/import boundary."""

        class _FakeBackend:
            def __init__(self) -> None:
                self.seen_ptr: int | None = None
                self.opened_handle: bytes | None = None

            def get_ipc_handle(self, ptr: int) -> bytes:
                self.seen_ptr = ptr
                return f"ipc:{ptr}".encode()

            def open_ipc_handle(self, handle: bytes) -> int:
                self.opened_handle = handle
                return 0x12345000

        backend = _FakeBackend()
        export = symmetric_heap._allocator.export_peer_memory(
            peer_rank=0,
            transport="ctypes_ipc",
            backend=backend,  # type: ignore[arg-type]
        )

        assert backend.seen_ptr == symmetric_heap.local_base
        assert export.peer_rank == 0
        assert export.allocator_name == symmetric_heap.allocator_name
        assert export.transport == "ctypes_ipc"
        assert export.size_bytes == symmetric_heap.size
        assert export.base_ptr == symmetric_heap.local_base
        assert export.segment_id == "heap"
        assert export.segment_kind == "device_heap"
        assert export.to_dict()["payload_type"] == "bytes"

        imported = symmetric_heap._allocator.import_peer_memory(
            export,
            backend=backend,  # type: ignore[arg-type]
        )
        assert backend.opened_handle == export.payload
        assert imported.peer_rank == 0
        assert imported.segment_id == "heap"
        assert imported.segment_kind == "device_heap"
        assert imported.transport == "ctypes_ipc"
        assert imported.mapped_ptr == 0x12345000
        assert imported.exported_base_ptr == symmetric_heap.local_base
        assert imported.cleanup_kind == "ipc_handle"
        assert imported.cleanup_resource == 0x12345000

    def test_allocator_exports_peer_pointer_exchange_descriptor(
        self,
        symmetric_heap,
    ) -> None:
        """The fallback peer-pointer path should also use allocator descriptors."""

        class _UnusedBackend:
            pass

        backend = _UnusedBackend()
        export = symmetric_heap._allocator.export_peer_memory(
            peer_rank=0,
            transport="peer_access_pointer_exchange",
            backend=backend,  # type: ignore[arg-type]
        )
        imported = symmetric_heap._allocator.import_peer_memory(
            export,
            backend=backend,  # type: ignore[arg-type]
        )

        assert export.transport == "peer_access_pointer_exchange"
        assert export.peer_rank == 0
        assert export.payload == symmetric_heap.local_base
        assert imported.peer_rank == 0
        assert imported.segment_id == "heap"
        assert imported.transport == "peer_access_pointer_exchange"
        assert imported.mapped_ptr == symmetric_heap.local_base
        assert imported.cleanup_kind == "none"
        assert imported.cleanup_resource is None


# ---------------------------------------------------------------------------
# Multi-GPU tests (require >= 2 GPUs)
# ---------------------------------------------------------------------------


@pytest.mark.multigpu
class TestSymmetricHeapMultiGPU:
    """Multi-GPU tests requiring >= 2 GPUs.

    Uses SymmetricHeap.create_all() (single-process mode) to avoid
    mp.spawn pickle issues (P1-003).
    """

    def _get_heap_cls(self):
        from xtile.memory.symmetric_heap import SymmetricHeap
        return SymmetricHeap

    def test_heap_bases_correct_world_size(self, skip_no_multigpu, device_info) -> None:
        """Heap bases have correct world_size entries after create_all."""
        SymmetricHeap = self._get_heap_cls()
        world_size = min(device_info.num_gpus, 4)
        heaps = SymmetricHeap.create_all(size=1024 * 1024, world_size=world_size)
        try:
            for rank in range(world_size):
                bases = heaps[rank].get_heap_bases()
                assert bases.shape == (world_size,), (
                    f"Rank {rank}: expected bases shape ({world_size},), got {bases.shape}"
                )
                assert bases.dtype == torch.int64
                # Each base should be non-zero
                for i in range(world_size):
                    assert bases[i].item() != 0, f"Rank {rank}: base[{i}] is zero"
        finally:
            for h in heaps:
                h.cleanup()

    def test_create_all_exposes_peer_memory_map_metadata(
        self,
        skip_no_multigpu,
        device_info,
    ) -> None:
        """Single-process peer-access heaps should expose full peer mapping metadata."""
        SymmetricHeap = self._get_heap_cls()
        world_size = 2
        heaps = SymmetricHeap.create_all(size=1024 * 1024, world_size=world_size)
        try:
            metadata = heaps[0].peer_memory_map_metadata()
            exports = heaps[0].peer_export_descriptors()
            export_metadata = heaps[0].peer_export_metadata()
            assert heaps[0].peer_export_descriptor(1).peer_rank == 1
            assert heaps[0].peer_import(1).peer_rank == 1
            imports = heaps[0].peer_import_metadata()

            assert len(metadata) == world_size
            assert len(exports) == world_size
            assert len(export_metadata) == world_size
            assert len(imports) == world_size
            assert {entry["peer_rank"] for entry in metadata} == {0, 1}
            assert {export.peer_rank for export in exports} == {0, 1}
            assert {entry["peer_rank"] for entry in export_metadata} == {0, 1}
            assert {entry["peer_rank"] for entry in imports} == {0, 1}
            assert {entry["segment_id"] for entry in metadata} == {"heap"}
            assert {entry["segment_kind"] for entry in metadata} == {"device_heap"}
            assert {entry["transport"] for entry in metadata} == {"peer_access"}
            assert {entry["transport"] for entry in export_metadata} == {"peer_access"}
            assert {entry["transport"] for entry in imports} == {"peer_access"}
            assert all(entry["size_bytes"] == heaps[0].size for entry in metadata)
            assert metadata[0]["is_local_rank"] is True
            assert metadata[1]["is_local_rank"] is False
            assert metadata[0]["cleanup_kind"] == "none"
            assert metadata[1]["cleanup_kind"] == "none"
            assert exports[0].segment_id == "heap"
            assert imports[1]["segment_kind"] == "device_heap"
            assert exports[1].transport == "peer_access"
        finally:
            for h in heaps:
                h.cleanup()

    def test_cross_gpu_pointer_translation(self, skip_no_multigpu, device_info) -> None:
        """Translated pointer computes correct remote address."""
        SymmetricHeap = self._get_heap_cls()
        world_size = min(device_info.num_gpus, 2)
        heaps = SymmetricHeap.create_all(size=1024 * 1024, world_size=world_size)
        try:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                t = heaps[rank].allocate_tensor(shape=(64,), dtype=torch.float32)
                local_ptr = t.data_ptr()
                offset = local_ptr - heaps[rank].local_base

                other_rank = 1 - rank
                translated = heaps[rank].translate(local_ptr, to_rank=other_rank)

                bases = heaps[rank].get_heap_bases()
                expected = int(bases[other_rank].item()) + offset
                assert translated == expected, (
                    f"Rank {rank}: translated=0x{translated:x}, expected=0x{expected:x}"
                )
        finally:
            for h in heaps:
                h.cleanup()

    def test_allocate_tensor_on_heap(self, skip_no_multigpu, device_info) -> None:
        """Tensor allocated on heap has correct data_ptr within heap bounds."""
        SymmetricHeap = self._get_heap_cls()
        world_size = min(device_info.num_gpus, 2)
        heaps = SymmetricHeap.create_all(size=1024 * 1024, world_size=world_size)
        try:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                t = heaps[rank].allocate_tensor(shape=(128, 64), dtype=torch.float16)
                ptr = t.data_ptr()
                base = heaps[rank].local_base
                assert base <= ptr < base + heaps[rank].size, (
                    f"Rank {rank}: tensor ptr 0x{ptr:x} outside heap bounds"
                )
                assert t.shape == (128, 64)
                assert t.dtype == torch.float16
                assert t.device.type == "cuda"
        finally:
            for h in heaps:
                h.cleanup()
