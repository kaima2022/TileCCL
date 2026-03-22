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
