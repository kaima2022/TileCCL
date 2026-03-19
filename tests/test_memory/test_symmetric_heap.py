"""Tests for xtile.memory.symmetric_heap.SymmetricHeap.

These tests exercise heap creation, tensor allocation, pointer translation,
context-manager cleanup, and multi-allocation overlap checks.  Multi-GPU
tests are marked with ``@pytest.mark.multigpu`` and are automatically
skipped on single-GPU systems.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Single-GPU tests (require at least 1 GPU)
# ---------------------------------------------------------------------------

class TestSymmetricHeap:
    """Unit tests for SymmetricHeap on a single GPU."""

    def test_heap_creation(self, symmetric_heap) -> None:
        """Create a heap and verify basic attributes."""
        heap = symmetric_heap
        assert heap.size == 1024 * 1024, "Heap size should be 1 MB"
        assert heap.rank == 0
        assert heap.world_size == 1

    def test_heap_allocate_tensor(self, symmetric_heap) -> None:
        """Allocate a tensor from the heap and verify shape/dtype."""
        heap = symmetric_heap
        t = heap.allocate_tensor(shape=(64, 64), dtype=torch.float32)
        assert t.shape == (64, 64)
        assert t.dtype == torch.float32
        assert t.device.type == "cuda"

    def test_heap_bases_shape(self, symmetric_heap) -> None:
        """Verify that heap_bases has the correct shape (world_size,)."""
        heap = symmetric_heap
        bases = heap.heap_bases
        assert bases.shape == (heap.world_size,)
        assert bases.dtype == torch.int64

    def test_pointer_translation_roundtrip(self, symmetric_heap) -> None:
        """Translate a pointer and verify the offset math round-trips."""
        heap = symmetric_heap
        t = heap.allocate_tensor(shape=(32,), dtype=torch.float32)
        ptr = t.data_ptr()
        base = heap.heap_bases[0].item()
        offset = ptr - base
        assert offset >= 0, "Tensor pointer should be >= heap base"
        assert offset < heap.size, "Tensor pointer should be within heap"
        reconstructed = base + offset
        assert reconstructed == ptr

    def test_heap_context_manager(self, device_info) -> None:
        """Test that the with-statement cleans up the heap."""
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

        from xtile.memory.symmetric_heap import SymmetricHeap

        with SymmetricHeap(
            size=512 * 1024,
            rank=0,
            world_size=1,
            device=device_info.device,
        ) as heap:
            assert heap.size == 512 * 1024
            t = heap.allocate_tensor(shape=(16,), dtype=torch.float32)
            assert t is not None
        # After exiting the context, the heap should be cleaned up
        assert heap._cleaned_up

    def test_heap_multiple_allocations(self, symmetric_heap) -> None:
        """Allocate multiple tensors and verify they do not overlap."""
        heap = symmetric_heap
        t1 = heap.allocate_tensor(shape=(128,), dtype=torch.float32)
        t2 = heap.allocate_tensor(shape=(128,), dtype=torch.float32)

        ptr1 = t1.data_ptr()
        size1 = t1.nelement() * t1.element_size()
        ptr2 = t2.data_ptr()
        size2 = t2.nelement() * t2.element_size()

        # Regions [ptr1, ptr1+size1) and [ptr2, ptr2+size2) must not overlap
        no_overlap = (ptr1 + size1 <= ptr2) or (ptr2 + size2 <= ptr1)
        assert no_overlap, "Allocated tensors must not overlap in memory"


# ---------------------------------------------------------------------------
# Multi-GPU tests
# ---------------------------------------------------------------------------

@pytest.mark.multigpu
class TestSymmetricHeapMultiGPU:
    """Tests that require >= 2 GPUs."""

    def test_heap_bases_multi_device(self, skip_no_multigpu, device_info) -> None:
        """With world_size > 1, heap_bases should have one entry per rank."""
        from xtile.memory.symmetric_heap import SymmetricHeap

        world_size = device_info.num_gpus
        heap = SymmetricHeap(
            size=1024 * 1024,
            rank=0,
            world_size=world_size,
            device=device_info.device,
        )
        try:
            bases = heap.heap_bases
            assert bases.shape == (world_size,), (
                f"Expected heap_bases of shape ({world_size},), got {bases.shape}"
            )
        finally:
            heap.cleanup()
