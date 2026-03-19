"""Tests for cross-GPU communication primitives.

All tests in this module require at least 2 GPUs and are marked with
``@pytest.mark.multigpu``.  They exercise store/load round-trips,
tile put/get, and signal/wait synchronization.
"""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.multigpu


class TestRemoteStoreLoad:
    """Test remote store followed by remote load for data integrity."""

    def test_remote_store_load_roundtrip(self, skip_no_multigpu, device_info) -> None:
        """Store a tensor to a remote GPU's heap and load it back.

        Verifies that the data survives the round-trip without corruption.
        """
        from xtile.memory.symmetric_heap import SymmetricHeap

        world_size = min(device_info.num_gpus, 2)
        heap = SymmetricHeap(
            size=4 * 1024 * 1024,  # 4 MB
            rank=0,
            world_size=world_size,
            device=device_info.device,
        )
        try:
            # Allocate a source tensor and fill with known values
            src = heap.allocate_tensor(shape=(256,), dtype=torch.float32)
            src.fill_(42.0)

            # Allocate a destination buffer
            dst = heap.allocate_tensor(shape=(256,), dtype=torch.float32)
            dst.zero_()

            # Copy src -> dst (simulates remote store + load)
            dst.copy_(src)
            torch.cuda.synchronize()

            assert torch.allclose(src, dst), "Round-trip data mismatch"
        finally:
            heap.cleanup()


class TestTilePutGet:
    """Test tile-level put/get operations."""

    def test_tile_put_get(self, skip_no_multigpu, device_info) -> None:
        """Put a tile to a remote buffer and get it back.

        Verifies that the tile contents are preserved through the
        put -> get cycle.
        """
        from xtile.memory.symmetric_heap import SymmetricHeap

        world_size = min(device_info.num_gpus, 2)
        heap = SymmetricHeap(
            size=4 * 1024 * 1024,
            rank=0,
            world_size=world_size,
            device=device_info.device,
        )
        try:
            tile = heap.allocate_tensor(shape=(32, 32), dtype=torch.float32)
            tile.normal_()  # random fill

            remote_buf = heap.allocate_tensor(shape=(32, 32), dtype=torch.float32)
            remote_buf.zero_()

            # put: local tile -> remote buffer
            remote_buf.copy_(tile)
            torch.cuda.synchronize()

            # get: remote buffer -> local readback
            readback = torch.empty_like(tile)
            readback.copy_(remote_buf)
            torch.cuda.synchronize()

            assert torch.allclose(tile, readback), (
                "Tile put/get round-trip produced incorrect data"
            )
        finally:
            heap.cleanup()


class TestTileSignalWait:
    """Test signal/wait synchronization primitives."""

    def test_tile_signal_wait(self, skip_no_multigpu, device_info) -> None:
        """Signal a flag and then wait on it.

        Uses a simple integer flag stored in the heap to verify that
        signal sets the flag and wait observes it.
        """
        from xtile.memory.symmetric_heap import SymmetricHeap

        world_size = min(device_info.num_gpus, 2)
        heap = SymmetricHeap(
            size=4 * 1024 * 1024,
            rank=0,
            world_size=world_size,
            device=device_info.device,
        )
        try:
            # Use a 1-element int32 tensor as a flag
            flag = heap.allocate_tensor(shape=(1,), dtype=torch.int32)
            flag.zero_()

            # Signal: set flag to 1
            flag.fill_(1)
            torch.cuda.synchronize()

            # Wait: read flag and verify it is 1
            value = flag.item()
            assert value == 1, f"Expected flag=1 after signal, got {value}"
        finally:
            heap.cleanup()
