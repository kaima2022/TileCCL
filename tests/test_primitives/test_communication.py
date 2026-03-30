# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-GPU communication primitives using real Triton kernels.

All tests in this module require at least 2 GPUs and use
``SymmetricHeap.create_all`` (single-process mode) for simplicity.
They exercise:
- translate_ptr-based remote store / load (as used in patterns)
- tile_put / tile_get (pointer-based)
- tile_signal / tile_wait (intra-kernel sync)
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.memory.translation import translate_ptr
from tncc.patterns._helpers import multicast_tile_to_peers
from tncc.sync.primitives import (
    tile_acquire_credit,
    tile_release_credit,
    tile_signal,
    tile_try_wait,
    tile_wait,
)

pytestmark = pytest.mark.multigpu

BLOCK_SIZE = 1024


def _skip_if_no_multigpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 GPUs with peer access")


# ---------------------------------------------------------------------------
# Triton kernels for testing communication primitives
# ---------------------------------------------------------------------------

@triton.jit
def _remote_store_via_translate_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Write local data to the same offset in remote rank's heap.

    This is the pattern used by scatter_tile_to_peer: translate a local
    pointer to the remote space and store there.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load from local
    data = tl.load(local_ptr + offsets)
    # Translate to remote and store
    remote_ptr = translate_ptr(
        local_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    tl.store(remote_ptr, data)


@triton.jit
def _remote_load_via_translate_kernel(
    local_ptr,
    result_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Read from remote rank's heap at same offset, write to result."""
    offsets = tl.arange(0, BLOCK_SIZE)
    remote_ptr = translate_ptr(
        local_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    data = tl.load(remote_ptr)
    tl.store(result_ptr + offsets, data)


@triton.jit
def _put_via_translate_kernel(
    src_ptr,
    dst_mirror_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Put: load from local src, translate dst mirror to remote, store.

    Simulates tile_put: local memory → remote memory.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    data = tl.load(src_ptr + offsets)
    remote_ptr = translate_ptr(
        dst_mirror_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    tl.store(remote_ptr, data)


@triton.jit
def _get_via_translate_kernel(
    local_mirror_ptr,
    result_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Get: translate local mirror to remote, load, store to local result.

    Simulates tile_get: remote memory → local memory.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    remote_ptr = translate_ptr(
        local_mirror_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    data = tl.load(remote_ptr)
    tl.store(result_ptr + offsets, data)


@triton.jit
def _signal_wait_producer_kernel(
    data_ptr,
    locks_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Producer: write data then signal completion."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(data_ptr + offsets, tl.full((BLOCK_SIZE,), 42.0, dtype=tl.float32), mask=mask)
    tile_signal(locks_ptr, 0)


@triton.jit
def _signal_wait_consumer_kernel(
    data_ptr,
    locks_ptr,
    result_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Consumer: wait for signal, then read data."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tile_wait(locks_ptr, 0)
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    tl.store(result_ptr + offsets, data, mask=mask)


@triton.jit
def _try_wait_kernel(
    locks_ptr,
    result_ptr,
    MAX_SPINS: tl.constexpr,
):
    """Probe a tile lock without blocking forever."""
    status = tile_try_wait(locks_ptr, 0, max_spins=MAX_SPINS)
    tl.store(result_ptr, status)


@triton.jit
def _credit_consumer_kernel(
    credit_ptr,
    data_ptr,
    result_ptr,
):
    """Consume one credit, then read one scalar payload."""
    tile_acquire_credit(credit_ptr, 0)
    value = tl.load(data_ptr)
    tl.store(result_ptr, value)


@triton.jit
def _credit_release_kernel(
    credit_ptr,
    heap_bases,
    rank,
    peer,
):
    """Release one remote credit to a peer rank."""
    tile_release_credit(credit_ptr, 0, peer, rank, heap_bases)


@triton.jit
def _software_multicast_kernel(
    C_ptr,
    heap_bases,
    M,
    N,
    stride_cm,
    stride_cn,
    rank,
    world_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Multicast one local tile to all peers."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tile_data = tl.load(ptrs, mask=mask, other=0.0)
    multicast_tile_to_peers(
        C_ptr,
        tile_data,
        offs_m,
        offs_n,
        rank,
        world_size,
        heap_bases,
        0,
        N,
        stride_cm,
        0,
        mask,
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestRemoteStoreLoad:
    """Test translate_ptr-based remote store and load for data integrity."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _skip_if_no_multigpu()
        self.heaps = SymmetricHeap.create_all(
            size=4 * 1024 * 1024, world_size=2,
        )
        yield
        for h in self.heaps:
            h.cleanup()

    def _symmetric_alloc(self, shape, dtype):
        """Allocate tensors at the same offset on both GPUs."""
        tensors = []
        for rank in range(2):
            torch.cuda.set_device(rank)
            t = self.heaps[rank].allocate_tensor(shape, dtype)
            tensors.append(t)
        off0 = tensors[0].data_ptr() - self.heaps[0].local_base
        off1 = tensors[1].data_ptr() - self.heaps[1].local_base
        assert off0 == off1, f"Offsets differ: {off0} vs {off1}"
        return tensors

    def test_remote_store_roundtrip(self) -> None:
        """GPU 0 stores data to GPU 1, then loads it back to verify."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(0)
        t0.fill_(42.0)
        result = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float32)
        result.fill_(0.0)
        torch.cuda.set_device(1)
        t1.fill_(0.0)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # GPU 0 writes its data to GPU 1 at same offset
        torch.cuda.set_device(0)
        bases = self.heaps[0].get_heap_bases()
        _remote_store_via_translate_kernel[(1,)](
            t0, bases, 0, 1, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # Verify GPU 1 received the data
        assert torch.allclose(t1, torch.full_like(t1, 42.0), atol=1e-5), (
            f"Remote store failed: t1[0]={t1[0].item()}"
        )

        # GPU 0 reads back from GPU 1
        _remote_load_via_translate_kernel[(1,)](
            t0, result, bases, 0, 1, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)

        assert torch.allclose(result, torch.full_like(result, 42.0), atol=1e-5), (
            f"Remote load failed: result[0]={result[0].item()}"
        )

    def test_remote_store_load_f16(self) -> None:
        """Round-trip with float16 dtype."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float16)

        torch.cuda.set_device(0)
        t0.fill_(3.14)
        result = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float16)
        result.fill_(0.0)
        torch.cuda.set_device(1)
        t1.fill_(0.0)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        torch.cuda.set_device(0)
        bases = self.heaps[0].get_heap_bases()
        _remote_store_via_translate_kernel[(1,)](
            t0, bases, 0, 1, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # Read back
        _remote_load_via_translate_kernel[(1,)](
            t0, result, bases, 0, 1, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)

        assert torch.allclose(result, torch.full_like(result, 3.14), atol=1e-2)


class TestTilePutGet:
    """Test put/get via translate_ptr (pointer-based memory-to-memory)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _skip_if_no_multigpu()
        self.heaps = SymmetricHeap.create_all(
            size=4 * 1024 * 1024, world_size=2,
        )
        yield
        for h in self.heaps:
            h.cleanup()

    def _symmetric_alloc(self, shape, dtype):
        tensors = []
        for rank in range(2):
            torch.cuda.set_device(rank)
            t = self.heaps[rank].allocate_tensor(shape, dtype)
            tensors.append(t)
        return tensors

    def test_tile_put_get(self) -> None:
        """Put local data to remote buffer, then get it back."""
        src0, _ = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)
        buf0, buf1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(0)
        src0.fill_(99.5)
        result = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float32)
        result.fill_(0.0)
        torch.cuda.set_device(1)
        buf1.fill_(0.0)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # Put: GPU 0 src → GPU 1's buf (via buf0 as mirror)
        torch.cuda.set_device(0)
        bases = self.heaps[0].get_heap_bases()
        _put_via_translate_kernel[(1,)](
            src0, buf0, bases, 0, 1, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        assert torch.allclose(buf1, torch.full_like(buf1, 99.5), atol=1e-5), (
            f"Put failed: buf1[0]={buf1[0].item()}"
        )

        # Get: GPU 0 pulls from GPU 1's buf
        _get_via_translate_kernel[(1,)](
            buf0, result, bases, 0, 1, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)

        assert torch.allclose(result, torch.full_like(result, 99.5), atol=1e-5), (
            f"Get failed: result[0]={result[0].item()}"
        )

    def test_tile_put_get_reverse(self) -> None:
        """GPU 1 puts to GPU 0, then gets back."""
        _, src1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)
        buf0, buf1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(1)
        src1.fill_(77.0)
        result = self.heaps[1].allocate_tensor((BLOCK_SIZE,), torch.float32)
        result.fill_(0.0)
        torch.cuda.set_device(0)
        buf0.fill_(0.0)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        torch.cuda.set_device(1)
        bases = self.heaps[1].get_heap_bases()
        _put_via_translate_kernel[(1,)](
            src1, buf1, bases, 1, 0, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        assert torch.allclose(buf0, torch.full_like(buf0, 77.0), atol=1e-5)

        _get_via_translate_kernel[(1,)](
            buf1, result, bases, 1, 0, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(1)

        assert torch.allclose(result, torch.full_like(result, 77.0), atol=1e-5)


class TestTileSignalWait:
    """Test signal/wait synchronization primitives via Triton kernels."""

    def test_tile_signal_wait(self) -> None:
        """Producer signals, consumer waits, data is visible after acquire."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

        torch.cuda.set_device(0)
        N = BLOCK_SIZE

        locks = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        data = torch.zeros(N, dtype=torch.float32, device="cuda:0")
        result = torch.zeros(N, dtype=torch.float32, device="cuda:0")

        prod_stream = torch.cuda.Stream()
        cons_stream = torch.cuda.Stream()

        with torch.cuda.stream(prod_stream):
            _signal_wait_producer_kernel[(1,)](
                data, locks, N, BLOCK_SIZE=BLOCK_SIZE,
            )

        with torch.cuda.stream(cons_stream):
            _signal_wait_consumer_kernel[(1,)](
                data, locks, result, N, BLOCK_SIZE=BLOCK_SIZE,
            )

        prod_stream.synchronize()
        cons_stream.synchronize()

        assert torch.allclose(result, torch.full_like(result, 42.0), atol=1e-5), (
            f"Signal/wait failed: result[0]={result[0].item()}"
        )

    def test_signal_wait_multiple_tiles(self) -> None:
        """Signal/wait works across multiple tile IDs."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

        torch.cuda.set_device(0)
        NUM_TILES = 4

        locks = torch.zeros(NUM_TILES, dtype=torch.int32, device="cuda:0")
        data = torch.zeros(NUM_TILES, dtype=torch.float32, device="cuda:0")

        @triton.jit
        def _multi_signal_kernel(data_ptr, locks_ptr):
            # Write 1.0, 2.0, 3.0, 4.0 and signal each
            offsets = tl.arange(0, 4)
            vals = (offsets + 1).to(tl.float32)
            tl.store(data_ptr + offsets, vals)
            tile_signal(locks_ptr, 0)
            tile_signal(locks_ptr, 1)
            tile_signal(locks_ptr, 2)
            tile_signal(locks_ptr, 3)

        @triton.jit
        def _multi_wait_kernel(data_ptr, locks_ptr, result_ptr):
            tile_wait(locks_ptr, 0)
            tile_wait(locks_ptr, 1)
            tile_wait(locks_ptr, 2)
            tile_wait(locks_ptr, 3)
            offsets = tl.arange(0, 4)
            vals = tl.load(data_ptr + offsets)
            tl.store(result_ptr + offsets, vals)

        result = torch.zeros(NUM_TILES, dtype=torch.float32, device="cuda:0")

        prod_stream = torch.cuda.Stream()
        cons_stream = torch.cuda.Stream()

        with torch.cuda.stream(prod_stream):
            _multi_signal_kernel[(1,)](data, locks)

        with torch.cuda.stream(cons_stream):
            _multi_wait_kernel[(1,)](data, locks, result)

        prod_stream.synchronize()
        cons_stream.synchronize()

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda:0")
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Multi-tile signal/wait failed: result={result.tolist()}"
        )


class TestNonBlockingWaitAndCredits:
    """Test the P0 synchronization additions: try_wait and credits."""

    def test_tile_try_wait_success_and_timeout(self) -> None:
        _skip_if_no_multigpu()
        torch.cuda.set_device(0)

        locks = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        result = torch.zeros(1, dtype=torch.int32, device="cuda:0")

        locks.fill_(1)
        _try_wait_kernel[(1,)](locks, result, MAX_SPINS=8)
        torch.cuda.synchronize(0)
        assert result.item() == 1
        assert locks.item() == 0

        result.zero_()
        _try_wait_kernel[(1,)](locks, result, MAX_SPINS=8)
        torch.cuda.synchronize(0)
        assert result.item() == 0
        assert locks.item() == 0

    def test_tile_acquire_credit_consumes_local_credit(self) -> None:
        _skip_if_no_multigpu()
        torch.cuda.set_device(0)

        credits = torch.ones(1, dtype=torch.int32, device="cuda:0")
        data = torch.tensor([7.0], dtype=torch.float32, device="cuda:0")
        result = torch.zeros(1, dtype=torch.float32, device="cuda:0")

        _credit_consumer_kernel[(1,)](credits, data, result)
        torch.cuda.synchronize(0)

        assert result.item() == pytest.approx(7.0)
        assert credits.item() == 0

    def test_tile_release_credit_updates_remote_peer(self) -> None:
        _skip_if_no_multigpu()
        heaps = SymmetricHeap.create_all(size=4 * 1024 * 1024, world_size=2)
        try:
            credits = []
            for rank in range(2):
                torch.cuda.set_device(rank)
                credits.append(heaps[rank].allocate_tensor((1,), torch.int32))
                credits[rank].zero_()

            torch.cuda.set_device(1)
            bases = heaps[1].get_heap_bases()
            _credit_release_kernel[(1,)](credits[1], bases, 1, 0)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)

            assert credits[0].item() == 1
            assert credits[1].item() == 0
        finally:
            for heap in heaps:
                heap.cleanup()


class TestSoftwareMulticast:
    """Test the portable software tile-multicast helper."""

    def test_multicast_tile_to_peer(self) -> None:
        _skip_if_no_multigpu()
        heaps = SymmetricHeap.create_all(size=4 * 1024 * 1024, world_size=2)
        try:
            matrices = []
            for rank in range(2):
                torch.cuda.set_device(rank)
                matrices.append(heaps[rank].allocate_tensor((16, 16), torch.float32))
                matrices[rank].zero_()

            torch.cuda.set_device(0)
            matrices[0].copy_(torch.arange(16 * 16, device="cuda:0", dtype=torch.float32).view(16, 16))
            bases = heaps[0].get_heap_bases()
            _software_multicast_kernel[(1,)](
                matrices[0],
                bases,
                16,
                16,
                matrices[0].stride(0),
                matrices[0].stride(1),
                0,
                2,
                BLOCK_M=16,
                BLOCK_N=16,
            )
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)

            assert torch.allclose(matrices[1].cpu(), matrices[0].cpu(), atol=1e-5)
        finally:
            for heap in heaps:
                heap.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
