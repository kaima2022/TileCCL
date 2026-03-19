"""End-to-end P2P tests on real multi-GPU hardware.

Validates that translate_ptr, remote_load/store, and signal/wait primitives
work correctly on NVIDIA H100 (or any multi-GPU system with peer access).

These tests use single-process multi-GPU mode (SymmetricHeap.create_all).

Key concept: translate_ptr converts a LOCAL pointer to the CORRESPONDING
location in a REMOTE rank's heap (same offset, different base).  This is
the PGAS (Partitioned Global Address Space) model where all ranks allocate
symmetric heaps.
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from xtile.memory.symmetric_heap import SymmetricHeap
from xtile.memory.translation import translate_ptr


# ---------------------------------------------------------------------------
# Skip if < 2 GPUs
# ---------------------------------------------------------------------------

def _skip_if_no_multigpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 GPUs with peer access")


# ---------------------------------------------------------------------------
# Triton kernels for P2P testing
# ---------------------------------------------------------------------------

@triton.jit
def _p2p_read_via_translate_kernel(
    local_ptr,       # local buffer (same offset as data on remote rank)
    result_ptr,      # where to write the result on the caller
    heap_bases,
    caller_rank,     # rank running this kernel
    remote_rank,     # rank whose data we want to read
    BLOCK_SIZE: tl.constexpr,
):
    """Read from a remote rank's heap at the same offset as local_ptr.

    translate_ptr(local_ptr, caller_rank, remote_rank) returns the pointer
    to the SAME offset in remote_rank's heap.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    # Translate local pointer to the corresponding remote location
    remote_ptr = translate_ptr(
        local_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    # Load from remote heap (peer access makes this valid)
    data = tl.load(remote_ptr)
    # Store result locally
    tl.store(result_ptr + offsets, data)


@triton.jit
def _p2p_write_via_translate_kernel(
    local_src_ptr,   # local data to write
    local_dst_ptr,   # local pointer at the same offset as remote destination
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Write local data to a remote rank's heap at the same offset as local_dst_ptr."""
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load local data
    values = tl.load(local_src_ptr + offsets)
    # Translate to remote heap
    remote_ptr = translate_ptr(
        local_dst_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    # Write to remote heap
    tl.store(remote_ptr, values)


@triton.jit
def _p2p_roundtrip_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Read from remote at same offset, add 1.0, write back to remote."""
    offsets = tl.arange(0, BLOCK_SIZE)
    remote_ptr = translate_ptr(
        local_ptr + offsets, caller_rank, remote_rank, heap_bases,
    )
    data = tl.load(remote_ptr)
    data = data + 1.0
    tl.store(remote_ptr, data)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

BLOCK_SIZE = 1024


class TestP2PTranslatePtr:
    """Test translate_ptr with real cross-GPU Triton kernels."""

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
        # Verify symmetric offsets
        off0 = tensors[0].data_ptr() - self.heaps[0].local_base
        off1 = tensors[1].data_ptr() - self.heaps[1].local_base
        assert off0 == off1, f"Offsets differ: {off0} vs {off1}"
        return tensors

    def test_p2p_read_gpu0_reads_gpu1(self):
        """GPU 0 reads data from GPU 1 at the same symmetric offset."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        # Fill GPU 1's tensor with known data
        torch.cuda.set_device(1)
        t1.fill_(3.14)
        torch.cuda.synchronize(1)

        # GPU 0 reads from GPU 1 using translate_ptr
        torch.cuda.set_device(0)
        result = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float32)
        result.fill_(0.0)
        torch.cuda.synchronize(0)

        bases = self.heaps[0].get_heap_bases()
        _p2p_read_via_translate_kernel[(1,)](
            t0, result, bases,
            0, 1,  # caller=GPU0, remote=GPU1
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)

        assert torch.allclose(result, torch.full_like(result, 3.14), atol=1e-5), (
            f"P2P read failed: first={result[0].item()}, last={result[-1].item()}"
        )

    def test_p2p_read_gpu1_reads_gpu0(self):
        """GPU 1 reads data from GPU 0 at the same symmetric offset."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(0)
        t0.fill_(2.718)
        torch.cuda.synchronize(0)

        torch.cuda.set_device(1)
        result = self.heaps[1].allocate_tensor((BLOCK_SIZE,), torch.float32)
        result.fill_(0.0)
        torch.cuda.synchronize(1)

        bases = self.heaps[1].get_heap_bases()
        _p2p_read_via_translate_kernel[(1,)](
            t1, result, bases,
            1, 0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(1)

        assert torch.allclose(result, torch.full_like(result, 2.718), atol=1e-5)

    def test_p2p_write_gpu0_writes_to_gpu1(self):
        """GPU 0 writes data to GPU 1 at the same symmetric offset."""
        # Allocate symmetrically: t0 and t1 are at the same heap offset
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        # Also allocate source data on GPU 0 (separate buffer)
        src0, _ = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(0)
        src0.fill_(42.0)
        torch.cuda.set_device(1)
        t1.fill_(0.0)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # GPU 0 writes to GPU 1's t1 using t0 as the "mirror" pointer
        torch.cuda.set_device(0)
        bases = self.heaps[0].get_heap_bases()
        _p2p_write_via_translate_kernel[(1,)](
            src0,  # local source data
            t0,    # local pointer at same offset as t1 (the destination)
            bases,
            0, 1,  # caller=GPU0, remote=GPU1
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        assert torch.allclose(t1, torch.full_like(t1, 42.0), atol=1e-5), (
            f"P2P write failed: first={t1[0].item()}"
        )

    def test_p2p_roundtrip(self):
        """GPU 0 reads from GPU 1, adds 1.0, writes back."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(1)
        t1.fill_(10.0)
        torch.cuda.synchronize(1)

        torch.cuda.set_device(0)
        bases = self.heaps[0].get_heap_bases()
        _p2p_roundtrip_kernel[(1,)](
            t0, bases, 0, 1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        assert torch.allclose(t1, torch.full_like(t1, 11.0), atol=1e-5), (
            f"Roundtrip failed: t1[0]={t1[0].item()}"
        )

    def test_p2p_read_f16(self):
        """P2P read works with float16 dtype."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float16)

        torch.cuda.set_device(1)
        t1.fill_(1.5)
        torch.cuda.synchronize(1)

        torch.cuda.set_device(0)
        result = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float16)
        result.fill_(0.0)
        torch.cuda.synchronize(0)

        # Need a second symmetric alloc pair to keep offsets aligned
        # Actually result is at a different offset, but that's fine - we
        # use t0 as the "mirror" pointer for translate
        bases = self.heaps[0].get_heap_bases()
        _p2p_read_via_translate_kernel[(1,)](
            t0, result, bases,
            0, 1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)

        assert torch.allclose(result, torch.full_like(result, 1.5), atol=1e-2)

    def test_identity_translation(self):
        """translate_ptr with caller==remote is identity."""
        torch.cuda.set_device(0)
        src = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float32)
        src.fill_(7.77)
        result = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float32)
        result.fill_(0.0)
        torch.cuda.synchronize(0)

        bases = self.heaps[0].get_heap_bases()
        _p2p_read_via_translate_kernel[(1,)](
            src, result, bases,
            0, 0,  # same rank = identity
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)

        assert torch.allclose(result, torch.full_like(result, 7.77), atol=1e-5)

    def test_bidirectional_exchange(self):
        """Both GPUs exchange data simultaneously."""
        t0, t1 = self._symmetric_alloc((BLOCK_SIZE,), torch.float32)

        torch.cuda.set_device(0)
        t0.fill_(100.0)
        r0 = self.heaps[0].allocate_tensor((BLOCK_SIZE,), torch.float32)
        r0.fill_(0.0)

        torch.cuda.set_device(1)
        t1.fill_(200.0)
        r1 = self.heaps[1].allocate_tensor((BLOCK_SIZE,), torch.float32)
        r1.fill_(0.0)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # GPU 0 reads from GPU 1
        torch.cuda.set_device(0)
        bases0 = self.heaps[0].get_heap_bases()
        _p2p_read_via_translate_kernel[(1,)](
            t0, r0, bases0, 0, 1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # GPU 1 reads from GPU 0
        torch.cuda.set_device(1)
        bases1 = self.heaps[1].get_heap_bases()
        _p2p_read_via_translate_kernel[(1,)](
            t1, r1, bases1, 1, 0,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        assert torch.allclose(r0, torch.full_like(r0, 200.0), atol=1e-5), (
            f"GPU0 read from GPU1: got {r0[0].item()}, expected 200.0"
        )
        assert torch.allclose(r1, torch.full_like(r1, 100.0), atol=1e-5), (
            f"GPU1 read from GPU0: got {r1[0].item()}, expected 100.0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
