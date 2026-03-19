"""Tests for xtile.patterns.BulkSyncPattern.

The BulkSyncPattern is the simplest overlap strategy (no overlap -- just
compute then communicate).  This test verifies correctness by comparing
the fused result against ``torch.matmul``.
"""

from __future__ import annotations

import pytest
import torch


def _make_ctx(rank, world_size, device, backend_name, heap_bases):
    """Build a minimal context object for pattern testing."""
    from xtile.backends import get_backend

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.rank = rank
    ctx.world_size = world_size
    ctx.device = device
    ctx.backend = get_backend(backend_name)
    ctx.heap_bases = heap_bases
    return ctx


@pytest.mark.multigpu
class TestBulkSyncPattern:
    """Correctness tests for BulkSyncPattern."""

    def test_bulk_sync_gemm_correctness(self, skip_no_multigpu, device_info) -> None:
        """Compare BulkSyncPattern GEMM output against torch.matmul.

        Uses a single GPU (world_size=1) so the scatter is a no-op.
        Verifies that the persistent GEMM kernel is numerically correct.
        """
        from xtile.patterns import BulkSyncPattern
        from xtile.memory.symmetric_heap import SymmetricHeap

        M, N, K = 512, 512, 512

        heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
        try:
            torch.cuda.set_device(0)
            A = torch.randn(M, K, device="cuda:0", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda:0", dtype=torch.float16)
            C = torch.zeros(M, N, device="cuda:0", dtype=torch.float16)

            ctx = _make_ctx(0, 1, "cuda:0", device_info.backend, heaps[0].get_heap_bases())

            pattern = BulkSyncPattern(ctx)
            pattern.execute(A, B, C)
            torch.cuda.synchronize()

            ref = torch.matmul(A.float(), B.float()).half()
            assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1), (
                f"BulkSyncPattern output differs from torch.matmul. "
                f"Max diff: {(C.float() - ref.float()).abs().max().item():.4f}"
            )
        finally:
            for h in heaps:
                h.cleanup()

    def test_bulk_sync_scatter_correctness(self, skip_no_multigpu, device_info) -> None:
        """Test scatter phase: GPU 0 computes and scatters to GPU 1.

        Verifies that after scatter, GPU 1's buffer contains the correct
        column shard of the GEMM result written by GPU 0.
        """
        if device_info.num_gpus < 2:
            pytest.skip("Requires 2+ GPUs")

        from xtile.patterns import BulkSyncPattern
        from xtile.memory.symmetric_heap import SymmetricHeap

        M, N, K = 256, 256, 256
        world_size = 2

        heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=world_size)
        try:
            torch.cuda.set_device(0)

            # Allocate C from the symmetric heap so translate_ptr works
            C0 = heaps[0].allocate_tensor((M, N), dtype=torch.float16)
            C1 = heaps[1].allocate_tensor((M, N), dtype=torch.float16)
            C0.zero_()
            C1.zero_()

            A = torch.randn(M, K, device="cuda:0", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda:0", dtype=torch.float16)

            ctx = _make_ctx(0, world_size, "cuda:0", device_info.backend,
                            heaps[0].get_heap_bases())

            pattern = BulkSyncPattern(ctx, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
            pattern.execute(A, B, C0)
            torch.cuda.synchronize()

            # After scatter from rank 0, GPU 1's buffer columns [0, N_per_rank)
            # should contain the first shard of the GEMM result
            N_per_rank = N // world_size
            ref = torch.matmul(A.float(), B.float()).half()

            C1_host = C1.cpu()
            expected_cols = ref[:, :N_per_rank].cpu()
            assert torch.allclose(C1_host[:, :N_per_rank], expected_cols, rtol=1e-2, atol=1e-1), (
                f"Scatter to GPU 1 failed. "
                f"Max diff: {(C1_host[:, :N_per_rank].float() - expected_cols.float()).abs().max().item():.4f}"
            )
        finally:
            for h in heaps:
                h.cleanup()
