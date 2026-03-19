"""Tests for xtile.patterns.BulkSyncPattern.

The BulkSyncPattern is the simplest overlap strategy (no overlap -- just
compute then communicate).  This test verifies correctness by comparing
the fused result against ``torch.matmul``.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.multigpu
class TestBulkSyncPattern:
    """Correctness tests for BulkSyncPattern."""

    def test_bulk_sync_correctness(self, skip_no_multigpu, device_info) -> None:
        """Compare BulkSyncPattern output against torch.matmul.

        Allocates A (M x K) and B (K x N), runs BulkSyncPattern.execute(),
        and verifies that the result matches ``torch.matmul(A, B)`` within
        a reasonable tolerance (fp16 accumulation).
        """
        from xtile.patterns import BulkSyncPattern
        from xtile.memory.symmetric_heap import SymmetricHeap

        M, N, K = 512, 512, 512
        world_size = min(device_info.num_gpus, 2)
        device = device_info.device

        heap = SymmetricHeap(
            size=64 * 1024 * 1024,  # 64 MB
            rank=0,
            world_size=world_size,
            device=device,
        )
        try:
            A = torch.randn(M, K, device=device, dtype=torch.float16)
            B = torch.randn(K, N, device=device, dtype=torch.float16)
            C = torch.zeros(M, N, device=device, dtype=torch.float16)

            # Build a minimal context object for the pattern
            class _Ctx:
                pass

            ctx = _Ctx()
            ctx.rank = 0
            ctx.world_size = world_size
            ctx.device = device
            ctx.backend = device_info.backend
            ctx.heap = heap

            pattern = BulkSyncPattern(ctx)
            pattern.execute(A, B, C)
            torch.cuda.synchronize()

            # Reference
            ref = torch.matmul(A.float(), B.float()).half()

            # fp16 tolerance: rtol=1e-2, atol=1e-1 is typical for large matmuls
            assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1), (
                f"BulkSyncPattern output differs from torch.matmul. "
                f"Max diff: {(C.float() - ref.float()).abs().max().item():.4f}"
            )
        finally:
            heap.cleanup()
