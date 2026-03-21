"""Integration tests for high-level xtile.ops entrypoints."""

from __future__ import annotations

import torch

import xtile


def test_gemm_allscatter_high_level_api(skip_no_multigpu, device_info) -> None:
    """Single-GPU smoke test for the high-level API contract."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    M, N, K = 256, 256, 256
    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        ctx = xtile.init(
            backend=device_info.backend,
            rank=0,
            world_size=1,
            heap=heaps[0],
            force_backend=True,
        )
        A = torch.randn(M, K, device=ctx.device, dtype=torch.float16)
        B = torch.randn(K, N, device=ctx.device, dtype=torch.float16)
        C = torch.zeros(M, N, device=ctx.device, dtype=torch.float16)

        xtile.ops.gemm_allscatter(A, B, C, ctx=ctx, pattern="bulk_sync")
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B.float()).half()
        assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()
