# SPDX-License-Identifier: Apache-2.0
"""Public smoke tests for the minimal high-level GEMM+collective API."""

from __future__ import annotations

import pytest
import torch

import tileccl


def _detect_backend() -> str:
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    return "hip" if getattr(torch.version, "hip", None) is not None else "cuda"


def test_gemm_allscatter_single_gpu_smoke() -> None:
    """The minimal single-GPU GEMM+allscatter API should run and match matmul."""
    from tileccl.memory.symmetric_heap import SymmetricHeap

    backend = _detect_backend()
    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        ctx = tileccl.init(
            backend=backend,
            rank=0,
            world_size=1,
            heap=heaps[0],
            force_backend=True,
        )
        a = torch.randn(128, 64, device=ctx.device, dtype=torch.float16)
        b = torch.randn(64, 128, device=ctx.device, dtype=torch.float16)
        c = torch.zeros(128, 128, device=ctx.device, dtype=torch.float16)

        tileccl.ops.gemm_allscatter(a, b, c, ctx=ctx, pattern="bulk_sync")
        torch.cuda.synchronize()

        ref = torch.matmul(a.float(), b.float()).half()
        assert torch.allclose(c, ref, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()
