# SPDX-License-Identifier: Apache-2.0
"""Public smoke tests for single-GPU context construction."""

from __future__ import annotations

import pytest
import torch

import tileccl


def _detect_backend() -> str:
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    return "hip" if getattr(torch.version, "hip", None) is not None else "cuda"


def test_init_single_gpu_context_smoke() -> None:
    """A minimal single-GPU runtime context should attach a heap and allocate tensors."""
    backend = _detect_backend()
    ctx = tileccl.init(
        backend=backend,
        rank=0,
        world_size=1,
        heap_size=8 * 1024 * 1024,
        force_backend=True,
    )
    try:
        assert ctx.has_heap
        assert ctx.heap is not None
        assert ctx.backend_name == backend

        zeros = ctx.zeros(8, 8, dtype=torch.float16)
        randn = ctx.randn(8, 8, dtype=torch.float16)
        empty = ctx.empty(8, dtype=torch.float32)

        assert zeros.shape == (8, 8)
        assert randn.shape == (8, 8)
        assert empty.shape == (8,)
        assert zeros.device.type == "cuda"
        assert torch.count_nonzero(zeros).item() == 0

        ctx.barrier()
    finally:
        if ctx.heap is not None:
            ctx.heap.cleanup()
