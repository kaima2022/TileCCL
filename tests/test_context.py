"""Tests for XTile runtime context construction and heap attachment."""

from __future__ import annotations

import pytest
import torch

import xtile


def test_init_with_heap_size_attaches_single_gpu_heap(skip_no_gpu, device_info) -> None:
    """``xtile.init(..., heap_size=...)`` should return a usable single-GPU context."""
    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        heap_size=8 * 1024 * 1024,
        force_backend=True,
    )
    try:
        assert xtile.current_context() is ctx
        assert ctx.backend_name == device_info.backend
        assert ctx.has_heap
        assert ctx.heap is not None
        assert ctx.heap.rank == 0
        assert ctx.heap.world_size == 1
        assert ctx.heap_bases.shape == (1,)
        assert ctx.heap_bases.dtype == torch.int64

        zeros = ctx.zeros(32, 16, dtype=torch.float16)
        randn = ctx.randn(32, 16, dtype=torch.float16)
        empty = ctx.empty(8, dtype=torch.float32)

        assert zeros.shape == (32, 16)
        assert randn.shape == (32, 16)
        assert empty.shape == (8,)
        assert zeros.device.type == "cuda"
        assert randn.device.type == "cuda"
        assert empty.device.type == "cuda"
        assert torch.count_nonzero(zeros).item() == 0

        # Validate these tensors truly live inside the attached heap.
        ctx.heap.get_offset(zeros.data_ptr())
        ctx.heap.get_offset(randn.data_ptr())
        ctx.heap.get_offset(empty.data_ptr())

        ctx.barrier()
    finally:
        if ctx.heap is not None:
            ctx.heap.cleanup()


@pytest.mark.multigpu
def test_init_local_returns_attached_contexts(skip_no_multigpu, device_info) -> None:
    """``xtile.init_local`` should build one attached context per visible rank."""
    contexts = xtile.init_local(
        world_size=2,
        heap_size=8 * 1024 * 1024,
        backend=device_info.backend,
        force_backend=True,
    )
    try:
        assert len(contexts) == 2
        for rank, ctx in enumerate(contexts):
            assert ctx.rank == rank
            assert ctx.world_size == 2
            assert ctx.device == f"cuda:{rank}"
            assert ctx.backend_name == device_info.backend
            assert ctx.has_heap
            assert ctx.heap is not None
            assert ctx.heap.rank == rank
            assert ctx.heap.world_size == 2
            assert ctx.heap_bases.shape == (2,)
            assert int(ctx.heap_bases[rank].item()) == ctx.heap.local_base

            tensor = ctx.zeros(4, 4, dtype=torch.float16)
            assert tensor.shape == (4, 4)
            ctx.heap.get_offset(tensor.data_ptr())
    finally:
        for ctx in contexts:
            if ctx.heap is not None:
                ctx.heap.cleanup()
