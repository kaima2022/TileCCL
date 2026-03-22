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

        heap_metadata = ctx.heap_metadata()
        runtime_metadata = ctx.runtime_metadata()
        assert heap_metadata["rank"] == 0
        assert heap_metadata["world_size"] == 1
        assert heap_metadata["mode"] == "single_process"
        assert heap_metadata["transport_strategy"] == "local_only"
        assert heap_metadata["allocator"]["name"] == "torch_bump"
        assert heap_metadata["allocator"]["capabilities"]["external_mapping"] is False
        assert heap_metadata["allocator"]["external_tensor_import_mode"] == "copy"
        assert heap_metadata["allocator"]["external_mapping_mode"] == "none"
        assert heap_metadata["allocator"]["peer_transport_modes"] == [
            "ctypes_ipc",
            "pytorch_ipc",
            "peer_access_pointer_exchange",
        ]
        assert heap_metadata["allocator"]["peer_import_access_kinds"] == [
            "local",
            "peer_direct",
            "mapped_remote",
            "remote_pointer",
        ]
        assert heap_metadata["allocator"]["memory_model"]["peer_import_model"] == (
            "per_rank_transport_resolved_imports"
        )
        assert heap_metadata["allocator"]["memory_model"]["external_mapping_mode"] == "none"
        assert len(heap_metadata["segments"]) == 1
        assert heap_metadata["segments"][0]["segment_id"] == "heap"
        assert len(heap_metadata["peer_exports"]) == 1
        assert heap_metadata["peer_exports"][0]["peer_rank"] == 0
        assert heap_metadata["peer_exports"][0]["segment_id"] == "heap"
        assert len(heap_metadata["peer_imports"]) == 1
        assert heap_metadata["peer_imports"][0]["peer_rank"] == 0
        assert heap_metadata["peer_imports"][0]["access_kind"] == "local"
        assert len(heap_metadata["peer_memory_map"]) == 1
        assert heap_metadata["peer_memory_map"][0]["peer_rank"] == 0
        assert heap_metadata["peer_memory_map"][0]["access_kind"] == "local"
        assert runtime_metadata["backend"] == device_info.backend
        assert runtime_metadata["has_heap"] is True
        assert runtime_metadata["heap"]["local_base"] == ctx.heap.local_base

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
            assert ctx.heap_metadata()["transport_strategy"] == "peer_access"
            assert len(ctx.heap_metadata()["segments"]) == 1
            assert len(ctx.heap_metadata()["peer_exports"]) == 2
            assert {entry["peer_rank"] for entry in ctx.heap_metadata()["peer_exports"]} == {0, 1}
            assert len(ctx.heap_metadata()["peer_imports"]) == 2
            assert {entry["peer_rank"] for entry in ctx.heap_metadata()["peer_imports"]} == {0, 1}
            assert {entry["access_kind"] for entry in ctx.heap_metadata()["peer_imports"]} == {"local", "peer_direct"}
            assert len(ctx.heap_metadata()["peer_memory_map"]) == 2
            assert {entry["peer_rank"] for entry in ctx.heap_metadata()["peer_memory_map"]} == {0, 1}
            assert {entry["access_kind"] for entry in ctx.heap_metadata()["peer_memory_map"]} == {"local", "peer_direct"}

            tensor = ctx.zeros(4, 4, dtype=torch.float16)
            assert tensor.shape == (4, 4)
            ctx.heap.get_offset(tensor.data_ptr())
    finally:
        for ctx in contexts:
            if ctx.heap is not None:
                ctx.heap.cleanup()
