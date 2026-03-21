"""Integration tests for high-level xtile.ops entrypoints."""

from __future__ import annotations

import pytest
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


def test_build_gemm_allscatter_plan_exposes_stable_metadata(
    skip_no_multigpu,
    device_info,
) -> None:
    """Plan building should resolve contract + pattern once, up front."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    M, N, K = 128, 256, 64
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

        plan = xtile.ops.build_gemm_allscatter_plan(A, B, C, ctx=ctx, pattern="bulk_sync")
        payload = plan.to_dict()

        assert plan.pattern_name == "bulk_sync"
        assert plan.execution.full_N == N
        assert plan.execution.rhs_layout == "full"
        assert plan.execution.output_layout == "full"
        assert payload["pattern"] == "bulk_sync"
        assert payload["execution"]["full_N"] == N
    finally:
        for heap in heaps:
            heap.cleanup()


def test_gemm_allscatter_sharded_requires_explicit_full_n() -> None:
    """The expert sharded API must stay explicit about the logical full shape."""
    with pytest.raises(TypeError):
        xtile.ops.gemm_allscatter_sharded(None, None, None)  # type: ignore[misc]


def test_gemm_allscatter_sharded_expert_api_smoke(skip_no_multigpu, device_info) -> None:
    """The expert shard/shard wrapper should execute through the same plan path."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    M, N, K = 128, 256, 64
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

        xtile.ops.gemm_allscatter_sharded(
            A,
            B,
            C,
            ctx=ctx,
            full_N=N,
            pattern="bulk_sync",
        )
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B.float()).half()
        assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_allgather_plan_exposes_stable_metadata(skip_no_multigpu, device_info) -> None:
    """AllGatherPlan should capture the validated collective contract."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        ctx = xtile.init(
            backend=device_info.backend,
            rank=0,
            world_size=1,
            heap=heaps[0],
            force_backend=True,
        )
        src = ctx.zeros(32, dtype=torch.float32)
        output = ctx.zeros(32, dtype=torch.float32)

        plan = xtile.ops.build_allgather_plan(src, output, ctx=ctx)
        payload = plan.to_dict()

        assert plan.block_size == 32
        assert payload["op"] == "allgather"
        assert payload["block_size"] == 32
        assert payload["input_shape"] == [32]
        assert payload["output_shape"] == [32]
    finally:
        for heap in heaps:
            heap.cleanup()


@pytest.mark.multigpu
def test_allgather_high_level_api_multigpu(skip_no_multigpu, device_info) -> None:
    """The high-level allgather API should wrap the collective launcher correctly."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    block_size = 128
    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=world_size)
    try:
        contexts = [
            xtile.init(
                backend=device_info.backend,
                rank=rank,
                world_size=world_size,
                heap=heaps[rank],
                force_backend=True,
            )
            for rank in range(world_size)
        ]

        src = []
        output = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            src.append(heaps[rank].allocate_tensor((block_size,), torch.float32))
            output.append(heaps[rank].allocate_tensor((block_size * world_size,), torch.float32))

        for rank in range(world_size):
            torch.cuda.set_device(rank)
            src[rank].fill_(float((rank + 1) * 10))
            output[rank].zero_()

        for rank in range(world_size):
            xtile.ops.allgather(src[rank], output[rank], ctx=contexts[rank])

        for rank in range(world_size):
            torch.cuda.synchronize(rank)
            assert torch.allclose(output[rank][:block_size], torch.full_like(output[rank][:block_size], 10.0))
            assert torch.allclose(output[rank][block_size:], torch.full_like(output[rank][block_size:], 20.0))
    finally:
        for heap in heaps:
            heap.cleanup()
