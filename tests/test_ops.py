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


def test_build_gemm_allscatter_plan_requires_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """The current GEMM+allscatter runtime should fail early without a heap."""
    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )
    A = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
    B = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
    C = torch.zeros(32, 32, device=ctx.device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="No SymmetricHeap is attached"):
        xtile.ops.build_gemm_allscatter_plan(A, B, C, ctx=ctx, pattern="bulk_sync")


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


@pytest.mark.multigpu
def test_gemm_allscatter_full_to_shard_wrapper_multigpu(
    skip_no_multigpu,
    device_info,
) -> None:
    """The high-level full/shard wrapper should return the rank-local shard."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    M, N, K = 128, 256, 64
    shard_cols = N // world_size
    heaps = SymmetricHeap.create_all(size=128 * 1024 * 1024, world_size=world_size)
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

        for rank, ctx in enumerate(contexts):
            torch.cuda.set_device(rank)
            A = torch.randn(M, K, device=ctx.device, dtype=torch.float16)
            B = torch.randn(K, N, device=ctx.device, dtype=torch.float16)
            C = torch.zeros(M, shard_cols, device=ctx.device, dtype=torch.float16)

            plan = xtile.ops.build_gemm_allscatter_plan(
                A,
                B,
                C,
                ctx=ctx,
                full_N=N,
                b_layout="full",
                c_layout="shard",
                pattern="bulk_sync",
            )
            payload = plan.to_dict()

            xtile.ops.gemm_allscatter(
                A,
                B,
                C,
                ctx=ctx,
                full_N=N,
                b_layout="full",
                c_layout="shard",
                pattern="bulk_sync",
            )
            torch.cuda.synchronize(rank)

            ref = torch.matmul(A.float(), B.float()).half()
            col_offset = rank * shard_cols
            expected = ref[:, col_offset:col_offset + shard_cols]

            assert payload["materialization"] == "full_to_shard"
            assert payload["public_contract"]["rhs_layout"] == "full"
            assert payload["public_contract"]["output_layout"] == "shard"
            assert torch.allclose(C, expected, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()


@pytest.mark.multigpu
def test_build_gemm_allscatter_shard_to_full_plan_is_rejected(
    skip_no_multigpu,
    device_info,
) -> None:
    """The inverse mixed layout remains intentionally unsupported."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    M, N, K = 64, 128, 32
    shard_cols = N // world_size
    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=world_size)
    try:
        ctx = xtile.init(
            backend=device_info.backend,
            rank=0,
            world_size=world_size,
            heap=heaps[0],
            force_backend=True,
        )
        torch.cuda.set_device(0)
        A = torch.randn(M, K, device=ctx.device, dtype=torch.float16)
        B = torch.randn(K, shard_cols, device=ctx.device, dtype=torch.float16)
        C = torch.zeros(M, N, device=ctx.device, dtype=torch.float16)

        with pytest.raises(ValueError, match="future gemm_allgather-style API"):
            xtile.ops.build_gemm_allscatter_plan(
                A,
                B,
                C,
                ctx=ctx,
                full_N=N,
                b_layout="shard",
                c_layout="full",
                pattern="bulk_sync",
            )
    finally:
        for heap in heaps:
            heap.cleanup()


def test_gemm_allgather_high_level_api(skip_no_multigpu, device_info) -> None:
    """Single-GPU smoke test for the high-level GEMM + allgather contract."""
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
        B_shard = torch.randn(K, N, device=ctx.device, dtype=torch.float16)
        C = ctx.zeros(M, N, dtype=torch.float16)

        xtile.ops.gemm_allgather(A, B_shard, C, ctx=ctx)
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B_shard.float()).half()
        assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_gemm_allgather_plan_requires_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """The current GEMM + allgather runtime should fail early without a heap."""
    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )
    A = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
    B_shard = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
    C = torch.zeros(32, 32, device=ctx.device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="No SymmetricHeap is attached"):
        xtile.ops.build_gemm_allgather_plan(A, B_shard, C, ctx=ctx)


def test_build_gemm_allgather_plan_requires_heap_backed_output(
    skip_no_multigpu,
    device_info,
) -> None:
    """Only the allgather output path is heap-backed; a plain device C must be rejected."""
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
        A = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
        B_shard = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
        C = torch.zeros(32, 32, device=ctx.device, dtype=torch.float16)

        with pytest.raises(ValueError, match="C must reside in the attached symmetric heap"):
            xtile.ops.build_gemm_allgather_plan(A, B_shard, C, ctx=ctx)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_gemm_allgather_plan_exposes_stable_metadata(
    skip_no_multigpu,
    device_info,
) -> None:
    """Plan building should resolve GEMM + allgather metadata once, up front."""
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
        B_shard = torch.randn(K, N, device=ctx.device, dtype=torch.float16)
        C = ctx.zeros(M, N, dtype=torch.float16)

        plan = xtile.ops.build_gemm_allgather_plan(A, B_shard, C, ctx=ctx)
        payload = plan.to_dict()

        assert plan.contract.full_N == N
        assert plan.contract.shard_cols == N
        assert payload["op"] == "gemm_allgather"
        assert payload["contract"]["full_N"] == N
        assert payload["allgather_plan"]["block_size"] == M * N
    finally:
        for heap in heaps:
            heap.cleanup()


@pytest.mark.multigpu
def test_gemm_allgather_multigpu(
    skip_no_multigpu,
    device_info,
) -> None:
    """2-GPU single-process correctness should assemble full GEMM output shards."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    M, N, K = 64, 128, 32
    shard_cols = N // world_size
    generator = torch.Generator().manual_seed(0)
    A_host = torch.randn(M, K, generator=generator, dtype=torch.float32)
    B_shards_host = [
        torch.randn(K, shard_cols, generator=generator, dtype=torch.float32)
        for _ in range(world_size)
    ]
    expected_full = torch.matmul(A_host, torch.cat(B_shards_host, dim=1))

    heaps = SymmetricHeap.create_all(size=128 * 1024 * 1024, world_size=world_size)
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

        outputs = []
        for rank, ctx in enumerate(contexts):
            torch.cuda.set_device(rank)
            A = A_host.to(device=ctx.device, dtype=torch.float16)
            B_shard = B_shards_host[rank].to(device=ctx.device, dtype=torch.float16)
            C = ctx.zeros(M, N, dtype=torch.float16)

            xtile.ops.gemm_allgather(A, B_shard, C, ctx=ctx)
            outputs.append(C)

        for rank in range(world_size):
            torch.cuda.synchronize(rank)

        expected = expected_full.to(dtype=torch.float16)
        for output in outputs:
            assert torch.allclose(output.cpu(), expected, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_gemm_reducescatter_high_level_api(skip_no_multigpu, device_info) -> None:
    """Single-GPU smoke test for the high-level GEMM + reduce-scatter contract."""
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
        C = ctx.zeros(M, N, dtype=torch.float16)

        xtile.ops.gemm_reducescatter(A, B, C, ctx=ctx)
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B.float()).half()
        assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_gemm_reducescatter_plan_requires_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """The current GEMM + reduce-scatter runtime should fail early without a heap."""
    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )
    A = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
    B = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
    C = torch.zeros(32, 32, device=ctx.device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="No SymmetricHeap is attached"):
        xtile.ops.build_gemm_reducescatter_plan(A, B, C, ctx=ctx)


def test_build_gemm_reducescatter_plan_requires_heap_backed_output(
    skip_no_multigpu,
    device_info,
) -> None:
    """Only the reduce_scatter output path is heap-backed; a plain device C must be rejected."""
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
        A = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
        B = torch.randn(32, 32, device=ctx.device, dtype=torch.float16)
        C = torch.zeros(32, 32, device=ctx.device, dtype=torch.float16)

        with pytest.raises(ValueError, match="C must reside in the attached symmetric heap"):
            xtile.ops.build_gemm_reducescatter_plan(A, B, C, ctx=ctx)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_gemm_reducescatter_plan_exposes_stable_metadata(
    skip_no_multigpu,
    device_info,
) -> None:
    """Plan building should resolve GEMM + reduce-scatter metadata once, up front."""
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
        C = ctx.zeros(M, N, dtype=torch.float16)

        plan = xtile.ops.build_gemm_reducescatter_plan(A, B, C, ctx=ctx)
        payload = plan.to_dict()

        assert plan.contract.full_N == N
        assert plan.contract.output_cols == N
        assert plan.implementation == "reference"
        assert payload["op"] == "gemm_reducescatter"
        assert payload["implementation"] == "reference"
        assert payload["contract"]["full_N"] == N
        assert payload["reduce_scatter_plan"]["implementation"] == "reference"
    finally:
        for heap in heaps:
            heap.cleanup()


@pytest.mark.multigpu
def test_gemm_reducescatter_multigpu(
    skip_no_multigpu,
    device_info,
) -> None:
    """2-GPU single-process correctness should match summed full GEMM shards."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    M, N, K = 64, 128, 32
    shard_cols = N // world_size
    generator = torch.Generator().manual_seed(0)
    A_host = [
        torch.randn(M, K, generator=generator, dtype=torch.float32)
        for _ in range(world_size)
    ]
    B_host = [
        torch.randn(K, N, generator=generator, dtype=torch.float32)
        for _ in range(world_size)
    ]
    expected_full = torch.zeros(M, N, dtype=torch.float32)
    for A_rank, B_rank in zip(A_host, B_host):
        expected_full += torch.matmul(A_rank, B_rank)

    heaps = SymmetricHeap.create_all(size=128 * 1024 * 1024, world_size=world_size)
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

        outputs = []
        for rank, ctx in enumerate(contexts):
            torch.cuda.set_device(rank)
            A = A_host[rank].to(device=ctx.device, dtype=torch.float16)
            B = B_host[rank].to(device=ctx.device, dtype=torch.float16)
            C = ctx.zeros(M, shard_cols, dtype=torch.float16)

            xtile.ops.gemm_reducescatter(A, B, C, ctx=ctx)
            outputs.append(C)

        for rank in range(world_size):
            torch.cuda.synchronize(rank)

        for rank, output in enumerate(outputs):
            col_offset = rank * shard_cols
            expected = expected_full[:, col_offset:col_offset + shard_cols].to(dtype=torch.float16)
            assert torch.allclose(output.cpu(), expected, rtol=1e-2, atol=1e-1)
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


def test_build_allreduce_plan_requires_heap(
    skip_no_gpu,
    device_info,
) -> None:
    """The current allreduce runtime should fail early without a heap."""
    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )
    tensor = torch.zeros(32, device=ctx.device, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="No SymmetricHeap is attached"):
        xtile.ops.build_allreduce_plan(tensor, ctx=ctx)


def test_build_allreduce_plan_exposes_stable_metadata(
    skip_no_multigpu,
    device_info,
) -> None:
    """AllReducePlan should capture the validated in-place collective contract."""
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
        tensor = ctx.zeros(32, dtype=torch.float32)

        plan = xtile.ops.build_allreduce_plan(tensor, ctx=ctx)
        payload = plan.to_dict()

        assert plan.block_size == 32
        assert plan.implementation == "noop"
        assert plan.protocol == "local_identity"
        assert plan.pipeline_slots == 0
        assert payload["op"] == "allreduce"
        assert payload["block_size"] == 32
        assert payload["tensor_shape"] == [32]
        assert payload["reduction"] == "sum"
        assert payload["workspace_bytes"] == 0
    finally:
        for heap in heaps:
            heap.cleanup()


@pytest.mark.multigpu
def test_allreduce_high_level_api_multigpu(skip_no_multigpu, device_info) -> None:
    """The high-level allreduce API should reduce in-place across local ranks."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    total_elements = 4097
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

        tensors = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            tensor = heaps[rank].allocate_tensor((total_elements,), torch.float32)
            tensor.fill_(float(rank + 1))
            tensors.append(tensor)

        plan = xtile.ops.build_allreduce_plan(tensors[0], ctx=contexts[0])
        assert plan.implementation == "device_staged_pipeline"
        assert plan.protocol == "slot_epoch_pipeline"
        assert plan.block_size == total_elements
        assert plan.chunk_elems > 0
        assert plan.num_chunks >= 2
        assert plan.pipeline_slots >= 2
        assert plan.grid_size == plan.pipeline_slots
        assert plan.workspace_bytes > 0

        for rank in range(world_size):
            xtile.ops.allreduce(tensors[rank], ctx=contexts[rank])

        for rank in range(world_size):
            torch.cuda.synchronize(rank)
            assert torch.allclose(
                tensors[rank],
                torch.full_like(tensors[rank], 3.0),
                atol=1e-4,
            )
    finally:
        for heap in heaps:
            heap.cleanup()


def test_context_as_symmetric_materializes_external_tensor(
    skip_no_multigpu,
    device_info,
) -> None:
    """Context helper should expose heap-backed external import."""
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
        external = torch.arange(32, device=ctx.device, dtype=torch.float32)

        imported = ctx.as_symmetric(external)

        assert ctx.is_symmetric(imported)
        assert imported.data_ptr() != external.data_ptr()
        assert torch.allclose(imported, external)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_allgather_plan_rejects_unvalidated_multiprocess_transport(
    skip_no_gpu,
    device_info,
) -> None:
    """High-level allgather should fail fast before entering an unsafe transport."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "pytorch_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=2,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    src = torch.zeros(8, device=ctx.device, dtype=torch.float32)
    output = torch.zeros(16, device=ctx.device, dtype=torch.float32)

    with pytest.raises(ValueError, match="remote dereference"):
        xtile.ops.build_allgather_plan(src, output, ctx=ctx)


def test_build_allreduce_plan_rejects_unvalidated_multiprocess_transport(
    skip_no_gpu,
    device_info,
) -> None:
    """High-level allreduce should fail fast before entering an unsafe transport."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "pytorch_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=2,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    tensor = torch.zeros(16, device=ctx.device, dtype=torch.float32)

    with pytest.raises(ValueError, match="remote dereference"):
        xtile.ops.build_allreduce_plan(tensor, ctx=ctx)


def test_build_gemm_allscatter_plan_rejects_unvalidated_multiprocess_transport(
    skip_no_gpu,
    device_info,
) -> None:
    """High-level GEMM+allscatter should fail fast before entering an unsafe transport."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "pytorch_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=2,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    A = torch.randn(16, 16, device=ctx.device, dtype=torch.float16)
    B = torch.randn(16, 16, device=ctx.device, dtype=torch.float16)
    C = torch.zeros(16, 16, device=ctx.device, dtype=torch.float16)

    with pytest.raises(ValueError, match="remote dereference"):
        xtile.ops.build_gemm_allscatter_plan(A, B, C, ctx=ctx, pattern="bulk_sync")


def test_bulk_sync_pattern_rejects_unvalidated_multiprocess_transport(
    skip_no_gpu,
    device_info,
) -> None:
    """Expert pattern surface should also fail fast before launching Triton."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "pytorch_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=2,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    A = torch.randn(16, 16, device=ctx.device, dtype=torch.float16)
    B = torch.randn(16, 16, device=ctx.device, dtype=torch.float16)
    C = torch.zeros(16, 16, device=ctx.device, dtype=torch.float16)

    pattern = xtile.patterns.BulkSyncPattern(ctx)
    with pytest.raises(ValueError, match="remote dereference"):
        pattern.execute(
            A,
            B,
            C,
            full_N=16,
            b_layout="full",
            c_layout="full",
        )


def test_build_reduce_scatter_plan_exposes_stable_metadata(
    skip_no_gpu,
    device_info,
) -> None:
    """ReduceScatterPlan should capture the validated collective contract."""
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

        plan = xtile.ops.build_reduce_scatter_plan(src, output, ctx=ctx)
        payload = plan.to_dict()

        assert plan.block_size == 32
        assert payload["op"] == "reduce_scatter"
        assert payload["block_size"] == 32
        assert payload["input_shape"] == [32]
        assert payload["output_shape"] == [32]
        assert payload["implementation"] == "reference"
    finally:
        for heap in heaps:
            heap.cleanup()


def test_build_reduce_scatter_plan_rejects_single_process_device_override(
    skip_no_gpu,
    device_info,
) -> None:
    """High-level plan building should reject an unvalidated device override."""
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

        with pytest.raises(ValueError, match="not validated for single-process symmetric heaps"):
            xtile.ops.build_reduce_scatter_plan(
                src,
                output,
                ctx=ctx,
                implementation="device",
            )
    finally:
        for heap in heaps:
            heap.cleanup()


def test_resolve_reduce_scatter_implementation_rejects_multiprocess_by_default(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """Multiprocess device collectives must stay behind an explicit feature gate."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "ctypes_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=4,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    monkeypatch.delenv(
        "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES",
        raising=False,
    )

    with pytest.raises(ValueError, match="disabled by default"):
        xtile.ops._resolve_reduce_scatter_implementation(  # type: ignore[attr-defined]
            ctx,
            implementation="auto",
        )

    with pytest.raises(ValueError, match="disabled by default"):
        xtile.ops._resolve_reduce_scatter_implementation(  # type: ignore[attr-defined]
            ctx,
            implementation="device",
        )


def test_resolve_reduce_scatter_implementation_accepts_validated_surface_without_opt_in(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """The validated 2-GPU ctypes_ipc surface should resolve without an env gate."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "ctypes_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=2,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    monkeypatch.delenv(
        "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES",
        raising=False,
    )

    assert (
        xtile.ops._resolve_reduce_scatter_implementation(  # type: ignore[attr-defined]
            ctx,
            implementation="auto",
        )
        == "device"
    )


def test_resolve_reduce_scatter_implementation_allows_explicit_multiprocess_opt_in(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """The explicit opt-in should still unlock broader diagnostic runs."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "ctypes_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=4,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    monkeypatch.setenv(
        "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES",
        "1",
    )

    assert (
        xtile.ops._resolve_reduce_scatter_implementation(  # type: ignore[attr-defined]
            ctx,
            implementation="auto",
        )
        == "device"
    )


def test_resolve_reduce_scatter_implementation_rejects_unvalidated_transport_even_with_opt_in(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """The gate must remain transport-aware after the matrix diagnostics."""

    class _DummyHeap:
        mode = "multiprocess"
        transport_strategy = "pytorch_ipc"

    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=2,
        force_backend=True,
    )
    ctx.heap = _DummyHeap()  # type: ignore[assignment]
    monkeypatch.setenv(
        "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES",
        "1",
    )

    with pytest.raises(ValueError, match="transport-sensitive"):
        xtile.ops._resolve_reduce_scatter_implementation(  # type: ignore[attr-defined]
            ctx,
            implementation="auto",
        )

    with pytest.raises(ValueError, match="transport-sensitive"):
        xtile.ops._resolve_reduce_scatter_implementation(  # type: ignore[attr-defined]
            ctx,
            implementation="device",
        )


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


@pytest.mark.multigpu
def test_reduce_scatter_high_level_api_multigpu(skip_no_multigpu, device_info) -> None:
    """The high-level reduce_scatter API should produce the reduced chunk."""
    from xtile.memory.symmetric_heap import SymmetricHeap

    world_size = 2
    block_size = 128
    total_elements = block_size * world_size
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
            src_rank = heaps[rank].allocate_tensor((total_elements,), torch.float32)
            out_rank = heaps[rank].allocate_tensor((block_size,), torch.float32)
            for chunk in range(world_size):
                start = chunk * block_size
                src_rank[start:start + block_size].fill_(float(rank * 2 + chunk + 1))
            out_rank.zero_()
            src.append(src_rank)
            output.append(out_rank)

        for rank in range(world_size):
            xtile.ops.reduce_scatter(src[rank], output[rank], ctx=contexts[rank])

        for rank in range(world_size):
            torch.cuda.synchronize(rank)
            expected = float((0 * 2 + rank + 1) + (1 * 2 + rank + 1))
            assert torch.allclose(
                output[rank],
                torch.full_like(output[rank], expected),
                atol=1e-4,
            ), f"Rank {rank}: expected {expected}, got {output[rank][0].item()}"
    finally:
        for heap in heaps:
            heap.cleanup()
