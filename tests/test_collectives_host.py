# SPDX-License-Identifier: Apache-2.0
"""Tests for host-side collective launchers."""

from __future__ import annotations

import pytest
import torch


def test_reduce_scatter_launcher_validates_input_shape(skip_no_gpu, device_info) -> None:
    """The host launcher should reject mismatched input/output sizes."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import reduce_scatter

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        heap = heaps[0]
        src = heap.allocate_tensor((16,), torch.float32)
        dst = heap.allocate_tensor((8,), torch.float32)

        with pytest.raises(ValueError, match="world_size \\* output.numel"):
            reduce_scatter(src, dst, heap)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_allgather_launcher_rejects_unvalidated_multiprocess_transport(
    skip_no_gpu,
) -> None:
    """Host allgather should fail fast on transports that are not device-safe."""
    from tncc.primitives import allgather

    class _DummyHeap:
        mode = "multiprocess"
        world_size = 2
        transport_strategy = "pytorch_ipc"

    src = torch.zeros(8, device="cuda:0", dtype=torch.float32)
    dst = torch.zeros(16, device="cuda:0", dtype=torch.float32)

    with pytest.raises(ValueError, match="remote dereference"):
        allgather(src, dst, _DummyHeap())


@pytest.mark.multigpu
def test_allreduce_launcher_supports_non_divisible_tensor_sizes(
    skip_no_multigpu,
    device_info,
) -> None:
    """The public allreduce path should handle odd-sized contiguous tensors."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import allreduce

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=2)
    try:
        tensors = []
        for rank in range(2):
            torch.cuda.set_device(rank)
            tensor = heaps[rank].allocate_tensor((15,), torch.float32)
            tensor.fill_(float(rank + 1))
            tensors.append(tensor)

        for rank in range(2):
            torch.cuda.set_device(rank)
            allreduce(tensors[rank], heaps[rank])

        for rank in range(2):
            torch.cuda.synchronize(rank)
            assert torch.allclose(
                tensors[rank],
                torch.full_like(tensors[rank], 3.0),
                atol=1e-4,
            )
    finally:
        for heap in heaps:
            heap.cleanup()


def test_allreduce_launcher_rejects_unvalidated_multiprocess_transport(
    skip_no_gpu,
) -> None:
    """Host allreduce should fail fast on transports outside the validated surface."""
    from tncc.primitives import allreduce

    class _DummyHeap:
        mode = "multiprocess"
        world_size = 2
        transport_strategy = "pytorch_ipc"
        rank = 0

        def owns_tensor(self, tensor):
            return True

    tensor = torch.zeros(16, device="cuda:0", dtype=torch.float32)

    with pytest.raises(ValueError, match="remote dereference"):
        allreduce(tensor, _DummyHeap())


@pytest.mark.multigpu
def test_reduce_scatter_launcher_multigpu_value_check(skip_no_multigpu, device_info) -> None:
    """The single-process reference path should produce the reduced chunk."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import reduce_scatter

    world_size = 2
    block_size = 128
    total_elements = world_size * block_size
    heaps = SymmetricHeap.create_all(
        size=64 * 1024 * 1024,
        world_size=world_size,
        backend=device_info.backend,
    )
    try:
        src = []
        dst = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            src_rank = heaps[rank].allocate_tensor((total_elements,), torch.float32)
            dst_rank = heaps[rank].allocate_tensor((block_size,), torch.float32)
            for chunk in range(world_size):
                start = chunk * block_size
                src_rank[start:start + block_size].fill_(float(rank * 2 + chunk + 1))
            dst_rank.zero_()
            src.append(src_rank)
            dst.append(dst_rank)

        for rank in range(world_size):
            torch.cuda.synchronize(rank)

        for rank in range(world_size):
            torch.cuda.set_device(rank)
            reduce_scatter(src[rank], dst[rank], heaps[rank])

        for rank in range(world_size):
            torch.cuda.synchronize(rank)
            expected = float((0 * 2 + rank + 1) + (1 * 2 + rank + 1))
            assert torch.allclose(
                dst[rank],
                torch.full_like(dst[rank], expected),
                atol=1e-4,
            ), f"Rank {rank}: expected {expected}, got {dst[rank][0].item()}"
    finally:
        for heap in heaps:
            heap.cleanup()


@pytest.mark.multigpu
def test_reduce_scatter_launcher_rejects_single_process_device_override(
    skip_no_multigpu,
    device_info,
) -> None:
    """The unvalidated single-process device path should be explicitly rejected."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import reduce_scatter

    world_size = 2
    block_size = 32
    heaps = SymmetricHeap.create_all(
        size=64 * 1024 * 1024,
        world_size=world_size,
        backend=device_info.backend,
    )
    try:
        torch.cuda.set_device(0)
        src = heaps[0].allocate_tensor((world_size * block_size,), torch.float32)
        dst = heaps[0].allocate_tensor((block_size,), torch.float32)
        src.zero_()
        dst.zero_()

        with pytest.raises(ValueError, match="not validated for single-process symmetric heaps"):
            reduce_scatter(src, dst, heaps[0], implementation="device")
    finally:
        for heap in heaps:
            heap.cleanup()
