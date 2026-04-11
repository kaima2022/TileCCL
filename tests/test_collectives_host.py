# SPDX-License-Identifier: Apache-2.0
"""Tests for host-side collective launchers."""

from __future__ import annotations

import pytest
import torch


def _make_dummy_heap(
    *,
    world_size: int,
    mode: str = "single_process",
    transport_strategy: str = "ctypes_ipc",
):
    """Build a minimal heap-like object for host precondition tests."""

    class _DummyHeap:
        rank = 0

        def __init__(self):
            self.mode = mode
            self.world_size = world_size
            self.transport_strategy = transport_strategy

        def owns_tensor(self, _tensor):
            return True

    return _DummyHeap()


@pytest.mark.parametrize(
    ("message_bytes", "expected_regime"),
    [
        (4 * 1024, "legacy"),
        (16 * 1024, "legacy"),
        (64 * 1024, "staged"),
    ],
)
def test_non_allreduce_execution_resolver_threshold_split_and_limits(
    message_bytes: int,
    expected_regime: str,
    monkeypatch,
) -> None:
    """Resolver should apply stable threshold and cap policies for non-allreduce."""
    import tncc.primitives.collectives as collectives

    class _Props:
        multi_processor_count = 6

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: _Props(),
    )

    element_size = torch.tensor([], dtype=torch.float32).element_size()
    input_numel = message_bytes // element_size
    spec = collectives._resolve_collective_execution(
        "allgather",
        input_numel=input_numel,
        world_size=2,
        element_size=element_size,
        device=torch.device("cuda:0"),
    )

    assert spec.path == expected_regime
    assert spec.message_regime == expected_regime
    assert spec.chunk_elems <= 4096
    assert spec.num_chunks == (input_numel + spec.chunk_elems - 1) // spec.chunk_elems
    if spec.num_chunks > 0:
        assert spec.pipeline_slots <= min(spec.num_chunks, _Props.multi_processor_count, 8)
        assert spec.pipeline_slots >= 1
    else:
        assert spec.pipeline_slots == 0


def test_non_allreduce_execution_resolver_pipeline_slot_cap(monkeypatch) -> None:
    """Resolver should cap pipeline slots to 8 even on larger SM counts."""
    import tncc.primitives.collectives as collectives

    class _Props:
        multi_processor_count = 80

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: _Props(),
    )

    element_size = torch.tensor([], dtype=torch.float32).element_size()
    input_numel = 64 * 4096
    spec = collectives._resolve_collective_execution(
        "allgather",
        input_numel=input_numel,
        world_size=2,
        element_size=element_size,
        device=torch.device("cuda:0"),
    )

    assert spec.path == "staged"
    assert spec.chunk_elems == 4096
    assert spec.num_chunks == 64
    assert spec.pipeline_slots == 8
    assert spec.pipeline_slots <= min(spec.num_chunks, _Props.multi_processor_count, 8)


def test_reduce_scatter_launcher_validates_input_shape(skip_no_gpu, device_info) -> None:
    """The host launcher should reject mismatched input/output sizes."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import reduce_scatter

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        heap = heaps[0]
        src = heap.allocate_tensor((16,), torch.float32)
        dst = heap.allocate_tensor((8,), torch.float32)

        with pytest.raises(
            ValueError,
            match="reduce_scatter tensor\\.numel must equal output\\.numel \\* world_size",
        ):
            reduce_scatter(src, dst, heap)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_allgather_launcher_validates_input_output_shape(skip_no_gpu, device_info) -> None:
    """allgather should reject shape contracts that do not match world size."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import allgather

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        heap = heaps[0]
        src = heap.allocate_tensor((8,), torch.float32)
        output = heap.allocate_tensor((7,), torch.float32)

        with pytest.raises(
            ValueError,
            match="allgather output\\.numel must equal tensor\\.numel \\* world_size",
        ):
            allgather(src, output, heap)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_scatter_launcher_validates_input_output_shape(skip_no_gpu, device_info) -> None:
    """scatter should reject shape contracts that do not match world size."""
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import scatter

    heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
    try:
        heap = heaps[0]
        src = heap.allocate_tensor((7,), torch.float32)
        output = heap.allocate_tensor((8,), torch.float32)

        with pytest.raises(
            ValueError,
            match="scatter tensor\\.numel must equal output\\.numel \\* world_size",
        ):
            scatter(src, output, heap)
    finally:
        for heap in heaps:
            heap.cleanup()


def test_broadcast_launcher_validates_root_range(skip_no_gpu) -> None:
    """broadcast should reject roots outside ``[0, world_size)``."""
    from tncc.primitives import broadcast

    tensor = torch.zeros(8, device="cuda:0", dtype=torch.float32)

    with pytest.raises(ValueError, match=r"root=2 out of range \[0, 2\)"):
        broadcast(tensor, _make_dummy_heap(world_size=2), root=2)


def test_scatter_launcher_validates_root_range(skip_no_gpu) -> None:
    """scatter should reject roots outside ``[0, world_size)``."""
    from tncc.primitives import scatter

    src = torch.zeros(16, device="cuda:0", dtype=torch.float32)
    output = torch.zeros(8, device="cuda:0", dtype=torch.float32)

    with pytest.raises(ValueError, match=r"root=2 out of range \[0, 2\)"):
        scatter(src, output, _make_dummy_heap(world_size=2), root=2)


@pytest.mark.parametrize(
    ("collective", "name", "kwargs", "expected"),
    [
        (
            "allgather",
            "tensor",
            {},
            "allgather currently requires tensor to be contiguous",
        ),
        (
            "allgather",
            "output",
            {},
            "allgather currently requires output to be contiguous",
        ),
        (
            "broadcast",
            "tensor",
            {"root": 0},
            "broadcast currently requires tensor to be contiguous",
        ),
        (
            "scatter",
            "tensor",
            {"root": 0},
            "scatter currently requires tensor to be contiguous",
        ),
        (
            "scatter",
            "output",
            {"root": 0},
            "scatter currently requires output to be contiguous",
        ),
        (
            "reduce_scatter",
            "tensor",
            {},
            "reduce_scatter currently requires tensor to be contiguous",
        ),
        (
            "reduce_scatter",
            "output",
            {},
            "reduce_scatter currently requires output to be contiguous",
        ),
    ],
)
def test_collective_launchers_reject_non_contiguous_inputs(
    skip_no_gpu,
    device_info,
    collective: str,
    name: str,
    kwargs: dict[str, int],
    expected: str,
) -> None:
    """Host collective launchers should reject non-contiguous tensors consistently."""
    from tncc.primitives import allgather, broadcast, reduce_scatter, scatter

    heap = _make_dummy_heap(world_size=1)
    tensor = torch.zeros((8,), device="cuda:0", dtype=torch.float32)
    output = torch.zeros((8,), device="cuda:0", dtype=torch.float32)

    if name == "tensor":
        tensor = tensor.view(2, 4).transpose(0, 1)
    else:
        output = output.view(2, 4).transpose(0, 1)

    launchers = {
        "allgather": lambda: allgather(tensor, output, heap),
        "broadcast": lambda: broadcast(tensor, heap, **kwargs),
        "scatter": lambda: scatter(tensor, output, heap, **kwargs),
        "reduce_scatter": lambda: reduce_scatter(tensor, output, heap),
    }

    with pytest.raises(ValueError, match=expected):
        launchers[collective]()


def _launch_multiprocess_collective_for_failure_matrix(
    *,
    collective: str,
    world_size: int,
    transport_strategy: str,
) -> None:
    """Launch one collective on a dummy multiprocess heap for gate failures."""
    from tncc.primitives import allgather, broadcast, reduce_scatter, scatter

    heap = _make_dummy_heap(
        mode="multiprocess",
        world_size=world_size,
        transport_strategy=transport_strategy,
    )
    block_size = 8

    if collective == "allgather":
        src = torch.zeros(block_size, device="cuda:0", dtype=torch.float32)
        dst = torch.zeros(block_size * world_size, device="cuda:0", dtype=torch.float32)
        allgather(src, dst, heap)
        return
    if collective == "broadcast":
        tensor = torch.zeros(block_size, device="cuda:0", dtype=torch.float32)
        broadcast(tensor, heap, root=0)
        return
    if collective == "scatter":
        src = torch.zeros(block_size * world_size, device="cuda:0", dtype=torch.float32)
        output = torch.zeros(block_size, device="cuda:0", dtype=torch.float32)
        scatter(src, output, heap, root=0)
        return
    if collective == "reduce_scatter":
        src = torch.zeros(block_size * world_size, device="cuda:0", dtype=torch.float32)
        output = torch.zeros(block_size, device="cuda:0", dtype=torch.float32)
        reduce_scatter(src, output, heap)
        return
    raise AssertionError(f"Unknown collective for failure matrix: {collective}")


@pytest.mark.parametrize("collective", ["allgather", "broadcast", "scatter", "reduce_scatter"])
def test_collective_launchers_reject_unvalidated_transport_with_surface_wording(
    skip_no_gpu,
    collective: str,
) -> None:
    """All host collectives should explain the ws2+ctypes validated transport surface."""
    with pytest.raises(ValueError) as exc_info:
        _launch_multiprocess_collective_for_failure_matrix(
            collective=collective,
            world_size=2,
            transport_strategy="pytorch_ipc",
        )

    message = str(exc_info.value)
    assert "transport_strategy='ctypes_ipc'" in message
    assert "Current transport_strategy='pytorch_ipc'" in message
    assert "Current world_size=2" in message
    assert "other transports are not yet safe" in message
    if collective == "reduce_scatter":
        assert "transport-sensitive" in message
    else:
        assert "remote dereference" in message


@pytest.mark.parametrize("collective", ["allgather", "broadcast", "scatter", "reduce_scatter"])
def test_collective_launchers_reject_world_size_3_outside_validated_surface(
    skip_no_gpu,
    collective: str,
) -> None:
    """All host collectives should reject ws=3 as outside the validated public surface."""
    with pytest.raises(ValueError) as exc_info:
        _launch_multiprocess_collective_for_failure_matrix(
            collective=collective,
            world_size=3,
            transport_strategy="ctypes_ipc",
        )

    message = str(exc_info.value)
    assert "Current transport_strategy='ctypes_ipc'" in message
    assert "Current world_size=3" in message
    if collective == "reduce_scatter":
        assert "outside the validated public surface" in message
        assert "world_size=2 with transport_strategy='ctypes_ipc'" in message
    else:
        assert "remote dereference" in message
        assert "2-GPU public surface" in message
        assert "broader world-size usage is not yet public-supported" in message


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
                src_rank[start : start + block_size].fill_(float(rank * 2 + chunk + 1))
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
