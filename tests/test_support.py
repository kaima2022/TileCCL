"""Tests for the runtime support matrix."""

from __future__ import annotations

import xtile


def test_support_matrix_without_heap(skip_no_gpu, device_info) -> None:
    """The support matrix should report heap-backed ops conservatively."""
    ctx = xtile.init(
        backend=device_info.backend,
        rank=0,
        world_size=1,
        force_backend=True,
    )

    matrix = xtile.describe_runtime_support(ctx)
    payload = matrix.to_dict()

    assert matrix.has_heap is False
    assert matrix.ops["gemm_allscatter"].state == "partial"
    assert matrix.ops["gemm_allgather"].state == "partial"
    assert matrix.ops["allgather"].state == "partial"
    assert matrix.ops["reduce_scatter"].state == "partial"
    assert matrix.ops["gemm_reducescatter"].state == "partial"
    assert matrix.contracts["gemm_allscatter.full/full"].state == "supported"
    assert matrix.contracts["gemm_allscatter.full/shard"].state == "supported"
    assert matrix.contracts["gemm_allscatter.shard/full"].state == "unsupported"
    assert matrix.contracts["gemm_allgather.shard/full"].state == "partial"
    assert matrix.contracts["gemm_reducescatter.full/shard"].state == "partial"
    assert matrix.execution_paths["reduce_scatter.reference"].state == "partial"
    assert matrix.execution_paths["reduce_scatter.device"].state == "partial"
    assert payload["context"]["has_heap"] is False
    assert matrix.memory["symmetric_heap.device_remote_access"].state == "partial"
    assert matrix.memory["symmetric_heap_allocator_first_import_map"].state == "unsupported"
    assert matrix.memory["symmetric_heap.external_import"].state == "partial"


def test_support_matrix_with_heap_matches_context_method(
    skip_no_gpu,
    device_info,
) -> None:
    """The top-level helper and context method should agree."""
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

        direct = xtile.describe_runtime_support(ctx)
        via_ctx = ctx.support_matrix()

        assert direct.to_dict() == via_ctx.to_dict()
        assert direct.has_heap is True
        assert direct.heap_mode == heaps[0].mode
        assert direct.transport_strategy == heaps[0].transport_strategy
        assert direct.ops["gemm_allgather"].state == "supported"
        assert direct.ops["allgather"].state == "supported"
        assert direct.ops["reduce_scatter"].state == "supported"
        assert direct.ops["gemm_reducescatter"].state == "supported"
        assert direct.contracts["gemm_allgather.shard/full"].state == "supported"
        assert direct.contracts["gemm_reducescatter.full/shard"].state == "supported"
        assert direct.execution_paths["reduce_scatter.reference"].state == "supported"
        assert direct.execution_paths["reduce_scatter.device"].state == "unsupported"
        assert direct.collectives["collectives.allgather_launcher"].state == "supported"
        assert direct.collectives["collectives.reduce_scatter_launcher"].state == "supported"
        assert direct.memory["symmetric_heap.device_remote_access"].state == "supported"
        assert direct.memory["symmetric_heap_allocator_first_import_map"].state == "partial"
        assert direct.memory["symmetric_heap.external_import"].state == "supported"
    finally:
        for heap in heaps:
            heap.cleanup()


def test_support_matrix_multigpu_reports_peer_access(
    skip_no_multigpu,
    device_info,
) -> None:
    """A local multi-GPU heap should surface its active transport strategy."""
    contexts = xtile.init_local(
        world_size=2,
        heap_size=64 * 1024 * 1024,
        backend=device_info.backend,
    )
    try:
        matrix = contexts[0].support_matrix()
        assert matrix.has_heap is True
        assert matrix.heap_mode == "single_process"
        assert matrix.transport_strategy == "peer_access"
        assert matrix.ops["gemm_allgather"].state == "supported"
        assert matrix.ops["allgather"].state == "supported"
        assert matrix.ops["reduce_scatter"].state == "supported"
        assert matrix.ops["gemm_reducescatter"].state == "supported"
        assert matrix.execution_paths["reduce_scatter.reference"].state == "supported"
        assert matrix.execution_paths["reduce_scatter.device"].state == "unsupported"
        assert matrix.memory["symmetric_heap.device_remote_access"].state == "supported"
        assert matrix.memory["symmetric_heap_allocator_first_import_map"].state == "partial"
    finally:
        for ctx in contexts:
            if ctx.heap is not None:
                ctx.heap.cleanup()


def test_support_matrix_multiprocess_defaults_to_unsupported(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """Unsafe multiprocess device collectives should not be public-supported by default."""

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
    monkeypatch.delenv(
        "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES",
        raising=False,
    )

    matrix = xtile.describe_runtime_support(ctx)
    assert matrix.ops["reduce_scatter"].state == "unsupported"
    assert matrix.ops["allgather"].state == "unsupported"
    assert matrix.ops["gemm_allscatter"].state == "unsupported"
    assert matrix.ops["gemm_allgather"].state == "unsupported"
    assert matrix.ops["gemm_reducescatter"].state == "unsupported"
    assert matrix.execution_paths["reduce_scatter.reference"].state == "unsupported"
    assert matrix.execution_paths["reduce_scatter.device"].state == "unsupported"
    assert matrix.memory["symmetric_heap.device_remote_access"].state == "unsupported"


def test_support_matrix_multiprocess_opt_in_remains_partial(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """Even with opt-in, multiprocess device collectives remain experimental."""

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
    monkeypatch.setenv(
        "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES",
        "1",
    )

    matrix = xtile.describe_runtime_support(ctx)
    assert matrix.ops["reduce_scatter"].state == "partial"
    assert matrix.ops["allgather"].state == "partial"
    assert matrix.ops["gemm_allscatter"].state == "partial"
    assert matrix.ops["gemm_allgather"].state == "partial"
    assert matrix.ops["gemm_reducescatter"].state == "partial"
    assert matrix.execution_paths["reduce_scatter.reference"].state == "unsupported"
    assert matrix.execution_paths["reduce_scatter.device"].state == "partial"
    assert matrix.memory["symmetric_heap.device_remote_access"].state == "supported"


def test_support_matrix_multiprocess_opt_in_rejects_unvalidated_transport(
    skip_no_gpu,
    device_info,
    monkeypatch,
) -> None:
    """Opt-in should still stay unsupported for transports that fail real diagnostics."""

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

    matrix = xtile.describe_runtime_support(ctx)
    assert matrix.ops["reduce_scatter"].state == "unsupported"
    assert matrix.ops["allgather"].state == "unsupported"
    assert matrix.ops["gemm_allscatter"].state == "unsupported"
    assert matrix.ops["gemm_allgather"].state == "unsupported"
    assert matrix.ops["gemm_reducescatter"].state == "unsupported"
    assert matrix.execution_paths["reduce_scatter.reference"].state == "unsupported"
    assert matrix.execution_paths["reduce_scatter.device"].state == "unsupported"
    assert matrix.memory["symmetric_heap.device_remote_access"].state == "unsupported"
