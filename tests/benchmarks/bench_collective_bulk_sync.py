#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Communication-only collectives: TNCC device path vs bulk_sync baseline.

This benchmark is intentionally different from the NCCL comparison:

- execution mode: single-process ``peer_access``
- optimized path: TNCC device collective kernels
- baseline: bulk-synchronous host orchestration that composes lower-level
  point-to-point primitives with explicit phase boundaries

The goal is to answer a narrower question: how much speedup do TNCC's
collective-specific kernels provide over a naive internal bulk_sync
composition built from simpler communication steps.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

import torch
import triton
import triton.language as tl

import tncc
from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.memory.translation import translate_ptr
from tncc.primitives.collectives import (
    _allgather_kernel,
    _broadcast_kernel,
    _reduce_scatter_kernel,
    _scatter_kernel,
)
from tncc.utils.benchmark_results import (
    benchmark_environment_health,
    canonical_benchmark_run,
    default_collective_bulk_sync_benchmark_path,
    emit_benchmark_environment_warnings,
    runtime_metadata_snapshot,
    runtime_support_snapshot,
    write_json,
)


@triton.jit
def _mirror_put_kernel(
    src_ptr,
    dst_mirror_ptr,
    heap_bases_ptr,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy local data into a peer's symmetric mirror allocation.

    ``dst_mirror_ptr`` must point at the local rank's own symmetric
    allocation that mirrors the destination buffer layout. The kernel then
    translates that mirror pointer into ``remote_rank``'s address space.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    data = tl.load(src_ptr + offsets)
    remote_ptr = translate_ptr(
        dst_mirror_ptr + offsets,
        caller_rank,
        remote_rank,
        heap_bases_ptr,
    )
    tl.store(remote_ptr, data)


_COLLECTIVES = (
    "allreduce",
    "allgather",
    "scatter",
    "reduce_scatter",
    "broadcast",
)
_DEFAULT_WARMUP = 5
_DEFAULT_ITERS = 20
_SMALL_MESSAGE_BATCH_THRESHOLD_BYTES = 16 * 1024
_MEDIUM_MESSAGE_BATCH_THRESHOLD_BYTES = 64 * 1024
_SMALL_MESSAGE_BATCH_REPEATS = 16
_MEDIUM_MESSAGE_BATCH_REPEATS = 8
_LARGE_MESSAGE_BATCH_REPEATS = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=["float32"],
        default="float32",
        help="Element dtype. Current benchmark is pinned to float32 for stable small-message comparisons.",
    )
    parser.add_argument(
        "--message-sizes",
        type=str,
        default="4096,16384,65536",
        help="Comma-separated rank-local message sizes in bytes.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=_DEFAULT_WARMUP,
        help="Warmup iterations per case.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=_DEFAULT_ITERS,
        help="Timed iterations per case.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Requested world size. Current validated configuration is 2 GPUs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(default_collective_bulk_sync_benchmark_path()),
        help="Structured JSON output path.",
    )
    return parser.parse_args()


def _parse_message_sizes(raw: str, *, element_size: int, world_size: int) -> list[int]:
    sizes: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        size_bytes = int(item)
        if size_bytes <= 0:
            raise ValueError("message sizes must be positive")
        if size_bytes % element_size != 0:
            raise ValueError(
                f"message size {size_bytes} is not divisible by dtype size {element_size}"
            )
        if size_bytes % world_size != 0:
            raise ValueError(
                f"message size {size_bytes} must be divisible by world_size {world_size}"
            )
        sizes.append(size_bytes)
    if not sizes:
        raise ValueError("at least one message size is required")
    return sorted(set(sizes))


def _sync_all(world_size: int) -> None:
    for rank in range(world_size):
        torch.cuda.synchronize(rank)


def _timed_batch_repeats_for_size(size_bytes: int) -> int:
    """Return how many logical collectives to batch into one timed sample."""
    if size_bytes <= _SMALL_MESSAGE_BATCH_THRESHOLD_BYTES:
        return _SMALL_MESSAGE_BATCH_REPEATS
    if size_bytes <= _MEDIUM_MESSAGE_BATCH_THRESHOLD_BYTES:
        return _MEDIUM_MESSAGE_BATCH_REPEATS
    return _LARGE_MESSAGE_BATCH_REPEATS


def _reset_heaps(heaps: list[SymmetricHeap]) -> None:
    for heap in heaps:
        heap._bump_offset = 0
        heap._alloc_records.clear()


def _alloc_symmetric(
    heaps: list[SymmetricHeap],
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for rank, heap in enumerate(heaps):
        torch.cuda.set_device(rank)
        tensors.append(heap.allocate_tensor(shape, dtype))
    if len(tensors) > 1:
        reference_offset = tensors[0].data_ptr() - heaps[0].local_base
        for rank in range(1, len(tensors)):
            current_offset = tensors[rank].data_ptr() - heaps[rank].local_base
            if current_offset != reference_offset:
                raise RuntimeError(
                    "bulk_sync benchmark requires symmetric allocations at identical "
                    f"heap offsets, got rank0={reference_offset} rank{rank}={current_offset}"
                )
    return tensors


def _stream_copy_(
    *,
    rank: int,
    dst: torch.Tensor,
    src: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    torch.cuda.set_device(rank)
    with torch.cuda.stream(stream):
        dst.copy_(src)


def _stream_add_(
    *,
    rank: int,
    dst: torch.Tensor,
    src: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    torch.cuda.set_device(rank)
    with torch.cuda.stream(stream):
        dst.add_(src)


def _launch_mirror_put(
    *,
    src_rank: int,
    dst_rank: int,
    src: torch.Tensor,
    dst_mirror: torch.Tensor,
    heaps: list[SymmetricHeap],
    streams: list[torch.cuda.Stream],
) -> None:
    if src.numel() != dst_mirror.numel():
        raise ValueError(
            "mirror put requires source and destination mirror to have the same number "
            f"of elements, got {src.numel()} vs {dst_mirror.numel()}"
        )
    torch.cuda.set_device(src_rank)
    with torch.cuda.stream(streams[src_rank]):
        _mirror_put_kernel[(1,)](
            src,
            dst_mirror,
            heaps[src_rank].get_heap_bases(),
            src_rank,
            dst_rank,
            BLOCK_SIZE=src.numel(),
        )


def _time_end_to_end(
    prepare_fn: Callable[[], None],
    run_fn: Callable[[], None],
    *,
    world_size: int,
    warmup: int,
    iters: int,
    operations_per_sample: int,
) -> dict[str, float]:
    for _ in range(warmup):
        prepare_fn()
        _sync_all(world_size)
        for _ in range(operations_per_sample):
            run_fn()
            _sync_all(world_size)

    times_ms: list[float] = []
    for _ in range(iters):
        prepare_fn()
        _sync_all(world_size)
        start_ns = time.perf_counter_ns()
        for _ in range(operations_per_sample):
            run_fn()
            _sync_all(world_size)
        elapsed_ms = float(time.perf_counter_ns() - start_ns) / 1_000_000.0
        times_ms.append(elapsed_ms / float(operations_per_sample))

    return {
        "mean_ms": float(sum(times_ms) / len(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def _effective_bytes(collective: str, size_bytes: int, world_size: int) -> float:
    if collective == "allreduce":
        return 2.0 * (world_size - 1) / world_size * size_bytes
    if collective in {"allgather", "reduce_scatter", "scatter", "broadcast"}:
        return (world_size - 1) * size_bytes
    raise ValueError(f"unsupported collective: {collective}")


def _bandwidth_gbps(effective_bytes: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return float(effective_bytes / (latency_ms * 1e-3) / 1e9)


def _allgather_expected(output: torch.Tensor, *, world_size: int, block_elements: int) -> torch.Tensor:
    expected = torch.empty_like(output)
    for rank in range(world_size):
        start = rank * block_elements
        expected[start:start + block_elements].fill_(float((rank + 1) * 10))
    return expected


def _reduce_scatter_expected(output: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    expected_value = float(sum(peer * world_size + rank + 1 for peer in range(world_size)))
    return torch.full_like(output, expected_value)


def _allreduce_expected(output: torch.Tensor, *, world_size: int) -> torch.Tensor:
    return torch.full_like(output, float(sum(range(1, world_size + 1))))


def _broadcast_expected(output: torch.Tensor) -> torch.Tensor:
    return torch.full_like(output, 111.0)


def _scatter_expected(output: torch.Tensor, *, rank: int) -> torch.Tensor:
    return torch.full_like(output, float((rank + 1) * 10))


def _benchmark_case(
    *,
    collective: str,
    size_bytes: int,
    heaps: list[SymmetricHeap],
    streams: list[torch.cuda.Stream],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    world_size: int,
) -> dict[str, Any]:
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_elements = size_bytes // element_size
    block_elements = total_elements if collective != "allreduce" else total_elements // world_size

    if collective == "allgather":
        xt_src = _alloc_symmetric(heaps, (block_elements,), dtype)
        xt_dst = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)
        bulk_src = _alloc_symmetric(heaps, (block_elements,), dtype)
        bulk_dst = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)

        def prepare_tncc() -> None:
            for rank in range(world_size):
                xt_src[rank].fill_(float((rank + 1) * 10))
                xt_dst[rank].zero_()

        def prepare_bulk() -> None:
            for rank in range(world_size):
                bulk_src[rank].fill_(float((rank + 1) * 10))
                bulk_dst[rank].zero_()

        def run_tncc() -> None:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                with torch.cuda.stream(streams[rank]):
                    _allgather_kernel[(1,)](
                        xt_src[rank],
                        xt_dst[rank],
                        heaps[rank].get_heap_bases(),
                        rank,
                        world_size,
                        BLOCK_SIZE=block_elements,
                    )

        def run_bulk() -> None:
            for rank in range(world_size):
                _stream_copy_(
                    rank=rank,
                    dst=bulk_dst[rank][rank * block_elements:(rank + 1) * block_elements],
                    src=bulk_src[rank],
                    stream=streams[rank],
                )
            _sync_all(world_size)
            for src_rank in range(world_size):
                for dst_rank in range(world_size):
                    if src_rank == dst_rank:
                        continue
                    _launch_mirror_put(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src=bulk_src[src_rank],
                        dst_mirror=bulk_dst[src_rank][
                            src_rank * block_elements:(src_rank + 1) * block_elements
                        ],
                        heaps=heaps,
                        streams=streams,
                    )
            _sync_all(world_size)

        def validate() -> tuple[bool, bool]:
            xt_ok = all(
                torch.allclose(
                    xt_dst[rank],
                    _allgather_expected(xt_dst[rank], world_size=world_size, block_elements=block_elements),
                    atol=1e-4,
                )
                for rank in range(world_size)
            )
            bulk_ok = all(
                torch.allclose(
                    bulk_dst[rank],
                    _allgather_expected(bulk_dst[rank], world_size=world_size, block_elements=block_elements),
                    atol=1e-4,
                )
                for rank in range(world_size)
            )
            return xt_ok, bulk_ok

    elif collective == "scatter":
        xt_src = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)
        xt_dst = _alloc_symmetric(heaps, (block_elements,), dtype)
        bulk_src = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)
        bulk_dst = _alloc_symmetric(heaps, (block_elements,), dtype)

        def _fill_scatter(src_list: list[torch.Tensor], dst_list: list[torch.Tensor]) -> None:
            for rank in range(world_size):
                src_list[rank].zero_()
                dst_list[rank].zero_()
            for peer in range(world_size):
                src_list[0][peer * block_elements:(peer + 1) * block_elements].fill_(float((peer + 1) * 10))

        def prepare_tncc() -> None:
            _fill_scatter(xt_src, xt_dst)

        def prepare_bulk() -> None:
            _fill_scatter(bulk_src, bulk_dst)

        def run_tncc() -> None:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                with torch.cuda.stream(streams[rank]):
                    _scatter_kernel[(1,)](
                        xt_src[rank],
                        xt_dst[rank],
                        heaps[rank].get_heap_bases(),
                        rank,
                        world_size,
                        0,
                        BLOCK_SIZE=block_elements,
                    )

        def run_bulk() -> None:
            _stream_copy_(
                rank=0,
                dst=bulk_dst[0],
                src=bulk_src[0][:block_elements],
                stream=streams[0],
            )
            _sync_all(world_size)
            for dst_rank in range(1, world_size):
                _launch_mirror_put(
                    src_rank=0,
                    dst_rank=dst_rank,
                    src=bulk_src[0][dst_rank * block_elements:(dst_rank + 1) * block_elements],
                    dst_mirror=bulk_dst[0],
                    heaps=heaps,
                    streams=streams,
                )
            _sync_all(world_size)

        def validate() -> tuple[bool, bool]:
            xt_ok = all(
                torch.allclose(xt_dst[rank], _scatter_expected(xt_dst[rank], rank=rank), atol=1e-4)
                for rank in range(world_size)
            )
            bulk_ok = all(
                torch.allclose(bulk_dst[rank], _scatter_expected(bulk_dst[rank], rank=rank), atol=1e-4)
                for rank in range(world_size)
            )
            return xt_ok, bulk_ok

    elif collective == "broadcast":
        xt_buf = _alloc_symmetric(heaps, (block_elements,), dtype)
        bulk_buf = _alloc_symmetric(heaps, (block_elements,), dtype)

        def _fill_broadcast(buffers: list[torch.Tensor]) -> None:
            for rank in range(world_size):
                buffers[rank].fill_(111.0 if rank == 0 else -1.0)

        def prepare_tncc() -> None:
            _fill_broadcast(xt_buf)

        def prepare_bulk() -> None:
            _fill_broadcast(bulk_buf)

        def run_tncc() -> None:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                with torch.cuda.stream(streams[rank]):
                    _broadcast_kernel[(1,)](
                        xt_buf[rank],
                        heaps[rank].get_heap_bases(),
                        rank,
                        world_size,
                        0,
                        BLOCK_SIZE=block_elements,
                    )

        def run_bulk() -> None:
            for dst_rank in range(1, world_size):
                _launch_mirror_put(
                    src_rank=0,
                    dst_rank=dst_rank,
                    src=bulk_buf[0],
                    dst_mirror=bulk_buf[0],
                    heaps=heaps,
                    streams=streams,
                )
            _sync_all(world_size)

        def validate() -> tuple[bool, bool]:
            xt_ok = all(
                torch.allclose(xt_buf[rank], _broadcast_expected(xt_buf[rank]), atol=1e-4)
                for rank in range(world_size)
            )
            bulk_ok = all(
                torch.allclose(bulk_buf[rank], _broadcast_expected(bulk_buf[rank]), atol=1e-4)
                for rank in range(world_size)
            )
            return xt_ok, bulk_ok

    elif collective == "reduce_scatter":
        xt_full = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)
        xt_shard = _alloc_symmetric(heaps, (block_elements,), dtype)
        bulk_full = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)
        bulk_shard = _alloc_symmetric(heaps, (block_elements,), dtype)
        bulk_stage = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)

        def _fill_reduce_scatter(full_list: list[torch.Tensor], shard_list: list[torch.Tensor]) -> None:
            for rank in range(world_size):
                shard_list[rank].zero_()
                for chunk in range(world_size):
                    full_list[rank][chunk * block_elements:(chunk + 1) * block_elements].fill_(
                        float(rank * world_size + chunk + 1)
                    )

        def prepare_tncc() -> None:
            _fill_reduce_scatter(xt_full, xt_shard)

        def prepare_bulk() -> None:
            _fill_reduce_scatter(bulk_full, bulk_shard)
            for rank in range(world_size):
                bulk_stage[rank].zero_()

        def run_tncc() -> None:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                with torch.cuda.stream(streams[rank]):
                    _reduce_scatter_kernel[(1,)](
                        xt_full[rank],
                        xt_shard[rank],
                        heaps[rank].get_heap_bases(),
                        rank,
                        world_size,
                        BLOCK_SIZE=block_elements,
                    )

        def run_bulk() -> None:
            for rank in range(world_size):
                _stream_copy_(
                    rank=rank,
                    dst=bulk_shard[rank],
                    src=bulk_full[rank][rank * block_elements:(rank + 1) * block_elements],
                    stream=streams[rank],
                )
            _sync_all(world_size)
            for src_rank in range(world_size):
                for dst_rank in range(world_size):
                    if src_rank == dst_rank:
                        continue
                    _launch_mirror_put(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src=bulk_full[src_rank][
                            dst_rank * block_elements:(dst_rank + 1) * block_elements
                        ],
                        dst_mirror=bulk_stage[src_rank][
                            src_rank * block_elements:(src_rank + 1) * block_elements
                        ],
                        heaps=heaps,
                        streams=streams,
                    )
            _sync_all(world_size)
            for rank in range(world_size):
                for src_rank in range(world_size):
                    if src_rank == rank:
                        continue
                    _stream_add_(
                        rank=rank,
                        dst=bulk_shard[rank],
                        src=bulk_stage[rank][
                            src_rank * block_elements:(src_rank + 1) * block_elements
                        ],
                        stream=streams[rank],
                    )
            _sync_all(world_size)

        def validate() -> tuple[bool, bool]:
            xt_ok = all(
                torch.allclose(
                    xt_shard[rank],
                    _reduce_scatter_expected(xt_shard[rank], rank=rank, world_size=world_size),
                    atol=1e-4,
                )
                for rank in range(world_size)
            )
            bulk_ok = all(
                torch.allclose(
                    bulk_shard[rank],
                    _reduce_scatter_expected(bulk_shard[rank], rank=rank, world_size=world_size),
                    atol=1e-4,
                )
                for rank in range(world_size)
            )
            return xt_ok, bulk_ok

    elif collective == "allreduce":
        xt_full = _alloc_symmetric(heaps, (total_elements,), dtype)
        xt_shard = _alloc_symmetric(heaps, (block_elements,), dtype)
        xt_gathered = _alloc_symmetric(heaps, (total_elements,), dtype)
        bulk_full = _alloc_symmetric(heaps, (total_elements,), dtype)
        bulk_shard = _alloc_symmetric(heaps, (block_elements,), dtype)
        bulk_stage = _alloc_symmetric(heaps, (block_elements * world_size,), dtype)
        bulk_gathered = _alloc_symmetric(heaps, (total_elements,), dtype)

        def prepare_tncc() -> None:
            for rank in range(world_size):
                xt_full[rank].fill_(float(rank + 1))
                xt_shard[rank].zero_()
                xt_gathered[rank].zero_()

        def prepare_bulk() -> None:
            for rank in range(world_size):
                bulk_full[rank].fill_(float(rank + 1))
                bulk_shard[rank].zero_()
                bulk_stage[rank].zero_()
                bulk_gathered[rank].zero_()

        def run_tncc() -> None:
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                with torch.cuda.stream(streams[rank]):
                    _reduce_scatter_kernel[(1,)](
                        xt_full[rank],
                        xt_shard[rank],
                        heaps[rank].get_heap_bases(),
                        rank,
                        world_size,
                        BLOCK_SIZE=block_elements,
                    )
            _sync_all(world_size)
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                with torch.cuda.stream(streams[rank]):
                    _allgather_kernel[(1,)](
                        xt_shard[rank],
                        xt_gathered[rank],
                        heaps[rank].get_heap_bases(),
                        rank,
                        world_size,
                        BLOCK_SIZE=block_elements,
                    )

        def run_bulk() -> None:
            for rank in range(world_size):
                _stream_copy_(
                    rank=rank,
                    dst=bulk_shard[rank],
                    src=bulk_full[rank][rank * block_elements:(rank + 1) * block_elements],
                    stream=streams[rank],
                )
            _sync_all(world_size)
            for src_rank in range(world_size):
                for dst_rank in range(world_size):
                    if src_rank == dst_rank:
                        continue
                    _launch_mirror_put(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src=bulk_full[src_rank][
                            dst_rank * block_elements:(dst_rank + 1) * block_elements
                        ],
                        dst_mirror=bulk_stage[src_rank][
                            src_rank * block_elements:(src_rank + 1) * block_elements
                        ],
                        heaps=heaps,
                        streams=streams,
                    )
            _sync_all(world_size)
            for rank in range(world_size):
                for src_rank in range(world_size):
                    if src_rank == rank:
                        continue
                    _stream_add_(
                        rank=rank,
                        dst=bulk_shard[rank],
                        src=bulk_stage[rank][
                            src_rank * block_elements:(src_rank + 1) * block_elements
                        ],
                        stream=streams[rank],
                    )
            _sync_all(world_size)

            for rank in range(world_size):
                _stream_copy_(
                    rank=rank,
                    dst=bulk_gathered[rank][rank * block_elements:(rank + 1) * block_elements],
                    src=bulk_shard[rank],
                    stream=streams[rank],
                )
            _sync_all(world_size)
            for src_rank in range(world_size):
                for dst_rank in range(world_size):
                    if src_rank == dst_rank:
                        continue
                    _launch_mirror_put(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src=bulk_shard[src_rank],
                        dst_mirror=bulk_gathered[src_rank][
                            src_rank * block_elements:(src_rank + 1) * block_elements
                        ],
                        heaps=heaps,
                        streams=streams,
                    )
            _sync_all(world_size)

        def validate() -> tuple[bool, bool]:
            xt_ok = all(
                torch.allclose(
                    xt_gathered[rank],
                    _allreduce_expected(xt_gathered[rank], world_size=world_size),
                    atol=1e-4,
                )
                for rank in range(world_size)
            )
            bulk_ok = all(
                torch.allclose(
                    bulk_gathered[rank],
                    _allreduce_expected(bulk_gathered[rank], world_size=world_size),
                    atol=1e-4,
                )
                for rank in range(world_size)
            )
            return xt_ok, bulk_ok

    else:
        raise ValueError(f"unsupported collective: {collective}")

    xt_stats = _time_end_to_end(
        prepare_tncc,
        run_tncc,
        world_size=world_size,
        warmup=warmup,
        iters=iters,
        operations_per_sample=_timed_batch_repeats_for_size(size_bytes),
    )
    bulk_stats = _time_end_to_end(
        prepare_bulk,
        run_bulk,
        world_size=world_size,
        warmup=warmup,
        iters=iters,
        operations_per_sample=_timed_batch_repeats_for_size(size_bytes),
    )

    prepare_tncc()
    run_tncc()
    _sync_all(world_size)
    prepare_bulk()
    run_bulk()
    _sync_all(world_size)
    xt_ok, bulk_ok = validate()

    effective_bytes = _effective_bytes(collective, size_bytes, world_size)
    xt_bw = _bandwidth_gbps(effective_bytes, xt_stats["median_ms"])
    bulk_bw = _bandwidth_gbps(effective_bytes, bulk_stats["median_ms"])
    speedup_vs_bulk = (
        bulk_stats["median_ms"] / xt_stats["median_ms"]
        if xt_stats["median_ms"] > 0
        else 0.0
    )

    return {
        "collective": collective,
        "size_bytes": size_bytes,
        "size_kib": float(size_bytes / 1024.0),
        "timing_budget": {
            "warmup": warmup,
            "iters": iters,
            "operations_per_timed_sample": _timed_batch_repeats_for_size(size_bytes),
            "latency_measurement": "host_wall_end_to_end_batched_per_operation",
        },
        "tncc": {
            **xt_stats,
            "median_bandwidth_gbps": xt_bw,
            "correct": xt_ok,
        },
        "bulk_sync": {
            **bulk_stats,
            "median_bandwidth_gbps": bulk_bw,
            "correct": bulk_ok,
        },
        "speedup_vs_bulk": float(speedup_vs_bulk),
    }


def main() -> None:
    args = _parse_args()
    world_size = min(torch.cuda.device_count(), int(args.world_size))
    if world_size < 2:
        raise SystemExit("Need >= 2 GPUs")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")

    dtype = torch.float32
    sizes = _parse_message_sizes(
        args.message_sizes,
        element_size=torch.tensor([], dtype=dtype).element_size(),
        world_size=world_size,
    )
    environment_health = benchmark_environment_health(
        visible_gpu_count=world_size,
    )
    emit_benchmark_environment_warnings(environment_health)

    output_path = args.output_json
    with canonical_benchmark_run(output_path):
        heaps = SymmetricHeap.create_all(size=256 * 1024 * 1024, world_size=world_size)
        try:
            streams: list[torch.cuda.Stream] = []
            for rank in range(world_size):
                torch.cuda.set_device(rank)
                streams.append(torch.cuda.Stream(device=rank))

            ctx = tncc.init(
                backend="cuda",
                rank=0,
                world_size=world_size,
                heap=heaps[0],
                force_backend=True,
            )

            cases: list[dict[str, Any]] = []
            for collective in _COLLECTIVES:
                for size_bytes in sizes:
                    _reset_heaps(heaps)
                    case = _benchmark_case(
                        collective=collective,
                        size_bytes=size_bytes,
                        heaps=heaps,
                        streams=streams,
                        dtype=dtype,
                        warmup=int(args.warmup),
                        iters=int(args.iters),
                        world_size=world_size,
                    )
                    cases.append(case)
                    status = "PASS" if case["tncc"]["correct"] and case["bulk_sync"]["correct"] else "FAIL"
                    print(
                        f"[{status}] {collective:14s} {size_bytes//1024:6.0f} KiB "
                        f"tncc={case['tncc']['median_ms']:.3f} ms "
                        f"bulk={case['bulk_sync']['median_ms']:.3f} ms "
                        f"speedup={case['speedup_vs_bulk']:.3f}x",
                        flush=True,
                    )

            best_case = max(cases, key=lambda item: item["speedup_vs_bulk"]) if cases else None
            peak_by_collective: dict[str, dict[str, float]] = {}
            for collective in _COLLECTIVES:
                current = [case for case in cases if case["collective"] == collective]
                if not current:
                    continue
                peak_by_collective[collective] = {
                    "best_speedup_vs_bulk": max(case["speedup_vs_bulk"] for case in current),
                    "peak_tncc_bandwidth_gbps": max(case["tncc"]["median_bandwidth_gbps"] for case in current),
                    "peak_bulk_bandwidth_gbps": max(case["bulk_sync"]["median_bandwidth_gbps"] for case in current),
                }

            payload = {
                "schema_version": 1,
                "benchmark": "collective_bulk_sync",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "environment": {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "visible_gpus": torch.cuda.device_count(),
                    "world_size": world_size,
                    "dtype": "float32",
                    "warmup": int(args.warmup),
                    "iters": int(args.iters),
                    "message_sizes_bytes": sizes,
                    "mode": "single_process",
                    "transport_strategy": "peer_access",
                    "sampling_policy": {
                        "base_warmup": int(args.warmup),
                        "base_iters": int(args.iters),
                        "small_message_batch_threshold_bytes": _SMALL_MESSAGE_BATCH_THRESHOLD_BYTES,
                        "medium_message_batch_threshold_bytes": _MEDIUM_MESSAGE_BATCH_THRESHOLD_BYTES,
                        "small_message_batch_repeats": _SMALL_MESSAGE_BATCH_REPEATS,
                        "medium_message_batch_repeats": _MEDIUM_MESSAGE_BATCH_REPEATS,
                        "large_message_batch_repeats": _LARGE_MESSAGE_BATCH_REPEATS,
                    },
                },
                "methodology": {
                    "optimized_path": "single-process peer_access device collective kernels",
                    "bulk_sync_baseline": (
                        "host-orchestrated composition of lower-level point-to-point "
                        "steps with explicit synchronization between phases"
                    ),
                    "latency_measurement": (
                        "host_wall_end_to_end_batched_per_operation"
                    ),
                    "allreduce_note": (
                        "allreduce optimized path is measured as device reduce_scatter "
                        "+ device allgather composition because the raw single-kernel "
                        "tile_allreduce path is not the stable benchmark surface here"
                    ),
                },
                "runtime_support": runtime_support_snapshot(ctx),
                "runtime_metadata": runtime_metadata_snapshot(ctx),
                "environment_health": environment_health,
                "cases": cases,
                "summary": {
                    "best_speedup_vs_bulk": best_case["speedup_vs_bulk"] if best_case is not None else None,
                    "best_case": {
                        "collective": best_case["collective"],
                        "size_bytes": best_case["size_bytes"],
                        "size_kib": best_case["size_kib"],
                        "speedup_vs_bulk": best_case["speedup_vs_bulk"],
                    }
                    if best_case is not None
                    else None,
                    "peak_by_collective": peak_by_collective,
                },
            }
            written = write_json(output_path, payload)
            print(f"Structured results written to: {written}", flush=True)
        finally:
            for heap in heaps:
                heap.cleanup()


if __name__ == "__main__":
    main()
