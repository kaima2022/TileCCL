#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real communication-only collective benchmark: TNCC kernels vs NCCL.

This benchmark measures the pure communication collectives exposed by TNCC's
tile primitives and compares them against the corresponding NCCL-backed
``torch.distributed`` collectives on the same host.

Covered collectives:
    - tile_allreduce
    - tile_allgather
    - tile_scatter
    - tile_reduce_scatter
    - tile_broadcast

The benchmark is multiprocess and uses a real NCCL process group plus a real
``SymmetricHeap`` transport. No synthetic fallback values are emitted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tncc.utils.benchmark_results import (
    benchmark_environment_health,
    canonical_benchmark_run,
    default_collective_comm_only_benchmark_path,
    emit_benchmark_environment_warnings,
    runtime_metadata_snapshot,
    runtime_support_snapshot,
    write_json,
)
from tncc.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_COLLECTIVES = (
    "allreduce",
    "allgather",
    "scatter",
    "reduce_scatter",
    "broadcast",
)
_DEFAULT_TIMING_MODE = "device_event"
_DEFAULT_WARMUP = 12
_DEFAULT_ITERS = 12
_PRECONDITION_MESSAGE_BYTES = 256 * 1024
_PRECONDITION_ITERS = 4
_TINY_MESSAGE_BATCH_THRESHOLD_BYTES = 4 * 1024
_SMALL_MESSAGE_BATCH_THRESHOLD_BYTES = 16 * 1024
_MEDIUM_MESSAGE_BATCH_THRESHOLD_BYTES = 64 * 1024
_LARGE_MESSAGE_THRESHOLD_BYTES = 256 * 1024
_VERY_LARGE_MESSAGE_THRESHOLD_BYTES = 1024 * 1024
_TINY_MESSAGE_BATCH_REPEATS = 64
_SMALL_MESSAGE_BATCH_REPEATS = 64
_MEDIUM_MESSAGE_BATCH_REPEATS = 16
_LARGE_MESSAGE_BATCH_REPEATS = 8
_VERY_LARGE_MESSAGE_BATCH_REPEATS = 4
_LARGE_MESSAGE_WARMUP_CAP = 10
_LARGE_MESSAGE_ITERS_CAP = 10
_VERY_LARGE_MESSAGE_WARMUP_CAP = 8
_VERY_LARGE_MESSAGE_ITERS_CAP = 8


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one comm-only benchmark run."""

    collectives: tuple[str, ...]
    message_sizes_bytes: tuple[int, ...]
    dtype_name: str
    timing_mode: str
    warmup: int
    iters: int
    world_size: int
    force_transport: str | None
    root: int
    heap_size_bytes: int
    output_dir: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="float32",
        help="Element dtype for both TNCC and NCCL buffers.",
    )
    parser.add_argument(
        "--message-sizes",
        type=str,
        default="4096,16384,65536,262144,1048576,2097152",
        help=(
            "Comma-separated rank-local message sizes in bytes. "
            "Default spans small-message latency points through multi-MiB bandwidth points."
        ),
    )
    parser.add_argument(
        "--collectives",
        type=str,
        default=",".join(_COLLECTIVES),
        help="Comma-separated subset of collectives to run.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["host_wall", "device_event"],
        default=_DEFAULT_TIMING_MODE,
        help="Measurement window to use for timed iterations.",
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
        help="Requested world size. Current validated configuration uses 2 GPUs.",
    )
    parser.add_argument(
        "--force-transport",
        choices=["auto", "ctypes_ipc", "pytorch_ipc", "peer_access_pointer_exchange"],
        default="auto",
        help="Force one specific TNCC multiprocess transport strategy.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(default_collective_comm_only_benchmark_path()),
        help="Structured JSON output path.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return _DTYPES[dtype_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(f"dtype must be one of {allowed}, got {dtype_name!r}") from exc


def _parse_message_sizes(raw: str, *, element_size: int, world_size: int) -> tuple[int, ...]:
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
        sizes.append(size_bytes)
    if not sizes:
        raise ValueError("at least one message size is required")
    return tuple(sorted(set(sizes)))


def _parse_collectives(raw: str) -> tuple[str, ...]:
    selected: list[str] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if item not in _COLLECTIVES:
            allowed = ", ".join(_COLLECTIVES)
            raise ValueError(f"unsupported collective {item!r}; expected one of {allowed}")
        selected.append(item)
    if not selected:
        raise ValueError("at least one collective is required")
    return tuple(dict.fromkeys(selected))


def _block_elements(*, collective: str, size_bytes: int, dtype: torch.dtype, world_size: int) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    if collective == "allreduce":
        return size_bytes // element_size
    return size_bytes // element_size


def _effective_bytes(collective: str, size_bytes: int, world_size: int) -> float:
    if collective == "allreduce":
        return 2.0 * (world_size - 1) / world_size * size_bytes
    if collective in {"allgather", "reduce_scatter", "scatter", "broadcast"}:
        return (world_size - 1) * size_bytes
    raise ValueError(f"unsupported collective: {collective}")


def _timing_stats(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(sum(times_ms) / len(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def _bandwidth_summary(
    *,
    collective: str,
    size_bytes: int,
    world_size: int,
    times_ms: list[float],
) -> dict[str, float]:
    effective_bytes = _effective_bytes(collective, size_bytes, world_size)
    stats = _timing_stats(times_ms)
    median_ms = max(stats["median_ms"], 1e-9)
    min_ms = max(stats["min_ms"], 1e-9)
    return {
        **stats,
        "effective_bytes": float(effective_bytes),
        "median_bandwidth_gbps": float(effective_bytes / (median_ms * 1e-3) / 1e9),
        "best_bandwidth_gbps": float(effective_bytes / (min_ms * 1e-3) / 1e9),
    }


def _sampling_budget_for_size(
    *,
    size_bytes: int,
    warmup: int,
    iters: int,
) -> tuple[int, int]:
    """Return a size-aware timing budget for one benchmark case."""
    if size_bytes < _LARGE_MESSAGE_THRESHOLD_BYTES:
        return warmup, iters
    if size_bytes >= _VERY_LARGE_MESSAGE_THRESHOLD_BYTES:
        return min(warmup, _VERY_LARGE_MESSAGE_WARMUP_CAP), min(
            iters,
            _VERY_LARGE_MESSAGE_ITERS_CAP,
        )
    return min(warmup, _LARGE_MESSAGE_WARMUP_CAP), min(iters, _LARGE_MESSAGE_ITERS_CAP)


def _timed_batch_repeats_for_size(size_bytes: int) -> int:
    """Return how many logical collectives to batch into one timed sample."""
    if size_bytes <= _TINY_MESSAGE_BATCH_THRESHOLD_BYTES:
        return _TINY_MESSAGE_BATCH_REPEATS
    if size_bytes <= _SMALL_MESSAGE_BATCH_THRESHOLD_BYTES:
        return _SMALL_MESSAGE_BATCH_REPEATS
    if size_bytes <= _MEDIUM_MESSAGE_BATCH_THRESHOLD_BYTES:
        return _MEDIUM_MESSAGE_BATCH_REPEATS
    if size_bytes <= _VERY_LARGE_MESSAGE_THRESHOLD_BYTES:
        return _LARGE_MESSAGE_BATCH_REPEATS
    return _VERY_LARGE_MESSAGE_BATCH_REPEATS


def _execute_collective_batch(
    fn,
    *,
    rank: int,
    operations_per_sample: int,
) -> None:
    """Execute one timed batch and fully complete every logical collective."""
    for _ in range(operations_per_sample):
        result = fn()
        if hasattr(result, "wait"):
            result.wait()
        torch.cuda.synchronize(rank)


def _timed_collective(
    fn,
    *,
    prepare_fn,
    rank: int,
    barrier_kwargs: dict[str, object],
    timing_mode: str,
    warmup: int,
    iters: int,
    operations_per_sample: int,
) -> list[float]:
    for _ in range(warmup):
        prepare_fn()
        dist.barrier(**barrier_kwargs)
        _execute_collective_batch(
            fn,
            rank=rank,
            operations_per_sample=operations_per_sample,
        )
        dist.barrier(**barrier_kwargs)

    times_ms: list[float] = []
    for _ in range(iters):
        prepare_fn()
        dist.barrier(**barrier_kwargs)
        if timing_mode == "device_event":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _execute_collective_batch(
                fn,
                rank=rank,
                operations_per_sample=operations_per_sample,
            )
            end.record()
            torch.cuda.synchronize(rank)
            elapsed_ms = float(start.elapsed_time(end)) / float(operations_per_sample)
        else:
            # Measure the public collective as the caller observes it:
            # launch/rendezvous plus device completion on this rank.
            start_ns = time.perf_counter_ns()
            _execute_collective_batch(
                fn,
                rank=rank,
                operations_per_sample=operations_per_sample,
            )
            end_ns = time.perf_counter_ns()
            elapsed_ms = float(end_ns - start_ns) / 1_000_000.0 / float(operations_per_sample)
        dist.barrier(**barrier_kwargs)
        times_ms.append(elapsed_ms)
    return times_ms


def _run_validation_collective(
    fn,
    *,
    prepare_fn,
    rank: int,
    barrier_kwargs: dict[str, object],
) -> None:
    """Run one post-timing validation replay on freshly prepared buffers."""
    prepare_fn()
    dist.barrier(**barrier_kwargs)
    result = fn()
    if hasattr(result, "wait"):
        result.wait()
    torch.cuda.synchronize(rank)
    dist.barrier(**barrier_kwargs)


def _precondition_collective_runtime(
    *,
    rank: int,
    barrier_kwargs: dict[str, object],
    dtype: torch.dtype,
    world_size: int,
    message_sizes_bytes: tuple[int, ...],
    heap,
    primitive_allreduce_fn,
    xt_allreduce: torch.Tensor,
    nccl_allreduce: torch.Tensor,
) -> dict[str, int]:
    """Warm up the first allreduce path so tiny-message timing starts hot."""
    precondition_size_bytes = min(
        max(message_sizes_bytes),
        _PRECONDITION_MESSAGE_BYTES,
    )
    precondition_elements = _block_elements(
        collective="allreduce",
        size_bytes=precondition_size_bytes,
        dtype=dtype,
        world_size=world_size,
    )
    xt_view = xt_allreduce.narrow(0, 0, precondition_elements)
    nccl_view = nccl_allreduce.narrow(0, 0, precondition_elements)

    for _ in range(_PRECONDITION_ITERS):
        _fill_allreduce_input(xt_view, rank=rank)
        dist.barrier(**barrier_kwargs)
        primitive_allreduce_fn(xt_view, heap)
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)

        _fill_allreduce_input(nccl_view, rank=rank)
        dist.barrier(**barrier_kwargs)
        work = dist.all_reduce(nccl_view, async_op=True)
        if hasattr(work, "wait"):
            work.wait()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)

    return {
        "collective": "allreduce",
        "size_bytes": int(precondition_size_bytes),
        "iterations": _PRECONDITION_ITERS,
    }


def _fill_allreduce_input(tensor: torch.Tensor, *, rank: int) -> None:
    tensor.fill_(float(rank + 1))


def _fill_allgather_input(src: torch.Tensor, dst: torch.Tensor, *, rank: int) -> None:
    src.fill_(float((rank + 1) * 10))
    dst.zero_()


def _fill_broadcast_input(tensor: torch.Tensor, *, rank: int, root: int) -> None:
    if rank == root:
        tensor.fill_(111.0)
    else:
        tensor.fill_(-1.0)


def _fill_scatter_input(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    block_elements: int,
    root: int,
) -> None:
    dst.zero_()
    if rank != root:
        src.zero_()
        return
    for peer in range(world_size):
        start = peer * block_elements
        src[start:start + block_elements].fill_(float((peer + 1) * 10))


def _fill_reduce_scatter_input(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    block_elements: int,
) -> None:
    dst.zero_()
    for chunk in range(world_size):
        start = chunk * block_elements
        src[start:start + block_elements].fill_(float(rank * world_size + chunk + 1))


def _expected_allreduce(tensor: torch.Tensor, *, world_size: int) -> torch.Tensor:
    return torch.full_like(tensor, float(sum(range(1, world_size + 1))))


def _expected_allgather(
    tensor: torch.Tensor,
    *,
    world_size: int,
    block_elements: int,
) -> torch.Tensor:
    expected = torch.empty_like(tensor)
    for peer in range(world_size):
        start = peer * block_elements
        expected[start:start + block_elements].fill_(float((peer + 1) * 10))
    return expected


def _expected_scatter_or_broadcast(
    tensor: torch.Tensor,
    *,
    rank: int,
    is_scatter: bool,
) -> torch.Tensor:
    value = float((rank + 1) * 10) if is_scatter else 111.0
    return torch.full_like(tensor, value)


def _expected_reduce_scatter(
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    expected_value = float(
        sum(peer * world_size + rank + 1 for peer in range(world_size))
    )
    return torch.full_like(tensor, expected_value)


def _worker(rank: int, store_path: str, config: _RunConfig) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    barrier_kwargs = {"device_ids": [rank]}
    dtype = _resolve_dtype(config.dtype_name)
    heap = None
    try:
        if config.force_transport is None:
            os.environ.pop(FORCE_MULTIPROCESS_TRANSPORT_ENV, None)
        else:
            os.environ[FORCE_MULTIPROCESS_TRANSPORT_ENV] = config.force_transport

        store = dist.FileStore(store_path, config.world_size)
        dist.init_process_group(
            "nccl",
            store=store,
            rank=rank,
            world_size=config.world_size,
            device_id=device,
        )

        import tncc
        from tncc.memory.symmetric_heap import SymmetricHeap
        from tncc.primitives import (
            allgather as primitive_allgather,
        )
        from tncc.primitives import (
            allreduce as primitive_allreduce,
        )
        from tncc.primitives import (
            broadcast as primitive_broadcast,
        )
        from tncc.primitives import (
            reduce_scatter as primitive_reduce_scatter,
        )
        from tncc.primitives import (
            scatter as primitive_scatter,
        )

        heap = SymmetricHeap(
            size=config.heap_size_bytes,
            rank=rank,
            world_size=config.world_size,
            backend="cuda",
        )
        ctx = tncc.init(
            backend="cuda",
            rank=rank,
            world_size=config.world_size,
            heap=heap,
            force_backend=True,
        )

        max_message_bytes = max(config.message_sizes_bytes)
        max_block_elements = max(
            _block_elements(
                collective=collective,
                size_bytes=max_message_bytes,
                dtype=dtype,
                world_size=config.world_size,
            )
            for collective in _COLLECTIVES
        )
        max_allreduce_block = _block_elements(
            collective="allreduce",
            size_bytes=max_message_bytes,
            dtype=dtype,
            world_size=config.world_size,
        )

        xt_allreduce = heap.allocate_tensor(
            (max_allreduce_block * config.world_size,),
            dtype,
        )
        xt_allgather_src = heap.allocate_tensor((max_block_elements,), dtype)
        xt_allgather_dst = heap.allocate_tensor((max_block_elements * config.world_size,), dtype)
        xt_scatter_src = heap.allocate_tensor((max_block_elements * config.world_size,), dtype)
        xt_scatter_dst = heap.allocate_tensor((max_block_elements,), dtype)
        xt_reduce_scatter_src = heap.allocate_tensor(
            (max_block_elements * config.world_size,),
            dtype,
        )
        xt_reduce_scatter_dst = heap.allocate_tensor((max_block_elements,), dtype)
        xt_broadcast = heap.allocate_tensor((max_block_elements,), dtype)

        nccl_allreduce = torch.empty_like(xt_allreduce)
        nccl_allgather_src = torch.empty_like(xt_allgather_src)
        nccl_allgather_dst = torch.empty_like(xt_allgather_dst)
        nccl_scatter_src = torch.empty_like(xt_scatter_src)
        nccl_scatter_dst = torch.empty_like(xt_scatter_dst)
        nccl_reduce_scatter_src = torch.empty_like(xt_reduce_scatter_src)
        nccl_reduce_scatter_dst = torch.empty_like(xt_reduce_scatter_dst)
        nccl_broadcast = torch.empty_like(xt_broadcast)

        preconditioning = _precondition_collective_runtime(
            rank=rank,
            barrier_kwargs=barrier_kwargs,
            dtype=dtype,
            world_size=config.world_size,
            message_sizes_bytes=config.message_sizes_bytes,
            heap=heap,
            primitive_allreduce_fn=primitive_allreduce,
            xt_allreduce=xt_allreduce,
            nccl_allreduce=nccl_allreduce,
        )

        results: list[dict[str, Any]] = []
        for collective in config.collectives:
            for size_bytes in config.message_sizes_bytes:
                nccl_async = config.timing_mode == "device_event"
                block_elements = _block_elements(
                    collective=collective,
                    size_bytes=size_bytes,
                    dtype=dtype,
                    world_size=config.world_size,
                )
                allreduce_plan = None
                case_warmup, case_iters = _sampling_budget_for_size(
                    size_bytes=size_bytes,
                    warmup=config.warmup,
                    iters=config.iters,
                )
                operations_per_sample = _timed_batch_repeats_for_size(size_bytes)

                if collective == "allreduce":
                    total_elements = block_elements
                    xt_view = xt_allreduce.narrow(0, 0, total_elements)
                    nccl_view = nccl_allreduce.narrow(0, 0, total_elements)
                    allreduce_plan = tncc.ops.build_allreduce_plan(xt_view, ctx=ctx)

                    def prepare_tncc() -> None:
                        _fill_allreduce_input(xt_view, rank=rank)

                    def prepare_nccl() -> None:
                        _fill_allreduce_input(nccl_view, rank=rank)

                    def tncc_fn():
                        return primitive_allreduce(xt_view, heap)

                    def nccl_fn():
                        return dist.all_reduce(nccl_view, async_op=nccl_async)

                    expected_xt = _expected_allreduce(xt_view, world_size=config.world_size)
                    expected_nccl = _expected_allreduce(nccl_view, world_size=config.world_size)

                elif collective == "allgather":
                    xt_src = xt_allgather_src.narrow(0, 0, block_elements)
                    xt_dst = xt_allgather_dst.narrow(0, 0, block_elements * config.world_size)
                    nccl_src = nccl_allgather_src.narrow(0, 0, block_elements)
                    nccl_dst = nccl_allgather_dst.narrow(0, 0, block_elements * config.world_size)

                    def prepare_tncc() -> None:
                        _fill_allgather_input(xt_src, xt_dst, rank=rank)

                    def prepare_nccl() -> None:
                        _fill_allgather_input(nccl_src, nccl_dst, rank=rank)

                    def tncc_fn():
                        return primitive_allgather(xt_src, xt_dst, heap)

                    def nccl_fn():
                        return dist.all_gather_into_tensor(
                            nccl_dst,
                            nccl_src,
                            async_op=nccl_async,
                        )

                    expected_xt = _expected_allgather(
                        xt_dst,
                        world_size=config.world_size,
                        block_elements=block_elements,
                    )
                    expected_nccl = _expected_allgather(
                        nccl_dst,
                        world_size=config.world_size,
                        block_elements=block_elements,
                    )

                elif collective == "scatter":
                    xt_src = xt_scatter_src.narrow(0, 0, block_elements * config.world_size)
                    xt_dst = xt_scatter_dst.narrow(0, 0, block_elements)
                    nccl_src = nccl_scatter_src.narrow(0, 0, block_elements * config.world_size)
                    nccl_dst = nccl_scatter_dst.narrow(0, 0, block_elements)
                    if rank == config.root:
                        nccl_scatter_list = [
                            nccl_src.narrow(0, peer * block_elements, block_elements)
                            for peer in range(config.world_size)
                        ]
                    else:
                        nccl_scatter_list = None

                    def prepare_tncc() -> None:
                        _fill_scatter_input(
                            xt_src,
                            xt_dst,
                            rank=rank,
                            world_size=config.world_size,
                            block_elements=block_elements,
                            root=config.root,
                        )

                    def prepare_nccl() -> None:
                        _fill_scatter_input(
                            nccl_src,
                            nccl_dst,
                            rank=rank,
                            world_size=config.world_size,
                            block_elements=block_elements,
                            root=config.root,
                        )

                    def tncc_fn():
                        return primitive_scatter(
                            xt_src,
                            xt_dst,
                            heap,
                            root=config.root,
                        )

                    def nccl_fn():
                        return dist.scatter(
                            nccl_dst,
                            scatter_list=nccl_scatter_list,
                            src=config.root,
                            async_op=nccl_async,
                        )

                    expected_xt = _expected_scatter_or_broadcast(
                        xt_dst,
                        rank=rank,
                        is_scatter=True,
                    )
                    expected_nccl = _expected_scatter_or_broadcast(
                        nccl_dst,
                        rank=rank,
                        is_scatter=True,
                    )

                elif collective == "reduce_scatter":
                    xt_src = xt_reduce_scatter_src.narrow(0, 0, block_elements * config.world_size)
                    xt_dst = xt_reduce_scatter_dst.narrow(0, 0, block_elements)
                    nccl_src = nccl_reduce_scatter_src.narrow(
                        0,
                        0,
                        block_elements * config.world_size,
                    )
                    nccl_dst = nccl_reduce_scatter_dst.narrow(0, 0, block_elements)

                    def prepare_tncc() -> None:
                        _fill_reduce_scatter_input(
                            xt_src,
                            xt_dst,
                            rank=rank,
                            world_size=config.world_size,
                            block_elements=block_elements,
                        )

                    def prepare_nccl() -> None:
                        _fill_reduce_scatter_input(
                            nccl_src,
                            nccl_dst,
                            rank=rank,
                            world_size=config.world_size,
                            block_elements=block_elements,
                        )

                    def tncc_fn():
                        return primitive_reduce_scatter(
                            xt_src,
                            xt_dst,
                            heap,
                            implementation="device",
                        )

                    def nccl_fn():
                        return dist.reduce_scatter_tensor(
                            nccl_dst,
                            nccl_src,
                            async_op=nccl_async,
                        )

                    expected_xt = _expected_reduce_scatter(
                        xt_dst,
                        rank=rank,
                        world_size=config.world_size,
                    )
                    expected_nccl = _expected_reduce_scatter(
                        nccl_dst,
                        rank=rank,
                        world_size=config.world_size,
                    )

                elif collective == "broadcast":
                    xt_view = xt_broadcast.narrow(0, 0, block_elements)
                    nccl_view = nccl_broadcast.narrow(0, 0, block_elements)

                    def prepare_tncc() -> None:
                        _fill_broadcast_input(xt_view, rank=rank, root=config.root)

                    def prepare_nccl() -> None:
                        _fill_broadcast_input(nccl_view, rank=rank, root=config.root)

                    def tncc_fn():
                        return primitive_broadcast(xt_view, heap, root=config.root)

                    def nccl_fn():
                        return dist.broadcast(
                            nccl_view,
                            src=config.root,
                            async_op=nccl_async,
                        )

                    expected_xt = _expected_scatter_or_broadcast(
                        xt_view,
                        rank=rank,
                        is_scatter=False,
                    )
                    expected_nccl = _expected_scatter_or_broadcast(
                        nccl_view,
                        rank=rank,
                        is_scatter=False,
                    )

                else:
                    raise ValueError(f"unsupported collective: {collective}")

                tncc_result: dict[str, Any] = {
                    "times_ms": {},
                    "correct": False,
                }
                if allreduce_plan is not None:
                    tncc_result.update(
                        {
                            "implementation": allreduce_plan.implementation,
                            "protocol": allreduce_plan.protocol,
                            "kernel_family": allreduce_plan.kernel_family,
                            "reuse_handshake": allreduce_plan.reuse_handshake,
                            "message_bytes": allreduce_plan.message_bytes,
                            "message_regime": allreduce_plan.message_regime,
                            "cta_policy": allreduce_plan.cta_policy,
                            "epoch_policy": allreduce_plan.epoch_policy,
                            "chunk_elems": allreduce_plan.chunk_elems,
                            "num_chunks": allreduce_plan.num_chunks,
                            "pipeline_slots": allreduce_plan.pipeline_slots,
                            "grid_size": allreduce_plan.grid_size,
                            "num_warps": allreduce_plan.num_warps,
                            "workspace_bytes": allreduce_plan.workspace_bytes,
                        }
                    )

                tncc_times_ms = _timed_collective(
                    tncc_fn,
                    prepare_fn=prepare_tncc,
                    rank=rank,
                    barrier_kwargs=barrier_kwargs,
                    timing_mode=config.timing_mode,
                    warmup=case_warmup,
                    iters=case_iters,
                    operations_per_sample=operations_per_sample,
                )
                nccl_times_ms = _timed_collective(
                    nccl_fn,
                    prepare_fn=prepare_nccl,
                    rank=rank,
                    barrier_kwargs=barrier_kwargs,
                    timing_mode=config.timing_mode,
                    warmup=case_warmup,
                    iters=case_iters,
                    operations_per_sample=operations_per_sample,
                )
                _run_validation_collective(
                    tncc_fn,
                    prepare_fn=prepare_tncc,
                    rank=rank,
                    barrier_kwargs=barrier_kwargs,
                )
                _run_validation_collective(
                    nccl_fn,
                    prepare_fn=prepare_nccl,
                    rank=rank,
                    barrier_kwargs=barrier_kwargs,
                )

                if collective == "allreduce":
                    tncc_ok = bool(torch.allclose(xt_view, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_view, expected_nccl, atol=1e-4))
                elif collective == "allgather":
                    tncc_ok = bool(torch.allclose(xt_dst, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_dst, expected_nccl, atol=1e-4))
                elif collective == "scatter":
                    tncc_ok = bool(torch.allclose(xt_dst, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_dst, expected_nccl, atol=1e-4))
                elif collective == "reduce_scatter":
                    tncc_ok = bool(torch.allclose(xt_dst, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_dst, expected_nccl, atol=1e-4))
                elif collective == "broadcast":
                    tncc_ok = bool(torch.allclose(xt_view, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_view, expected_nccl, atol=1e-4))

                results.append(
                    {
                        "collective": collective,
                        "size_bytes": size_bytes,
                        "block_elements": block_elements,
                        "timing_budget": {
                            "warmup": case_warmup,
                            "iters": case_iters,
                            "operations_per_timed_sample": operations_per_sample,
                            "size_adaptive": bool(size_bytes >= _LARGE_MESSAGE_THRESHOLD_BYTES),
                            "timing_mode": config.timing_mode,
                        },
                        "tncc": {
                            **tncc_result,
                            "times_ms": tncc_times_ms,
                            "correct": tncc_ok,
                        },
                        "nccl": {
                            "times_ms": nccl_times_ms,
                            "correct": nccl_ok,
                        },
                    }
                )

        payload = {
            "rank": rank,
            "dtype": config.dtype_name,
            "world_size": config.world_size,
            "warmup": config.warmup,
            "iters": config.iters,
            "force_transport": config.force_transport or "auto",
            "root": config.root,
            "transport_strategy": heap.transport_strategy,
            "heap_mode": heap.mode,
            "preconditioning": preconditioning if rank == 0 else None,
            "runtime_support": runtime_support_snapshot(ctx) if rank == 0 else None,
            "runtime_metadata": runtime_metadata_snapshot(ctx) if rank == 0 else None,
            "results": results,
        }
        out_path = Path(config.output_dir) / f"rank_{rank}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        failures = [
            {
                "collective": entry["collective"],
                "size_bytes": entry["size_bytes"],
                "tncc_correct": entry["tncc"]["correct"],
                "nccl_correct": entry["nccl"]["correct"],
            }
            for entry in results
            if not (entry["tncc"]["correct"] and entry["nccl"]["correct"])
        ]
        if failures:
            raise RuntimeError(
                f"collective correctness failed on rank {rank}: {json.dumps(failures, ensure_ascii=False)}"
            )
    except BaseException:
        error_path = Path(config.output_dir) / f"rank_{rank}_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.barrier(**barrier_kwargs)
            except Exception:
                pass
        try:
            if heap is not None:
                heap.cleanup()
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass


def _aggregate_rank_results(
    rank_payloads: list[dict[str, Any]],
    *,
    world_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for payload in rank_payloads:
        for result in payload["results"]:
            key = (str(result["collective"]), int(result["size_bytes"]))
            grouped.setdefault(key, []).append(result)

    cases: list[dict[str, Any]] = []
    peak_summary: dict[str, dict[str, float]] = {}
    for collective, size_bytes in sorted(
        grouped,
        key=lambda key: (_COLLECTIVES.index(key[0]), key[1]),
    ):
        per_rank = grouped[(collective, size_bytes)]
        tncc_execution_metadata = _shared_rank_metadata(
            per_rank,
            side="tncc",
            keys=(
                "implementation",
                "protocol",
                "kernel_family",
                "reuse_handshake",
                "message_bytes",
                "message_regime",
                "cta_policy",
                "epoch_policy",
                "chunk_elems",
                "num_chunks",
                "pipeline_slots",
                "grid_size",
                "num_warps",
                "workspace_bytes",
            ),
        )
        tncc_times_by_rank = [list(entry["tncc"]["times_ms"]) for entry in per_rank]
        nccl_times_by_rank = [list(entry["nccl"]["times_ms"]) for entry in per_rank]
        tncc_aggregate_times = [
            max(rank_times[idx] for rank_times in tncc_times_by_rank)
            for idx in range(len(tncc_times_by_rank[0]))
        ]
        nccl_aggregate_times = [
            max(rank_times[idx] for rank_times in nccl_times_by_rank)
            for idx in range(len(nccl_times_by_rank[0]))
        ]

        tncc_summary = _bandwidth_summary(
            collective=collective,
            size_bytes=size_bytes,
            world_size=world_size,
            times_ms=tncc_aggregate_times,
        )
        nccl_summary = _bandwidth_summary(
            collective=collective,
            size_bytes=size_bytes,
            world_size=world_size,
            times_ms=nccl_aggregate_times,
        )
        speedup = (
            tncc_summary["median_bandwidth_gbps"] / nccl_summary["median_bandwidth_gbps"]
            if nccl_summary["median_bandwidth_gbps"] > 0
            else 0.0
        )

        case = {
            "collective": collective,
            "size_bytes": size_bytes,
            "size_mib": float(size_bytes / (1024 ** 2)),
            "world_size": world_size,
            "timing_budget": per_rank[0].get("timing_budget"),
            "tncc": {
                **tncc_summary,
                **tncc_execution_metadata,
                "correct_all_ranks": all(bool(entry["tncc"]["correct"]) for entry in per_rank),
                "rank_times_ms": tncc_times_by_rank,
                "aggregate_times_ms": tncc_aggregate_times,
            },
            "nccl": {
                **nccl_summary,
                "correct_all_ranks": all(bool(entry["nccl"]["correct"]) for entry in per_rank),
                "rank_times_ms": nccl_times_by_rank,
                "aggregate_times_ms": nccl_aggregate_times,
            },
            "tncc_vs_nccl_bandwidth_ratio": float(speedup),
        }
        cases.append(case)

        summary_entry = peak_summary.setdefault(
            collective,
            {
                "peak_tncc_bandwidth_gbps": 0.0,
                "peak_nccl_bandwidth_gbps": 0.0,
                "best_tncc_vs_nccl_ratio": 0.0,
            },
        )
        summary_entry["peak_tncc_bandwidth_gbps"] = max(
            summary_entry["peak_tncc_bandwidth_gbps"],
            case["tncc"]["median_bandwidth_gbps"],
        )
        summary_entry["peak_nccl_bandwidth_gbps"] = max(
            summary_entry["peak_nccl_bandwidth_gbps"],
            case["nccl"]["median_bandwidth_gbps"],
        )
        summary_entry["best_tncc_vs_nccl_ratio"] = max(
            summary_entry["best_tncc_vs_nccl_ratio"],
            case["tncc_vs_nccl_bandwidth_ratio"],
        )

    best_case = max(cases, key=lambda item: item["tncc_vs_nccl_bandwidth_ratio"]) if cases else None
    summary: dict[str, Any] = {
        "peak_by_collective": peak_summary,
    }
    if best_case is not None:
        summary["best_tncc_vs_nccl_case"] = {
            "collective": best_case["collective"],
            "size_bytes": best_case["size_bytes"],
            "size_mib": best_case["size_mib"],
            "ratio": best_case["tncc_vs_nccl_bandwidth_ratio"],
        }
    return cases, summary


def _shared_rank_metadata(
    per_rank: list[dict[str, Any]],
    *,
    side: str,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Return rank-invariant metadata for one aggregated benchmark case."""
    if not per_rank:
        return {}

    result: dict[str, Any] = {}
    for key in keys:
        if key not in per_rank[0][side]:
            continue
        values = [entry[side].get(key) for entry in per_rank]
        if all(value == values[0] for value in values[1:]):
            result[key] = values[0]
        else:
            result[f"{key}_per_rank"] = values
    return result


def main() -> None:
    args = _parse_args()
    world_size = min(torch.cuda.device_count(), int(args.world_size))
    if world_size < 2:
        raise SystemExit("Need >= 2 GPUs")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")

    dtype = _resolve_dtype(args.dtype)
    collectives = _parse_collectives(args.collectives)
    message_sizes_bytes = _parse_message_sizes(
        args.message_sizes,
        element_size=torch.tensor([], dtype=dtype).element_size(),
        world_size=world_size,
    )

    max_message_bytes = max(message_sizes_bytes)
    heap_size_bytes = max(512 * 1024 * 1024, max_message_bytes * 12)

    config = _RunConfig(
        collectives=collectives,
        message_sizes_bytes=message_sizes_bytes,
        dtype_name=args.dtype,
        timing_mode=str(args.timing_mode),
        warmup=int(args.warmup),
        iters=int(args.iters),
        world_size=world_size,
        force_transport=None if args.force_transport == "auto" else args.force_transport,
        root=0,
        heap_size_bytes=heap_size_bytes,
        output_dir="",
    )
    environment_health = benchmark_environment_health(
        visible_gpu_count=world_size,
    )
    emit_benchmark_environment_warnings(environment_health)

    output_path = Path(args.output_json)
    store_fd, store_path = tempfile.mkstemp(prefix="tncc_collective_comm_store_")
    os.close(store_fd)
    os.unlink(store_path)

    with tempfile.TemporaryDirectory(prefix="tncc_collective_comm_rank_") as temp_output_dir:
        run_config = _RunConfig(
            collectives=config.collectives,
            message_sizes_bytes=config.message_sizes_bytes,
            dtype_name=config.dtype_name,
            timing_mode=config.timing_mode,
            warmup=config.warmup,
            iters=config.iters,
            world_size=config.world_size,
            force_transport=config.force_transport,
            root=config.root,
            heap_size_bytes=config.heap_size_bytes,
            output_dir=temp_output_dir,
        )
        with canonical_benchmark_run(output_path):
            try:
                mp.start_processes(
                    _worker,
                    args=(store_path, run_config),
                    nprocs=world_size,
                    join=True,
                    start_method="spawn",
                )
            except Exception as exc:
                error_messages: list[str] = []
                for rank in range(world_size):
                    error_path = Path(temp_output_dir) / f"rank_{rank}_error.txt"
                    if error_path.exists():
                        error_messages.append(
                            f"--- rank {rank} ---\n{error_path.read_text(encoding='utf-8')}"
                        )
                if error_messages:
                    raise RuntimeError("\n".join(error_messages)) from exc
                raise
            finally:
                try:
                    os.unlink(store_path)
                except FileNotFoundError:
                    pass

            rank_payloads = [
                json.loads(
                    (Path(temp_output_dir) / f"rank_{rank}.json").read_text(encoding="utf-8")
                )
                for rank in range(world_size)
            ]

    cases, summary = _aggregate_rank_results(rank_payloads, world_size=world_size)
    runtime_support = next(
        (
            payload["runtime_support"]
            for payload in rank_payloads
            if isinstance(payload.get("runtime_support"), dict)
        ),
        None,
    )
    runtime_metadata = next(
        (
            payload["runtime_metadata"]
            for payload in rank_payloads
            if isinstance(payload.get("runtime_metadata"), dict)
        ),
        None,
    )
    preconditioning = next(
        (
            payload["preconditioning"]
            for payload in rank_payloads
            if isinstance(payload.get("preconditioning"), dict)
        ),
        None,
    )
    transport_strategy = next(
        (payload["transport_strategy"] for payload in rank_payloads if payload.get("transport_strategy")),
        None,
    )

    payload = {
        "schema_version": 1,
        "benchmark": "collective_comm_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(),
            "world_size": world_size,
            "collectives": list(collectives),
            "dtype": args.dtype,
            "timing_mode": str(args.timing_mode),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "preconditioning": preconditioning,
            "message_sizes_bytes": list(message_sizes_bytes),
            "transport_strategy": transport_strategy,
            "latency_measurement": (
                "host_wall_end_to_end_with_cuda_completion"
                if args.timing_mode == "host_wall"
                else "device_event_collective_completion"
            ),
            "sampling_policy": {
                "mode": "size_adaptive",
                "base_warmup": int(args.warmup),
                "base_iters": int(args.iters),
                "tiny_message_batch_threshold_bytes": _TINY_MESSAGE_BATCH_THRESHOLD_BYTES,
                "small_message_batch_threshold_bytes": _SMALL_MESSAGE_BATCH_THRESHOLD_BYTES,
                "medium_message_batch_threshold_bytes": _MEDIUM_MESSAGE_BATCH_THRESHOLD_BYTES,
                "tiny_message_batch_repeats": _TINY_MESSAGE_BATCH_REPEATS,
                "small_message_batch_repeats": _SMALL_MESSAGE_BATCH_REPEATS,
                "medium_message_batch_repeats": _MEDIUM_MESSAGE_BATCH_REPEATS,
                "large_message_batch_repeats": _LARGE_MESSAGE_BATCH_REPEATS,
                "very_large_message_batch_repeats": _VERY_LARGE_MESSAGE_BATCH_REPEATS,
                "large_message_threshold_bytes": _LARGE_MESSAGE_THRESHOLD_BYTES,
                "very_large_message_threshold_bytes": _VERY_LARGE_MESSAGE_THRESHOLD_BYTES,
                "large_message_warmup_cap": _LARGE_MESSAGE_WARMUP_CAP,
                "large_message_iters_cap": _LARGE_MESSAGE_ITERS_CAP,
                "very_large_message_warmup_cap": _VERY_LARGE_MESSAGE_WARMUP_CAP,
                "very_large_message_iters_cap": _VERY_LARGE_MESSAGE_ITERS_CAP,
            },
        },
        "bandwidth_definition": {
            "allreduce": "2*(world_size-1)/world_size * rank_local_bytes / latency",
            "allgather": "(world_size-1) * rank_local_input_bytes / latency",
            "reduce_scatter": "(world_size-1) * rank_local_output_bytes / latency",
            "scatter": "(world_size-1) * rank_local_output_bytes / latency",
            "broadcast": "(world_size-1) * tensor_bytes / latency",
        },
        "runtime_support": runtime_support,
        "runtime_metadata": runtime_metadata,
        "environment_health": environment_health,
        "rank_payloads": rank_payloads,
        "cases": cases,
        "summary": summary,
    }
    written = write_json(output_path, payload)
    print(f"Structured results written to: {written}", flush=True)
    for collective in collectives:
        collective_cases = [case for case in cases if case["collective"] == collective]
        if not collective_cases:
            continue
        best_case = max(
            collective_cases,
            key=lambda item: item["tncc"]["median_bandwidth_gbps"],
        )
        print(
            f"[{collective}] "
            f"best_tncc={best_case['tncc']['median_bandwidth_gbps']:.2f} GB/s "
            f"best_nccl={best_case['nccl']['median_bandwidth_gbps']:.2f} GB/s "
            f"best_ratio={best_case['tncc_vs_nccl_bandwidth_ratio']:.3f}x "
            f"size={best_case['size_mib']:.4g} MiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
