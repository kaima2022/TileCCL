#!/usr/bin/env python3
"""Real communication-only collective benchmark: XTile kernels vs NCCL.

This benchmark measures the pure communication collectives exposed by XTile's
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
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xtile.utils.benchmark_results import (
    canonical_benchmark_run,
    default_collective_comm_only_benchmark_path,
    runtime_metadata_snapshot,
    runtime_support_snapshot,
    write_json,
)
from xtile.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV


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


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one comm-only benchmark run."""

    message_sizes_bytes: tuple[int, ...]
    dtype_name: str
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
        help="Element dtype for both XTile and NCCL buffers.",
    )
    parser.add_argument(
        "--message-sizes",
        type=str,
        default="4096,65536",
        help="Comma-separated rank-local message sizes in bytes.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations per case.")
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
        help="Force one specific XTile multiprocess transport strategy.",
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


def _timed_collective(
    fn,
    *,
    prepare_fn,
    rank: int,
    barrier_kwargs: dict[str, object],
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        prepare_fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)
        fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)

    times_ms: list[float] = []
    for _ in range(iters):
        prepare_fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)
        # Measure the public collective as the caller observes it:
        # launch/rendezvous plus device completion on this rank.
        start_ns = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize(rank)
        end_ns = time.perf_counter_ns()
        dist.barrier(**barrier_kwargs)
        times_ms.append(float(end_ns - start_ns) / 1_000_000.0)
    return times_ms


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

        import xtile
        from xtile.memory.symmetric_heap import SymmetricHeap
        from xtile.primitives import (
            allgather as primitive_allgather,
            allreduce as primitive_allreduce,
            broadcast as primitive_broadcast,
            reduce_scatter as primitive_reduce_scatter,
            scatter as primitive_scatter,
        )

        heap = SymmetricHeap(
            size=config.heap_size_bytes,
            rank=rank,
            world_size=config.world_size,
            backend="cuda",
        )
        ctx = xtile.init(
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

        results: list[dict[str, Any]] = []
        for collective in _COLLECTIVES:
            for size_bytes in config.message_sizes_bytes:
                block_elements = _block_elements(
                    collective=collective,
                    size_bytes=size_bytes,
                    dtype=dtype,
                    world_size=config.world_size,
                )
                allreduce_plan = None

                if collective == "allreduce":
                    total_elements = block_elements
                    xt_view = xt_allreduce.narrow(0, 0, total_elements)
                    nccl_view = nccl_allreduce.narrow(0, 0, total_elements)
                    allreduce_plan = xtile.ops.build_allreduce_plan(xt_view, ctx=ctx)

                    def prepare_xtile() -> None:
                        _fill_allreduce_input(xt_view, rank=rank)

                    def prepare_nccl() -> None:
                        _fill_allreduce_input(nccl_view, rank=rank)

                    xtile_fn = lambda: primitive_allreduce(xt_view, heap)
                    nccl_fn = lambda: dist.all_reduce(nccl_view)
                    expected_xt = _expected_allreduce(xt_view, world_size=config.world_size)
                    expected_nccl = _expected_allreduce(nccl_view, world_size=config.world_size)

                elif collective == "allgather":
                    xt_src = xt_allgather_src.narrow(0, 0, block_elements)
                    xt_dst = xt_allgather_dst.narrow(0, 0, block_elements * config.world_size)
                    nccl_src = nccl_allgather_src.narrow(0, 0, block_elements)
                    nccl_dst = nccl_allgather_dst.narrow(0, 0, block_elements * config.world_size)

                    def prepare_xtile() -> None:
                        _fill_allgather_input(xt_src, xt_dst, rank=rank)

                    def prepare_nccl() -> None:
                        _fill_allgather_input(nccl_src, nccl_dst, rank=rank)

                    xtile_fn = lambda: primitive_allgather(xt_src, xt_dst, heap)
                    nccl_fn = lambda: dist.all_gather_into_tensor(nccl_dst, nccl_src)
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

                    def prepare_xtile() -> None:
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

                    xtile_fn = lambda: primitive_scatter(
                        xt_src,
                        xt_dst,
                        heap,
                        root=config.root,
                    )
                    nccl_fn = lambda: dist.scatter(
                        nccl_dst,
                        scatter_list=nccl_scatter_list,
                        src=config.root,
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

                    def prepare_xtile() -> None:
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

                    xtile_fn = lambda: primitive_reduce_scatter(
                        xt_src,
                        xt_dst,
                        heap,
                        implementation="device",
                    )
                    nccl_fn = lambda: dist.reduce_scatter_tensor(nccl_dst, nccl_src)
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

                    def prepare_xtile() -> None:
                        _fill_broadcast_input(xt_view, rank=rank, root=config.root)

                    def prepare_nccl() -> None:
                        _fill_broadcast_input(nccl_view, rank=rank, root=config.root)

                    xtile_fn = lambda: primitive_broadcast(xt_view, heap, root=config.root)
                    nccl_fn = lambda: dist.broadcast(nccl_view, src=config.root)
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

                xtile_result: dict[str, Any] = {
                    "times_ms": {},
                    "correct": False,
                }
                if allreduce_plan is not None:
                    xtile_result.update(
                        {
                            "implementation": allreduce_plan.implementation,
                            "protocol": allreduce_plan.protocol,
                            "chunk_elems": allreduce_plan.chunk_elems,
                            "num_chunks": allreduce_plan.num_chunks,
                            "pipeline_slots": allreduce_plan.pipeline_slots,
                            "grid_size": allreduce_plan.grid_size,
                            "num_warps": allreduce_plan.num_warps,
                            "workspace_bytes": allreduce_plan.workspace_bytes,
                        }
                    )

                xtile_times_ms = _timed_collective(
                    xtile_fn,
                    prepare_fn=prepare_xtile,
                    rank=rank,
                    barrier_kwargs=barrier_kwargs,
                    warmup=config.warmup,
                    iters=config.iters,
                )
                nccl_times_ms = _timed_collective(
                    nccl_fn,
                    prepare_fn=prepare_nccl,
                    rank=rank,
                    barrier_kwargs=barrier_kwargs,
                    warmup=config.warmup,
                    iters=config.iters,
                )

                if collective == "allreduce":
                    xtile_ok = bool(torch.allclose(xt_view, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_view, expected_nccl, atol=1e-4))
                elif collective == "allgather":
                    xtile_ok = bool(torch.allclose(xt_dst, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_dst, expected_nccl, atol=1e-4))
                elif collective == "scatter":
                    xtile_ok = bool(torch.allclose(xt_dst, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_dst, expected_nccl, atol=1e-4))
                elif collective == "reduce_scatter":
                    xtile_ok = bool(torch.allclose(xt_dst, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_dst, expected_nccl, atol=1e-4))
                elif collective == "broadcast":
                    xtile_ok = bool(torch.allclose(xt_view, expected_xt, atol=1e-4))
                    nccl_ok = bool(torch.allclose(nccl_view, expected_nccl, atol=1e-4))

                results.append(
                    {
                        "collective": collective,
                        "size_bytes": size_bytes,
                        "block_elements": block_elements,
                        "xtile": {
                            **xtile_result,
                            "times_ms": xtile_times_ms,
                            "correct": xtile_ok,
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
                "xtile_correct": entry["xtile"]["correct"],
                "nccl_correct": entry["nccl"]["correct"],
            }
            for entry in results
            if not (entry["xtile"]["correct"] and entry["nccl"]["correct"])
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
        xtile_execution_metadata = _shared_rank_metadata(
            per_rank,
            side="xtile",
            keys=(
                "implementation",
                "protocol",
                "chunk_elems",
                "num_chunks",
                "pipeline_slots",
                "grid_size",
                "num_warps",
                "workspace_bytes",
            ),
        )
        xtile_times_by_rank = [list(entry["xtile"]["times_ms"]) for entry in per_rank]
        nccl_times_by_rank = [list(entry["nccl"]["times_ms"]) for entry in per_rank]
        xtile_aggregate_times = [
            max(rank_times[idx] for rank_times in xtile_times_by_rank)
            for idx in range(len(xtile_times_by_rank[0]))
        ]
        nccl_aggregate_times = [
            max(rank_times[idx] for rank_times in nccl_times_by_rank)
            for idx in range(len(nccl_times_by_rank[0]))
        ]

        xtile_summary = _bandwidth_summary(
            collective=collective,
            size_bytes=size_bytes,
            world_size=world_size,
            times_ms=xtile_aggregate_times,
        )
        nccl_summary = _bandwidth_summary(
            collective=collective,
            size_bytes=size_bytes,
            world_size=world_size,
            times_ms=nccl_aggregate_times,
        )
        speedup = (
            xtile_summary["median_bandwidth_gbps"] / nccl_summary["median_bandwidth_gbps"]
            if nccl_summary["median_bandwidth_gbps"] > 0
            else 0.0
        )

        case = {
            "collective": collective,
            "size_bytes": size_bytes,
            "size_mib": float(size_bytes / (1024 ** 2)),
            "world_size": world_size,
            "xtile": {
                **xtile_summary,
                **xtile_execution_metadata,
                "correct_all_ranks": all(bool(entry["xtile"]["correct"]) for entry in per_rank),
                "rank_times_ms": xtile_times_by_rank,
                "aggregate_times_ms": xtile_aggregate_times,
            },
            "nccl": {
                **nccl_summary,
                "correct_all_ranks": all(bool(entry["nccl"]["correct"]) for entry in per_rank),
                "rank_times_ms": nccl_times_by_rank,
                "aggregate_times_ms": nccl_aggregate_times,
            },
            "xtile_vs_nccl_bandwidth_ratio": float(speedup),
        }
        cases.append(case)

        summary_entry = peak_summary.setdefault(
            collective,
            {
                "peak_xtile_bandwidth_gbps": 0.0,
                "peak_nccl_bandwidth_gbps": 0.0,
                "best_xtile_vs_nccl_ratio": 0.0,
            },
        )
        summary_entry["peak_xtile_bandwidth_gbps"] = max(
            summary_entry["peak_xtile_bandwidth_gbps"],
            case["xtile"]["median_bandwidth_gbps"],
        )
        summary_entry["peak_nccl_bandwidth_gbps"] = max(
            summary_entry["peak_nccl_bandwidth_gbps"],
            case["nccl"]["median_bandwidth_gbps"],
        )
        summary_entry["best_xtile_vs_nccl_ratio"] = max(
            summary_entry["best_xtile_vs_nccl_ratio"],
            case["xtile_vs_nccl_bandwidth_ratio"],
        )

    best_case = max(cases, key=lambda item: item["xtile_vs_nccl_bandwidth_ratio"]) if cases else None
    summary: dict[str, Any] = {
        "peak_by_collective": peak_summary,
    }
    if best_case is not None:
        summary["best_xtile_vs_nccl_case"] = {
            "collective": best_case["collective"],
            "size_bytes": best_case["size_bytes"],
            "size_mib": best_case["size_mib"],
            "ratio": best_case["xtile_vs_nccl_bandwidth_ratio"],
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
    message_sizes_bytes = _parse_message_sizes(
        args.message_sizes,
        element_size=torch.tensor([], dtype=dtype).element_size(),
        world_size=world_size,
    )

    max_message_bytes = max(message_sizes_bytes)
    heap_size_bytes = max(512 * 1024 * 1024, max_message_bytes * 12)

    config = _RunConfig(
        message_sizes_bytes=message_sizes_bytes,
        dtype_name=args.dtype,
        warmup=int(args.warmup),
        iters=int(args.iters),
        world_size=world_size,
        force_transport=None if args.force_transport == "auto" else args.force_transport,
        root=0,
        heap_size_bytes=heap_size_bytes,
        output_dir="",
    )

    output_path = Path(args.output_json)
    store_fd, store_path = tempfile.mkstemp(prefix="xtile_collective_comm_store_")
    os.close(store_fd)
    os.unlink(store_path)

    with tempfile.TemporaryDirectory(prefix="xtile_collective_comm_rank_") as temp_output_dir:
        run_config = _RunConfig(
            message_sizes_bytes=config.message_sizes_bytes,
            dtype_name=config.dtype_name,
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
            "dtype": args.dtype,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "message_sizes_bytes": list(message_sizes_bytes),
            "transport_strategy": transport_strategy,
            "latency_measurement": "host_wall_end_to_end_with_cuda_completion",
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
        "rank_payloads": rank_payloads,
        "cases": cases,
        "summary": summary,
    }
    written = write_json(output_path, payload)
    print(f"Structured results written to: {written}", flush=True)
    for collective in _COLLECTIVES:
        collective_cases = [case for case in cases if case["collective"] == collective]
        if not collective_cases:
            continue
        best_case = max(
            collective_cases,
            key=lambda item: item["xtile"]["median_bandwidth_gbps"],
        )
        print(
            f"[{collective}] "
            f"best_xtile={best_case['xtile']['median_bandwidth_gbps']:.2f} GB/s "
            f"best_nccl={best_case['nccl']['median_bandwidth_gbps']:.2f} GB/s "
            f"best_ratio={best_case['xtile_vs_nccl_bandwidth_ratio']:.3f}x "
            f"size={best_case['size_mib']:.4g} MiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
