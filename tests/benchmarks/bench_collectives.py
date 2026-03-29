#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collective communication bandwidth benchmark.

Measures the bandwidth of allreduce, allgather, and broadcast operations
at various buffer sizes, and reports normalized bandwidth relative to
the theoretical NVLink peak.

Note: Ring-based collectives (allreduce, reduce_scatter) are cooperative
and require concurrent execution across GPUs.  This benchmark uses
per-device CUDA streams to launch kernels concurrently.

Usage:
    python3 tests/benchmarks/bench_collectives.py

Requires >= 2 GPUs with peer access.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.primitives.collectives import (
    tile_allreduce,
    tile_allgather,
    tile_broadcast,
)


# ---------------------------------------------------------------------------
# Wrapper kernels
# ---------------------------------------------------------------------------

@triton.jit
def _allreduce_kernel(
    data_ptr, heap_bases_ptr, rank, world_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_allreduce(
        data_ptr, offsets, rank, world_size, heap_bases_ptr,
        BLOCK_SIZE, op="sum",
    )


@triton.jit
def _allgather_kernel(
    src_ptr, dst_ptr, heap_bases_ptr, rank, world_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_allgather(
        src_ptr, dst_ptr, offsets, rank, world_size,
        heap_bases_ptr, BLOCK_SIZE,
    )


@triton.jit
def _broadcast_kernel(
    data_ptr, heap_bases_ptr, rank, world_size, root,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_broadcast(
        data_ptr, offsets, rank, world_size, root,
        heap_bases_ptr, BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class CollBenchResult:
    collective: str
    size_bytes: int
    bandwidth_gbps: float
    time_us: float
    normalized_bw: float  # fraction of peak


def _launch_on_streams(
    heaps, world_size, collective, tensors, block_size,
    streams, src_tensors=None, dst_tensors=None,
):
    """Launch collective kernels concurrently on per-device streams."""
    for rank in range(world_size):
        torch.cuda.set_device(rank)
        bases = heaps[rank].get_heap_bases()
        with torch.cuda.stream(streams[rank]):
            if collective == "allreduce":
                _allreduce_kernel[(1,)](
                    tensors[rank], bases, rank, world_size,
                    BLOCK_SIZE=block_size,
                )
            elif collective == "allgather":
                _allgather_kernel[(1,)](
                    src_tensors[rank], dst_tensors[rank], bases,
                    rank, world_size,
                    BLOCK_SIZE=block_size,
                )
            elif collective == "broadcast":
                _broadcast_kernel[(1,)](
                    tensors[rank], bases, rank, world_size, 0,
                    BLOCK_SIZE=block_size,
                )


def benchmark_collective(
    collective: str,
    heaps: list[SymmetricHeap],
    block_size: int,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    iters: int = 100,
    theoretical_peak: float = 300.0,
) -> CollBenchResult:
    """Benchmark a collective operation with concurrent multi-GPU execution."""
    world_size = len(heaps)
    el_size = torch.tensor([], dtype=dtype).element_size()

    # Create per-device streams for concurrent kernel launches
    streams = []
    for rank in range(world_size):
        torch.cuda.set_device(rank)
        streams.append(torch.cuda.Stream(device=rank))

    src_tensors = None
    dst_tensors = None

    if collective == "allreduce":
        total_elements = block_size * world_size
        size_bytes = total_elements * el_size

        tensors = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            t = heaps[rank].allocate_tensor((total_elements,), dtype)
            t.fill_(float(rank + 1))
            tensors.append(t)

    elif collective == "allgather":
        size_bytes = block_size * el_size

        tensors = None
        src_tensors = []
        dst_tensors = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            src = heaps[rank].allocate_tensor((block_size,), dtype)
            src.fill_(float(rank + 1))
            dst = heaps[rank].allocate_tensor((block_size * world_size,), dtype)
            dst.fill_(0.0)
            src_tensors.append(src)
            dst_tensors.append(dst)

    elif collective == "broadcast":
        size_bytes = block_size * el_size

        tensors = []
        for rank in range(world_size):
            torch.cuda.set_device(rank)
            t = heaps[rank].allocate_tensor((block_size,), dtype)
            t.fill_(float(rank + 1))
            tensors.append(t)

    else:
        raise ValueError(f"Unknown collective: {collective}")

    for rank in range(world_size):
        torch.cuda.synchronize(rank)

    # Warmup with concurrent launches
    for _ in range(warmup):
        _launch_on_streams(
            heaps, world_size, collective, tensors, block_size,
            streams, src_tensors, dst_tensors,
        )
    for rank in range(world_size):
        torch.cuda.synchronize(rank)

    # Timed iterations using device 0 events
    torch.cuda.set_device(0)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record(streams[0])
        _launch_on_streams(
            heaps, world_size, collective, tensors, block_size,
            streams, src_tensors, dst_tensors,
        )
        # Record end event on device 0's stream after all launches
        end_events[i].record(streams[0])

    torch.cuda.synchronize(0)
    for rank in range(1, world_size):
        torch.cuda.synchronize(rank)

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    min_ms = min(times_ms)
    mean_us = (sum(times_ms) / len(times_ms)) * 1000.0

    # Bandwidth calculation:
    # For allreduce: 2 * (N-1)/N * data_size (ring optimal)
    # For allgather: (N-1)/N * data_size
    # For broadcast: data_size
    if collective == "allreduce":
        algo_bytes = 2 * (world_size - 1) / world_size * size_bytes
    elif collective == "allgather":
        algo_bytes = (world_size - 1) / world_size * size_bytes * world_size
    else:
        algo_bytes = size_bytes

    bandwidth_gbps = algo_bytes / (min_ms * 1e-3) / 1e9
    normalized_bw = bandwidth_gbps / theoretical_peak

    return CollBenchResult(
        collective=collective,
        size_bytes=size_bytes,
        bandwidth_gbps=bandwidth_gbps,
        time_us=mean_us,
        normalized_bw=normalized_bw,
    )


def main():
    assert torch.cuda.device_count() >= 2, "Need >= 2 GPUs"

    theoretical_peak = 300.0  # GB/s
    print("=== TNCC Collective Bandwidth Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"Theoretical peak: {theoretical_peak:.1f} GB/s (per direction)")
    print()

    heap_size = 256 * 1024 * 1024  # 256 MB
    world_size = 2

    collectives = ["allgather", "broadcast", "allreduce"]

    # Block sizes: per-chunk element count (float32).
    # Keep moderate to avoid Triton register pressure.
    block_sizes_kb = [4, 16, 64, 256]

    heaps = SymmetricHeap.create_all(size=heap_size, world_size=world_size)
    try:
        for collective in collectives:
            print(f"--- {collective} ---")
            for size_kb in block_sizes_kb:
                # Reset bump allocator for each test
                for h in heaps:
                    h._bump_offset = 0
                    h._alloc_records.clear()
                try:
                    block_size = size_kb * 1024 // 4  # float32 elements
                    r = benchmark_collective(
                        collective, heaps, block_size,
                        warmup=5, iters=50,
                        theoretical_peak=theoretical_peak,
                    )
                    print(
                        f"  {r.size_bytes/1e6:7.1f} MB | "
                        f"{r.bandwidth_gbps:7.2f} GB/s | "
                        f"{r.normalized_bw*100:5.1f}% peak | "
                        f"{r.time_us:8.1f} us"
                    )
                except Exception as e:
                    import traceback
                    print(f"  {size_kb:5d} KB | ERROR: {e}")
                    traceback.print_exc()
    finally:
        for h in heaps:
            h.cleanup()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
