#!/usr/bin/env python3
"""P2P bandwidth microbenchmark using translate_ptr.

Measures actual NVLink bandwidth achieved by translate_ptr-based remote
reads/writes at various block sizes, and compares against the theoretical
peak.

Usage:
    python3 tests/benchmarks/bench_p2p_translate.py

Requires >= 2 GPUs with peer access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from xtile.memory.symmetric_heap import SymmetricHeap
from xtile.memory.translation import translate_ptr


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _p2p_read_bw_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    """Persistent kernel: reads from remote heap via translate_ptr."""
    pid = tl.program_id(0)
    num_blocks = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in range(pid, num_blocks, tl.num_programs(0)):
        start = block_id * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        remote_ptr = translate_ptr(
            local_ptr + offsets, caller_rank, remote_rank, heap_bases,
            HINT=BLOCK_SIZE,
        )
        data = tl.load(remote_ptr, mask=mask)
        # Write to local memory to prevent dead code elimination
        tl.store(local_ptr + offsets, data, mask=mask)


@triton.jit
def _p2p_write_bw_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    """Persistent kernel: writes to remote heap via translate_ptr."""
    pid = tl.program_id(0)
    num_blocks = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in range(pid, num_blocks, tl.num_programs(0)):
        start = block_id * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        # Load from local
        data = tl.load(local_ptr + offsets, mask=mask)
        # Write to remote
        remote_ptr = translate_ptr(
            local_ptr + offsets, caller_rank, remote_rank, heap_bases,
            HINT=BLOCK_SIZE,
        )
        tl.store(remote_ptr, data, mask=mask)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    direction: str
    dtype: str
    size_bytes: int
    block_size: int
    bandwidth_gbps: float
    time_us: float
    num_sms: int


def benchmark_p2p(
    heaps: list[SymmetricHeap],
    n_elements: int,
    block_size: int,
    dtype: torch.dtype,
    direction: str = "read",
    warmup: int = 10,
    iters: int = 100,
    num_sms: int = 0,
) -> BenchResult:
    """Run P2P bandwidth benchmark."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    size_bytes = n_elements * element_size

    # Symmetric allocation
    tensors = []
    for rank in range(2):
        torch.cuda.set_device(rank)
        t = heaps[rank].allocate_tensor((n_elements,), dtype)
        t.fill_(float(rank + 1))
        tensors.append(t)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    # Determine number of SMs
    if num_sms <= 0:
        props = torch.cuda.get_device_properties(0)
        num_sms = props.multi_processor_count

    torch.cuda.set_device(0)
    bases = heaps[0].get_heap_bases()

    kernel = _p2p_read_bw_kernel if direction == "read" else _p2p_write_bw_kernel

    # Warmup
    for _ in range(warmup):
        kernel[(num_sms,)](
            tensors[0], bases, 0, 1, n_elements,
            BLOCK_SIZE=block_size,
        )
    torch.cuda.synchronize(0)

    # Timed iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        kernel[(num_sms,)](
            tensors[0], bases, 0, 1, n_elements,
            BLOCK_SIZE=block_size,
        )
        end_events[i].record()

    torch.cuda.synchronize(0)

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    mean_us = mean_ms * 1000.0

    # Bandwidth: bytes / time
    bandwidth_gbps = size_bytes / (min_ms * 1e-3) / 1e9

    return BenchResult(
        direction=direction,
        dtype=str(dtype).split(".")[-1],
        size_bytes=size_bytes,
        block_size=block_size,
        bandwidth_gbps=bandwidth_gbps,
        time_us=mean_us,
        num_sms=num_sms,
    )


def main():
    assert torch.cuda.device_count() >= 2, "Need >= 2 GPUs"

    # Get topology info
    link = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "GPU0" in line and "NV" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("NV"):
                        link = p
                        break
    except Exception:
        pass

    # H100 PCIe NV12: 12 × 25 GB/s = 300 GB/s per direction
    theoretical_peak = 300.0  # GB/s
    print(f"=== XTile P2P Bandwidth Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"Link: {link}")
    print(f"Theoretical peak: {theoretical_peak:.1f} GB/s (per direction)")
    print()

    # Heap size: enough for largest test
    heap_size = 256 * 1024 * 1024  # 256 MB
    heaps = SymmetricHeap.create_all(size=heap_size, world_size=2)

    results: list[BenchResult] = []

    # Test matrix
    sizes_mb = [1, 4, 16, 64, 128]
    block_sizes = [1024, 4096]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        el_size = torch.tensor([], dtype=dtype).element_size()
        for size_mb in sizes_mb:
            n_elements = size_mb * 1024 * 1024 // el_size
            for bs in block_sizes:
                for direction in ["read", "write"]:
                    # Reset heaps for each test (bump allocator)
                    for h in heaps:
                        h._bump_offset = 0
                        h._alloc_records.clear()

                    r = benchmark_p2p(
                        heaps, n_elements, bs, dtype,
                        direction=direction, warmup=5, iters=50,
                    )
                    results.append(r)
                    pct = r.bandwidth_gbps / theoretical_peak * 100
                    print(
                        f"  {direction:5s} | {r.dtype:7s} | "
                        f"{r.size_bytes/1e6:7.1f} MB | "
                        f"BS={r.block_size:5d} | "
                        f"{r.bandwidth_gbps:7.2f} GB/s | "
                        f"{pct:5.1f}% peak | "
                        f"{r.time_us:8.1f} us"
                    )

    # Summary
    print()
    print("=== Summary ===")
    read_results = [r for r in results if r.direction == "read" and r.size_bytes >= 16e6]
    write_results = [r for r in results if r.direction == "write" and r.size_bytes >= 16e6]

    if read_results:
        best_read = max(read_results, key=lambda r: r.bandwidth_gbps)
        print(f"Best read:  {best_read.bandwidth_gbps:.2f} GB/s "
              f"({best_read.bandwidth_gbps/theoretical_peak*100:.1f}% peak)")
    if write_results:
        best_write = max(write_results, key=lambda r: r.bandwidth_gbps)
        print(f"Best write: {best_write.bandwidth_gbps:.2f} GB/s "
              f"({best_write.bandwidth_gbps/theoretical_peak*100:.1f}% peak)")

    for h in heaps:
        h.cleanup()


if __name__ == "__main__":
    main()
