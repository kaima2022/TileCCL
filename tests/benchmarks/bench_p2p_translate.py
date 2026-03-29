#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P2P bandwidth microbenchmark using translate_ptr.

Measures actual NVLink bandwidth achieved by translate_ptr-based remote
reads/writes at various block sizes and cache modifiers, and compares
against the theoretical peak.

Optimizations tested:
- Cache modifiers: .cg (cache-global, bypass L1 for reads),
  .wt (write-through, bypass L2 for writes)
- Eviction policies: evict_first for streaming data
- BLOCK_SIZE sweep: 1024, 2048, 4096, 8192
- Grid size sweep: num_sms, num_sms * 2

Usage:
    python3 tests/benchmarks/bench_p2p_translate.py
    python3 tests/benchmarks/bench_p2p_translate.py --quick   # fast mode

Requires >= 2 GPUs with peer access.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.memory.translation import translate_ptr
from tncc.utils.benchmark_results import (
    canonical_benchmark_run,
    default_p2p_benchmark_path,
    describe_runtime_metadata_snapshot,
    describe_runtime_support_snapshot,
    write_json,
)


# ---------------------------------------------------------------------------
# Triton kernels -- baseline (no cache modifier)
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
    """Persistent kernel: reads from remote heap via translate_ptr (baseline)."""
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
    """Persistent kernel: writes to remote heap via translate_ptr (baseline)."""
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
# Triton kernels -- optimized with cache modifiers
# ---------------------------------------------------------------------------

@triton.jit
def _p2p_read_bw_cg_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    """Persistent read kernel with .cg cache modifier (bypass L1).

    Remote data is streaming (not reused locally), so bypassing L1
    avoids polluting the L1 cache and reduces cache thrashing.
    """
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
        data = tl.load(remote_ptr, mask=mask, cache_modifier=".cg")
        tl.store(local_ptr + offsets, data, mask=mask)


@triton.jit
def _p2p_write_bw_wt_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    """Persistent write kernel with .wt cache modifier (write-through).

    Remote writes bypass L2 cache on the way out, avoiding L2 pollution
    and allowing the NVLink to be fed directly.
    """
    pid = tl.program_id(0)
    num_blocks = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in range(pid, num_blocks, tl.num_programs(0)):
        start = block_id * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        data = tl.load(local_ptr + offsets, mask=mask)
        remote_ptr = translate_ptr(
            local_ptr + offsets, caller_rank, remote_rank, heap_bases,
            HINT=BLOCK_SIZE,
        )
        tl.store(remote_ptr, data, mask=mask, cache_modifier=".wt")


# ---------------------------------------------------------------------------
# Triton kernels -- optimized with eviction policy
# ---------------------------------------------------------------------------

@triton.jit
def _p2p_read_bw_evict_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    """Read kernel with evict_first policy for streaming remote loads."""
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
        data = tl.load(remote_ptr, mask=mask, eviction_policy="evict_first")
        tl.store(local_ptr + offsets, data, mask=mask)


@triton.jit
def _p2p_write_bw_evict_kernel(
    local_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    """Write kernel with evict_first for streaming local loads."""
    pid = tl.program_id(0)
    num_blocks = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in range(pid, num_blocks, tl.num_programs(0)):
        start = block_id * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        data = tl.load(local_ptr + offsets, mask=mask, eviction_policy="evict_first")
        remote_ptr = translate_ptr(
            local_ptr + offsets, caller_rank, remote_rank, heap_bases,
            HINT=BLOCK_SIZE,
        )
        tl.store(remote_ptr, data, mask=mask, cache_modifier=".wt")


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

_READ_KERNELS = {
    "baseline": _p2p_read_bw_kernel,
    "cg": _p2p_read_bw_cg_kernel,
    "evict_first": _p2p_read_bw_evict_kernel,
}

_WRITE_KERNELS = {
    "baseline": _p2p_write_bw_kernel,
    "wt": _p2p_write_bw_wt_kernel,
    "wt+evict": _p2p_write_bw_evict_kernel,
}


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
    variant: str


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse standalone P2P translate benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a reduced sweep")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(default_p2p_benchmark_path()),
        help="Structured JSON output path.",
    )
    return parser.parse_args(argv)


def _summarize_by_size(results: list[BenchResult], *, dtype: str = "float32") -> list[dict[str, object]]:
    """Summarize best read/write bandwidth per transfer size for one dtype."""
    sizes = sorted({result.size_bytes for result in results if result.dtype == dtype})
    summary: list[dict[str, object]] = []
    for size_bytes in sizes:
        subset = [
            result for result in results
            if result.dtype == dtype and result.size_bytes == size_bytes
        ]
        read = max((result for result in subset if result.direction == "read"), key=lambda item: item.bandwidth_gbps, default=None)
        write = max((result for result in subset if result.direction == "write"), key=lambda item: item.bandwidth_gbps, default=None)
        if read is None or write is None:
            continue
        summary.append({
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1e6, 1),
            "dtype": dtype,
            "best_read_gbps": round(read.bandwidth_gbps, 3),
            "best_write_gbps": round(write.bandwidth_gbps, 3),
            "best_read_variant": read.variant,
            "best_write_variant": write.variant,
            "best_read_block_size": read.block_size,
            "best_write_block_size": write.block_size,
            "best_read_grid": read.num_sms,
            "best_write_grid": write.num_sms,
        })
    return summary


def _p2p_payload(
    *,
    results: list[BenchResult],
    quick: bool,
    link: str,
    theoretical_peak: float,
    heaps: list[SymmetricHeap],
    runtime_support: dict[str, object],
    runtime_metadata: dict[str, object],
) -> dict[str, object]:
    """Build a structured P2P benchmark payload."""
    best_read = max((r for r in results if r.direction == "read"), key=lambda item: item.bandwidth_gbps)
    best_write = max((r for r in results if r.direction == "write"), key=lambda item: item.bandwidth_gbps)
    return {
        "schema_version": 1,
        "benchmark": "p2p_translate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(),
            "world_size": 2,
            "link": link,
            "theoretical_peak_gbps": theoretical_peak,
            "heap_mode": heaps[0].mode,
            "transport_strategy": heaps[0].transport_strategy,
            "quick_mode": quick,
        },
        "runtime_support": runtime_support,
        "runtime_metadata": runtime_metadata,
        "results": [
            {
                "direction": result.direction,
                "dtype": result.dtype,
                "size_bytes": result.size_bytes,
                "block_size": result.block_size,
                "bandwidth_gbps": result.bandwidth_gbps,
                "time_us": result.time_us,
                "num_sms": result.num_sms,
                "variant": result.variant,
            }
            for result in results
        ],
        "summary": {
            "best_read": {
                "bandwidth_gbps": best_read.bandwidth_gbps,
                "dtype": best_read.dtype,
                "size_bytes": best_read.size_bytes,
                "variant": best_read.variant,
                "block_size": best_read.block_size,
                "grid": best_read.num_sms,
            },
            "best_write": {
                "bandwidth_gbps": best_write.bandwidth_gbps,
                "dtype": best_write.dtype,
                "size_bytes": best_write.size_bytes,
                "variant": best_write.variant,
                "block_size": best_write.block_size,
                "grid": best_write.num_sms,
            },
            "float32_by_size": _summarize_by_size(results, dtype="float32"),
            "float16_by_size": _summarize_by_size(results, dtype="float16"),
        },
    }


def benchmark_p2p(
    heaps: list[SymmetricHeap],
    n_elements: int,
    block_size: int,
    dtype: torch.dtype,
    direction: str = "read",
    variant: str = "baseline",
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

    # Select kernel
    if direction == "read":
        kernel = _READ_KERNELS.get(variant, _p2p_read_bw_kernel)
    else:
        kernel = _WRITE_KERNELS.get(variant, _p2p_write_bw_kernel)

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
        variant=variant,
    )


def main():
    assert torch.cuda.device_count() >= 2, "Need >= 2 GPUs"

    args = _parse_args(sys.argv[1:])
    quick = args.quick
    with canonical_benchmark_run(args.output_json):
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

        theoretical_peak = 300.0  # GB/s
        print(f"=== TNCC P2P Bandwidth Benchmark (Optimized) ===")
        print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
        print(f"Link: {link}")
        print(f"Theoretical peak: {theoretical_peak:.1f} GB/s (per direction)")
        print()

        heap_size = 256 * 1024 * 1024  # 256 MB
        heaps = SymmetricHeap.create_all(size=heap_size, world_size=2)
        try:
            runtime_support = describe_runtime_support_snapshot(
                backend=getattr(heaps[0], "_backend_name", "auto"),
                rank=heaps[0].rank,
                world_size=heaps[0].world_size,
                heap=heaps[0],
                force_backend=True,
            )
            runtime_metadata = describe_runtime_metadata_snapshot(
                backend=getattr(heaps[0], "_backend_name", "auto"),
                rank=heaps[0].rank,
                world_size=heaps[0].world_size,
                heap=heaps[0],
                force_backend=True,
            )

            num_sms = torch.cuda.get_device_properties(0).multi_processor_count
            results: list[BenchResult] = []

            if quick:
                sizes_mb = [128]
                block_sizes = [4096]
                grid_scales = [1]
                dtypes = [torch.float32]
                read_variants = ["baseline", "cg", "evict_first"]
                write_variants = ["baseline", "wt", "wt+evict"]
            else:
                sizes_mb = [1, 4, 16, 64, 128]
                block_sizes = [1024, 2048, 4096, 8192]
                grid_scales = [1, 2]
                dtypes = [torch.float32, torch.float16]
                read_variants = ["baseline", "cg", "evict_first"]
                write_variants = ["baseline", "wt", "wt+evict"]

            best_read = None
            best_write = None

            for dtype in dtypes:
                el_size = torch.tensor([], dtype=dtype).element_size()
                print(f"--- dtype={dtype} ---")

                for size_mb in sizes_mb:
                    n_elements = size_mb * 1024 * 1024 // el_size

                    for bs in block_sizes:
                        for grid_scale in grid_scales:
                            gs = num_sms * grid_scale

                            for variant in read_variants:
                                for h in heaps:
                                    h._bump_offset = 0
                                    h._alloc_records.clear()

                                r = benchmark_p2p(
                                    heaps, n_elements, bs, dtype,
                                    direction="read", variant=variant,
                                    warmup=5, iters=50,
                                    num_sms=gs,
                                )
                                results.append(r)
                                pct = r.bandwidth_gbps / theoretical_peak * 100
                                if best_read is None or r.bandwidth_gbps > best_read.bandwidth_gbps:
                                    if r.size_bytes >= 16e6:
                                        best_read = r
                                print(
                                    f"  read  | {variant:14s} | {r.dtype:7s} | "
                                    f"{r.size_bytes/1e6:7.1f} MB | "
                                    f"BS={r.block_size:5d} | grid={gs:4d} | "
                                    f"{r.bandwidth_gbps:7.2f} GB/s | "
                                    f"{pct:5.1f}% peak | "
                                    f"{r.time_us:8.1f} us"
                                )

                            for variant in write_variants:
                                for h in heaps:
                                    h._bump_offset = 0
                                    h._alloc_records.clear()

                                r = benchmark_p2p(
                                    heaps, n_elements, bs, dtype,
                                    direction="write", variant=variant,
                                    warmup=5, iters=50,
                                    num_sms=gs,
                                )
                                results.append(r)
                                pct = r.bandwidth_gbps / theoretical_peak * 100
                                if best_write is None or r.bandwidth_gbps > best_write.bandwidth_gbps:
                                    if r.size_bytes >= 16e6:
                                        best_write = r
                                print(
                                    f"  write | {variant:14s} | {r.dtype:7s} | "
                                    f"{r.size_bytes/1e6:7.1f} MB | "
                                    f"BS={r.block_size:5d} | grid={gs:4d} | "
                                    f"{r.bandwidth_gbps:7.2f} GB/s | "
                                    f"{pct:5.1f}% peak | "
                                    f"{r.time_us:8.1f} us"
                                )

            print()
            print("=" * 80)
            print("=== Summary ===")
            if best_read:
                pct = best_read.bandwidth_gbps / theoretical_peak * 100
                print(
                    f"Best read:  {best_read.bandwidth_gbps:.2f} GB/s ({pct:.1f}% peak) "
                    f"[{best_read.variant}, BS={best_read.block_size}, "
                    f"grid={best_read.num_sms}, {best_read.dtype}]"
                )
            if best_write:
                pct = best_write.bandwidth_gbps / theoretical_peak * 100
                print(
                    f"Best write: {best_write.bandwidth_gbps:.2f} GB/s ({pct:.1f}% peak) "
                    f"[{best_write.variant}, BS={best_write.block_size}, "
                    f"grid={best_write.num_sms}, {best_write.dtype}]"
                )

            target_met = False
            if best_read and best_write:
                target_met = (best_read.bandwidth_gbps >= 285.0 and
                              best_write.bandwidth_gbps >= 285.0)
            print(f"\nTarget (≥95% = 285 GB/s): {'MET' if target_met else 'NOT MET'}")

            output_path = write_json(Path(args.output_json), _p2p_payload(
                results=results,
                quick=quick,
                link=link,
                theoretical_peak=theoretical_peak,
                heaps=heaps,
                runtime_support=runtime_support,
                runtime_metadata=runtime_metadata,
            ))
            print(f"Structured results written to: {output_path}")
        finally:
            for h in heaps:
                h.cleanup()


if __name__ == "__main__":
    main()
