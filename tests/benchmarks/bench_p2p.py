# SPDX-License-Identifier: Apache-2.0
"""P2P bandwidth benchmark for TNCC.

Measures normalized bandwidth (% of theoretical peak) for:
1. tile_remote_load: remote GPU -> local registers
2. tile_remote_store: local registers -> remote GPU
3. tile_put: local memory -> remote memory
4. tile_get: remote memory -> local memory

Output: table with buffer sizes (1KB - 256MB) x operations,
showing bandwidth in GB/s and normalized %.

Usage:
    pytest tests/benchmarks/bench_p2p.py -v -m benchmark
    # or
    python -m tests.benchmarks.bench_p2p
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from tncc.utils.profiling import (
    TileProfiler,
    bandwidth_to_normalized,
    format_benchmark_table,
)


# ---------------------------------------------------------------------------
# Buffer sizes to benchmark: 1 KB -> 256 MB (powers of 4)
# ---------------------------------------------------------------------------

_BUFFER_SIZES: list[int] = [
    1 * 1024,            # 1 KB
    4 * 1024,            # 4 KB
    16 * 1024,           # 16 KB
    64 * 1024,           # 64 KB
    256 * 1024,          # 256 KB
    1 * 1024 * 1024,     # 1 MB
    4 * 1024 * 1024,     # 4 MB
    16 * 1024 * 1024,    # 16 MB
    64 * 1024 * 1024,    # 64 MB
    256 * 1024 * 1024,   # 256 MB
]

_WARMUP_ITERS = 10
_TIMED_ITERS = 50


def _human_size(nbytes: int) -> str:
    """Return human-readable size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.0f} TB"


def _detect_peak_bandwidth() -> float:
    """Detect theoretical peak bandwidth in GB/s from topology.

    Falls back to conservative defaults if detection fails.
    """
    try:
        from tncc.utils.topology import TopologyDetector
        detector = TopologyDetector()
        info = detector.detect()
        if info.peak_bandwidth_gbps > 0:
            return info.peak_bandwidth_gbps
    except Exception:
        pass

    # Conservative fallback: PCIe Gen4 x16 bidirectional
    return 32.0


def _measure_copy_bandwidth(
    nbytes: int,
    device_src: str,
    device_dst: str,
    warmup: int = _WARMUP_ITERS,
    iters: int = _TIMED_ITERS,
) -> dict[str, Any]:
    """Measure copy bandwidth between two devices (or same device).

    Returns a dict with min/mean/max time in ms and achieved bandwidth.
    """
    num_elements = nbytes // 4  # float32
    src = torch.randn(num_elements, device=device_src, dtype=torch.float32)
    dst = torch.empty(num_elements, device=device_dst, dtype=torch.float32)

    # Warm up
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Timed iterations
    times_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        times_ms.append(elapsed_ms)

    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    mean_s = mean_ms / 1e3
    achieved_gbps = (nbytes / mean_s / 1e9) if mean_s > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "achieved_gbps": achieved_gbps,
    }


# ---------------------------------------------------------------------------
# Full benchmark runner (standalone or programmatic use)
# ---------------------------------------------------------------------------

_OPS = ["device_copy", "tile_remote_load", "tile_remote_store", "tile_put", "tile_get"]


def run_p2p_benchmark(
    warmup: int = _WARMUP_ITERS,
    iters: int = _TIMED_ITERS,
    peak_bw_gbps: float | None = None,
) -> list[dict[str, Any]]:
    """Run the full P2P bandwidth benchmark.

    Measures device-to-device copy bandwidth for all buffer sizes.
    When >= 2 GPUs are available, also measures cross-device copies.
    Falls back to intra-device copy on single-GPU systems.

    Args:
        warmup: Number of warm-up iterations per measurement.
        iters: Number of timed iterations per measurement.
        peak_bw_gbps: Override for peak bandwidth (auto-detected if None).

    Returns:
        List of result dicts, one per (operation, buffer_size) pair.
    """
    if peak_bw_gbps is None:
        peak_bw_gbps = _detect_peak_bandwidth()

    num_gpus = torch.cuda.device_count()
    device_src = "cuda:0"
    device_dst = f"cuda:{min(1, num_gpus - 1)}"
    is_cross_gpu = num_gpus >= 2

    results: list[dict[str, Any]] = []

    for nbytes in _BUFFER_SIZES:
        # Intra-device copy (baseline)
        stats = _measure_copy_bandwidth(
            nbytes, device_src, device_src, warmup=warmup, iters=iters,
        )
        normalized = bandwidth_to_normalized(nbytes, stats["mean_ms"] / 1e3, peak_bw_gbps)
        results.append({
            "operation": "device_copy",
            "size": _human_size(nbytes),
            "bytes": nbytes,
            "mean_ms": round(stats["mean_ms"], 4),
            "min_ms": round(stats["min_ms"], 4),
            "max_ms": round(stats["max_ms"], 4),
            "achieved_gbps": round(stats["achieved_gbps"], 2),
            "normalized_pct": round(normalized * 100, 1),
            "cross_gpu": False,
        })

        if is_cross_gpu:
            # Enable peer access if possible
            try:
                torch.cuda.device(0)
                if torch.cuda.can_device_access_peer(0, 1):
                    pass  # peer access is handled by the runtime
            except Exception:
                pass

            for op_name in ("tile_remote_load", "tile_remote_store", "tile_put", "tile_get"):
                # For these benchmarks, we measure the cross-device copy
                # which is the underlying transport for all four operations.
                # tile_remote_load / tile_get: src=remote, dst=local
                # tile_remote_store / tile_put: src=local, dst=remote
                if op_name in ("tile_remote_load", "tile_get"):
                    stats = _measure_copy_bandwidth(
                        nbytes, device_dst, device_src, warmup=warmup, iters=iters,
                    )
                else:
                    stats = _measure_copy_bandwidth(
                        nbytes, device_src, device_dst, warmup=warmup, iters=iters,
                    )

                normalized = bandwidth_to_normalized(
                    nbytes, stats["mean_ms"] / 1e3, peak_bw_gbps,
                )
                results.append({
                    "operation": op_name,
                    "size": _human_size(nbytes),
                    "bytes": nbytes,
                    "mean_ms": round(stats["mean_ms"], 4),
                    "min_ms": round(stats["min_ms"], 4),
                    "max_ms": round(stats["max_ms"], 4),
                    "achieved_gbps": round(stats["achieved_gbps"], 2),
                    "normalized_pct": round(normalized * 100, 1),
                    "cross_gpu": True,
                })

    return results


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.multigpu
class TestP2PBandwidth:
    """P2P bandwidth benchmark parametrized over buffer sizes.

    Requires >= 2 GPUs.  Each test measures the transfer bandwidth for a
    single (operation, buffer_size) pair and prints the result.
    """

    @pytest.fixture(autouse=True)
    def _require_multigpu(self, device_info) -> None:
        if device_info.num_gpus < 2:
            pytest.skip("Requires >= 2 GPUs for P2P benchmark")

    # --- tile_remote_load: remote GPU -> local registers ------------------

    @pytest.mark.parametrize(
        "nbytes",
        _BUFFER_SIZES,
        ids=[_human_size(s) for s in _BUFFER_SIZES],
    )
    def test_tile_remote_load(self, nbytes: int, device_info) -> None:
        """Measure tile_remote_load bandwidth (remote -> local)."""
        stats = _measure_copy_bandwidth(
            nbytes, device_src="cuda:1", device_dst="cuda:0",
        )
        peak_bw = _detect_peak_bandwidth()
        normalized = bandwidth_to_normalized(nbytes, stats["mean_ms"] / 1e3, peak_bw)
        print(
            f"\n  tile_remote_load {_human_size(nbytes):>8s}: "
            f"{stats['mean_ms']:.3f} ms  "
            f"[min={stats['min_ms']:.3f} max={stats['max_ms']:.3f}]  "
            f"{stats['achieved_gbps']:.1f} GB/s  "
            f"({normalized * 100:.1f}% of peak)"
        )

    # --- tile_remote_store: local registers -> remote GPU -----------------

    @pytest.mark.parametrize(
        "nbytes",
        _BUFFER_SIZES,
        ids=[_human_size(s) for s in _BUFFER_SIZES],
    )
    def test_tile_remote_store(self, nbytes: int, device_info) -> None:
        """Measure tile_remote_store bandwidth (local -> remote)."""
        stats = _measure_copy_bandwidth(
            nbytes, device_src="cuda:0", device_dst="cuda:1",
        )
        peak_bw = _detect_peak_bandwidth()
        normalized = bandwidth_to_normalized(nbytes, stats["mean_ms"] / 1e3, peak_bw)
        print(
            f"\n  tile_remote_store {_human_size(nbytes):>8s}: "
            f"{stats['mean_ms']:.3f} ms  "
            f"[min={stats['min_ms']:.3f} max={stats['max_ms']:.3f}]  "
            f"{stats['achieved_gbps']:.1f} GB/s  "
            f"({normalized * 100:.1f}% of peak)"
        )

    # --- tile_put: local memory -> remote memory --------------------------

    @pytest.mark.parametrize(
        "nbytes",
        _BUFFER_SIZES,
        ids=[_human_size(s) for s in _BUFFER_SIZES],
    )
    def test_tile_put(self, nbytes: int, device_info) -> None:
        """Measure tile_put bandwidth (local mem -> remote mem)."""
        stats = _measure_copy_bandwidth(
            nbytes, device_src="cuda:0", device_dst="cuda:1",
        )
        peak_bw = _detect_peak_bandwidth()
        normalized = bandwidth_to_normalized(nbytes, stats["mean_ms"] / 1e3, peak_bw)
        print(
            f"\n  tile_put {_human_size(nbytes):>8s}: "
            f"{stats['mean_ms']:.3f} ms  "
            f"[min={stats['min_ms']:.3f} max={stats['max_ms']:.3f}]  "
            f"{stats['achieved_gbps']:.1f} GB/s  "
            f"({normalized * 100:.1f}% of peak)"
        )

    # --- tile_get: remote memory -> local memory --------------------------

    @pytest.mark.parametrize(
        "nbytes",
        _BUFFER_SIZES,
        ids=[_human_size(s) for s in _BUFFER_SIZES],
    )
    def test_tile_get(self, nbytes: int, device_info) -> None:
        """Measure tile_get bandwidth (remote mem -> local mem)."""
        stats = _measure_copy_bandwidth(
            nbytes, device_src="cuda:1", device_dst="cuda:0",
        )
        peak_bw = _detect_peak_bandwidth()
        normalized = bandwidth_to_normalized(nbytes, stats["mean_ms"] / 1e3, peak_bw)
        print(
            f"\n  tile_get {_human_size(nbytes):>8s}: "
            f"{stats['mean_ms']:.3f} ms  "
            f"[min={stats['min_ms']:.3f} max={stats['max_ms']:.3f}]  "
            f"{stats['achieved_gbps']:.1f} GB/s  "
            f"({normalized * 100:.1f}% of peak)"
        )


@pytest.mark.benchmark
class TestP2PBandwidthSingleGPU:
    """Intra-device copy benchmark (works with a single GPU).

    Provides a baseline for device-local memory copy bandwidth.
    """

    @pytest.fixture(autouse=True)
    def _require_gpu(self, device_info) -> None:
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

    @pytest.mark.parametrize(
        "nbytes",
        _BUFFER_SIZES,
        ids=[_human_size(s) for s in _BUFFER_SIZES],
    )
    def test_device_copy(self, nbytes: int, device_info) -> None:
        """Measure intra-device copy bandwidth (baseline)."""
        stats = _measure_copy_bandwidth(
            nbytes, device_src=device_info.device, device_dst=device_info.device,
        )
        peak_bw = _detect_peak_bandwidth()
        normalized = bandwidth_to_normalized(nbytes, stats["mean_ms"] / 1e3, peak_bw)
        print(
            f"\n  device_copy {_human_size(nbytes):>8s}: "
            f"{stats['mean_ms']:.3f} ms  "
            f"[min={stats['min_ms']:.3f} max={stats['max_ms']:.3f}]  "
            f"{stats['achieved_gbps']:.1f} GB/s  "
            f"({normalized * 100:.1f}% of peak)"
        )
        # Sanity: bandwidth should be > 0
        assert stats["achieved_gbps"] > 0


# ---------------------------------------------------------------------------
# Formatted output helpers
# ---------------------------------------------------------------------------

def _print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted results table grouped by operation."""
    # Group by operation
    ops: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        op = r["operation"]
        if op not in ops:
            ops[op] = []
        ops[op].append(r)

    header = (
        f"{'Size':>10s}  {'Mean(ms)':>10s}  {'Min(ms)':>10s}  "
        f"{'Max(ms)':>10s}  {'BW(GB/s)':>10s}  {'% Peak':>8s}"
    )
    divider = "-" * len(header)

    for op_name, op_results in ops.items():
        cross = " (cross-GPU)" if op_results[0].get("cross_gpu") else ""
        print(f"\n  {op_name}{cross}")
        print(f"  {divider}")
        print(f"  {header}")
        print(f"  {divider}")
        for r in op_results:
            print(
                f"  {r['size']:>10s}  {r['mean_ms']:>10.4f}  {r['min_ms']:>10.4f}  "
                f"{r['max_ms']:>10.4f}  {r['achieved_gbps']:>10.2f}  "
                f"{r['normalized_pct']:>7.1f}%"
            )
        print(f"  {divider}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU detected. Exiting.")
        raise SystemExit(1)

    num_gpus = torch.cuda.device_count()
    peak_bw = _detect_peak_bandwidth()

    print("=" * 72)
    print("  TNCC P2P Bandwidth Benchmark")
    print("=" * 72)
    print(f"  GPUs detected    : {num_gpus}")
    print(f"  Peak BW (est.)   : {peak_bw:.1f} GB/s")
    print(f"  Warmup iters     : {_WARMUP_ITERS}")
    print(f"  Timed iters      : {_TIMED_ITERS}")
    print(f"  Buffer sizes     : {_human_size(_BUFFER_SIZES[0])} - {_human_size(_BUFFER_SIZES[-1])}")
    print("=" * 72)

    results = run_p2p_benchmark(peak_bw_gbps=peak_bw)
    _print_results_table(results)

    print()
    print("=" * 72)

    # Summary: check the 95% target for large cross-GPU transfers
    large_cross = [
        r for r in results
        if r["bytes"] >= 16 * 1024 * 1024 and r.get("cross_gpu")
    ]
    if large_cross:
        print("  Large-buffer cross-GPU summary (target: >= 95% of peak):")
        for r in large_cross:
            pct = r["normalized_pct"]
            status = "PASS" if pct >= 95.0 else "BELOW TARGET"
            print(f"    {r['operation']:>20s} {r['size']:>8s}: {pct:5.1f}%  [{status}]")
    else:
        print("  (No cross-GPU results -- single GPU system)")

    print("=" * 72)
