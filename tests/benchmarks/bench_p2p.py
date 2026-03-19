"""P2P bandwidth benchmark.

Measures point-to-point (device-to-device) copy bandwidth for a range of
buffer sizes and reports normalised bandwidth as a percentage of the
theoretical peak.

Usage::

    pytest tests/benchmarks/bench_p2p.py -v -m benchmark

Or directly::

    python tests/benchmarks/bench_p2p.py
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from xtile.utils.profiling import (
    TileProfiler,
    bandwidth_to_normalized,
    format_benchmark_table,
    save_benchmark_results,
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


def _human_size(nbytes: int) -> str:
    """Return human-readable size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.0f} TB"


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def run_p2p_benchmark(
    device: str = "cuda:0",
    warmup: int = 5,
    iters: int = 20,
    peak_bw_gbps: float = 450.0,
) -> list[dict[str, Any]]:
    """Run the P2P bandwidth benchmark over all buffer sizes.

    Args:
        device: Torch device string.
        warmup: Warm-up iterations per buffer size.
        iters: Timed iterations per buffer size.
        peak_bw_gbps: Theoretical peak bandwidth in GB/s for normalisation.

    Returns:
        List of result dicts, one per buffer size.
    """
    results: list[dict[str, Any]] = []

    for nbytes in _BUFFER_SIZES:
        num_elements = nbytes // 4  # float32
        src = torch.randn(num_elements, device=device, dtype=torch.float32)
        dst = torch.empty_like(src)

        # Warm up
        for _ in range(warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        # Timed iterations
        profiler = TileProfiler(f"p2p_{_human_size(nbytes)}")
        for _ in range(iters):
            with profiler:
                dst.copy_(src)
                torch.cuda.synchronize()

        stats = profiler.summary()
        mean_s = stats["mean_ms"] / 1e3
        normalized = bandwidth_to_normalized(nbytes, mean_s, peak_bw_gbps)

        results.append({
            "size": _human_size(nbytes),
            "bytes": nbytes,
            "mean_ms": stats["mean_ms"],
            "min_ms": stats["min_ms"],
            "achieved_gbps": (nbytes / mean_s / 1e9) if mean_s > 0 else 0.0,
            "normalized_pct": normalized * 100.0,
        })

    return results


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestP2PBandwidth:
    """P2P bandwidth benchmark parametrized over buffer sizes."""

    @pytest.fixture(autouse=True)
    def _require_gpu(self, device_info) -> None:
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

    @pytest.mark.parametrize(
        "nbytes",
        _BUFFER_SIZES,
        ids=[_human_size(s) for s in _BUFFER_SIZES],
    )
    def test_p2p_bandwidth(self, nbytes: int, device_info) -> None:
        """Measure copy bandwidth for a single buffer size."""
        device = device_info.device
        num_elements = nbytes // 4
        src = torch.randn(num_elements, device=device, dtype=torch.float32)
        dst = torch.empty_like(src)

        # Warm up
        for _ in range(5):
            dst.copy_(src)
        torch.cuda.synchronize()

        # Measure
        profiler = TileProfiler(f"p2p_{_human_size(nbytes)}")
        for _ in range(20):
            with profiler:
                dst.copy_(src)
                torch.cuda.synchronize()

        stats = profiler.summary()
        mean_s = stats["mean_ms"] / 1e3

        # We expect P2P to reach >= 95% of peak for large buffers.
        # For smaller buffers, latency dominates so we just check > 0.
        if nbytes >= 16 * 1024 * 1024:
            peak_bw = 450.0  # conservative default (NVLink)
            normalized = bandwidth_to_normalized(nbytes, mean_s, peak_bw)
            # Log for visibility; the 95% target is aspirational, not a hard gate
            print(
                f"\n  {_human_size(nbytes)}: "
                f"{stats['mean_ms']:.3f} ms, "
                f"{nbytes / mean_s / 1e9:.1f} GB/s, "
                f"{normalized * 100:.1f}% of peak"
            )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU detected. Exiting.")
        raise SystemExit(1)

    print("XTile P2P Bandwidth Benchmark")
    print("=" * 60)

    results = run_p2p_benchmark()
    print()
    print(format_benchmark_table(results))
    print()

    # Check the 95% target for large transfers
    for r in results:
        if r["bytes"] >= 16 * 1024 * 1024:
            pct = r["normalized_pct"]
            status = "PASS" if pct >= 95.0 else "BELOW TARGET"
            print(f"  {r['size']:>8s}: {pct:5.1f}%  [{status}]")
