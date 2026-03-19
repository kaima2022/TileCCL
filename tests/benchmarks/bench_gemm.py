"""GEMM performance benchmark.

Compares xtile.kernels.gemm against torch.matmul across various
matrix sizes in float16 and bfloat16.

Target: >=90% of torch.matmul performance (which uses cuBLAS/hipBLAS).

Usage:
    pytest tests/benchmarks/bench_gemm.py -v -m benchmark
    # or
    python -m tests.benchmarks.bench_gemm
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MATRIX_SIZES: list[tuple[int, int, int]] = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

_DTYPES: list[torch.dtype] = [torch.float16, torch.bfloat16]

_WARMUP_ITERS = 10
_TIMED_ITERS = 20

_TARGET_RATIO = 0.90  # xtile gemm should reach >= 90% of torch.matmul


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_str(dtype: torch.dtype) -> str:
    """Short string for dtype."""
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(dtype, str(dtype))


def _compute_tflops(M: int, N: int, K: int, time_s: float) -> float:
    """Compute TFLOPS for a GEMM of size (M, N, K).

    FLOP count for C = A @ B is 2 * M * N * K (multiply-add).
    """
    if time_s <= 0:
        return 0.0
    flops = 2.0 * M * N * K
    return flops / time_s / 1e12


def _benchmark_kernel(
    fn,
    warmup: int = _WARMUP_ITERS,
    iters: int = _TIMED_ITERS,
) -> dict[str, float]:
    """Time a callable with warmup and return min/mean/max in ms.

    Uses CUDA events for precise GPU timing, avoiding Python overhead
    between event recording and kernel launch.

    Args:
        fn: Zero-argument callable that runs the kernel.
        warmup: Number of warm-up calls.
        iters: Number of timed calls.

    Returns:
        Dict with keys mean_ms, min_ms, max_ms.
    """
    # Warm up
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Use CUDA events for precise GPU-side timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def _run_gemm_comparison(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run xtile gemm and torch.matmul, return comparison metrics."""
    from xtile.kernels.gemm import gemm as xtile_gemm

    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    # Pre-allocate output tensors to eliminate allocation overhead during timing
    C_torch = torch.empty((M, N), device=device, dtype=dtype)
    C_xtile = torch.empty((M, N), device=device, dtype=dtype)

    # --- torch.matmul (cuBLAS / hipBLAS) ---
    torch_stats = _benchmark_kernel(lambda: torch.matmul(A, B, out=C_torch))
    torch_tflops = _compute_tflops(M, N, K, torch_stats["mean_ms"] / 1e3)

    # --- xtile gemm (with pre-allocated output) ---
    xtile_stats = _benchmark_kernel(lambda: xtile_gemm(A, B, C=C_xtile))
    xtile_tflops = _compute_tflops(M, N, K, xtile_stats["mean_ms"] / 1e3)

    # --- correctness check (relative tolerance) ---
    C_ref = torch.matmul(A, B)
    C_xtile = xtile_gemm(A, B)
    # Use relaxed tolerance for half-precision
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    atol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    correct = torch.allclose(C_ref, C_xtile, rtol=rtol, atol=atol)

    ratio = xtile_tflops / torch_tflops if torch_tflops > 0 else 0.0

    return {
        "M": M,
        "N": N,
        "K": K,
        "dtype": _dtype_str(dtype),
        "torch_ms": round(torch_stats["mean_ms"], 3),
        "torch_tflops": round(torch_tflops, 2),
        "xtile_ms": round(xtile_stats["mean_ms"], 3),
        "xtile_tflops": round(xtile_tflops, 2),
        "ratio": round(ratio, 4),
        "ratio_pct": round(ratio * 100, 1),
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_gemm_benchmark(device: str = "cuda:0") -> list[dict[str, Any]]:
    """Run the full GEMM benchmark across all sizes and dtypes.

    Returns:
        List of result dicts, one per (size, dtype) combination.
    """
    results: list[dict[str, Any]] = []
    for M, N, K in _MATRIX_SIZES:
        for dtype in _DTYPES:
            result = _run_gemm_comparison(M, N, K, dtype, device=device)
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------

# Build parametrize IDs
_SIZE_IDS = [f"{M}x{N}x{K}" for M, N, K in _MATRIX_SIZES]
_DTYPE_IDS = [_dtype_str(d) for d in _DTYPES]


@pytest.mark.benchmark
class TestGEMMPerformance:
    """GEMM benchmark comparing xtile.kernels.gemm vs torch.matmul."""

    @pytest.fixture(autouse=True)
    def _require_gpu(self, device_info) -> None:
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

    @pytest.mark.parametrize(
        "size",
        _MATRIX_SIZES,
        ids=_SIZE_IDS,
    )
    @pytest.mark.parametrize(
        "dtype",
        _DTYPES,
        ids=_DTYPE_IDS,
    )
    def test_gemm_performance(
        self, size: tuple[int, int, int], dtype: torch.dtype, device_info
    ) -> None:
        """Compare xtile gemm vs torch.matmul for a single (size, dtype)."""
        M, N, K = size
        result = _run_gemm_comparison(M, N, K, dtype, device=device_info.device)

        print(
            f"\n  GEMM {M}x{N}x{K} {result['dtype']}:"
            f"\n    torch.matmul : {result['torch_ms']:.3f} ms "
            f"({result['torch_tflops']:.2f} TFLOPS)"
            f"\n    xtile.gemm   : {result['xtile_ms']:.3f} ms "
            f"({result['xtile_tflops']:.2f} TFLOPS)"
            f"\n    ratio        : {result['ratio_pct']:.1f}%"
            f"\n    correct      : {result['correct']}"
        )

        # Correctness is a hard requirement
        assert result["correct"], (
            f"xtile gemm produced incorrect results for "
            f"{M}x{N}x{K} {result['dtype']}"
        )

    @pytest.mark.parametrize(
        "size",
        _MATRIX_SIZES,
        ids=_SIZE_IDS,
    )
    @pytest.mark.parametrize(
        "dtype",
        _DTYPES,
        ids=_DTYPE_IDS,
    )
    def test_gemm_meets_target(
        self, size: tuple[int, int, int], dtype: torch.dtype, device_info
    ) -> None:
        """Verify xtile gemm reaches >= 90% of torch.matmul TFLOPS.

        This test logs a warning rather than failing hard, since
        performance can vary with hardware and system load.
        """
        M, N, K = size
        result = _run_gemm_comparison(M, N, K, dtype, device=device_info.device)

        ratio = result["ratio"]
        target_met = ratio >= _TARGET_RATIO

        status = "PASS" if target_met else "BELOW TARGET"
        print(
            f"\n  [{status}] GEMM {M}x{N}x{K} {result['dtype']}: "
            f"{result['ratio_pct']:.1f}% of torch.matmul "
            f"(target: {_TARGET_RATIO * 100:.0f}%)"
        )

        if not target_met:
            import warnings
            warnings.warn(
                f"xtile gemm at {result['ratio_pct']:.1f}% of torch.matmul "
                f"for {M}x{N}x{K} {result['dtype']} "
                f"(target: {_TARGET_RATIO * 100:.0f}%)",
                stacklevel=1,
            )


@pytest.mark.benchmark
class TestGEMMCorrectness:
    """Correctness-only tests for xtile gemm (no performance measurement)."""

    @pytest.fixture(autouse=True)
    def _require_gpu(self, device_info) -> None:
        if not device_info.has_gpu:
            pytest.skip("No GPU available")

    @pytest.mark.parametrize(
        "dtype",
        _DTYPES,
        ids=_DTYPE_IDS,
    )
    def test_gemm_identity(self, dtype: torch.dtype, device_info) -> None:
        """C = I @ B should equal B."""
        from xtile.kernels.gemm import gemm as xtile_gemm

        N = 256
        I_mat = torch.eye(N, device=device_info.device, dtype=dtype)
        B = torch.randn(N, N, device=device_info.device, dtype=dtype)
        C = xtile_gemm(I_mat, B)

        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        atol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        assert torch.allclose(C, B, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "dtype",
        _DTYPES,
        ids=_DTYPE_IDS,
    )
    def test_gemm_zero(self, dtype: torch.dtype, device_info) -> None:
        """C = A @ 0 should be all zeros."""
        from xtile.kernels.gemm import gemm as xtile_gemm

        M, K, N = 128, 64, 128
        A = torch.randn(M, K, device=device_info.device, dtype=dtype)
        B = torch.zeros(K, N, device=device_info.device, dtype=dtype)
        C = xtile_gemm(A, B)
        assert torch.all(C == 0)

    @pytest.mark.parametrize(
        "shape",
        [(128, 256, 64), (256, 128, 512), (1, 1024, 1)],
        ids=["128x256x64", "256x128x512", "1x1024x1"],
    )
    def test_gemm_non_square(
        self, shape: tuple[int, int, int], device_info
    ) -> None:
        """Non-square matrices produce correct results."""
        from xtile.kernels.gemm import gemm as xtile_gemm

        M, N, K = shape
        A = torch.randn(M, K, device=device_info.device, dtype=torch.float16)
        B = torch.randn(K, N, device=device_info.device, dtype=torch.float16)
        C_ref = torch.matmul(A, B)
        C_xtile = xtile_gemm(A, B)
        assert torch.allclose(C_ref, C_xtile, rtol=1e-2, atol=1e-1)


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def _print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted GEMM results table."""
    header = (
        f"{'Size':>16s}  {'DType':>5s}  "
        f"{'torch(ms)':>10s}  {'torch(TF)':>10s}  "
        f"{'xtile(ms)':>10s}  {'xtile(TF)':>10s}  "
        f"{'Ratio':>8s}  {'OK':>4s}"
    )
    divider = "-" * len(header)

    print(f"\n  {header}")
    print(f"  {divider}")

    for r in results:
        size_str = f"{r['M']}x{r['N']}x{r['K']}"
        ok_str = "Y" if r["correct"] else "N"
        ratio_str = f"{r['ratio_pct']:.1f}%"
        print(
            f"  {size_str:>16s}  {r['dtype']:>5s}  "
            f"{r['torch_ms']:>10.3f}  {r['torch_tflops']:>10.2f}  "
            f"{r['xtile_ms']:>10.3f}  {r['xtile_tflops']:>10.2f}  "
            f"{ratio_str:>8s}  {ok_str:>4s}"
        )

    print(f"  {divider}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU detected. Exiting.")
        raise SystemExit(1)

    device = "cuda:0"
    props = torch.cuda.get_device_properties(device)

    print("=" * 80)
    print("  XTile GEMM Performance Benchmark")
    print("=" * 80)
    print(f"  Device         : {props.name}")
    print(f"  SMs            : {props.multi_processor_count}")
    print(f"  VRAM           : {props.total_memory / (1024**3):.1f} GB")
    print(f"  Warmup iters   : {_WARMUP_ITERS}")
    print(f"  Timed iters    : {_TIMED_ITERS}")
    print(f"  Target ratio   : >= {_TARGET_RATIO * 100:.0f}% of torch.matmul")
    print("=" * 80)

    results = run_gemm_benchmark(device=device)
    _print_results_table(results)

    # Summary
    print()
    passing = [r for r in results if r["ratio"] >= _TARGET_RATIO]
    failing = [r for r in results if r["ratio"] < _TARGET_RATIO]
    incorrect = [r for r in results if not r["correct"]]

    print(
        f"  Results: {len(passing)}/{len(results)} meet the "
        f"{_TARGET_RATIO * 100:.0f}% target"
    )
    if incorrect:
        print(f"  WARNING: {len(incorrect)} results had incorrect output!")
    if failing:
        print("  Below-target configurations:")
        for r in failing:
            print(
                f"    {r['M']}x{r['N']}x{r['K']} {r['dtype']}: "
                f"{r['ratio_pct']:.1f}%"
            )

    print("=" * 80)
