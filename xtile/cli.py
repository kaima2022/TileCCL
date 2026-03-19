"""xtile.cli - Command-line interface for XTile.

Provides the ``xtile`` console entry point for hardware diagnostics and
benchmarking.

Usage::

    xtile info            # print hardware / topology info
    xtile bench           # run benchmarks (all patterns, default sizes)
    xtile bench --pattern bulk_sync --M 4096 --N 4096 --K 4096
"""

from __future__ import annotations

import argparse
import sys


def _handle_info(args: argparse.Namespace) -> None:
    """Print hardware and topology information."""
    try:
        from xtile.utils.topology import TopologyDetector
        detector = TopologyDetector()
        info = detector.detect()
        detector.print_info(info)
    except RuntimeError as exc:
        print(f"[xtile info] Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _handle_bench(args: argparse.Namespace) -> None:
    """Run benchmark suite."""
    try:
        import torch
    except ImportError:
        print("[xtile bench] PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[xtile bench] No GPU detected. Benchmarks require a CUDA or ROCm GPU.", file=sys.stderr)
        sys.exit(1)

    M, N, K = args.M, args.N, args.K
    pattern = args.pattern or "all"

    print(f"XTile Benchmark: pattern={pattern}, M={M}, N={N}, K={K}")
    print("-" * 60)

    # Import GEMM kernel and run a basic benchmark
    from xtile.kernels.gemm import gemm
    from xtile.utils.profiling import TileProfiler, format_benchmark_table

    device = torch.cuda.current_device()
    A = torch.randn(M, K, device=f"cuda:{device}", dtype=torch.float16)
    B = torch.randn(K, N, device=f"cuda:{device}", dtype=torch.float16)

    # Warm up
    warmup_iters = 10
    for _ in range(warmup_iters):
        gemm(A, B)
    torch.cuda.synchronize()

    # Timed iterations
    profiler = TileProfiler("gemm_baseline")
    num_iters = 50
    for _ in range(num_iters):
        with profiler:
            gemm(A, B)
            torch.cuda.synchronize()

    results = [profiler.summary()]

    # If a specific pattern is requested, note it for future expansion
    if pattern != "all":
        print(f"(Pattern '{pattern}' -- full pattern benchmarks coming soon)")

    print()
    print(format_benchmark_table(results))
    print()

    # Compute TFLOPS
    flops = 2.0 * M * N * K
    mean_s = results[0]["mean_ms"] / 1e3
    if mean_s > 0:
        tflops = flops / mean_s / 1e12
        print(f"Throughput: {tflops:.1f} TFLOPS")


def main() -> None:
    """XTile CLI -- benchmark and diagnostics tool."""
    parser = argparse.ArgumentParser(
        prog="xtile",
        description="XTile: cross-platform tile communication library CLI",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # xtile info
    subparsers.add_parser("info", help="Print hardware and topology info")

    # xtile bench
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument(
        "--pattern",
        choices=["all", "bulk_sync", "fused_seq", "pc", "wg_spec"],
        default=None,
        help="Pattern to benchmark (default: all)",
    )
    bench_parser.add_argument("--M", type=int, default=8192, help="Matrix M dimension")
    bench_parser.add_argument("--N", type=int, default=8192, help="Matrix N dimension")
    bench_parser.add_argument("--K", type=int, default=8192, help="Matrix K dimension")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "info":
        _handle_info(args)
    elif args.command == "bench":
        _handle_bench(args)
    else:
        parser.print_help()
        sys.exit(1)


def _get_version() -> str:
    """Return the package version string."""
    try:
        from xtile import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
