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

    print()
    print(format_benchmark_table(results))
    print()

    # Compute TFLOPS
    flops = 2.0 * M * N * K
    mean_s = results[0]["mean_ms"] / 1e3
    if mean_s > 0:
        tflops = flops / mean_s / 1e12
        print(f"Throughput: {tflops:.1f} TFLOPS")

    # Pattern benchmark (requires >= 2 GPUs)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2 and pattern in ("all", "auto"):
        print()
        print("=== Pattern Benchmark (2 GPUs) ===")
        _run_pattern_bench(M, N, K, pattern)
    elif num_gpus < 2 and pattern in ("all", "auto"):
        print("\n(Skipping pattern benchmark: requires >= 2 GPUs)")


def _run_pattern_bench(M: int, N: int, K: int, pattern: str) -> None:
    """Run pattern benchmark with auto-select."""
    import torch
    from xtile.memory.symmetric_heap import SymmetricHeap
    from xtile.backends import get_backend, detect_hardware
    from xtile.patterns.auto_select import auto_select, benchmark_all_patterns

    world_size = 2
    heap_size = max(512 * 1024 * 1024, M * K * 2 + K * (N // world_size) * 2 + M * N * 2)
    heaps = SymmetricHeap.create_all(size=heap_size, world_size=world_size)

    try:
        rank = 0
        torch.cuda.set_device(rank)
        hw = detect_hardware()
        backend = get_backend(hw)
        bases = heaps[rank].get_heap_bases()

        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.heap_bases = bases
        ctx.backend = backend

        N_per_rank = N // world_size
        A = heaps[rank].allocate_tensor((M, K), torch.float16)
        B = heaps[rank].allocate_tensor((K, N_per_rank), torch.float16)
        C = heaps[rank].allocate_tensor((M, N_per_rank), torch.float16)
        A.normal_()
        B.normal_()
        C.zero_()
        torch.cuda.synchronize()

        if pattern == "auto":
            # Show auto-select decision
            selected = auto_select(
                "gemm_allscatter", M=M, N=N, K=K,
                world_size=world_size,
            )
            print(f"Auto-selected pattern: {selected.name if hasattr(selected, 'name') else selected.__name__}")

        # Benchmark all patterns
        results = benchmark_all_patterns(A, B, C, ctx, warmup=5, iters=20)
        print(f"\n{'Pattern':25s} | {'Mean (ms)':>10s} | {'Min (ms)':>10s} | {'Speedup':>8s}")
        print("-" * 65)
        bulk_min = None
        for r in results["results"]:
            if r["pattern"] == "bulk_sync":
                bulk_min = r["min_ms"]
        for r in results["results"]:
            speedup = ""
            if bulk_min and bulk_min > 0 and r["min_ms"] < float("inf"):
                speedup = f"{bulk_min / r['min_ms']:.3f}x"
            print(f"{r['pattern']:25s} | {r['mean_ms']:10.3f} | {r['min_ms']:10.3f} | {speedup:>8s}")
        print(f"\nBest: {results['best']}")
    finally:
        for h in heaps:
            h.cleanup()


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
        choices=["all", "auto", "bulk_sync", "fused_seq", "pc", "wg_spec"],
        default=None,
        help="Pattern to benchmark (default: all, 'auto' uses auto-select)",
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
