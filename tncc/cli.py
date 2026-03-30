# SPDX-License-Identifier: Apache-2.0
"""tncc.cli - Command-line interface for TNCC (Tile Native Collective Communication).

Provides the ``tncc`` console entry point for hardware diagnostics and
benchmarking.

Usage::

    tncc info                    # print hardware / topology info
    tncc bench p2p               # P2P bandwidth sweep
    tncc bench collective        # collective bandwidth benchmark
    tncc bench pattern           # pattern overlap efficiency
    tncc bench gemm              # GEMM vs torch.matmul
    tncc bench all               # run all benchmarks
    tncc bench --pattern auto    # auto-select pattern benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _handle_info(args: argparse.Namespace) -> None:
    """Print hardware and topology information."""
    try:
        from tncc.utils.topology import TopologyDetector
        detector = TopologyDetector()
        info = detector.detect()
        detector.print_info(info)
    except RuntimeError as exc:
        print(f"[tncc info] Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _format_support_matrix(matrix) -> str:
    """Render a support matrix into a human-readable CLI summary."""
    lines: list[str] = []
    lines.append("=== TNCC Runtime Support Matrix ===")
    lines.append(
        "Context: "
        f"backend={matrix.backend}, device={matrix.device}, "
        f"rank={matrix.rank}, world_size={matrix.world_size}, "
        f"has_heap={matrix.has_heap}"
    )
    if matrix.has_heap:
        lines.append(
            "Heap: "
            f"mode={matrix.heap_mode}, transport_strategy={matrix.transport_strategy}"
        )

    def _section(title: str, entries: dict) -> None:
        lines.append("")
        lines.append(f"{title}:")
        for name, status in entries.items():
            lines.append(f"  - {name}: {status.state} | {status.detail}")

    _section("Ops", matrix.ops)
    _section("Contracts", matrix.contracts)
    _section("Execution Paths", matrix.execution_paths)
    _section("Collectives", matrix.collectives)
    _section("Memory", matrix.memory)
    return "\n".join(lines)


def _build_support_context(args: argparse.Namespace):
    """Create a context for the support-matrix command.

    Returns ``(ctx, cleanup_fn)``.
    """
    import tncc

    backend = getattr(args, "backend", "auto")
    world_size = int(getattr(args, "world_size", 1) or 1)
    heap_size_mb = getattr(args, "heap_size_mb", None)
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    if heap_size_mb is not None:
        heap_size = int(heap_size_mb) * 1024 * 1024
        if heap_size <= 0:
            raise ValueError(f"heap_size_mb must be positive, got {heap_size_mb}")
        if world_size > 1:
            contexts = tncc.init_local(
                world_size=world_size,
                heap_size=heap_size,
                backend=backend,
            )

            def _cleanup() -> None:
                for ctx in contexts:
                    if ctx.heap is not None:
                        ctx.heap.cleanup()

            return contexts[0], _cleanup

        ctx = tncc.init(
            backend=backend,
            rank=0,
            world_size=1,
            heap_size=heap_size,
            force_backend=True,
        )

        def _cleanup() -> None:
            if ctx.heap is not None:
                ctx.heap.cleanup()

        return ctx, _cleanup

    ctx = tncc.init(
        backend=backend,
        rank=0,
        world_size=world_size,
        force_backend=True,
    )
    return ctx, (lambda: None)


def _handle_support(args: argparse.Namespace) -> None:
    """Print the current runtime support matrix."""
    _ensure_gpu()
    import tncc

    try:
        ctx, cleanup = _build_support_context(args)
    except Exception as exc:
        print(f"[tncc support] Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        matrix = tncc.describe_runtime_support(ctx)
        if getattr(args, "json", False):
            print(json.dumps(matrix.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(_format_support_matrix(matrix))
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Benchmark sub-handlers
# ---------------------------------------------------------------------------

def _ensure_gpu():
    """Check GPU availability, exit if none."""
    try:
        import torch
    except ImportError:
        print("[tncc bench] PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)
    if not torch.cuda.is_available():
        print("[tncc bench] No GPU detected. Benchmarks require a CUDA or ROCm GPU.", file=sys.stderr)
        sys.exit(1)
    return torch


def _ensure_multi_gpu(torch_mod, min_gpus: int = 2):
    """Check multi-GPU availability."""
    if torch_mod.cuda.device_count() < min_gpus:
        print(f"[tncc bench] This benchmark requires >= {min_gpus} GPUs. "
              f"Found {torch_mod.cuda.device_count()}.", file=sys.stderr)
        sys.exit(1)


def _results_dir() -> Path:
    """Return ~/.tncc/benchmark_results/, creating it if needed."""
    d = Path.home() / ".tncc" / "benchmark_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_results(name: str, data: dict) -> Path:
    """Save benchmark results to JSON with timestamp."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = _results_dir() / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")
    return path


def _project_root() -> str:
    """Return the TNCC project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bench_env() -> dict:
    """Return environment with PYTHONPATH set for benchmark subprocesses."""
    env = os.environ.copy()
    root = _project_root()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _bench_p2p(args: argparse.Namespace) -> None:
    """Run P2P bandwidth benchmark."""
    torch = _ensure_gpu()
    _ensure_multi_gpu(torch)

    print("=== TNCC P2P Bandwidth Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print()

    import subprocess
    quick_flag = ["--quick"] if getattr(args, "quick", False) else []
    result = subprocess.run(
        [sys.executable, "-u", "tests/benchmarks/bench_p2p_translate.py"] + quick_flag,
        cwd=_project_root(), env=_bench_env(),
    )
    sys.exit(result.returncode)


def _bench_collective(args: argparse.Namespace) -> None:
    """Run collective bandwidth benchmark."""
    torch = _ensure_gpu()
    _ensure_multi_gpu(torch)

    print("=== TNCC Collective Bandwidth Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print()

    import subprocess
    result = subprocess.run(
        [sys.executable, "-u", "tests/benchmarks/bench_collectives.py"],
        cwd=_project_root(), env=_bench_env(),
    )
    sys.exit(result.returncode)


def _bench_pattern(args: argparse.Namespace) -> None:
    """Run pattern overlap efficiency benchmark."""
    torch = _ensure_gpu()
    _ensure_multi_gpu(torch)

    print("=== TNCC Pattern Overlap Benchmark ===")
    print(f"GPUs: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print()

    import subprocess
    cmd = [sys.executable, "-u", "tests/benchmarks/bench_patterns.py"]
    if getattr(args, "quick", False):
        cmd.append("--quick")
    if getattr(args, "warmup", None) is not None:
        cmd.extend(["--warmup", str(args.warmup)])
    if getattr(args, "iters", None) is not None:
        cmd.extend(["--iters", str(args.iters)])
    if getattr(args, "heap_size_mb", None) is not None:
        cmd.extend(["--heap-size-mb", str(args.heap_size_mb)])
    if getattr(args, "output_json", None):
        cmd.extend(["--output-json", str(args.output_json)])
    result = subprocess.run(
        cmd,
        cwd=_project_root(), env=_bench_env(),
    )
    sys.exit(result.returncode)


def _bench_gemm(args: argparse.Namespace) -> None:
    """Run GEMM performance benchmark."""
    torch = _ensure_gpu()

    print("=== TNCC GEMM Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    import subprocess
    result = subprocess.run(
        [sys.executable, "-u", "tests/benchmarks/bench_gemm.py"],
        cwd=_project_root(), env=_bench_env(),
    )
    sys.exit(result.returncode)


def _bench_all(args: argparse.Namespace) -> None:
    """Run all benchmarks sequentially."""
    torch = _ensure_gpu()

    print("=" * 70)
    print("  TNCC Full Benchmark Suite")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print()

    project_root = _project_root()
    num_gpus = torch.cuda.device_count()

    benchmarks = []
    # Always run GEMM (single GPU)
    benchmarks.append(("GEMM", "tests/benchmarks/bench_gemm.py", []))

    if num_gpus >= 2:
        quick_flag = ["--quick"] if getattr(args, "quick", False) else []
        benchmarks.append(("P2P", "tests/benchmarks/bench_p2p_translate.py", quick_flag))
        benchmarks.append(("Collective", "tests/benchmarks/bench_collectives.py", []))
        benchmarks.append(("Pattern", "tests/benchmarks/bench_patterns.py", quick_flag))
    else:
        print("(Skipping P2P/collective/pattern benchmarks: requires >= 2 GPUs)")

    import subprocess
    for name, script, extra_args in benchmarks:
        print()
        print(f"{'='*70}")
        print(f"  Running: {name}")
        print(f"{'='*70}")
        result = subprocess.run(
            [sys.executable, "-u", script] + extra_args,
            cwd=project_root, env=_bench_env(),
        )
        if result.returncode != 0:
            print(f"\n  WARNING: {name} benchmark returned non-zero exit code: {result.returncode}")

    print()
    print("=" * 70)
    print("  All benchmarks complete.")
    print("=" * 70)


def _handle_bench(args: argparse.Namespace) -> None:
    """Run benchmark suite."""
    bench_type = getattr(args, "bench_type", None)

    # Dispatch to specific benchmark
    if bench_type == "p2p":
        _bench_p2p(args)
        return
    elif bench_type == "collective":
        _bench_collective(args)
        return
    elif bench_type == "pattern":
        _bench_pattern(args)
        return
    elif bench_type == "gemm":
        _bench_gemm(args)
        return
    elif bench_type == "all":
        _bench_all(args)
        return

    # Legacy behavior: GEMM baseline + optional pattern benchmark
    torch = _ensure_gpu()

    M, N, K = args.M, args.N, args.K
    pattern = args.pattern or "all"

    print(f"TNCC Benchmark: pattern={pattern}, M={M}, N={N}, K={K}")
    print("-" * 60)

    from tncc.kernels.gemm import gemm
    from tncc.utils.profiling import TileProfiler, format_benchmark_table

    device = torch.cuda.current_device()
    A = torch.randn(M, K, device=f"cuda:{device}", dtype=torch.float16)
    B = torch.randn(K, N, device=f"cuda:{device}", dtype=torch.float16)

    # Warm up
    for _ in range(10):
        gemm(A, B)
    torch.cuda.synchronize()

    # Timed iterations
    profiler = TileProfiler("gemm_baseline")
    for _ in range(50):
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

    import tncc
    from tncc.patterns.auto_select import auto_select, benchmark_all_patterns

    world_size = 2
    heap_size = max(512 * 1024 * 1024, M * K * 2 + K * (N // world_size) * 2 + M * N * 2)
    contexts = tncc.init_local(world_size=world_size, heap_size=heap_size)

    try:
        rank = 0
        torch.cuda.set_device(rank)
        ctx = contexts[rank]
        heap = ctx.require_heap()

        N_per_rank = N // world_size
        A = heap.allocate_tensor((M, K), torch.float16)
        B = heap.allocate_tensor((K, N_per_rank), torch.float16)
        C = heap.allocate_tensor((M, N_per_rank), torch.float16)
        A.normal_()
        B.normal_()
        C.zero_()
        torch.cuda.synchronize()

        if pattern == "auto":
            selected = auto_select(
                "gemm_allscatter", M=M, N=N, K=K,
                world_size=world_size,
            )
            print(f"Auto-selected pattern: {selected.name if hasattr(selected, 'name') else selected.__name__}")

        results = benchmark_all_patterns(
            A,
            B,
            C,
            ctx,
            warmup=5,
            iters=20,
            full_N=N,
            b_layout="shard",
            c_layout="shard",
        )
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
        for ctx in contexts:
            if ctx.heap is not None:
                ctx.heap.cleanup()


# ---------------------------------------------------------------------------
# Benchmark result comparison
# ---------------------------------------------------------------------------

def _handle_compare(args: argparse.Namespace) -> None:
    """Compare two benchmark result files."""
    results_dir = _results_dir()
    files = sorted(results_dir.glob("*.json"))

    if not files:
        print("No benchmark results found in ~/.tncc/benchmark_results/")
        return

    if len(files) < 2:
        print("Need at least 2 result files to compare. Run benchmarks first.")
        return

    # Compare the two most recent results
    print("Comparing last 2 results:")
    print(f"  Old: {files[-2].name}")
    print(f"  New: {files[-1].name}")

    with open(files[-2]) as f:
        old = json.load(f)
    with open(files[-1]) as f:
        new = json.load(f)

    print(f"\n  Old: {json.dumps(old, indent=2, default=str)[:500]}...")
    print(f"\n  New: {json.dumps(new, indent=2, default=str)[:500]}...")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """TNCC CLI -- benchmark and diagnostics tool."""
    parser = argparse.ArgumentParser(
        prog="tncc",
        description="TNCC (Tile Native Collective Communication): cross-platform tile communication library CLI",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # tncc info
    subparsers.add_parser("info", help="Print hardware and topology info")

    # tncc support
    support_parser = subparsers.add_parser(
        "support",
        help="Print the current runtime support / capability matrix",
    )
    support_parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cuda", "hip"],
        help="Backend used to construct the inspection context",
    )
    support_parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Logical world size for the inspection context",
    )
    support_parser.add_argument(
        "--heap-size-mb",
        type=int,
        default=None,
        help="Attach a symmetric heap of this size (MiB) before collecting support status",
    )
    support_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the support matrix as JSON",
    )

    # tncc bench
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
    bench_parser.add_argument("--quick", action="store_true", help="Quick mode (fewer sizes)")

    # bench subcommands: p2p, collective, pattern, gemm, all
    bench_sub = bench_parser.add_subparsers(dest="bench_type")
    p2p_parser = bench_sub.add_parser("p2p", help="P2P bandwidth benchmark")
    p2p_parser.add_argument("--quick", action="store_true", help="Quick mode")

    bench_sub.add_parser("collective", help="Collective bandwidth benchmark")

    pat_parser = bench_sub.add_parser("pattern", help="Pattern overlap efficiency benchmark")
    pat_parser.add_argument("--quick", action="store_true", help="Quick mode")
    pat_parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per pattern")
    pat_parser.add_argument("--iters", type=int, default=10, help="Timed iterations per pattern")
    pat_parser.add_argument(
        "--heap-size-mb",
        type=int,
        default=None,
        help="Per-rank symmetric heap size override in MiB",
    )
    pat_parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Structured JSON output path for benchmark results",
    )

    bench_sub.add_parser("gemm", help="GEMM vs torch.matmul benchmark")

    all_parser = bench_sub.add_parser("all", help="Run all benchmarks")
    all_parser.add_argument("--quick", action="store_true", help="Quick mode for P2P and pattern")

    # tncc compare
    subparsers.add_parser("compare", help="Compare benchmark results")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "info":
        _handle_info(args)
    elif args.command == "support":
        _handle_support(args)
    elif args.command == "bench":
        _handle_bench(args)
    elif args.command == "compare":
        _handle_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


def _get_version() -> str:
    """Return the package version string."""
    try:
        from tncc import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
