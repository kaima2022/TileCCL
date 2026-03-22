#!/usr/bin/env python3
"""Real multiprocess gemm_allscatter validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xtile.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess GEMM + all-scatter diagnostic run."""

    M: int
    N: int
    K: int
    dtype_name: str
    warmup: int
    iters: int
    force_transport: str | None
    contract: str
    launcher: str
    pattern: str
    expect_pattern: str | None
    heap_size_mb: int


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_CONTRACTS = {"full_full", "full_shard"}
_LAUNCHERS = {"plan", "ops", "all"}
_PATTERN_NAMES = {
    "auto",
    "bulk_sync",
    "fused_sequential",
    "fused_seq",
    "producer_consumer",
    "pc",
    "wg_specialized",
    "wg_spec",
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a user-facing dtype name into a torch dtype."""
    try:
        return _DTYPES[dtype_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(f"dtype must be one of {allowed}, got {dtype_name!r}") from exc


def _resolve_contract(contract: str) -> str:
    """Validate the requested public contract name."""
    if contract not in _CONTRACTS:
        allowed = ", ".join(sorted(_CONTRACTS))
        raise ValueError(f"contract must be one of {allowed}, got {contract!r}")
    return contract


def _resolve_launcher(launcher: str) -> str:
    """Validate the requested launcher selection."""
    if launcher not in _LAUNCHERS:
        allowed = ", ".join(sorted(_LAUNCHERS))
        raise ValueError(f"launcher must be one of {allowed}, got {launcher!r}")
    return launcher


def _resolve_pattern(pattern: str) -> str:
    """Validate the requested pattern argument."""
    if pattern not in _PATTERN_NAMES:
        allowed = ", ".join(sorted(_PATTERN_NAMES))
        raise ValueError(f"pattern must be one of {allowed}, got {pattern!r}")
    return pattern


def _resolve_expect_pattern(pattern: str | None) -> str | None:
    """Validate the optional expected resolved pattern name."""
    if pattern is None:
        return None
    canonical = {
        "bulk_sync",
        "fused_sequential",
        "producer_consumer",
        "wg_specialized",
    }
    if pattern not in canonical:
        allowed = ", ".join(sorted(canonical))
        raise ValueError(
            f"expect-pattern must be one of {allowed}, got {pattern!r}"
        )
    return pattern


def _rtol_atol(dtype: torch.dtype) -> tuple[float, float]:
    """Return conservative comparison tolerances for the output dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 2e-1
    return 1e-3, 5e-2


def _fill_replicated_matrix(
    tensor: torch.Tensor,
    *,
    row_scale: int,
    col_scale: int,
) -> None:
    """Fill a heap-backed matrix with a deterministic replicated pattern."""
    rows = torch.arange(tensor.shape[0], device=tensor.device, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(tensor.shape[1], device=tensor.device, dtype=torch.float32).unsqueeze(0)
    values = ((rows * row_scale + cols * col_scale) % 29.0 - 14.0) / 7.0
    tensor.copy_(values.to(dtype=tensor.dtype))


def _timed_call(
    fn,
    *,
    rank: int,
    barrier_kwargs: dict[str, object],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Time one launcher using CUDA events and rank barriers."""
    for _ in range(warmup):
        dist.barrier(**barrier_kwargs)
        fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier(**barrier_kwargs)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)
        times_ms.append(float(start.elapsed_time(end)))

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def _local_expected_output(
    A: torch.Tensor,
    B_full: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    contract: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the rank-local expected public output."""
    expected_full = torch.matmul(A.float(), B_full.float()).to(dtype=dtype)
    if contract == "full_full":
        return expected_full
    shard_cols = expected_full.shape[1] // world_size
    col_start = rank * shard_cols
    return expected_full[:, col_start:col_start + shard_cols].contiguous()


def _summarize_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[bool, float, float]:
    """Return correctness and scalar diagnostics for one output buffer."""
    rtol, atol = _rtol_atol(dtype)
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    sample_value = float(actual.reshape(-1)[0].item())
    return bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)), max_abs_diff, sample_value


def _recommended_heap_size_bytes(
    *,
    M: int,
    N: int,
    K: int,
    world_size: int,
    dtype: torch.dtype,
    contract: str,
) -> int:
    """Return a conservative lower bound for the symmetric heap size."""
    element_size = torch.empty((), dtype=dtype).element_size()
    output_cols = N if contract == "full_full" else (N if world_size == 1 else N // world_size)
    full_output_bytes = M * N * element_size
    required_bytes = (
        M * K * element_size
        + K * N * element_size
        + M * output_cols * element_size
    )
    if contract == "full_shard" and world_size > 1:
        required_bytes += full_output_bytes
    # Leave ample room for alignment slop and repeated workspace reuse.
    return int(required_bytes + max(64 * 1024 * 1024, required_bytes * 0.25))


def _worker(rank: int, world_size: int, store_path: str, config: _RunConfig) -> None:
    """Per-rank multiprocess validation worker."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    barrier_kwargs = {"device_ids": [rank]}
    dtype = _resolve_dtype(config.dtype_name)
    contract = _resolve_contract(config.contract)
    _resolve_launcher(config.launcher)
    _resolve_pattern(config.pattern)
    expected_pattern = _resolve_expect_pattern(config.expect_pattern)

    if config.force_transport is None:
        os.environ.pop(FORCE_MULTIPROCESS_TRANSPORT_ENV, None)
    else:
        os.environ[FORCE_MULTIPROCESS_TRANSPORT_ENV] = config.force_transport

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        "nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    import xtile
    from xtile.memory.symmetric_heap import SymmetricHeap

    heap_size_bytes = max(
        config.heap_size_mb * 1024 * 1024,
        _recommended_heap_size_bytes(
            M=config.M,
            N=config.N,
            K=config.K,
            world_size=world_size,
            dtype=dtype,
            contract=contract,
        ),
    )
    heap = SymmetricHeap(
        size=heap_size_bytes,
        rank=rank,
        world_size=world_size,
        backend="cuda",
    )
    try:
        M = config.M
        N = config.N
        K = config.K
        shard_cols = N if world_size == 1 else N // world_size

        A = heap.allocate_tensor((M, K), dtype)
        B_full = heap.allocate_tensor((K, N), dtype)
        C = heap.allocate_tensor((M, N if contract == "full_full" else shard_cols), dtype)
        _fill_replicated_matrix(A, row_scale=13, col_scale=7)
        _fill_replicated_matrix(B_full, row_scale=5, col_scale=11)
        C.zero_()

        torch.cuda.synchronize(rank)

        ctx = xtile.init(
            backend="cuda",
            rank=rank,
            world_size=world_size,
            heap=heap,
            force_backend=True,
        )

        plan_kwargs: dict[str, object] = {
            "ctx": ctx,
            "pattern": config.pattern,
        }
        if contract == "full_shard":
            plan_kwargs.update({
                "full_N": N,
                "b_layout": "full",
                "c_layout": "shard",
            })

        plan = xtile.ops.build_gemm_allscatter_plan(
            A,
            B_full,
            C,
            **plan_kwargs,
        )

        plan_timing = None
        plan_ok = None
        plan_max_abs_diff = None
        plan_sample = None
        plan_pattern_name = getattr(plan, "pattern_name", None)
        if config.launcher in {"plan", "all"}:
            plan_timing = _timed_call(
                lambda: plan.execute(A, B_full, C, validate=False),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_plan = _local_expected_output(
                A,
                B_full,
                rank=rank,
                world_size=world_size,
                contract=contract,
                dtype=dtype,
            )
            plan_ok, plan_max_abs_diff, plan_sample = _summarize_output(
                C,
                expected_plan,
                dtype=dtype,
            )
            C.zero_()

        high_level_timing = None
        high_level_ok = None
        high_level_max_abs_diff = None
        high_level_sample = None
        if config.launcher in {"ops", "all"}:
            high_level_timing = _timed_call(
                lambda: xtile.ops.gemm_allscatter(
                    A,
                    B_full,
                    C,
                    ctx=ctx,
                    full_N=N if contract == "full_shard" else None,
                    b_layout="full" if contract == "full_shard" else None,
                    c_layout="shard" if contract == "full_shard" else None,
                    pattern=config.pattern,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_ops = _local_expected_output(
                A,
                B_full,
                rank=rank,
                world_size=world_size,
                contract=contract,
                dtype=dtype,
            )
            high_level_ok, high_level_max_abs_diff, high_level_sample = _summarize_output(
                C,
                expected_ops,
                dtype=dtype,
            )

        if expected_pattern is not None and plan_pattern_name != expected_pattern:
            raise SystemExit(
                f"Expected plan_pattern_name={expected_pattern!r}, got "
                f"{plan_pattern_name!r} for shape M={M}, N={N}, K={K}."
            )

        payload = {
            "rank": rank,
            "dtype": config.dtype_name,
            "M": M,
            "N": N,
            "K": K,
            "contract": contract,
            "launcher": config.launcher,
            "pattern": config.pattern,
            "forced_transport": config.force_transport or "auto",
            "transport_strategy": heap.transport_strategy,
            "mode": heap.mode,
            "heap_size_mb": heap_size_bytes // (1024 * 1024),
            "plan_pattern_name": plan_pattern_name,
            "plan_ok": plan_ok,
            "plan_max_abs_diff": plan_max_abs_diff,
            "plan_sample": plan_sample,
            "plan_timing_ms": plan_timing,
            "high_level_ok": high_level_ok,
            "high_level_max_abs_diff": high_level_max_abs_diff,
            "high_level_sample": high_level_sample,
            "high_level_timing_ms": high_level_timing,
        }
        print(json.dumps(payload), flush=True)

        checks = [result for result in (plan_ok, high_level_ok) if result is not None]
        if not checks or not all(checks):
            raise SystemExit(2)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier(**barrier_kwargs)
            except Exception:
                pass
            dist.destroy_process_group()
        heap.cleanup()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multiprocess diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=256, help="Rows of A/C.")
    parser.add_argument("--N", type=int, default=512, help="Full output columns.")
    parser.add_argument("--K", type=int, default=256, help="Reduction dimension.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="One of float16,bfloat16,float32.",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default="full_full",
        help="Public contract to validate: full_full or full_shard.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="all",
        help="Which launchers to exercise: plan, ops, or all.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="bulk_sync",
        help="Pattern name passed to xtile.ops.gemm_allscatter(...).",
    )
    parser.add_argument(
        "--expect-pattern",
        type=str,
        default=None,
        help="Optional expected resolved plan pattern name.",
    )
    parser.add_argument(
        "--heap-size-mb",
        type=int,
        default=128,
        help="Minimum symmetric heap size per rank. The script may auto-raise this for large shapes.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations.")
    parser.add_argument(
        "--force-transport",
        type=str,
        default=None,
        help="Force one multiprocess transport strategy for diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    """Spawn one worker per visible GPU."""
    args = _parse_args()
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit("Need >= 2 GPUs")
    if args.N % world_size != 0:
        raise SystemExit(f"N must be divisible by world_size: {args.N} % {world_size} != 0")

    config = _RunConfig(
        M=args.M,
        N=args.N,
        K=args.K,
        dtype_name=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        force_transport=args.force_transport,
        contract=_resolve_contract(args.contract),
        launcher=_resolve_launcher(args.launcher),
        pattern=_resolve_pattern(args.pattern),
        expect_pattern=_resolve_expect_pattern(args.expect_pattern),
        heap_size_mb=args.heap_size_mb,
    )

    handle = tempfile.NamedTemporaryFile(
        prefix="xtile_gemm_allscatter_",
        delete=False,
    )
    handle.close()
    store_path = handle.name
    try:
        mp.start_processes(
            _worker,
            args=(world_size, store_path, config),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
    finally:
        path = Path(store_path)
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
