#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess gemm_reducescatter validation."""

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

from tncc.utils.feature_gates import (
    FORCE_MULTIPROCESS_TRANSPORT_ENV,
    MULTIPROCESS_DEVICE_COLLECTIVES_ENV,
)


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess GEMM + reduce-scatter diagnostic run."""

    M: int
    N: int
    K: int
    dtype_name: str
    warmup: int
    iters: int
    force_transport: str | None
    launcher: str
    heap_size_mb: int


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_LAUNCHERS = {"plan", "ops", "all"}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a user-facing dtype name into a torch dtype."""
    try:
        return _DTYPES[dtype_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(f"dtype must be one of {allowed}, got {dtype_name!r}") from exc


def _resolve_launcher(launcher: str) -> str:
    """Validate the requested launcher selection."""
    if launcher not in _LAUNCHERS:
        allowed = ", ".join(sorted(_LAUNCHERS))
        raise ValueError(f"launcher must be one of {allowed}, got {launcher!r}")
    return launcher


def _rtol_atol(dtype: torch.dtype) -> tuple[float, float]:
    """Return conservative comparison tolerances for the output dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 2e-1
    return 1e-3, 5e-2


def _make_ranked_matrix(
    rows: int,
    cols: int,
    *,
    rank: int,
    row_scale: int,
    col_scale: int,
    bias_scale: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a deterministic rank-dependent matrix."""
    row_idx = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    col_idx = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(0)
    values = (
        row_idx * float(row_scale + rank)
        + col_idx * float(col_scale + 2 * rank)
        + float(rank * bias_scale)
    ) % 31.0
    return (values - 15.0) / 8.0


def _fill_ranked_matrix(
    tensor: torch.Tensor,
    *,
    rank: int,
    row_scale: int,
    col_scale: int,
    bias_scale: int,
) -> None:
    """Fill one device tensor using the deterministic rank-dependent pattern."""
    values = _make_ranked_matrix(
        tensor.shape[0],
        tensor.shape[1],
        rank=rank,
        row_scale=row_scale,
        col_scale=col_scale,
        bias_scale=bias_scale,
        device=tensor.device,
    )
    tensor.copy_(values.to(dtype=tensor.dtype))


def _expected_local_shard(
    *,
    rank: int,
    world_size: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the rank-local expected shard of the reduced full output."""
    full = torch.zeros(M, N, dtype=torch.float32)
    for peer_rank in range(world_size):
        A_peer = _make_ranked_matrix(
            M,
            K,
            rank=peer_rank,
            row_scale=13,
            col_scale=7,
            bias_scale=5,
        )
        B_peer = _make_ranked_matrix(
            K,
            N,
            rank=peer_rank,
            row_scale=11,
            col_scale=3,
            bias_scale=9,
        )
        full += torch.matmul(A_peer, B_peer)
    shard_cols = N // world_size
    col_start = rank * shard_cols
    return full[:, col_start:col_start + shard_cols].to(dtype=dtype).contiguous()


def _recommended_heap_size_bytes(
    *,
    M: int,
    N: int,
    K: int,
    world_size: int,
    dtype: torch.dtype,
) -> int:
    """Return a conservative lower bound for the symmetric heap size."""
    element_size = torch.empty((), dtype=dtype).element_size()
    shard_cols = N if world_size == 1 else N // world_size
    required_bytes = (
        M * shard_cols * element_size  # C
        + M * N * element_size         # local_full workspace
        + M * N * element_size         # packed_input workspace
    )
    return int(required_bytes + max(64 * 1024 * 1024, required_bytes * 0.25))


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


def _worker(rank: int, world_size: int, store_path: str, config: _RunConfig) -> None:
    """Per-rank multiprocess validation worker."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    barrier_kwargs = {"device_ids": [rank]}
    dtype = _resolve_dtype(config.dtype_name)
    _resolve_launcher(config.launcher)

    os.environ[MULTIPROCESS_DEVICE_COLLECTIVES_ENV] = "1"
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

    import tncc
    from tncc.memory.symmetric_heap import SymmetricHeap

    heap_size_bytes = max(
        config.heap_size_mb * 1024 * 1024,
        _recommended_heap_size_bytes(
            M=config.M,
            N=config.N,
            K=config.K,
            world_size=world_size,
            dtype=dtype,
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

        A = torch.empty((M, K), device=device, dtype=dtype)
        B = torch.empty((K, N), device=device, dtype=dtype)
        _fill_ranked_matrix(A, rank=rank, row_scale=13, col_scale=7, bias_scale=5)
        _fill_ranked_matrix(B, rank=rank, row_scale=11, col_scale=3, bias_scale=9)

        C = heap.allocate_tensor((M, shard_cols), dtype)
        C.zero_()
        torch.cuda.synchronize(rank)

        ctx = tncc.init(
            backend="cuda",
            rank=rank,
            world_size=world_size,
            heap=heap,
            force_backend=True,
        )

        plan = tncc.ops.build_gemm_reducescatter_plan(
            A,
            B,
            C,
            ctx=ctx,
            implementation="device",
        )

        plan_timing = None
        plan_ok = None
        plan_max_abs_diff = None
        plan_sample = None
        if config.launcher in {"plan", "all"}:
            plan_timing = _timed_call(
                lambda: plan.execute(A, B, C, validate=False),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_plan = _expected_local_shard(
                rank=rank,
                world_size=world_size,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )
            plan_ok, plan_max_abs_diff, plan_sample = _summarize_output(
                C,
                expected_plan.to(device=device),
                dtype=dtype,
            )
            C.zero_()

        high_level_timing = None
        high_level_ok = None
        high_level_max_abs_diff = None
        high_level_sample = None
        if config.launcher in {"ops", "all"}:
            high_level_timing = _timed_call(
                lambda: tncc.ops.gemm_reducescatter(
                    A,
                    B,
                    C,
                    ctx=ctx,
                    implementation="device",
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_ops = _expected_local_shard(
                rank=rank,
                world_size=world_size,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )
            high_level_ok, high_level_max_abs_diff, high_level_sample = _summarize_output(
                C,
                expected_ops.to(device=device),
                dtype=dtype,
            )

        payload = {
            "rank": rank,
            "dtype": config.dtype_name,
            "M": M,
            "N": N,
            "K": K,
            "launcher": config.launcher,
            "forced_transport": config.force_transport or "auto",
            "transport_strategy": heap.transport_strategy,
            "mode": heap.mode,
            "heap_size_mb": heap_size_bytes // (1024 * 1024),
            "plan_implementation": plan.implementation,
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
    parser.add_argument("--M", type=int, default=128, help="Rows of A/C.")
    parser.add_argument("--N", type=int, default=256, help="Full output columns.")
    parser.add_argument("--K", type=int, default=128, help="Reduction dimension.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="One of float16,bfloat16,float32.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="all",
        help="Which launchers to exercise: plan, ops, or all.",
    )
    parser.add_argument(
        "--heap-size-mb",
        type=int,
        default=128,
        help="Minimum symmetric heap size per rank. The script may auto-raise this for large shapes.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=2, help="Timed iterations.")
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
        launcher=_resolve_launcher(args.launcher),
        heap_size_mb=args.heap_size_mb,
    )

    handle = tempfile.NamedTemporaryFile(
        prefix="tncc_gemm_reducescatter_",
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
