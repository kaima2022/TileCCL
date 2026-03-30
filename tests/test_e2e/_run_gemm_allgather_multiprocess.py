#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess gemm_allgather validation."""

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

from tncc.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess GEMM + allgather diagnostic run."""

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


def _fill_matrix_from_rank_pattern(
    tensor: torch.Tensor,
    *,
    source_rank: int,
    row_scale: int,
    col_scale: int,
    bias_scale: int,
) -> None:
    """Fill one device tensor using a deterministic pattern derived from one rank."""
    values = _make_ranked_matrix(
        tensor.shape[0],
        tensor.shape[1],
        rank=source_rank,
        row_scale=row_scale,
        col_scale=col_scale,
        bias_scale=bias_scale,
        device=tensor.device,
    )
    tensor.copy_(values.to(dtype=tensor.dtype))


def _expected_full_output(
    *,
    world_size: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the expected full GEMM output for every rank."""
    A_full = _make_ranked_matrix(
        M,
        K,
        rank=0,
        row_scale=13,
        col_scale=7,
        bias_scale=5,
    )
    B_full = torch.cat(
        [
            _make_ranked_matrix(
                K,
                N // world_size,
                rank=peer_rank,
                row_scale=11,
                col_scale=3,
                bias_scale=9,
            )
            for peer_rank in range(world_size)
        ],
        dim=1,
    )
    return torch.matmul(A_full, B_full).to(dtype=dtype).contiguous()


def _recommended_heap_size_bytes(
    *,
    M: int,
    N: int,
    world_size: int,
    dtype: torch.dtype,
) -> int:
    """Return a conservative lower bound for the symmetric heap size."""
    element_size = torch.empty((), dtype=dtype).element_size()
    shard_cols = N if world_size == 1 else N // world_size
    required_bytes = (
        M * N * element_size              # C
        + M * shard_cols * element_size   # local_shard workspace
        + M * N * element_size            # gathered_shards workspace
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
        B_shard = torch.empty((K, shard_cols), device=device, dtype=dtype)
        _fill_matrix_from_rank_pattern(
            A,
            source_rank=0,
            row_scale=13,
            col_scale=7,
            bias_scale=5,
        )
        _fill_matrix_from_rank_pattern(
            B_shard,
            source_rank=rank,
            row_scale=11,
            col_scale=3,
            bias_scale=9,
        )

        C = heap.allocate_tensor((M, N), dtype)
        C.zero_()
        torch.cuda.synchronize(rank)

        ctx = tncc.init(
            backend="cuda",
            rank=rank,
            world_size=world_size,
            heap=heap,
            force_backend=True,
        )

        plan = tncc.ops.build_gemm_allgather_plan(
            A,
            B_shard,
            C,
            ctx=ctx,
        )

        plan_timing = None
        plan_ok = None
        plan_max_abs_diff = None
        plan_sample = None
        if config.launcher in {"plan", "all"}:
            plan_timing = _timed_call(
                lambda: plan.execute(A, B_shard, C, validate=False),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_plan = _expected_full_output(
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
                lambda: tncc.ops.gemm_allgather(
                    A,
                    B_shard,
                    C,
                    ctx=ctx,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_ops = _expected_full_output(
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
            "plan_queue_slots": plan.execution.slot_count,
            "plan_credit_window": plan.execution.credit_window,
            "plan_tile_rows": plan.execution.tile_rows,
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
        checks = [
            result
            for result in (plan_ok, high_level_ok)
            if result is not None
        ]
        if not checks or not all(checks):
            raise SystemExit(2)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier(**barrier_kwargs)
            except Exception:
                pass
        try:
            heap.cleanup()
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass


def main() -> None:
    """Run the real multiprocess GEMM + allgather validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=128, help="Rows of A/C.")
    parser.add_argument("--N", type=int, default=256, help="Full output columns.")
    parser.add_argument("--K", type=int, default=128, help="Reduction dimension.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=sorted(_DTYPES),
        help="Element dtype for the diagnostic tensors.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per launcher.")
    parser.add_argument("--iters", type=int, default=2, help="Timed iterations per launcher.")
    parser.add_argument(
        "--force-transport",
        type=str,
        default=None,
        help="Explicit multiprocess transport to force for diagnostics.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="all",
        choices=sorted(_LAUNCHERS),
        help="Which launcher(s) to validate: plan, ops, or all.",
    )
    parser.add_argument(
        "--heap-size-mb",
        type=int,
        default=64,
        help="Requested heap size in MiB (auto-raised if required).",
    )
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 GPUs for multiprocess GEMM + allgather validation.")
    if args.N <= 0 or args.K <= 0 or args.M <= 0:
        raise SystemExit("M/N/K must be positive.")
    if args.N % 2 != 0:
        raise SystemExit("This diagnostic expects world_size=2 and requires N divisible by 2.")

    config = _RunConfig(
        M=args.M,
        N=args.N,
        K=args.K,
        dtype_name=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        force_transport=args.force_transport,
        launcher=args.launcher,
        heap_size_mb=args.heap_size_mb,
    )

    with tempfile.TemporaryDirectory(prefix="tncc_gemm_allgather_") as tmpdir:
        store_path = str(Path(tmpdir) / "dist_store")
        mp.spawn(
            _worker,
            args=(2, store_path, config),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    main()
