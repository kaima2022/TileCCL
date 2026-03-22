#!/usr/bin/env python3
"""Real multiprocess reduce_scatter(device) validation.

This script is meant to be launched as a standalone module so
``torch.multiprocessing.spawn`` can re-import it safely.  It validates the
current multiprocess/device execution path used by
``xtile.primitives.reduce_scatter(..., implementation="device")``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xtile.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess reduce_scatter diagnostic run."""

    block_size: int
    dtype_name: str
    warmup: int
    iters: int
    force_transport: str | None
    launcher: str


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a user-facing dtype name into a torch dtype."""
    try:
        return _DTYPES[dtype_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(f"dtype must be one of {allowed}, got {dtype_name!r}") from exc


def _fill_reduce_scatter_input(
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    block_size: int,
) -> None:
    """Fill the source tensor with a stable integer pattern."""
    for chunk in range(world_size):
        start = chunk * block_size
        value = float(rank * 2 + chunk + 1)
        tensor[start:start + block_size].fill_(value)


def _timed_collective(
    fn,
    *,
    rank: int,
    barrier_kwargs: dict[str, object],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Time one collective call using CUDA events and rank barriers."""
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


def _worker(rank: int, world_size: int, store_path: str, config: _RunConfig) -> None:
    """Per-rank multiprocess validation worker."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    barrier_kwargs = {"device_ids": [rank]}
    dtype = _resolve_dtype(config.dtype_name)
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
    from xtile.primitives.collectives import _reduce_scatter_kernel
    from xtile.primitives import reduce_scatter as primitive_reduce_scatter

    heap = SymmetricHeap(
        size=64 * 1024 * 1024,
        rank=rank,
        world_size=world_size,
        backend="cuda",
    )
    try:
        block_size = config.block_size
        total_elements = block_size * world_size

        src_primitive = heap.allocate_tensor((total_elements,), dtype)
        src_high_level = heap.allocate_tensor((total_elements,), dtype)
        dst_primitive = heap.allocate_tensor((block_size,), dtype)
        dst_high_level = heap.allocate_tensor((block_size,), dtype)

        _fill_reduce_scatter_input(
            src_primitive,
            rank=rank,
            world_size=world_size,
            block_size=block_size,
        )
        _fill_reduce_scatter_input(
            src_high_level,
            rank=rank,
            world_size=world_size,
            block_size=block_size,
        )
        src_kernel = None
        dst_kernel = None
        kernel_timing = None
        kernel_ok = None
        kernel_value = None
        if config.launcher in {"kernel", "all"}:
            src_kernel = heap.allocate_tensor((total_elements,), dtype)
            dst_kernel = heap.allocate_tensor((block_size,), dtype)
            _fill_reduce_scatter_input(
                src_kernel,
                rank=rank,
                world_size=world_size,
                block_size=block_size,
            )
            dst_kernel.zero_()
        dst_primitive.zero_()
        dst_high_level.zero_()
        torch.cuda.synchronize(rank)

        primitive_timing = None
        primitive_ok = None
        primitive_value = None
        if config.launcher in {"primitive", "all"}:
            primitive_timing = _timed_collective(
                lambda: primitive_reduce_scatter(
                    src_primitive,
                    dst_primitive,
                    heap,
                    implementation="device",
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        if config.launcher in {"kernel", "all"}:
            kernel_timing = _timed_collective(
                lambda: _reduce_scatter_kernel[(1,)](
                    src_kernel,
                    dst_kernel,
                    heap.get_heap_bases(),
                    rank,
                    world_size,
                    BLOCK_SIZE=block_size,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        ctx = xtile.init(
            backend="cuda",
            rank=rank,
            world_size=world_size,
            heap=heap,
            force_backend=True,
        )
        high_level_timing = None
        high_level_ok = None
        high_level_value = None
        if config.launcher in {"ops", "all"}:
            high_level_timing = _timed_collective(
                lambda: xtile.ops.reduce_scatter(
                    src_high_level,
                    dst_high_level,
                    ctx=ctx,
                    implementation="device",
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        expected = float((0 * 2 + rank + 1) + (1 * 2 + rank + 1))
        expected_tensor = torch.full_like(dst_primitive, expected)
        if config.launcher in {"primitive", "all"}:
            primitive_ok = bool(
                torch.allclose(dst_primitive, expected_tensor, atol=1e-4)
            )
            primitive_value = float(dst_primitive[0].item())
        if config.launcher in {"ops", "all"}:
            high_level_ok = bool(
                torch.allclose(dst_high_level, expected_tensor, atol=1e-4)
            )
            high_level_value = float(dst_high_level[0].item())
        if config.launcher in {"kernel", "all"}:
            assert dst_kernel is not None
            kernel_ok = bool(torch.allclose(dst_kernel, expected_tensor, atol=1e-4))
            kernel_value = float(dst_kernel[0].item())
        payload = {
            "rank": rank,
            "expected": expected,
            "primitive_value": primitive_value,
            "high_level_value": high_level_value,
            "kernel_value": kernel_value,
            "primitive_ok": primitive_ok,
            "high_level_ok": high_level_ok,
            "kernel_ok": kernel_ok,
            "transport_strategy": heap.transport_strategy,
            "mode": heap.mode,
            "dtype": config.dtype_name,
            "block_size": block_size,
            "warmup": config.warmup,
            "iters": config.iters,
            "forced_transport": config.force_transport or "auto",
            "launcher": config.launcher,
            "primitive_timing_ms": primitive_timing,
            "high_level_timing_ms": high_level_timing,
            "kernel_timing_ms": kernel_timing,
        }
        print(json.dumps(payload), flush=True)
        checks = [
            result
            for result in (primitive_ok, high_level_ok, kernel_ok)
            if result is not None
        ]
        if not checks or not all(checks):
            raise SystemExit(2)
    finally:
        dist.barrier(**barrier_kwargs)
        heap.cleanup()
        dist.destroy_process_group()


def main() -> None:
    """Run the real multiprocess validation on up to 2 visible GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=sorted(_DTYPES),
        help="Element dtype for the diagnostic source/output tensors.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Elements per rank-local reduce_scatter chunk.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Timed iterations for primitive/high-level latency statistics.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Requested world size. Must be <= visible GPUs. Current host runs up to 2.",
    )
    parser.add_argument(
        "--force-transport",
        type=str,
        default="auto",
        choices=["auto", "ctypes_ipc", "pytorch_ipc", "peer_access_pointer_exchange"],
        help="Force one specific multiprocess transport strategy for diagnostics.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="all",
        choices=["primitive", "ops", "kernel", "all"],
        help="Which launcher surface to validate: host primitive, high-level op, raw kernel, or all.",
    )
    args = parser.parse_args()

    world_size = min(torch.cuda.device_count(), int(args.world_size))
    if world_size < 2:
        raise SystemExit("Need >= 2 GPUs")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")

    config = _RunConfig(
        block_size=int(args.block_size),
        dtype_name=args.dtype,
        warmup=int(args.warmup),
        iters=int(args.iters),
        force_transport=None if args.force_transport == "auto" else args.force_transport,
        launcher=args.launcher,
    )

    store_fd, store_path = tempfile.mkstemp(prefix="xtile_rs_store_")
    os.close(store_fd)
    os.unlink(store_path)

    try:
        mp.start_processes(
            _worker,
            args=(world_size, store_path, config),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
    finally:
        try:
            os.unlink(store_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
