#!/usr/bin/env python3
"""Real multiprocess allgather validation."""

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
    """Configuration for one multiprocess allgather diagnostic run."""

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


def _fill_allgather_input(tensor: torch.Tensor, *, rank: int) -> None:
    """Fill one rank-local block with a stable scalar pattern."""
    tensor.fill_(float((rank + 1) * 10))


def _expected_output(
    output: torch.Tensor,
    *,
    world_size: int,
    block_size: int,
) -> torch.Tensor:
    """Build the expected allgather output on the local device."""
    expected = torch.empty_like(output)
    for rank in range(world_size):
        start = rank * block_size
        expected[start:start + block_size].fill_(float((rank + 1) * 10))
    return expected


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
    from xtile.primitives.collectives import _allgather_kernel
    from xtile.primitives import allgather as primitive_allgather

    heap = SymmetricHeap(
        size=64 * 1024 * 1024,
        rank=rank,
        world_size=world_size,
        backend="cuda",
    )
    try:
        block_size = config.block_size

        src_primitive = heap.allocate_tensor((block_size,), dtype)
        dst_primitive = heap.allocate_tensor((block_size * world_size,), dtype)
        _fill_allgather_input(src_primitive, rank=rank)
        dst_primitive.zero_()

        src_ops = heap.allocate_tensor((block_size,), dtype)
        dst_ops = heap.allocate_tensor((block_size * world_size,), dtype)
        _fill_allgather_input(src_ops, rank=rank)
        dst_ops.zero_()

        src_kernel = None
        dst_kernel = None
        if config.launcher in {"kernel", "all"}:
            src_kernel = heap.allocate_tensor((block_size,), dtype)
            dst_kernel = heap.allocate_tensor((block_size * world_size,), dtype)
            _fill_allgather_input(src_kernel, rank=rank)
            dst_kernel.zero_()

        torch.cuda.synchronize(rank)

        primitive_timing = None
        primitive_ok = None
        primitive_first_chunk = None
        if config.launcher in {"primitive", "all"}:
            primitive_timing = _timed_collective(
                lambda: primitive_allgather(
                    src_primitive,
                    dst_primitive,
                    heap,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        kernel_timing = None
        kernel_ok = None
        kernel_first_chunk = None
        if config.launcher in {"kernel", "all"}:
            kernel_timing = _timed_collective(
                lambda: _allgather_kernel[(1,)](
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
        high_level_first_chunk = None
        if config.launcher in {"ops", "all"}:
            high_level_timing = _timed_collective(
                lambda: xtile.ops.allgather(
                    src_ops,
                    dst_ops,
                    ctx=ctx,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        expected = _expected_output(
            dst_primitive,
            world_size=world_size,
            block_size=block_size,
        )
        if config.launcher in {"primitive", "all"}:
            primitive_ok = bool(torch.allclose(dst_primitive, expected, atol=1e-4))
            primitive_first_chunk = float(dst_primitive[0].item())
        if config.launcher in {"ops", "all"}:
            high_level_ok = bool(torch.allclose(dst_ops, expected, atol=1e-4))
            high_level_first_chunk = float(dst_ops[0].item())
        if config.launcher in {"kernel", "all"}:
            assert dst_kernel is not None
            kernel_ok = bool(torch.allclose(dst_kernel, expected, atol=1e-4))
            kernel_first_chunk = float(dst_kernel[0].item())

        payload = {
            "rank": rank,
            "dtype": config.dtype_name,
            "block_size": block_size,
            "warmup": config.warmup,
            "iters": config.iters,
            "forced_transport": config.force_transport or "auto",
            "launcher": config.launcher,
            "transport_strategy": heap.transport_strategy,
            "mode": heap.mode,
            "primitive_first_chunk": primitive_first_chunk,
            "primitive_ok": primitive_ok,
            "primitive_timing_ms": primitive_timing,
            "high_level_first_chunk": high_level_first_chunk,
            "high_level_ok": high_level_ok,
            "high_level_timing_ms": high_level_timing,
            "kernel_first_chunk": kernel_first_chunk,
            "kernel_ok": kernel_ok,
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
    """Run the real multiprocess allgather validation."""
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
        help="Elements per rank-local allgather contribution.",
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

    store_fd, store_path = tempfile.mkstemp(prefix="xtile_ag_store_")
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
