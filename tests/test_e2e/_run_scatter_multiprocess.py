#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess scatter validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tncc.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV

_KERNEL_MAX_BLOCK_SIZE = 4096


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess scatter diagnostic run."""

    block_size: int
    dtype_name: str
    warmup: int
    iters: int
    force_transport: str | None
    launcher: str
    root: int


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


def _is_power_of_two(value: int) -> bool:
    """Return ``True`` when *value* is a power of two."""
    return value > 0 and (value & (value - 1)) == 0


def _kernel_block_elems(remaining: int) -> int:
    """Return one Triton-safe kernel block size for the remaining tail."""
    capped = max(1, min(remaining, _KERNEL_MAX_BLOCK_SIZE))
    return 1 << (capped.bit_length() - 1)


def _launch_scatter_kernel(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    heap_bases: torch.Tensor,
    rank: int,
    world_size: int,
    root: int,
    block_size: int,
    scatter_kernel,
) -> None:
    """Launch raw scatter kernels with a power-of-two-safe fallback path."""
    if block_size <= _KERNEL_MAX_BLOCK_SIZE and _is_power_of_two(block_size):
        scatter_kernel[(1,)](
            src,
            dst,
            heap_bases,
            rank,
            world_size,
            root,
            BLOCK_SIZE=block_size,
        )
        return

    src_view = src.view(world_size, block_size)
    packed_chunk = None
    if rank == root:
        packed_chunk = torch.empty(
            (world_size * _KERNEL_MAX_BLOCK_SIZE,),
            dtype=src.dtype,
            device=src.device,
        )

    chunk_start = 0
    while chunk_start < block_size:
        chunk = _kernel_block_elems(block_size - chunk_start)
        dst_chunk = dst.narrow(0, chunk_start, chunk)
        if rank == root:
            assert packed_chunk is not None
            packed_chunk_view = packed_chunk.narrow(0, 0, world_size * chunk).view(
                world_size,
                chunk,
            )
            packed_chunk_view.copy_(src_view[:, chunk_start : chunk_start + chunk])
            src_chunk = packed_chunk.narrow(0, 0, world_size * chunk)
        else:
            src_chunk = src.narrow(0, 0, world_size * chunk)

        scatter_kernel[(1,)](
            src_chunk,
            dst_chunk,
            heap_bases,
            rank,
            world_size,
            root,
            BLOCK_SIZE=chunk,
        )
        chunk_start += chunk


def _timed_collective(
    fn,
    *,
    prepare_fn,
    rank: int,
    barrier_kwargs: dict[str, object],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Time one collective call using CUDA events and rank barriers."""
    for _ in range(warmup):
        prepare_fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)
        fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)

    times_ms: list[float] = []
    for _ in range(iters):
        prepare_fn()
        torch.cuda.synchronize(rank)
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


def _fill_scatter_input(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    rank: int,
    root: int,
    world_size: int,
    block_size: int,
) -> None:
    """Fill rank-local scatter input/output with a stable pattern."""
    if rank == root:
        for chunk_idx in range(world_size):
            start = chunk_idx * block_size
            src[start : start + block_size].fill_(float((chunk_idx + 1) * 10))
    else:
        src.fill_(-1.0)
    dst.fill_(-2.0)


def _launch_scatter_ops_compat(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    heap,
    root: int,
) -> None:
    """Compatibility launcher for the worker's "ops" surface.

    ``tncc.ops`` does not currently expose ``scatter``. Keep launcher-matrix
    parity in this worker by routing the "ops" lane through the same validated
    primitive entrypoint on independent tensors.
    """
    from tncc.primitives import scatter as primitive_scatter

    primitive_scatter(src, dst, heap, root=root)


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

    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import scatter as primitive_scatter
    from tncc.primitives.collectives import _resolve_collective_execution, _scatter_kernel
    from tncc.utils.feature_gates import (
        multiprocess_device_remote_access_detail,
        multiprocess_device_remote_access_runtime_supported,
    )

    heap = None
    try:
        heap = SymmetricHeap(
            size=64 * 1024 * 1024,
            rank=rank,
            world_size=world_size,
            backend="cuda",
        )

        if not multiprocess_device_remote_access_runtime_supported(
            transport_strategy=heap.transport_strategy,
            world_size=world_size,
        ):
            raise ValueError(
                multiprocess_device_remote_access_detail(
                    transport_strategy=heap.transport_strategy,
                    operation="tests.test_e2e._run_scatter_multiprocess",
                    world_size=world_size,
                )
            )

        block_size = config.block_size
        total_elements = block_size * world_size
        src_primitive = heap.allocate_tensor((total_elements,), dtype)
        dst_primitive = heap.allocate_tensor((block_size,), dtype)

        src_ops = heap.allocate_tensor((total_elements,), dtype)
        dst_ops = heap.allocate_tensor((block_size,), dtype)

        src_kernel = None
        dst_kernel = None
        if config.launcher in {"kernel", "all"}:
            src_kernel = heap.allocate_tensor((total_elements,), dtype)
            dst_kernel = heap.allocate_tensor((block_size,), dtype)

        _fill_scatter_input(
            src_primitive,
            dst_primitive,
            rank=rank,
            root=config.root,
            world_size=world_size,
            block_size=block_size,
        )
        _fill_scatter_input(
            src_ops,
            dst_ops,
            rank=rank,
            root=config.root,
            world_size=world_size,
            block_size=block_size,
        )
        if config.launcher in {"kernel", "all"}:
            assert src_kernel is not None
            assert dst_kernel is not None
            _fill_scatter_input(
                src_kernel,
                dst_kernel,
                rank=rank,
                root=config.root,
                world_size=world_size,
                block_size=block_size,
            )
        torch.cuda.synchronize(rank)

        primitive_execution = asdict(
            _resolve_collective_execution(
                "scatter",
                input_numel=dst_primitive.numel(),
                world_size=world_size,
                element_size=src_primitive.element_size(),
                device=src_primitive.device,
                root=config.root,
            )
        )

        primitive_timing = None
        primitive_ok = None
        primitive_first_value = None
        if config.launcher in {"primitive", "all"}:
            primitive_timing = _timed_collective(
                lambda: primitive_scatter(
                    src_primitive,
                    dst_primitive,
                    heap,
                    root=config.root,
                ),
                prepare_fn=lambda: _fill_scatter_input(
                    src_primitive,
                    dst_primitive,
                    rank=rank,
                    root=config.root,
                    world_size=world_size,
                    block_size=block_size,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        high_level_timing = None
        high_level_ok = None
        high_level_first_value = None
        if config.launcher in {"ops", "all"}:
            high_level_timing = _timed_collective(
                lambda: _launch_scatter_ops_compat(
                    src=src_ops,
                    dst=dst_ops,
                    heap=heap,
                    root=config.root,
                ),
                prepare_fn=lambda: _fill_scatter_input(
                    src_ops,
                    dst_ops,
                    rank=rank,
                    root=config.root,
                    world_size=world_size,
                    block_size=block_size,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        kernel_timing = None
        kernel_ok = None
        kernel_first_value = None
        if config.launcher in {"kernel", "all"}:
            assert src_kernel is not None
            assert dst_kernel is not None
            kernel_timing = _timed_collective(
                lambda: _launch_scatter_kernel(
                    src=src_kernel,
                    dst=dst_kernel,
                    heap_bases=heap.get_heap_bases(),
                    rank=rank,
                    world_size=world_size,
                    root=config.root,
                    block_size=block_size,
                    scatter_kernel=_scatter_kernel,
                ),
                prepare_fn=lambda: _fill_scatter_input(
                    src_kernel,
                    dst_kernel,
                    rank=rank,
                    root=config.root,
                    world_size=world_size,
                    block_size=block_size,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        expected_value = float((rank + 1) * 10)
        expected = torch.full_like(dst_primitive, expected_value)
        if config.launcher in {"primitive", "all"}:
            primitive_ok = bool(torch.allclose(dst_primitive, expected, atol=1e-4))
            primitive_first_value = float(dst_primitive[0].item())
        if config.launcher in {"ops", "all"}:
            high_level_ok = bool(torch.allclose(dst_ops, expected, atol=1e-4))
            high_level_first_value = float(dst_ops[0].item())
        if config.launcher in {"kernel", "all"}:
            assert dst_kernel is not None
            kernel_ok = bool(torch.allclose(dst_kernel, expected, atol=1e-4))
            kernel_first_value = float(dst_kernel[0].item())

        payload = {
            "rank": rank,
            "dtype": config.dtype_name,
            "block_size": block_size,
            "world_size": world_size,
            "root": config.root,
            "warmup": config.warmup,
            "iters": config.iters,
            "forced_transport": config.force_transport or "auto",
            "launcher": config.launcher,
            "transport_strategy": heap.transport_strategy,
            "mode": heap.mode,
            "primitive_first_value": primitive_first_value,
            "primitive_ok": primitive_ok,
            "primitive_timing_ms": primitive_timing,
            "primitive_execution": primitive_execution,
            "high_level_first_value": high_level_first_value,
            "high_level_ok": high_level_ok,
            "high_level_timing_ms": high_level_timing,
            "kernel_first_value": kernel_first_value,
            "kernel_ok": kernel_ok,
            "kernel_timing_ms": kernel_timing,
        }
        print(json.dumps(payload), flush=True)

        checks = [
            result for result in (primitive_ok, high_level_ok, kernel_ok) if result is not None
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
            if heap is not None:
                heap.cleanup()
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multiprocess diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4097)
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Requested world size. Current validated public surface is world_size=2.",
    )
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument(
        "--force-transport",
        type=str,
        default="auto",
        choices=["auto", "ctypes_ipc", "pytorch_ipc", "peer_access_pointer_exchange"],
        help="Force one specific multiprocess transport strategy for diagnostics.",
    )
    parser.add_argument(
        "--launcher",
        choices=["primitive", "ops", "kernel", "all"],
        default="all",
        help="Which launcher(s) to validate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the real multiprocess scatter validation."""
    args = _parse_args()
    requested_world_size = int(args.world_size)
    if requested_world_size != 2:
        raise SystemExit("--world-size must be 2 on the current validated public surface")
    if torch.cuda.device_count() < requested_world_size:
        raise SystemExit("Need >= 2 GPUs for multiprocess scatter validation.")
    if int(args.block_size) <= 0:
        raise SystemExit("--block-size must be positive")
    if int(args.warmup) < 0:
        raise SystemExit("--warmup must be >= 0")
    if int(args.iters) <= 0:
        raise SystemExit("--iters must be > 0")
    if int(args.root) < 0 or int(args.root) >= requested_world_size:
        raise SystemExit("--root must be in [0, world_size)")

    config = _RunConfig(
        block_size=int(args.block_size),
        dtype_name=str(args.dtype),
        warmup=int(args.warmup),
        iters=int(args.iters),
        force_transport=None if args.force_transport == "auto" else args.force_transport,
        launcher=str(args.launcher),
        root=int(args.root),
    )

    store_fd, store_path = tempfile.mkstemp(prefix="tncc_scatter_store_")
    os.close(store_fd)
    os.unlink(store_path)

    try:
        mp.start_processes(
            _worker,
            args=(requested_world_size, store_path, config),
            nprocs=requested_world_size,
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
