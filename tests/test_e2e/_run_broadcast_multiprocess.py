#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess broadcast validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tncc.utils.feature_gates import (
    FORCE_MULTIPROCESS_TRANSPORT_ENV,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_runtime_supported,
)


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess broadcast diagnostic run."""

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

_KERNEL_MAX_BLOCK_SIZE = 4096


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


def _launch_broadcast_kernel(
    *,
    tensor: torch.Tensor,
    heap_bases: torch.Tensor,
    rank: int,
    world_size: int,
    root: int,
    block_size: int,
    broadcast_kernel,
) -> None:
    """Launch raw broadcast kernels with a power-of-two-safe fallback path."""
    if block_size <= _KERNEL_MAX_BLOCK_SIZE and _is_power_of_two(block_size):
        broadcast_kernel[(1,)](
            tensor,
            heap_bases,
            rank,
            world_size,
            root,
            BLOCK_SIZE=block_size,
        )
        return

    chunk_start = 0
    while chunk_start < block_size:
        chunk = _kernel_block_elems(block_size - chunk_start)
        tensor_chunk = tensor.narrow(0, chunk_start, chunk)
        broadcast_kernel[(1,)](
            tensor_chunk,
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


def _fill_broadcast_input(tensor: torch.Tensor, *, rank: int, root: int) -> None:
    """Fill one rank-local tensor with a root/non-root distinguishable pattern."""
    if rank == root:
        tensor.fill_(111.0)
    else:
        tensor.fill_(-1.0)


def _launch_broadcast_ops_compat(
    *,
    tensor: torch.Tensor,
    heap,
    root: int,
) -> None:
    """Compatibility launcher for the worker's "ops" surface.

    ``tncc.ops`` does not currently expose ``broadcast``. Keep launcher-matrix
    parity in this worker by routing the "ops" lane through the same validated
    primitive entrypoint on an independent tensor.
    """
    from tncc.primitives import broadcast as primitive_broadcast

    primitive_broadcast(tensor, heap, root=root)


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
    from tncc.primitives import broadcast as primitive_broadcast
    from tncc.primitives.collectives import _broadcast_kernel, _resolve_collective_execution

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
                    operation="tests.test_e2e._run_broadcast_multiprocess",
                    world_size=world_size,
                )
            )

        block_size = config.block_size
        tensor_primitive = heap.allocate_tensor((block_size,), dtype)
        tensor_ops = heap.allocate_tensor((block_size,), dtype)
        tensor_kernel = None
        if config.launcher in {"kernel", "all"}:
            tensor_kernel = heap.allocate_tensor((block_size,), dtype)

        _fill_broadcast_input(tensor_primitive, rank=rank, root=config.root)
        _fill_broadcast_input(tensor_ops, rank=rank, root=config.root)
        if tensor_kernel is not None:
            _fill_broadcast_input(tensor_kernel, rank=rank, root=config.root)
        torch.cuda.synchronize(rank)

        primitive_execution = asdict(
            _resolve_collective_execution(
                "broadcast",
                input_numel=tensor_primitive.numel(),
                world_size=world_size,
                element_size=tensor_primitive.element_size(),
                device=tensor_primitive.device,
                root=config.root,
            )
        )

        primitive_timing = None
        primitive_ok = None
        primitive_first_value = None
        if config.launcher in {"primitive", "all"}:
            primitive_timing = _timed_collective(
                lambda: primitive_broadcast(tensor_primitive, heap, root=config.root),
                prepare_fn=lambda: _fill_broadcast_input(
                    tensor_primitive,
                    rank=rank,
                    root=config.root,
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
                lambda: _launch_broadcast_ops_compat(
                    tensor=tensor_ops,
                    heap=heap,
                    root=config.root,
                ),
                prepare_fn=lambda: _fill_broadcast_input(
                    tensor_ops,
                    rank=rank,
                    root=config.root,
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
            assert tensor_kernel is not None
            kernel_timing = _timed_collective(
                lambda: _launch_broadcast_kernel(
                    tensor=tensor_kernel,
                    heap_bases=heap.get_heap_bases(),
                    rank=rank,
                    world_size=world_size,
                    root=config.root,
                    block_size=block_size,
                    broadcast_kernel=_broadcast_kernel,
                ),
                prepare_fn=lambda: _fill_broadcast_input(
                    tensor_kernel,
                    rank=rank,
                    root=config.root,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        expected = torch.full_like(tensor_primitive, 111.0)
        if config.launcher in {"primitive", "all"}:
            primitive_ok = bool(torch.allclose(tensor_primitive, expected, atol=1e-4))
            primitive_first_value = float(tensor_primitive[0].item())
        if config.launcher in {"ops", "all"}:
            high_level_ok = bool(torch.allclose(tensor_ops, expected, atol=1e-4))
            high_level_first_value = float(tensor_ops[0].item())
        if config.launcher in {"kernel", "all"}:
            assert tensor_kernel is not None
            kernel_ok = bool(torch.allclose(tensor_kernel, expected, atol=1e-4))
            kernel_first_value = float(tensor_kernel[0].item())

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


def main() -> None:
    """Run the real multiprocess broadcast validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=sorted(_DTYPES),
        help="Element dtype for the diagnostic tensor.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Elements in the rank-local broadcast tensor.",
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
        help="Timed iterations for primitive/high-level/kernel latency statistics.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Requested world size. Current validated public surface is world_size=2.",
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
        help="Which launcher surface to validate: host primitive, compatibility ops lane, raw kernel, or all.",
    )
    parser.add_argument(
        "--root",
        type=int,
        default=0,
        help="Root rank for broadcast.",
    )
    args = parser.parse_args()

    requested_world_size = int(args.world_size)
    if requested_world_size != 2:
        raise SystemExit("--world-size must be 2 on the current validated public surface")
    if torch.cuda.device_count() < requested_world_size:
        raise SystemExit("Need >= 2 GPUs")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")
    if args.root < 0 or args.root >= requested_world_size:
        raise SystemExit("--root must be in [0, world_size)")

    config = _RunConfig(
        block_size=args.block_size,
        dtype_name=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        force_transport=None if args.force_transport == "auto" else args.force_transport,
        launcher=args.launcher,
        root=args.root,
    )

    store_fd, store_path = tempfile.mkstemp(prefix="tncc_broadcast_store_")
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
