#!/usr/bin/env python3
"""Real multiprocess Triton remote-access validation matrix worker.

This isolates the minimal device-side requirement behind GEMM scatter and
device collectives: ``translate_ptr + tl.load/tl.store`` across ranks.
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
import triton
import triton.language as tl

from xtile.memory.translation import translate_ptr
from xtile.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess Triton remote-access diagnostic run."""

    block_size: int
    dtype_name: str
    warmup: int
    iters: int
    force_transport: str | None
    operation: str


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


@triton.jit
def _remote_load_kernel(
    local_mirror_ptr,
    output_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Read a remote symmetric block into a local output buffer."""
    offsets = tl.arange(0, BLOCK_SIZE)
    remote_ptr = translate_ptr(
        local_mirror_ptr + offsets,
        caller_rank,
        remote_rank,
        heap_bases,
        HINT=BLOCK_SIZE,
    )
    values = tl.load(remote_ptr)
    tl.store(output_ptr + offsets, values)


@triton.jit
def _remote_store_kernel(
    local_src_ptr,
    local_mirror_dst_ptr,
    heap_bases,
    caller_rank,
    remote_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Write a local block into the peer's symmetric destination buffer."""
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(local_src_ptr + offsets)
    remote_ptr = translate_ptr(
        local_mirror_dst_ptr + offsets,
        caller_rank,
        remote_rank,
        heap_bases,
        HINT=BLOCK_SIZE,
    )
    tl.store(remote_ptr, values)


def _timed_kernel(
    fn,
    *,
    rank: int,
    barrier_kwargs: dict[str, object],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Time one Triton kernel launch using CUDA events and rank barriers."""
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

    from xtile.memory.symmetric_heap import SymmetricHeap

    heap = SymmetricHeap(
        size=64 * 1024 * 1024,
        rank=rank,
        world_size=world_size,
        backend="cuda",
    )
    try:
        block_size = config.block_size
        remote_load_mirror = heap.allocate_tensor((block_size,), dtype)
        remote_load_output = heap.allocate_tensor((block_size,), dtype)
        remote_store_src = heap.allocate_tensor((block_size,), dtype)
        remote_store_dst_mirror = heap.allocate_tensor((block_size,), dtype)

        remote_load_mirror.fill_(float(rank + 1))
        remote_load_output.zero_()
        remote_store_src.fill_(float(10 + rank))
        remote_store_dst_mirror.zero_()
        torch.cuda.synchronize(rank)

        peer_rank = (rank + 1) % world_size
        heap_bases = heap.get_heap_bases()

        load_timing = None
        load_ok = None
        load_value = None
        if config.operation in {"load", "both"}:
            load_timing = _timed_kernel(
                lambda: _remote_load_kernel[(1,)](
                    remote_load_mirror,
                    remote_load_output,
                    heap_bases,
                    rank,
                    peer_rank,
                    BLOCK_SIZE=block_size,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            expected_load = torch.full_like(remote_load_output, float(peer_rank + 1))
            load_ok = bool(torch.allclose(remote_load_output, expected_load, atol=1e-4))
            load_value = float(remote_load_output[0].item())

        store_timing = None
        store_ok = None
        store_value = None
        if config.operation in {"store", "both"}:
            remote_store_dst_mirror.zero_()
            torch.cuda.synchronize(rank)
            dist.barrier(**barrier_kwargs)
            store_timing = _timed_kernel(
                lambda: _remote_store_kernel[(1,)](
                    remote_store_src,
                    remote_store_dst_mirror,
                    heap_bases,
                    rank,
                    peer_rank,
                    BLOCK_SIZE=block_size,
                ),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )
            dist.barrier(**barrier_kwargs)
            expected_store = torch.full_like(
                remote_store_dst_mirror,
                float(10 + ((rank - 1 + world_size) % world_size)),
            )
            store_ok = bool(
                torch.allclose(remote_store_dst_mirror, expected_store, atol=1e-4)
            )
            store_value = float(remote_store_dst_mirror[0].item())

        payload = {
            "rank": rank,
            "peer_rank": peer_rank,
            "dtype": config.dtype_name,
            "block_size": block_size,
            "warmup": config.warmup,
            "iters": config.iters,
            "forced_transport": config.force_transport or "auto",
            "operation": config.operation,
            "transport_strategy": heap.transport_strategy,
            "mode": heap.mode,
            "load_value": load_value,
            "load_ok": load_ok,
            "load_timing_ms": load_timing,
            "store_value": store_value,
            "store_ok": store_ok,
            "store_timing_ms": store_timing,
        }
        print(json.dumps(payload), flush=True)

        checks = [result for result in (load_ok, store_ok) if result is not None]
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
    """Run the real multiprocess Triton remote-access validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=sorted(_DTYPES),
        help="Element dtype for the diagnostic tensors.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Elements per rank-local block.",
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
        help="Timed iterations for latency statistics.",
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
        "--operation",
        type=str,
        default="both",
        choices=["load", "store", "both"],
        help="Which minimal Triton remote-access operation to validate.",
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
        operation=args.operation,
    )

    store_fd, store_path = tempfile.mkstemp(prefix="xtile_triton_remote_store_")
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
