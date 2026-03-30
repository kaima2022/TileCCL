#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess allreduce validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tncc.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for one multiprocess allreduce diagnostic run."""

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
    prepare_fn=None,
    rank: int,
    barrier_kwargs: dict[str, object],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Time one collective call using CUDA events and rank barriers."""
    for _ in range(warmup):
        if prepare_fn is not None:
            prepare_fn()
            torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)
        fn()
        torch.cuda.synchronize(rank)
        dist.barrier(**barrier_kwargs)

    times_ms: list[float] = []
    for _ in range(iters):
        if prepare_fn is not None:
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


def _fill_allreduce_input(tensor: torch.Tensor, *, rank: int) -> None:
    """Fill one rank-local tensor with a stable scalar pattern."""
    tensor.fill_(float(rank + 1))


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

    import tncc
    from tncc.memory.symmetric_heap import SymmetricHeap
    from tncc.primitives import allreduce as primitive_allreduce
    from tncc.primitives.collectives import _allreduce_kernel, resolve_allreduce_execution

    heap = SymmetricHeap(
        size=64 * 1024 * 1024,
        rank=rank,
        world_size=world_size,
        backend="cuda",
    )
    try:
        total_elements = config.block_size * world_size

        tensor_primitive = heap.allocate_tensor((total_elements,), dtype)
        tensor_ops = heap.allocate_tensor((total_elements,), dtype)
        _fill_allreduce_input(tensor_primitive, rank=rank)
        _fill_allreduce_input(tensor_ops, rank=rank)

        tensor_kernel = None
        if config.launcher in {"kernel", "all"}:
            tensor_kernel = heap.allocate_tensor((total_elements,), dtype)
            _fill_allreduce_input(tensor_kernel, rank=rank)

        torch.cuda.synchronize(rank)

        primitive_timing = None
        primitive_ok = None
        primitive_first_value = None
        primitive_execution = None
        if config.launcher in {"primitive", "primitive_ops", "all"}:
            primitive_execution = resolve_allreduce_execution(
                tensor_primitive,
                heap=heap,
                op="sum",
            ).to_dict()
            primitive_timing = _timed_collective(
                lambda: primitive_allreduce(tensor_primitive, heap),
                prepare_fn=lambda: _fill_allreduce_input(tensor_primitive, rank=rank),
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
                lambda: _allreduce_kernel[(1,)](
                    tensor_kernel,
                    heap.get_heap_bases(),
                    rank,
                    world_size,
                    BLOCK_SIZE=config.block_size,
                ),
                prepare_fn=lambda: _fill_allreduce_input(tensor_kernel, rank=rank),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        ctx = tncc.init(
            backend="cuda",
            rank=rank,
            world_size=world_size,
            heap=heap,
            force_backend=True,
        )
        high_level_timing = None
        high_level_ok = None
        high_level_first_value = None
        high_level_plan = None
        if config.launcher in {"ops", "primitive_ops", "all"}:
            high_level_plan = tncc.ops.build_allreduce_plan(
                tensor_ops,
                ctx=ctx,
            ).to_dict()
            high_level_timing = _timed_collective(
                lambda: tncc.ops.allreduce(
                    tensor_ops,
                    ctx=ctx,
                ),
                prepare_fn=lambda: _fill_allreduce_input(tensor_ops, rank=rank),
                rank=rank,
                barrier_kwargs=barrier_kwargs,
                warmup=config.warmup,
                iters=config.iters,
            )

        expected = torch.full_like(tensor_primitive, float(sum(range(1, world_size + 1))))
        if config.launcher in {"primitive", "primitive_ops", "all"}:
            primitive_ok = bool(torch.allclose(tensor_primitive, expected, atol=1e-4))
            primitive_first_value = float(tensor_primitive[0].item())
        if config.launcher in {"ops", "primitive_ops", "all"}:
            high_level_ok = bool(torch.allclose(tensor_ops, expected, atol=1e-4))
            high_level_first_value = float(tensor_ops[0].item())
        if config.launcher in {"kernel", "all"}:
            assert tensor_kernel is not None
            kernel_ok = bool(torch.allclose(tensor_kernel, expected, atol=1e-4))
            kernel_first_value = float(tensor_kernel[0].item())

        payload = {
            "rank": rank,
            "dtype": config.dtype_name,
            "block_size": config.block_size,
            "total_elements": total_elements,
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
            "high_level_plan": high_level_plan,
            "kernel_first_value": kernel_first_value,
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
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multiprocess diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--force-transport",
        choices=["ctypes_ipc", "pytorch_ipc", "peer_access_pointer_exchange"],
        default=None,
        help="Force one specific multiprocess transport strategy for diagnostics.",
    )
    parser.add_argument(
        "--launcher",
        choices=["primitive", "ops", "primitive_ops", "kernel", "all"],
        default="primitive_ops",
        help="Which launcher(s) to validate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the real multiprocess allreduce validation."""
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 GPUs for multiprocess allreduce validation.")

    args = _parse_args()
    config = _RunConfig(
        block_size=int(args.block_size),
        dtype_name=str(args.dtype),
        warmup=int(args.warmup),
        iters=int(args.iters),
        force_transport=args.force_transport,
        launcher=str(args.launcher),
    )

    with tempfile.TemporaryDirectory(prefix="tncc_allreduce_mp_") as tmpdir:
        store_path = os.path.join(tmpdir, "store")
        mp.spawn(
            _worker,
            args=(2, store_path, config),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    main()
