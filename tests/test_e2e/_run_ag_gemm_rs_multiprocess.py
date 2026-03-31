#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess validation for allgather_gemm_reducescatter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tncc.utils.feature_gates import FORCE_MULTIPROCESS_TRANSPORT_ENV

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _make_ranked_matrix(
    rows: int,
    cols: int,
    *,
    rank: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    row_idx = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    col_idx = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(0)
    values = (row_idx * float(13 + rank + seed) + col_idx * float(7 + 2 * rank) + float(rank * 5)) % 31.0
    return (values - 15.0) / 8.0


def _expected_output(
    *,
    world_size: int,
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    K_shard = K // world_size
    shards = [
        _make_ranked_matrix(M, K_shard, rank=r, seed=0)
        for r in range(world_size)
    ]
    full_input = torch.cat(shards, dim=1)
    W = _make_ranked_matrix(K, N, rank=0, seed=100)
    return torch.matmul(full_input, W).to(dtype=dtype).contiguous()


def _worker(rank: int, world_size: int, store_path: str, M: int, K: int, N: int, dtype_name: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = _DTYPES[dtype_name]

    os.environ.pop(FORCE_MULTIPROCESS_TRANSPORT_ENV, None)

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size, device_id=device)

    import tncc
    from tncc.memory.symmetric_heap import SymmetricHeap

    element_size = torch.empty((), dtype=dtype).element_size()
    heap_size = max(128 * 1024 * 1024, M * K * element_size * 8)
    heap = SymmetricHeap(size=heap_size, rank=rank, world_size=world_size, backend="cuda")
    try:
        K_shard = K // world_size
        input_shard = heap.allocate_tensor((M, K_shard), dtype)
        pattern = _make_ranked_matrix(M, K_shard, rank=rank, seed=0, device=device)
        input_shard.copy_(pattern.to(dtype=dtype))

        W = torch.empty((K, N), device=device, dtype=dtype)
        W.copy_(_make_ranked_matrix(K, N, rank=0, seed=100, device=device).to(dtype=dtype))

        output = heap.allocate_tensor((M, N), dtype)
        output.zero_()
        torch.cuda.synchronize(rank)

        ctx = tncc.init(backend="cuda", rank=rank, world_size=world_size, heap=heap, force_backend=True)

        # Test plan builder
        plan = tncc.ops.build_allgather_gemm_reducescatter_plan(
            input_shard, W, output, ctx=ctx
        )
        dist.barrier(device_ids=[rank])
        plan.execute(input_shard, W, output, validate=False)
        torch.cuda.synchronize(rank)
        dist.barrier(device_ids=[rank])

        expected = _expected_output(world_size=world_size, M=M, K=K, N=N, dtype=dtype).to(device=device)

        rtol, atol = (1e-2, 2e-1) if dtype in (torch.float16, torch.bfloat16) else (1e-3, 5e-2)
        plan_ok = bool(torch.allclose(output, expected, rtol=rtol, atol=atol))
        plan_max_diff = float((output.float() - expected.float()).abs().max().item())

        # Test high-level API
        output.zero_()
        dist.barrier(device_ids=[rank])
        tncc.ops.allgather_gemm_reducescatter(input_shard, W, output, ctx=ctx)
        torch.cuda.synchronize(rank)
        dist.barrier(device_ids=[rank])

        hl_ok = bool(torch.allclose(output, expected, rtol=rtol, atol=atol))
        hl_max_diff = float((output.float() - expected.float()).abs().max().item())

        payload = {
            "rank": rank,
            "dtype": dtype_name,
            "M": M, "K": K, "N": N,
            "mode": heap.mode,
            "transport_strategy": heap.transport_strategy,
            "plan_ok": plan_ok,
            "plan_max_abs_diff": plan_max_diff,
            "high_level_ok": hl_ok,
            "high_level_max_abs_diff": hl_max_diff,
        }
        print(json.dumps(payload), flush=True)
        if not plan_ok or not hl_ok:
            raise SystemExit(2)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier(device_ids=[rank])
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32", choices=sorted(_DTYPES))
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 GPUs.")
    if args.K % 2 != 0:
        raise SystemExit("K must be divisible by world_size (2).")

    with tempfile.TemporaryDirectory(prefix="tncc_ag_gemm_rs_") as tmpdir:
        store_path = str(Path(tmpdir) / "dist_store")
        mp.spawn(_worker, args=(2, store_path, args.M, args.K, args.N, args.dtype), nprocs=2, join=True)


if __name__ == "__main__":
    main()
