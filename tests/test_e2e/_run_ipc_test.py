#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU IPC test using file-based store.

Usage:
    python3 -m tests.test_e2e._run_ipc_test
"""

import ctypes
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, store_path: str):
    """Per-rank worker: create heap, exchange IPC, verify P2P."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    barrier_kwargs = {"device_ids": [rank]}

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        "nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    print(f"Rank {rank}/{world_size}: init on cuda:{rank}", flush=True)

    from tncc.memory.symmetric_heap import SymmetricHeap

    try:
        heap = SymmetricHeap(
            size=4 * 1024 * 1024, rank=rank, world_size=world_size,
        )
    except Exception as e:
        print(f"Rank {rank}: heap creation failed: {e}", flush=True)
        dist.destroy_process_group()
        raise

    print(f"Rank {rank}: heap OK, base=0x{heap.local_base:x}", flush=True)

    # Allocate and fill
    t = heap.allocate_tensor((1024,), dtype=torch.float32)
    t.fill_(float(rank + 1))
    torch.cuda.synchronize()

    bases = heap.get_heap_bases()
    print(
        f"Rank {rank}: bases=[{', '.join(hex(b.item()) for b in bases)}]",
        flush=True,
    )

    # Host-side translation
    other = 1 - rank
    offset = t.data_ptr() - heap.local_base
    translated = heap.translate(t.data_ptr(), to_rank=other)
    expected_ptr = bases[other].item() + offset
    assert translated == expected_ptr, "translate mismatch"
    print(f"Rank {rank}: translate OK", flush=True)

    heap.barrier()

    # P2P read via cudaMemcpy
    remote_ptr = bases[other].item() + offset
    local_verify = torch.empty(1, dtype=torch.float32, device=f"cuda:{rank}")
    cuda = ctypes.CDLL("libcudart.so")
    err = cuda.cudaMemcpy(
        ctypes.c_void_p(local_verify.data_ptr()),
        ctypes.c_void_p(remote_ptr),
        ctypes.c_size_t(4),
        ctypes.c_int(3),
    )
    torch.cuda.synchronize()

    expected_val = float(other + 1)
    actual_val = local_verify[0].item()
    assert err == 0, f"cudaMemcpy err={err}"
    assert abs(actual_val - expected_val) < 1e-6, (
        f"P2P: got {actual_val}, expected {expected_val}"
    )
    print(f"Rank {rank}: P2P read OK ({actual_val} from rank {other})", flush=True)

    heap.barrier()
    heap.cleanup()
    dist.barrier(**barrier_kwargs)
    dist.destroy_process_group()

    if rank == 0:
        print("ALL MULTI-GPU IPC TESTS PASSED", flush=True)


def main():
    world_size = min(torch.cuda.device_count(), 2)
    assert world_size >= 2, f"Need >= 2 GPUs, got {world_size}"

    store_fd, store_path = tempfile.mkstemp(prefix="tncc_store_")
    os.close(store_fd)
    os.unlink(store_path)

    try:
        mp.start_processes(
            _worker,
            args=(world_size, store_path),
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
