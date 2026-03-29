# SPDX-License-Identifier: Apache-2.0
"""Worker function for multi-GPU IPC tests (must be importable for mp.spawn)."""

import ctypes
import os
import torch
import torch.distributed as dist


def ipc_test_worker(rank: int, world_size: int, results_dict: dict) -> None:
    """Test SymmetricHeap IPC and P2P access between two GPUs."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29510"
    torch.cuda.set_device(rank)
    # Use gloo backend: all_gather_object requires CPU-capable backend,
    # and gloo is more reliable for local single-node testing.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        from tncc.memory.symmetric_heap import SymmetricHeap

        heap = SymmetricHeap(size=4 * 1024 * 1024, rank=rank, world_size=world_size)

        # Allocate and fill a tensor with rank-specific value
        t = heap.allocate_tensor((1024,), dtype=torch.float32)
        t.fill_(float(rank + 1))  # rank 0 -> 1.0, rank 1 -> 2.0
        torch.cuda.synchronize()

        bases = heap.get_heap_bases()
        other = 1 - rank

        # Host-side translation test
        offset = t.data_ptr() - heap.local_base
        translated = heap.translate(t.data_ptr(), to_rank=other)
        expected_ptr = bases[other].item() + offset

        heap.barrier()

        # Device-side P2P read: read first element from other rank's tensor
        remote_ptr = bases[other].item() + offset
        local_verify = torch.empty(1, dtype=torch.float32, device=f"cuda:{rank}")

        cuda = ctypes.CDLL("libcudart.so")
        err = cuda.cudaMemcpy(
            ctypes.c_void_p(local_verify.data_ptr()),
            ctypes.c_void_p(remote_ptr),
            ctypes.c_size_t(4),
            ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
        )
        torch.cuda.synchronize()

        expected_val = float(other + 1)
        actual_val = local_verify[0].item()

        results_dict[rank] = {
            "translate_ok": translated == expected_ptr,
            "p2p_read_ok": abs(actual_val - expected_val) < 1e-6,
            "actual_val": actual_val,
            "expected_val": expected_val,
            "cuda_err": err,
        }

        heap.barrier()
        heap.cleanup()
    finally:
        dist.destroy_process_group()
