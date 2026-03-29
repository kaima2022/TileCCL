# SPDX-License-Identifier: Apache-2.0
"""Multi-process GEMM + AllScatter via torchrun.

Each process owns one GPU.  TNCC detects rank and world_size from the
torch.distributed environment and sets up IPC-based symmetric memory.

Requirements: 2x NVIDIA GPUs with NVLink.

Usage:
    torchrun --nproc_per_node=2 examples/multiprocess.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist

import tncc


def main() -> None:
    dist.init_process_group(backend="nccl")

    ctx = tncc.init(backend="auto", heap_size=512 * 1024 * 1024)

    M, K, N = 4096, 4096, 8192
    A = ctx.randn(M, K, dtype=torch.float16)
    B = ctx.randn(K, N, dtype=torch.float16)
    C = ctx.zeros(M, N, dtype=torch.float16)

    # Fused GEMM + all-scatter.
    tncc.ops.gemm_allscatter(A, B, C, ctx=ctx)
    torch.cuda.synchronize()

    # Verify against a local torch.matmul reference.
    ref = torch.matmul(A.float(), B.float()).half()
    max_err = (C - ref).abs().max().item()
    print(f"[rank {ctx.rank}] max error vs torch.matmul: {max_err:.4e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
