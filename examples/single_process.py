# SPDX-License-Identifier: Apache-2.0
"""Single-process multi-GPU GEMM + AllScatter.

Demonstrates the simplest usage path: two GPUs managed from a single
process via ``tncc.init_local()``.  No torchrun or distributed setup
required.

Requirements: 2x NVIDIA GPUs with NVLink.

Usage:
    python examples/single_process.py
"""

from __future__ import annotations

import torch

import tncc


def main() -> None:
    # Initialize a symmetric heap across 2 GPUs in this process.
    ctxs = tncc.init_local(world_size=2, heap_size=512 * 1024 * 1024)

    M, K, N = 4096, 4096, 8192

    for ctx in ctxs:
        # Allocate tensors directly on the symmetric heap.
        A = ctx.randn(M, K, dtype=torch.float16)
        B = ctx.randn(K, N, dtype=torch.float16)
        C = ctx.zeros(M, N, dtype=torch.float16)

        # Fused GEMM + all-scatter with automatic pattern selection.
        tncc.ops.gemm_allscatter(A, B, C, ctx=ctx)

    torch.cuda.synchronize()

    print(f"rank 0  C shape: {ctxs[0].device}  ->  done")
    print(f"rank 1  C shape: {ctxs[1].device}  ->  done")


if __name__ == "__main__":
    main()
