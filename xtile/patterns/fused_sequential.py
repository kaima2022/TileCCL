"""
xtile.patterns.fused_sequential - Fused Sequential overlap pattern.

Reference: Iris Listing 4.

A single fused kernel where each tile performs GEMM accumulation and then
*immediately* scatters the result to peer GPUs before moving on to the next
tile.  Because the hardware can overlap the memory traffic of the remote
store (previous tile) with the arithmetic of the next tile's GEMM, this
pattern achieves moderate compute-communication overlap with minimal
software complexity.

Overlap mechanism:
    tile[i] GEMM  ──────────┐
                             ├── hardware overlap
    tile[i] scatter  ────┐   │
    tile[i+1] GEMM  ─────┘──┘

The kernel is written in the *persistent* style: each SM loops over tiles
in round-robin order, so the total grid size equals ``NUM_SMS``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import triton
import triton.language as tl

from xtile.patterns import Pattern
from xtile.patterns._helpers import scatter_tile_to_peer

if TYPE_CHECKING:
    import torch


class FusedSequentialPattern(Pattern):
    """Fused sequential GEMM + scatter in a single kernel.

    For each output tile, the kernel:
        1. Computes the full GEMM accumulation (inner loop over K).
        2. Immediately scatters the result tile to all peers via remote store.
        3. Moves to the next tile (persistent loop).

    Args:
        ctx: Distributed context (rank, world_size, heap_bases, backend).
        BLOCK_M: Tile height.
        BLOCK_N: Tile width.
        BLOCK_K: Reduction tile depth.
        NUM_SMS: Number of SMs to use (0 = all available).
    """

    name: str = "fused_sequential"

    def __init__(
        self,
        ctx: Any,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 64,
        NUM_SMS: int = 0,
    ) -> None:
        super().__init__(ctx)
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.NUM_SMS = NUM_SMS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        A: "torch.Tensor",
        B: "torch.Tensor",
        C: "torch.Tensor",
        **kwargs: Any,
    ) -> None:
        """Run the fused GEMM + scatter kernel.

        Args:
            A: Input matrix of shape ``(M, K)``.
            B: Input matrix of shape ``(K, N)``.
            C: Output matrix of shape ``(M, N_local)`` -- written locally
               and simultaneously scattered to peers.
        """
        import torch

        M, K = A.shape
        _, N = B.shape

        num_sms = self.NUM_SMS or self.ctx.backend.get_device_properties().compute_units

        num_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n

        world_size = self.ctx.world_size
        heap_bases = self.ctx.heap_bases
        N_per_rank = N // world_size

        grid = (min(num_sms, total_tiles),)
        self._fused_kernel[grid](
            A, B, C,
            heap_bases,
            M, N, K, N_per_rank,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            self.ctx.rank,
            world_size,
            num_tiles_m=num_tiles_m,
            num_tiles_n=num_tiles_n,
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
            NUM_SMS=num_sms,
        )

    # ------------------------------------------------------------------
    # Triton kernel
    # ------------------------------------------------------------------

    @staticmethod
    @triton.jit
    def _fused_kernel(
        # Pointers
        A_ptr, B_ptr, C_ptr,
        heap_bases,
        # Dimensions
        M, N, K, N_per_rank,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Distribution info
        rank, world_size,
        # Tile counts
        num_tiles_m, num_tiles_n,
        # Compile-time constants
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """Fused GEMM + scatter kernel (Iris Listing 4 style).

        Persistent kernel: each SM round-robins over output tiles.
        For each tile:
            1. Accumulate C[tile] = A[rows, :] @ B[:, cols]
            2. Store to local C
            3. Scatter the tile to every peer via translate_ptr

        Hardware overlap occurs because the remote stores from tile[i] can
        overlap with the GEMM arithmetic of tile[i+1] at the memory
        subsystem level.
        """
        pid = tl.program_id(0)
        total_tiles = num_tiles_m * num_tiles_n

        for tile_id in range(pid, total_tiles, NUM_SMS):
            # ---- Tile coordinates ----
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # ---- Phase 1: GEMM accumulation (pipelined K-loop) ----
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Prefetch first K-tile
            offs_k_0 = tl.arange(0, BLOCK_K)
            a = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k_0[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k_0[None, :] < K), other=0.0,
            )
            b = tl.load(
                B_ptr + offs_k_0[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(offs_k_0[:, None] < K) & (offs_n[None, :] < N), other=0.0,
            )

            num_k_iters = tl.cdiv(K, BLOCK_K)
            for k_iter in range(0, num_k_iters):
                acc = tl.dot(a, b, acc, allow_tf32=True)
                next_k = (k_iter + 1) * BLOCK_K
                if next_k < K:
                    offs_k_n = next_k + tl.arange(0, BLOCK_K)
                    a = tl.load(
                        A_ptr + offs_m[:, None] * stride_am + offs_k_n[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & (offs_k_n[None, :] < K), other=0.0,
                        eviction_policy="evict_last",
                    )
                    b = tl.load(
                        B_ptr + offs_k_n[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(offs_k_n[:, None] < K) & (offs_n[None, :] < N), other=0.0,
                        eviction_policy="evict_last",
                    )

            # Cast accumulator to output dtype
            result = acc.to(C_ptr.dtype.element_ty)
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

            # ---- Phase 2: Local store ----
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, result, mask=c_mask)

            # ---- Phase 3: Scatter to all peers via translate_ptr ----
            # Immediately after computing the tile, push it to every peer.
            # The hardware memory subsystem can overlap these remote writes
            # with the next tile's GEMM computation.
            for peer in range(world_size):
                if peer != rank:
                    dst_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_per_rank)
                    scatter_tile_to_peer(
                        C_ptr, result, offs_m, offs_n,
                        rank, peer, N, N_per_rank, heap_bases, dst_mask,
                    )
