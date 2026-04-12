# SPDX-License-Identifier: Apache-2.0
"""
tileccl.patterns.fused_sequential - Fused Sequential overlap pattern.

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

from typing import TYPE_CHECKING, Any

import triton
import triton.language as tl

from tileccl.patterns import Pattern
from tileccl.patterns._helpers import multicast_tile_to_peers

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

        self.require_device_remote_access_runtime(
            operation="fused_sequential pattern execution"
        )
        spec = self.resolve_execution(
            A,
            B,
            C,
            spec=kwargs.get("spec"),
            full_N=kwargs.get("full_N"),
            b_layout=kwargs.get("b_layout"),
            c_layout=kwargs.get("c_layout"),
            storage_kind=kwargs.get("storage_kind", "symmetric"),
        )
        M, K = A.shape
        N = spec.local_N

        num_sms = self.NUM_SMS or self.ctx.backend.get_device_properties().compute_units

        num_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n

        world_size = self.ctx.world_size
        heap_bases = self.ctx.heap_bases

        grid = (min(num_sms, total_tiles),)
        EVEN_K = (K % self.BLOCK_K == 0)
        self._fused_kernel[grid](
            A, B, C,
            heap_bases,
            M, N, K,
            spec.scatter_src_col_offset,
            spec.scatter_cols,
            spec.scatter_dst_leading_dim,
            spec.scatter_dst_col_offset,
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
            EVEN_K=EVEN_K,
            num_warps=4,
            num_stages=4,
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
        M, N, K,
        scatter_src_col_offset,
        scatter_cols,
        scatter_dst_leading_dim,
        scatter_dst_col_offset,
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
        EVEN_K: tl.constexpr,
    ):
        """Fused GEMM + scatter kernel (Iris Listing 4 style).

        Persistent kernel: each SM round-robins over output tiles.
        For each tile:
            1. Accumulate C[tile] = A[rows, :] @ B[:, cols]
            2. Store to local C
            3. Scatter the tile to every peer via translate_ptr

        Uses Iris-style mask-free K-loop with modular index wrapping
        and compiler vectorization hints.
        """
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)

        pid = tl.program_id(0)
        total_tiles = num_tiles_m * num_tiles_n

        for tile_id in range(pid, total_tiles, NUM_SMS):
            # ---- Tile coordinates ----
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            # Original offsets for C store and scatter
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Wrapped offsets for mask-free A/B loads
            rm = offs_m % M
            rn = offs_n % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

            # ---- Phase 1: GEMM accumulation (optimized K-loop) ----
            rk = tl.arange(0, BLOCK_K)
            A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            loop_k = tl.cdiv(K, BLOCK_K)
            if not EVEN_K:
                loop_k -= 1

            for k in range(0, loop_k):
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
                acc = tl.dot(a, b, acc, allow_tf32=True)
                A_BASE += BLOCK_K * stride_ak
                B_BASE += BLOCK_K * stride_bk

            if not EVEN_K:
                rk_rem = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
                A_REM = A_ptr + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                B_REM = B_ptr + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0)
                b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0)
                acc = tl.dot(a, b, acc, allow_tf32=True)

            # Cast accumulator to output dtype
            result = acc.to(C_ptr.dtype.element_ty)
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

            # ---- Phase 2: Local store ----
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, result, mask=c_mask)

            # ---- Phase 3: Scatter to all peers (.wt write-through) ----
            multicast_tile_to_peers(
                C_ptr,
                result,
                offs_m,
                offs_n,
                rank,
                world_size,
                heap_bases,
                scatter_src_col_offset,
                scatter_cols,
                scatter_dst_leading_dim,
                scatter_dst_col_offset,
                c_mask,
            )
