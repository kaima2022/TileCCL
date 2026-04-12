# SPDX-License-Identifier: Apache-2.0
"""
tileccl.patterns.bulk_sync - Bulk-Synchronous overlap pattern.

Reference: Iris Listing 3.

This is the simplest overlap strategy and serves as the baseline.
Two completely separate kernels are launched in sequence:

    1. **GEMM kernel** -- a standard persistent Triton matmul that computes
       ``C_local = A @ B`` into a local output buffer.
    2. **Barrier** -- a device-wide synchronization ensuring the entire GEMM
       is complete before any communication begins.
    3. **Scatter kernel** -- reads the local output tiles and uses
       ``tile_remote_store`` to push each tile to the appropriate peer GPU.

Because the GEMM must finish entirely before communication starts, there is
*no* compute-communication overlap.  This pattern is useful as a correctness
reference and performance lower-bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import triton
import triton.language as tl

from tileccl.patterns import Pattern
from tileccl.patterns._helpers import multicast_tile_to_peers

if TYPE_CHECKING:
    import torch


class BulkSyncPattern(Pattern):
    """Bulk-synchronous GEMM followed by all-scatter.

    Args:
        ctx: Distributed context (rank, world_size, heap_bases, backend).
        BLOCK_M: Tile height for the GEMM.
        BLOCK_N: Tile width for the GEMM.
        BLOCK_K: Reduction tile depth for the GEMM.
        NUM_SMS: Number of SMs to use (0 = all available).
    """

    name: str = "bulk_sync"

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
        # If NUM_SMS is 0, we will resolve it at launch time from device props
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
        """Run GEMM -> barrier -> scatter sequentially.

        Args:
            A: Input matrix of shape ``(M, K)`` on the local device.
            B: Input matrix of shape ``(K, N)`` on the local device.
            C: Output matrix of shape ``(M, N_local)`` on the local device.
        """

        self.require_device_remote_access_runtime(
            operation="bulk_sync pattern execution"
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

        # Tile grid dimensions
        num_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n

        # Phase 1: GEMM kernel (persistent)
        grid = (min(num_sms, total_tiles),)
        EVEN_K = (K % self.BLOCK_K == 0)
        self._gemm_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
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

        # Phase 2: Device-wide barrier -- GEMM must be fully complete
        self.ctx.backend.synchronize()

        # Phase 3: Scatter kernel -- push local C tiles to all peers
        world_size = self.ctx.world_size
        heap_bases = self.ctx.heap_bases  # (world_size,) int64 heap base pointers
        num_scatter_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_scatter_tiles_n = triton.cdiv(spec.scatter_cols, self.BLOCK_N)
        total_scatter_tiles = num_scatter_tiles_m * num_scatter_tiles_n

        scatter_grid = (min(num_sms, total_scatter_tiles),)
        self._scatter_kernel[scatter_grid](
            C,
            heap_bases,
            M,
            spec.local_N,
            spec.scatter_src_col_offset,
            spec.scatter_cols,
            spec.scatter_dst_leading_dim,
            spec.scatter_dst_col_offset,
            C.stride(0), C.stride(1),
            self.ctx.rank,
            world_size,
            num_scatter_tiles_m=num_scatter_tiles_m,
            num_scatter_tiles_n=num_scatter_tiles_n,
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=self.BLOCK_N,
            NUM_SMS=num_sms,
        )

    # ------------------------------------------------------------------
    # Triton kernels
    # ------------------------------------------------------------------

    @staticmethod
    @triton.jit
    def _gemm_kernel(
        # Pointers
        A_ptr, B_ptr, C_ptr,
        # Dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Tile counts
        num_tiles_m, num_tiles_n,
        # Compile-time constants
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """Persistent GEMM kernel: C = A @ B.

        Each program instance loops over tiles in a round-robin fashion
        (persistent kernel style), computing one BLOCK_M x BLOCK_N output
        tile per iteration.  Uses Iris-style mask-free K-loop with
        modular index wrapping and compiler vectorization hints.
        """
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)

        pid = tl.program_id(0)
        total_tiles = num_tiles_m * num_tiles_n

        # Persistent loop: each SM processes multiple tiles
        for tile_id in range(pid, total_tiles, NUM_SMS):
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            # Wrapped offsets for mask-free A/B loads
            rm = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
            rn = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

            rk = tl.arange(0, BLOCK_K)
            A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Split K-loop: main iterations without masks
            loop_k = tl.cdiv(K, BLOCK_K)
            if not EVEN_K:
                loop_k -= 1

            for k in range(0, loop_k):
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
                acc = tl.dot(a, b, acc, allow_tf32=True)
                A_BASE += BLOCK_K * stride_ak
                B_BASE += BLOCK_K * stride_bk

            # Remainder: K-mask only
            if not EVEN_K:
                rk_rem = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
                A_REM = A_ptr + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                B_REM = B_ptr + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0)
                b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0)
                acc = tl.dot(a, b, acc, allow_tf32=True)

            # Store result tile (M/N boundary mask only here)
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)

    @staticmethod
    @triton.jit
    def _scatter_kernel(
        # Pointers
        C_ptr,
        heap_bases,  # int64 tensor of heap base pointers, shape (world_size,)
        # Dimensions
        M,
        local_N,
        scatter_src_col_offset,
        scatter_cols,
        scatter_dst_leading_dim,
        scatter_dst_col_offset,
        # Strides
        stride_cm, stride_cn,
        # Distribution info
        rank, world_size,
        # Tile counts
        num_scatter_tiles_m, num_scatter_tiles_n,
        # Compile-time constants
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """Scatter kernel: push each local C tile to every peer once.

        Each program instance reads one local tile and fans it out to all
        peers via the shared software-multicast helper. This avoids the
        previous per-peer reload of the same tile.
        """
        pid = tl.program_id(0)
        total_tiles = num_scatter_tiles_m * num_scatter_tiles_n

        for tile_id in range(pid, total_tiles, NUM_SMS):
            tile_m = tile_id // num_scatter_tiles_n
            tile_n = tile_id % num_scatter_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = scatter_src_col_offset + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            src_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            mask = (offs_m[:, None] < M) & (
                offs_n[None, :] < scatter_src_col_offset + scatter_cols
            )
            tile_data = tl.load(src_ptrs, mask=mask, other=0.0)

            multicast_tile_to_peers(
                C_ptr,
                tile_data,
                offs_m,
                offs_n,
                rank,
                world_size,
                heap_bases,
                scatter_src_col_offset,
                scatter_cols,
                scatter_dst_leading_dim,
                scatter_dst_col_offset,
                mask,
            )
