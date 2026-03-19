"""
xtile.patterns.bulk_sync - Bulk-Synchronous overlap pattern.

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

from typing import Any, TYPE_CHECKING

import triton
import triton.language as tl

from xtile.patterns import Pattern

if TYPE_CHECKING:
    import torch


class BulkSyncPattern(Pattern):
    """Bulk-synchronous GEMM followed by all-scatter.

    Args:
        ctx: Distributed context (rank, world_size, remote_ptrs, backend).
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
        import torch

        M, K = A.shape
        _, N = B.shape

        num_sms = self.NUM_SMS or self.ctx.backend.get_device_properties().compute_units

        # Tile grid dimensions
        num_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n

        # Phase 1: GEMM kernel (persistent)
        grid = (min(num_sms, total_tiles),)
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
        )

        # Phase 2: Device-wide barrier -- GEMM must be fully complete
        self.ctx.backend.synchronize()

        # Phase 3: Scatter kernel -- push local C tiles to all peers
        world_size = self.ctx.world_size
        remote_ptrs = self.ctx.remote_ptrs  # tensor of int64 base pointers
        N_per_rank = N // world_size
        num_scatter_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_scatter_tiles_n = triton.cdiv(N_per_rank, self.BLOCK_N)
        total_scatter_tiles = num_scatter_tiles_m * num_scatter_tiles_n * world_size

        scatter_grid = (min(num_sms, total_scatter_tiles),)
        self._scatter_kernel[scatter_grid](
            C,
            remote_ptrs,
            M, N, N_per_rank,
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
    ):
        """Persistent GEMM kernel: C = A @ B.

        Each program instance loops over tiles in a round-robin fashion
        (persistent kernel style), computing one BLOCK_M x BLOCK_N output
        tile per iteration.
        """
        pid = tl.program_id(0)
        total_tiles = num_tiles_m * num_tiles_n

        # Persistent loop: each SM processes multiple tiles
        for tile_id in range(pid, total_tiles, NUM_SMS):
            # Decompose linear tile_id into 2D tile coordinates
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            # Compute row/col offsets for this tile
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Inner loop over K dimension
            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)

                # Load A tile: (BLOCK_M, BLOCK_K)
                a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                # Load B tile: (BLOCK_K, BLOCK_N)
                b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                # Accumulate
                acc = tl.dot(a, b, acc, allow_tf32=True)

            # Store result tile
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)

    @staticmethod
    @triton.jit
    def _scatter_kernel(
        # Pointers
        C_ptr,
        remote_ptrs,  # int64 tensor of base pointers, shape (world_size,)
        # Dimensions
        M, N, N_per_rank,
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
        """Scatter kernel: push local C tiles to every peer via remote store.

        Each program instance loops over (peer, tile_m, tile_n) triples in
        round-robin order.  For each triple it reads a BLOCK_M x BLOCK_N
        tile from the local C buffer and writes it to the corresponding
        location in the peer's remote buffer.

        TODO: Integrate with xtile.primitives.tile_remote_store when available.
        TODO: Add support for non-uniform tile sizes at matrix boundaries.
        """
        pid = tl.program_id(0)
        tiles_per_peer = num_scatter_tiles_m * num_scatter_tiles_n
        total_tiles = tiles_per_peer * world_size

        for tile_id in range(pid, total_tiles, NUM_SMS):
            # Decompose into (peer_idx, tile_m, tile_n)
            peer_idx = tile_id // tiles_per_peer
            local_tile = tile_id % tiles_per_peer
            tile_m = local_tile // num_scatter_tiles_n
            tile_n = local_tile % num_scatter_tiles_n

            # Skip self -- local rank already has this data
            # TODO: Consider whether self-scatter is needed for API consistency
            if peer_idx == rank:
                continue

            # Source offsets in local C
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Load local tile
            src_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_per_rank)
            tile_data = tl.load(src_ptrs, mask=mask, other=0.0)

            # Compute destination offset in remote buffer
            # Remote layout: each peer expects data from rank `rank` at a
            # specific shard offset.  The offset depends on the scatter scheme.
            # TODO: Replace with tile_remote_store primitive once available.
            remote_base = tl.load(remote_ptrs + peer_idx)
            # Destination column offset accounts for which rank is sending
            dst_col_offset = rank * N_per_rank
            dst_ptrs = (remote_base
                        + (offs_m[:, None] * N + (dst_col_offset + offs_n[None, :])).to(tl.int64)
                        * tile_data.dtype.primitive_bitwidth // 8)
            tl.store(dst_ptrs, tile_data, mask=mask)
