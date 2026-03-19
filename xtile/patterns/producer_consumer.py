"""
xtile.patterns.producer_consumer - Producer-Consumer overlap pattern.

Reference: Iris Section 4.1.2.

Two kernels run on *separate CUDA/HIP streams*, communicating via a
shared lock tensor for tile-level synchronization:

    **Producer** (runs on ``COMPUTE_SMS`` SMs):
        Persistent GEMM kernel.  After computing each output tile, it
        calls ``tile_signal`` to mark the tile as ready.

    **Consumer** (runs on ``COMM_SMS`` SMs):
        Persistent scatter kernel.  For each tile, it calls ``tile_wait``
        until the producer has signalled completion, then performs the
        remote store to all peers.

The SM partition is configurable: by default the split is chosen so that
~80% of SMs run the producer and ~20% run the consumer (communication
is typically memory-bandwidth-bound and needs fewer SMs).

Overlap mechanism:
    Producer:  tile[i] GEMM  ─── signal ─── tile[i+1] GEMM ─── signal
    Consumer:            wait ─── scatter ────── wait ─── scatter
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import triton
import triton.language as tl

from xtile.patterns import Pattern

if TYPE_CHECKING:
    import torch


class ProducerConsumerPattern(Pattern):
    """Producer-consumer GEMM + scatter on separate streams.

    The producer kernel performs persistent GEMM and signals tile
    completion.  The consumer kernel waits for each tile and scatters
    it to peer GPUs.

    Args:
        ctx: Distributed context (rank, world_size, remote_ptrs, backend).
        BLOCK_M: Tile height.
        BLOCK_N: Tile width.
        BLOCK_K: Reduction tile depth.
        COMPUTE_SMS: Number of SMs for the GEMM producer (0 = auto).
        COMM_SMS: Number of SMs for the scatter consumer (0 = auto).
    """

    name: str = "producer_consumer"

    # Default fraction of SMs allocated to communication
    _COMM_SM_FRACTION: float = 0.2

    def __init__(
        self,
        ctx: Any,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 64,
        COMPUTE_SMS: int = 0,
        COMM_SMS: int = 0,
    ) -> None:
        super().__init__(ctx)
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.COMPUTE_SMS = COMPUTE_SMS
        self.COMM_SMS = COMM_SMS

    def _resolve_sm_split(self) -> tuple[int, int]:
        """Determine the compute/comm SM split.

        Returns:
            (compute_sms, comm_sms) tuple.
        """
        total_sms = self.ctx.backend.get_device_properties().compute_units
        if self.COMPUTE_SMS > 0 and self.COMM_SMS > 0:
            return self.COMPUTE_SMS, self.COMM_SMS
        comm = max(1, int(total_sms * self._COMM_SM_FRACTION))
        compute = total_sms - comm
        return compute, comm

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
        """Launch producer and consumer kernels on separate streams.

        Args:
            A: Input matrix ``(M, K)``.
            B: Input matrix ``(K, N)``.
            C: Output matrix ``(M, N_local)``.
        """
        import torch

        M, K = A.shape
        _, N = B.shape

        compute_sms, comm_sms = self._resolve_sm_split()

        num_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n

        world_size = self.ctx.world_size
        remote_ptrs = self.ctx.remote_ptrs
        N_per_rank = N // world_size

        # Lock tensor for tile-level synchronization: one int32 per tile.
        # Producer writes 1, consumer waits for 1.
        locks = torch.zeros(total_tiles, dtype=torch.int32, device=A.device)

        # Create separate streams for producer and consumer
        # TODO: Use stream pool from ctx when available.
        compute_stream = torch.cuda.Stream()
        comm_stream = torch.cuda.Stream()

        # Launch producer on compute stream
        producer_grid = (min(compute_sms, total_tiles),)
        with torch.cuda.stream(compute_stream):
            self._producer_kernel[producer_grid](
                A, B, C,
                locks,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                num_tiles_m=num_tiles_m,
                num_tiles_n=num_tiles_n,
                BLOCK_M=self.BLOCK_M,
                BLOCK_N=self.BLOCK_N,
                BLOCK_K=self.BLOCK_K,
                NUM_SMS=compute_sms,
            )

        # Launch consumer on comm stream
        consumer_grid = (min(comm_sms, total_tiles),)
        with torch.cuda.stream(comm_stream):
            self._consumer_kernel[consumer_grid](
                C,
                locks,
                remote_ptrs,
                M, N, N_per_rank,
                C.stride(0), C.stride(1),
                self.ctx.rank,
                world_size,
                num_tiles_m=num_tiles_m,
                num_tiles_n=num_tiles_n,
                total_tiles=total_tiles,
                BLOCK_M=self.BLOCK_M,
                BLOCK_N=self.BLOCK_N,
                NUM_SMS=comm_sms,
            )

        # Synchronize both streams
        compute_stream.synchronize()
        comm_stream.synchronize()

    # ------------------------------------------------------------------
    # Triton kernels
    # ------------------------------------------------------------------

    @staticmethod
    @triton.jit
    def _producer_kernel(
        # Pointers
        A_ptr, B_ptr, C_ptr,
        locks_ptr,
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
        """Producer: persistent GEMM with tile-level signalling.

        After computing each output tile, atomically sets the
        corresponding lock to 1 so the consumer knows the tile is ready.

        TODO: Replace atomic with xtile.primitives.tile_signal when available.
        TODO: Consider using tl.atomic_xchg for release semantics.
        """
        pid = tl.program_id(0)
        total_tiles = num_tiles_m * num_tiles_n

        for tile_id in range(pid, total_tiles, NUM_SMS):
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # ---- GEMM accumulation ----
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)

                a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                acc += tl.dot(a, b)

            # ---- Store result tile ----
            result = acc.to(C_ptr.dtype.element_ty)
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, result, mask=c_mask)

            # ---- Signal tile completion (tile_signal) ----
            # Atomic store ensures memory ordering: the consumer will
            # see the tile data after it observes lock == 1.
            # TODO: Replace with xtile.sync.tile_signal primitive.
            tl.atomic_xchg(locks_ptr + tile_id, 1)

    @staticmethod
    @triton.jit
    def _consumer_kernel(
        # Pointers
        C_ptr,
        locks_ptr,
        remote_ptrs,
        # Dimensions
        M, N, N_per_rank,
        # Strides
        stride_cm, stride_cn,
        # Distribution info
        rank, world_size,
        # Tile counts
        num_tiles_m, num_tiles_n, total_tiles,
        # Compile-time constants
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """Consumer: wait for tiles and scatter them to peers.

        For each tile (round-robin), spin-waits on the corresponding
        lock until the producer signals completion, then performs remote
        stores to all peer GPUs.

        TODO: Replace spin-wait with xtile.sync.tile_wait primitive.
        TODO: Replace scatter with xtile.primitives.tile_remote_store.
        TODO: Add backoff strategy to reduce spin-wait power consumption.
        """
        pid = tl.program_id(0)

        for tile_id in range(pid, total_tiles, NUM_SMS):
            # ---- tile_wait: spin until producer signals ----
            while tl.atomic_cas(locks_ptr + tile_id, 1, 1) != 1:
                pass  # spin -- TODO: add exponential backoff

            # ---- Read the completed tile ----
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tile_data = tl.load(c_ptrs, mask=c_mask, other=0.0)

            # ---- Scatter to all peers (tile_remote_store) ----
            for peer in range(world_size):
                if peer == rank:
                    continue

                remote_base = tl.load(remote_ptrs + peer)
                dst_col_offset = rank * N_per_rank
                dst_ptrs = (
                    remote_base
                    + (offs_m[:, None] * N
                       + (dst_col_offset + offs_n[None, :])).to(tl.int64)
                    * tile_data.dtype.primitive_bitwidth // 8
                )
                dst_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_per_rank)
                tl.store(dst_ptrs, tile_data, mask=dst_mask)
