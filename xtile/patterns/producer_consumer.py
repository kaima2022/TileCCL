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
from xtile.patterns._helpers import scatter_tile_to_peer
from xtile.sync.primitives import tile_signal, tile_wait

if TYPE_CHECKING:
    import torch


class ProducerConsumerPattern(Pattern):
    """Producer-consumer GEMM + scatter on separate streams.

    The producer kernel performs persistent GEMM and signals tile
    completion.  The consumer kernel waits for each tile and scatters
    it to peer GPUs.

    Args:
        ctx: Distributed context (rank, world_size, heap_bases, backend).
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
        self._lock_buffer: Any = None
        self._lock_capacity: int = 0
        self._lock_device_index: int | None = None
        self._compute_stream: Any = None
        self._comm_stream: Any = None
        self._stream_device_index: int | None = None

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

    def _get_locks(self, total_tiles: int, device: "torch.device") -> "torch.Tensor":
        """Return a zeroed lock buffer, reusing storage across launches."""
        import torch

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        if (
            self._lock_buffer is None
            or self._lock_capacity < total_tiles
            or self._lock_device_index != device_index
        ):
            self._lock_buffer = torch.empty(
                total_tiles,
                dtype=torch.int32,
                device=device,
            )
            self._lock_capacity = total_tiles
            self._lock_device_index = device_index

        locks = self._lock_buffer[:total_tiles]
        locks.zero_()
        return locks

    def _get_streams(self, device: "torch.device") -> tuple["torch.cuda.Stream", "torch.cuda.Stream"]:
        """Create or reuse dedicated compute/communication streams."""
        import torch

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        if (
            self._compute_stream is None
            or self._comm_stream is None
            or self._stream_device_index != device_index
        ):
            self._compute_stream = torch.cuda.Stream(device=device)
            self._comm_stream = torch.cuda.Stream(device=device)
            self._stream_device_index = device_index

        return self._compute_stream, self._comm_stream

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
        heap_bases = self.ctx.heap_bases
        N_per_rank = N // world_size

        # Lock tensor for tile-level synchronization: one int32 per tile.
        # Producer writes 1, consumer waits for 1.
        locks = self._get_locks(total_tiles, A.device)

        # Reuse dedicated streams to keep benchmarked launches focused on
        # kernel work rather than per-iteration stream allocation overhead.
        compute_stream, comm_stream = self._get_streams(A.device)

        # Launch producer on compute stream
        producer_grid = (min(compute_sms, total_tiles),)
        with torch.cuda.stream(compute_stream):
            EVEN_K = (K % self.BLOCK_K == 0)
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
                EVEN_K=EVEN_K,
                num_warps=4,
                num_stages=4,
            )

        # Launch consumer on comm stream
        consumer_grid = (min(comm_sms, total_tiles),)
        with torch.cuda.stream(comm_stream):
            self._consumer_kernel[consumer_grid](
                C,
                locks,
                heap_bases,
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
        EVEN_K: tl.constexpr,
    ):
        """Producer: persistent GEMM with tile-level signalling.

        Uses Iris-style mask-free K-loop. After computing each output
        tile, signals the corresponding lock via tile_signal (release
        semantics) so the consumer knows the tile is ready.
        """
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)

        pid = tl.program_id(0)
        total_tiles = num_tiles_m * num_tiles_n

        for tile_id in range(pid, total_tiles, NUM_SMS):
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Wrapped offsets for mask-free A/B loads
            rm = offs_m % M
            rn = offs_n % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

            # ---- GEMM accumulation (optimized K-loop) ----
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

            # ---- Store result tile ----
            result = acc.to(C_ptr.dtype.element_ty)
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, result, mask=c_mask)

            # ---- Signal tile completion ----
            tile_signal(locks_ptr, tile_id)

    @staticmethod
    @triton.jit
    def _consumer_kernel(
        # Pointers
        C_ptr,
        locks_ptr,
        heap_bases,
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

        For each tile (round-robin), waits via
        :func:`~xtile.sync.primitives.tile_wait` (acquire semantics) until
        the producer signals completion, then scatters the tile to all
        peer GPUs via :func:`~xtile.patterns._helpers.scatter_tile_to_peer`.
        """
        pid = tl.program_id(0)

        for tile_id in range(pid, total_tiles, NUM_SMS):
            # ---- Wait for producer to signal tile completion ----
            # Acquire semantics ensure we see the tile data written by
            # the producer before its release-signal.
            tile_wait(locks_ptr, tile_id)

            # ---- Read the completed tile ----
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tile_data = tl.load(c_ptrs, mask=c_mask, other=0.0)

            # ---- Scatter to all peers via translate_ptr ----
            for peer in range(world_size):
                if peer != rank:
                    dst_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_per_rank)
                    scatter_tile_to_peer(
                        C_ptr, tile_data, offs_m, offs_n,
                        rank, peer, N, N_per_rank, heap_bases, dst_mask,
                    )
