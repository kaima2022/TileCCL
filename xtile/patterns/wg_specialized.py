"""
xtile.patterns.wg_specialized - Workgroup Specialization overlap pattern.

Reference: Iris Listing 5.

A single fused kernel where the program instances are partitioned into
two roles based on their ``program_id``:

    * **Compute workers** (``pid < COMPUTE_SMS``): Run the persistent
      GEMM loop and signal tile completion via the lock tensor.
    * **Comm workers** (``pid >= COMPUTE_SMS``): Run a persistent
      wait+scatter loop, polling the lock tensor and issuing remote
      stores as tiles become ready.

This is the most complex pattern but achieves the highest overlap because
compute and communication are truly concurrent at the CU/SM level within
a single kernel launch, eliminating inter-kernel launch overhead.

Best suited for large N and K where there are enough tiles to keep both
worker pools busy.

Overlap mechanism:
    SM 0..C-1 (compute):  tile GEMM → signal → tile GEMM → signal → ...
    SM C..C+M-1 (comm):          wait → scatter → wait → scatter → ...
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                           fully parallel on different SMs
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import triton
import triton.language as tl

from xtile.patterns import Pattern

if TYPE_CHECKING:
    import torch


class WGSpecializedPattern(Pattern):
    """Workgroup-specialized fused GEMM + scatter kernel.

    A single kernel launch where program instances are split into
    compute workers (GEMM + signal) and comm workers (wait + scatter).

    Args:
        ctx: Distributed context (rank, world_size, remote_ptrs, backend).
        BLOCK_M: Tile height.
        BLOCK_N: Tile width.
        BLOCK_K: Reduction tile depth.
        COMPUTE_SMS: SMs for compute workers (0 = auto, ~80%).
        COMM_SMS: SMs for comm workers (0 = auto, ~20%).
    """

    name: str = "wg_specialized"

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

        TODO: Auto-tune the split based on problem shape and hardware.
              The optimal split depends on the ratio of compute to memory
              bandwidth requirements, which varies with M, N, K and the
              interconnect bandwidth.
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
        """Launch the single workgroup-specialized kernel.

        Args:
            A: Input matrix ``(M, K)``.
            B: Input matrix ``(K, N)``.
            C: Output matrix ``(M, N_local)``.
        """
        import torch

        M, K = A.shape
        _, N = B.shape

        compute_sms, comm_sms = self._resolve_sm_split()
        total_sms = compute_sms + comm_sms

        num_tiles_m = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n

        world_size = self.ctx.world_size
        remote_ptrs = self.ctx.remote_ptrs
        N_per_rank = N // world_size

        # Lock tensor for tile-level synchronization between compute and
        # comm workers within the same kernel.
        locks = torch.zeros(total_tiles, dtype=torch.int32, device=A.device)

        # Single kernel launch -- grid size = COMPUTE_SMS + COMM_SMS
        grid = (total_sms,)
        self._wg_specialized_kernel[grid](
            A, B, C,
            locks,
            remote_ptrs,
            M, N, K, N_per_rank,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            self.ctx.rank,
            world_size,
            compute_sms,
            num_tiles_m=num_tiles_m,
            num_tiles_n=num_tiles_n,
            total_tiles=total_tiles,
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
            COMPUTE_SMS=compute_sms,
            COMM_SMS=comm_sms,
        )

    # ------------------------------------------------------------------
    # Triton kernel
    # ------------------------------------------------------------------

    @staticmethod
    @triton.jit
    def _wg_specialized_kernel(
        # Pointers
        A_ptr, B_ptr, C_ptr,
        locks_ptr,
        remote_ptrs,
        # Dimensions
        M, N, K, N_per_rank,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Distribution info
        rank, world_size,
        compute_sms_runtime,
        # Tile counts
        num_tiles_m, num_tiles_n, total_tiles,
        # Compile-time constants
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COMPUTE_SMS: tl.constexpr,
        COMM_SMS: tl.constexpr,
    ):
        """Workgroup-specialized kernel (Iris Listing 5 style).

        Program instances 0..COMPUTE_SMS-1 are **compute workers** that
        execute the persistent GEMM and signal tile completion.

        Program instances COMPUTE_SMS..COMPUTE_SMS+COMM_SMS-1 are
        **comm workers** that wait for tile signals and perform remote
        scatter.

        The two worker pools operate concurrently on different SMs,
        achieving true SM-level overlap of compute and communication.

        TODO: Integrate with xtile.sync.tile_signal / tile_wait primitives.
        TODO: Integrate with xtile.primitives.tile_remote_store.
        TODO: Experiment with tile scheduling order (Hilbert curve, etc.)
              to improve L2 cache locality for compute workers.
        TODO: Add support for asymmetric tile sizes at matrix edges.
        """
        pid = tl.program_id(0)

        # ================================================================
        # Compute worker: persistent GEMM + signal
        # ================================================================
        if pid < COMPUTE_SMS:
            for tile_id in range(pid, total_tiles, COMPUTE_SMS):
                tile_m = tile_id // num_tiles_n
                tile_n = tile_id % num_tiles_n

                offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

                # ---- GEMM accumulation ----
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, K, BLOCK_K):
                    offs_k = k_start + tl.arange(0, BLOCK_K)

                    # Load A tile
                    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                    # Load B tile
                    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b)

                # ---- Store result tile locally ----
                result = acc.to(C_ptr.dtype.element_ty)
                c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.store(c_ptrs, result, mask=c_mask)

                # ---- Signal tile completion (tile_signal) ----
                # The atomic exchange ensures the store to C is visible
                # to the comm worker before it reads the tile.
                # TODO: Replace with xtile.sync.tile_signal primitive.
                tl.atomic_xchg(locks_ptr + tile_id, 1)

        # ================================================================
        # Comm worker: wait + scatter
        # ================================================================
        else:
            # Comm worker index (0-based within the comm pool)
            comm_pid = pid - COMPUTE_SMS

            for tile_id in range(comm_pid, total_tiles, COMM_SMS):
                # ---- tile_wait: spin until compute worker signals ----
                # TODO: Replace with xtile.sync.tile_wait primitive.
                # TODO: Add exponential backoff to reduce power/contention.
                while tl.atomic_cas(locks_ptr + tile_id, 1, 1) != 1:
                    pass  # spin

                # ---- Read the completed tile ----
                tile_m = tile_id // num_tiles_n
                tile_n = tile_id % num_tiles_n

                offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

                c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tile_data = tl.load(c_ptrs, mask=c_mask, other=0.0)

                # ---- Scatter to all peers (tile_remote_store) ----
                # TODO: Replace with xtile.primitives.tile_remote_store.
                # TODO: Consider coalescing stores or using async copy
                #       engines where available (e.g., TMA on H100).
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
