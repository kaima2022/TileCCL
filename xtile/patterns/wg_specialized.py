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
from xtile.patterns._helpers import scatter_tile_to_peer
from xtile.sync.primitives import tile_signal, tile_wait

if TYPE_CHECKING:
    import torch


class WGSpecializedPattern(Pattern):
    """Workgroup-specialized fused GEMM + scatter kernel.

    A single kernel launch where program instances are split into
    compute workers (GEMM + signal) and comm workers (wait + scatter).

    Args:
        ctx: Distributed context (rank, world_size, heap_bases, backend).
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
        self._lock_buffer: Any = None
        self._lock_capacity: int = 0
        self._lock_device_index: int | None = None

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
        heap_bases = self.ctx.heap_bases
        N_per_rank = N // world_size

        # Lock tensor for tile-level synchronization between compute and
        # comm workers within the same kernel.
        locks = self._get_locks(total_tiles, A.device)

        # Single kernel launch -- grid size = COMPUTE_SMS + COMM_SMS
        grid = (total_sms,)
        EVEN_K = (K % self.BLOCK_K == 0)
        self._wg_specialized_kernel[grid](
            A, B, C,
            locks,
            heap_bases,
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
            EVEN_K=EVEN_K,
            num_warps=4,
            num_stages=4,
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
        heap_bases,
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
        EVEN_K: tl.constexpr,
    ):
        """Workgroup-specialized kernel (Iris Listing 5 style).

        Program instances 0..COMPUTE_SMS-1 are **compute workers** that
        execute the persistent GEMM and signal tile completion via
        tile_signal (release semantics).

        Program instances COMPUTE_SMS..COMPUTE_SMS+COMM_SMS-1 are
        **comm workers** that wait via tile_wait (acquire semantics)
        and scatter tiles to peers via scatter_tile_to_peer.

        Uses Iris-style mask-free K-loop with modular index wrapping
        and compiler vectorization hints.
        """
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)

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

                # ---- Store result tile locally ----
                result = acc.to(C_ptr.dtype.element_ty)
                c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.store(c_ptrs, result, mask=c_mask)

                # ---- Signal tile completion ----
                tile_signal(locks_ptr, tile_id)

        # ================================================================
        # Comm worker: wait + scatter
        # ================================================================
        else:
            # Comm worker index (0-based within the comm pool)
            comm_pid = pid - COMPUTE_SMS

            for tile_id in range(comm_pid, total_tiles, COMM_SMS):
                # ---- Wait for compute worker to signal tile completion ----
                # Acquire semantics ensure we see the tile data written
                # by the compute worker before its release-signal.
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
