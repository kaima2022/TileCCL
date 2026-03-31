# SPDX-License-Identifier: Apache-2.0
"""
tncc.patterns.chained_ag_gemm_rs - Three-stage AllGather → GEMM → Scatter.

A single fused kernel where program instances are partitioned into three
roles using :class:`~tncc.patterns.runtime.StageRoleScheduler`:

    * **Gather workers** (pid < GATHER_SMS): Remote-load input shards from
      all peers into a shared gathered-input workspace and signal readiness.
    * **Compute workers** (GATHER_SMS <= pid < GATHER_SMS + COMPUTE_SMS):
      Wait for gathered input, run persistent GEMM, signal completion.
    * **Scatter workers** (pid >= GATHER_SMS + COMPUTE_SMS): Wait for
      GEMM completion, extract and write each rank's output shard.

This achieves three-way overlap: while gather workers fetch segment S+1,
compute workers process segment S, and scatter workers write segment S-1.

Usage::

    from tncc.patterns.chained_ag_gemm_rs import ChainedAGGemmRSPattern
    pattern = ChainedAGGemmRSPattern(ctx)
    pattern.execute(input_shard, W, output_shard)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import triton
import triton.language as tl

from tncc.memory.translation import translate_ptr
from tncc.patterns.runtime import resolve_stage_role_scheduler
from tncc.sync.primitives import tile_signal, tile_try_wait, tile_wait

if TYPE_CHECKING:
    import torch


class ChainedAGGemmRSPattern:
    """Three-stage AllGather → GEMM → Scatter with SM-level overlap.

    Args:
        ctx: Distributed context (rank, world_size, heap_bases, backend).
        BLOCK_M: Tile height for GEMM and scatter.
        BLOCK_N: Tile width for GEMM output.
        BLOCK_K: Reduction tile depth.
        GATHER_SMS: SMs for gather workers (0 = auto).
        COMPUTE_SMS: SMs for compute workers (0 = auto).
        SCATTER_SMS: SMs for scatter workers (0 = auto).
    """

    name: str = "chained_ag_gemm_rs"

    def __init__(
        self,
        ctx: Any,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 64,
        GATHER_SMS: int = 0,
        COMPUTE_SMS: int = 0,
        SCATTER_SMS: int = 0,
    ) -> None:
        self.ctx = ctx
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GATHER_SMS = GATHER_SMS
        self.COMPUTE_SMS = COMPUTE_SMS
        self.SCATTER_SMS = SCATTER_SMS
        self._gather_locks: Any = None
        self._compute_locks: Any = None
        self._lock_capacity: tuple[int, int] = (0, 0)

    def _resolve_sm_split(self) -> tuple[int, int, int]:
        scheduler = resolve_stage_role_scheduler(
            self.ctx.backend.get_device_properties().compute_units,
            gather_sms=self.GATHER_SMS,
            compute_sms=self.COMPUTE_SMS,
            scatter_sms=self.SCATTER_SMS,
        )
        return scheduler.gather_sms, scheduler.compute_sms, scheduler.scatter_sms

    def _get_locks(
        self,
        num_gather_segs: int,
        num_compute_tiles: int,
        device: "torch.device",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        if (
            self._gather_locks is None
            or self._lock_capacity[0] < num_gather_segs
            or self._lock_capacity[1] < num_compute_tiles
        ):
            self._gather_locks = torch.zeros(
                num_gather_segs, dtype=torch.int32, device=device
            )
            self._compute_locks = torch.zeros(
                num_compute_tiles, dtype=torch.int32, device=device
            )
            self._lock_capacity = (num_gather_segs, num_compute_tiles)
        else:
            self._gather_locks[:num_gather_segs].zero_()
            self._compute_locks[:num_compute_tiles].zero_()
        return (
            self._gather_locks[:num_gather_segs],
            self._compute_locks[:num_compute_tiles],
        )

    def execute(
        self,
        input_shard: "torch.Tensor",
        W: "torch.Tensor",
        output: "torch.Tensor",
        gathered_workspace: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Execute the three-stage pipeline.

        Args:
            input_shard: (M, K_shard) per rank - column shard of input.
            W: (K, N) full weight matrix.
            output: (M, N) full output (local copy on this rank).
            gathered_workspace: Optional pre-allocated (M, K) workspace for
                the gathered input.  Allocated on the heap if not provided.
        """
        import torch

        M, K_shard = input_shard.shape
        K, N = W.shape
        world_size = self.ctx.world_size

        if K_shard * world_size != K:
            raise ValueError(
                f"input_shard columns ({K_shard}) * world_size ({world_size}) "
                f"!= W rows ({K})"
            )

        if gathered_workspace is None:
            gathered_workspace = self.ctx.workspace(
                "chained_ag_gemm_rs.gathered_input",
                M,
                K,
                dtype=input_shard.dtype,
            )

        gather_sms, compute_sms, scatter_sms = self._resolve_sm_split()
        total_sms = gather_sms + compute_sms + scatter_sms

        num_row_segs = triton.cdiv(M, self.BLOCK_M)
        num_tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_compute_tiles = num_row_segs * num_tiles_n

        gather_locks, compute_locks = self._get_locks(
            num_row_segs, total_compute_tiles, input_shard.device
        )
        heap_bases = self.ctx.heap_bases
        EVEN_K = K % self.BLOCK_K == 0

        grid = (total_sms,)
        self._chained_kernel[grid](
            input_shard,
            W,
            output,
            gathered_workspace,
            gather_locks,
            compute_locks,
            heap_bases,
            M,
            N,
            K,
            K_shard,
            input_shard.stride(0),
            input_shard.stride(1),
            W.stride(0),
            W.stride(1),
            output.stride(0),
            output.stride(1),
            gathered_workspace.stride(0),
            gathered_workspace.stride(1),
            self.ctx.rank,
            world_size,
            num_row_segs,
            num_tiles_n,
            total_compute_tiles,
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
            GATHER_SMS=gather_sms,
            COMPUTE_SMS=compute_sms,
            SCATTER_SMS=scatter_sms,
            EVEN_K=EVEN_K,
            num_warps=4,
            num_stages=4,
        )
        torch.cuda.synchronize(input_shard.device)
        return output

    @staticmethod
    @triton.jit
    def _chained_kernel(
        input_shard_ptr,
        W_ptr,
        output_ptr,
        gathered_ptr,
        gather_locks_ptr,
        compute_locks_ptr,
        heap_bases,
        M,
        N,
        K,
        K_shard,
        stride_is_m,
        stride_is_k,
        stride_wk,
        stride_wn,
        stride_om,
        stride_on,
        stride_gm,
        stride_gk,
        rank,
        world_size,
        num_row_segs,
        num_tiles_n,
        total_compute_tiles,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GATHER_SMS: tl.constexpr,
        COMPUTE_SMS: tl.constexpr,
        SCATTER_SMS: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """Three-role kernel: gather → compute → scatter."""
        tl.assume(stride_is_m > 0)
        tl.assume(stride_wk > 0)
        tl.assume(stride_wn > 0)
        tl.assume(stride_gm > 0)
        tl.assume(stride_gk > 0)

        pid = tl.program_id(0)

        # ==============================================================
        # GATHER WORKER: allgather input shards into gathered workspace
        # ==============================================================
        if pid < GATHER_SMS:
            for seg_id in range(pid, num_row_segs, GATHER_SMS):
                row_start = seg_id * BLOCK_M
                offs_m = row_start + tl.arange(0, BLOCK_M)
                row_mask = offs_m < M

                # Copy local shard: gathered[:, rank*K_shard:(rank+1)*K_shard]
                local_col_start = rank * K_shard
                for kb in tl.static_range(0, 32):
                    col_off = kb * BLOCK_K
                    if col_off < K_shard:
                        actual_cols = tl.minimum(BLOCK_K, K_shard - col_off)
                        offs_k = tl.arange(0, BLOCK_K)
                        k_mask = offs_k < actual_cols

                        src_ptrs = (
                            input_shard_ptr
                            + offs_m[:, None] * stride_is_m
                            + (col_off + offs_k[None, :]) * stride_is_k
                        )
                        full_mask = row_mask[:, None] & k_mask[None, :]
                        data = tl.load(src_ptrs, mask=full_mask, other=0.0)

                        dst_ptrs = (
                            gathered_ptr
                            + offs_m[:, None] * stride_gk
                            + (local_col_start + col_off + offs_k[None, :])
                        )
                        tl.store(dst_ptrs, data, mask=full_mask)

                # Remote-load from each peer
                for peer in tl.static_range(0, 33):
                    if peer < world_size and peer != rank:
                        peer_col_start = peer * K_shard
                        remote_shard = translate_ptr(
                            input_shard_ptr, rank, peer, heap_bases
                        )
                        for kb in tl.static_range(0, 32):
                            col_off = kb * BLOCK_K
                            if col_off < K_shard:
                                actual_cols = tl.minimum(
                                    BLOCK_K, K_shard - col_off
                                )
                                offs_k = tl.arange(0, BLOCK_K)
                                k_mask = offs_k < actual_cols

                                src_ptrs = (
                                    remote_shard
                                    + offs_m[:, None] * stride_is_m
                                    + (col_off + offs_k[None, :]) * stride_is_k
                                )
                                full_mask = row_mask[:, None] & k_mask[None, :]
                                data = tl.load(
                                    src_ptrs, mask=full_mask, other=0.0
                                )

                                dst_ptrs = (
                                    gathered_ptr
                                    + offs_m[:, None] * stride_gk
                                    + (
                                        peer_col_start
                                        + col_off
                                        + offs_k[None, :]
                                    )
                                )
                                tl.store(dst_ptrs, data, mask=full_mask)

                # Signal gather complete for this row segment
                tile_signal(gather_locks_ptr, seg_id)

        # ==============================================================
        # COMPUTE WORKER: persistent GEMM on gathered input × W
        # ==============================================================
        elif pid < GATHER_SMS + COMPUTE_SMS:
            compute_pid = pid - GATHER_SMS

            for tile_id in range(compute_pid, total_compute_tiles, COMPUTE_SMS):
                tile_m = tile_id // num_tiles_n
                tile_n = tile_id % num_tiles_n

                # Wait for this row segment's gather to finish
                if tile_try_wait(gather_locks_ptr, tile_m) == 0:
                    tile_wait(gather_locks_ptr, tile_m)

                offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
                rm = offs_m % M
                rn = offs_n % N
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

                rk = tl.arange(0, BLOCK_K)
                A_BASE = (
                    gathered_ptr + rm[:, None] * stride_gk + rk[None, :] * 1
                )
                B_BASE = (
                    W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
                )

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                loop_k = tl.cdiv(K, BLOCK_K)
                if not EVEN_K:
                    loop_k -= 1

                for _k in range(0, loop_k):
                    a = tl.load(A_BASE)
                    b = tl.load(B_BASE)
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                    A_BASE += BLOCK_K * 1
                    B_BASE += BLOCK_K * stride_wk

                if not EVEN_K:
                    rk_rem = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
                    A_REM = (
                        gathered_ptr
                        + rm[:, None] * stride_gk
                        + rk_rem[None, :] * 1
                    )
                    B_REM = (
                        W_ptr
                        + rk_rem[:, None] * stride_wk
                        + rn[None, :] * stride_wn
                    )
                    a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0)
                    b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0)
                    acc = tl.dot(a, b, acc, allow_tf32=True)

                result = acc.to(output_ptr.dtype.element_ty)
                out_ptrs = (
                    output_ptr
                    + offs_m[:, None] * stride_om
                    + offs_n[None, :] * stride_on
                )
                out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.store(out_ptrs, result, mask=out_mask)

                tile_signal(compute_locks_ptr, tile_id)

        # ==============================================================
        # SCATTER WORKER: write computed tiles to peers (allscatter)
        # ==============================================================
        else:
            scatter_pid = pid - GATHER_SMS - COMPUTE_SMS

            for tile_id in range(scatter_pid, total_compute_tiles, SCATTER_SMS):
                if tile_try_wait(compute_locks_ptr, tile_id) == 0:
                    tile_wait(compute_locks_ptr, tile_id)

                tile_m = tile_id // num_tiles_n
                tile_n = tile_id % num_tiles_n

                offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

                out_ptrs = (
                    output_ptr
                    + offs_m[:, None] * stride_om
                    + offs_n[None, :] * stride_on
                )
                out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tile_data = tl.load(out_ptrs, mask=out_mask, other=0.0)

                for peer in tl.static_range(0, 33):
                    if peer < world_size and peer != rank:
                        remote_out = translate_ptr(
                            output_ptr, rank, peer, heap_bases
                        )
                        remote_ptrs = (
                            remote_out
                            + offs_m[:, None] * stride_om
                            + offs_n[None, :] * stride_on
                        )
                        tl.store(
                            remote_ptrs,
                            tile_data,
                            mask=out_mask,
                            cache_modifier=".wt",
                        )
