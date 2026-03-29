# SPDX-License-Identifier: Apache-2.0
"""Tests for all 4 overlap patterns: GEMM correctness and scatter validation.

Each pattern is tested for:
1. GEMM correctness (single GPU, world_size=1, scatter is no-op)
2. Scatter correctness (2 GPUs, verifies cross-GPU data transfer via translate_ptr)
"""

from __future__ import annotations

import pytest
import torch

import tncc
from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.patterns import (
    BulkSyncPattern,
    FusedSequentialPattern,
    ProducerConsumerPattern,
    WGSpecializedPattern,
)


# -----------------------------------------------------------------------
# GEMM correctness (single GPU, no scatter)
# -----------------------------------------------------------------------


@pytest.mark.multigpu
class TestPatternGEMMCorrectness:
    """Verify that each pattern's persistent GEMM produces correct results."""

    @pytest.fixture(autouse=True)
    def _setup(self, skip_no_multigpu, device_info):
        self.device = device_info.device
        self.backend_name = device_info.backend

    @pytest.fixture
    def single_gpu_ctx(self):
        """Create a single-GPU ctx with SymmetricHeap."""
        heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=1)
        torch.cuda.set_device(0)
        ctx = tncc.init(
            backend=self.backend_name,
            rank=0,
            world_size=1,
            heap=heaps[0],
            force_backend=True,
        )
        yield ctx
        for h in heaps:
            h.cleanup()

    def _check_gemm(self, pattern_cls, ctx, M=512, N=512, K=512):
        """Run a pattern and compare output to torch.matmul."""
        A = torch.randn(M, K, device=ctx.device, dtype=torch.float16)
        B = torch.randn(K, N, device=ctx.device, dtype=torch.float16)
        C = torch.zeros(M, N, device=ctx.device, dtype=torch.float16)

        pattern = pattern_cls(ctx)
        pattern.execute(A, B, C)
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B.float()).half()
        assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1), (
            f"{pattern_cls.name} GEMM failed. "
            f"Max diff: {(C.float() - ref.float()).abs().max().item():.4f}"
        )

    def test_bulk_sync_gemm(self, single_gpu_ctx) -> None:
        self._check_gemm(BulkSyncPattern, single_gpu_ctx)

    def test_fused_sequential_gemm(self, single_gpu_ctx) -> None:
        self._check_gemm(FusedSequentialPattern, single_gpu_ctx)

    def test_producer_consumer_gemm(self, single_gpu_ctx) -> None:
        self._check_gemm(ProducerConsumerPattern, single_gpu_ctx)

    def test_wg_specialized_gemm(self, single_gpu_ctx) -> None:
        self._check_gemm(WGSpecializedPattern, single_gpu_ctx)


# -----------------------------------------------------------------------
# Scatter correctness (2 GPUs)
# -----------------------------------------------------------------------


@pytest.mark.multigpu
class TestPatternScatterCorrectness:
    """Verify that each pattern correctly scatters tiles to peer GPUs."""

    @pytest.fixture(autouse=True)
    def _setup(self, skip_no_multigpu, device_info):
        if device_info.num_gpus < 2:
            pytest.skip("Requires 2+ GPUs")
        self.device = device_info.device
        self.backend_name = device_info.backend

    @pytest.fixture
    def dual_gpu_heaps(self):
        """Create 2-GPU SymmetricHeap set."""
        heaps = SymmetricHeap.create_all(size=64 * 1024 * 1024, world_size=2)
        yield heaps
        for h in heaps:
            h.cleanup()

    def _check_scatter(self, pattern_cls, dual_gpu_heaps):
        """Run a pattern on GPU 0 and verify scatter to GPU 1."""
        M, N, K = 256, 256, 256
        world_size = 2
        N_per_rank = N // world_size

        heaps = dual_gpu_heaps
        torch.cuda.set_device(0)

        # Allocate C from symmetric heap (required for translate_ptr)
        C0 = heaps[0].allocate_tensor((M, N), dtype=torch.float16)
        C1 = heaps[1].allocate_tensor((M, N), dtype=torch.float16)
        C0.zero_()
        C1.zero_()

        A = torch.randn(M, K, device="cuda:0", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda:0", dtype=torch.float16)

        ctx = tncc.init(
            backend=self.backend_name,
            rank=0,
            world_size=world_size,
            heap=heaps[0],
            force_backend=True,
        )

        pattern = pattern_cls(ctx, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
        pattern.execute(A, B, C0, full_N=N, b_layout="full", c_layout="full")
        torch.cuda.synchronize()

        # Verify GEMM correctness on GPU 0
        ref = torch.matmul(A.float(), B.float()).half()
        assert torch.allclose(C0.cpu(), ref.cpu(), rtol=1e-2, atol=1e-1), (
            f"{pattern_cls.name} GEMM on GPU 0 failed."
        )

        # Verify scatter: GPU 1 should have rank 0's column shard
        C1_host = C1.cpu()
        expected = ref[:, :N_per_rank].cpu()
        assert torch.allclose(C1_host[:, :N_per_rank], expected, rtol=1e-2, atol=1e-1), (
            f"{pattern_cls.name} scatter to GPU 1 failed. "
            f"Max diff: {(C1_host[:, :N_per_rank].float() - expected.float()).abs().max().item():.4f}"
        )

    def test_bulk_sync_scatter(self, dual_gpu_heaps) -> None:
        self._check_scatter(BulkSyncPattern, dual_gpu_heaps)

    def test_fused_sequential_scatter(self, dual_gpu_heaps) -> None:
        self._check_scatter(FusedSequentialPattern, dual_gpu_heaps)

    def test_producer_consumer_scatter(self, dual_gpu_heaps) -> None:
        self._check_scatter(ProducerConsumerPattern, dual_gpu_heaps)

    def test_wg_specialized_scatter(self, dual_gpu_heaps) -> None:
        self._check_scatter(WGSpecializedPattern, dual_gpu_heaps)
