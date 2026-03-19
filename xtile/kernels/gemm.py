"""xtile.kernels.gemm - Optimized Triton GEMM kernel.

Provides a persistent-style GEMM (C = A @ B) with:
- Tile-level swizzling (GROUP_SIZE_M) for L2 locality
- K-loop software pipelining (double-buffering) for latency hiding
- Persistent launch via NUM_SMS
- TF32 support

The host-side :func:`gemm` launcher sets up the grid, allocates the output
tensor (if needed), and invokes the kernel.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton JIT kernel -- optimized with software pipelining
# ---------------------------------------------------------------------------

@triton.jit
def gemm_kernel(
    # Pointers
    A_ptr,
    B_ptr,
    C_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ALLOW_TF32: tl.constexpr = True,
):
    """Persistent tile-swizzled GEMM kernel with K-loop pipelining: C = A @ B.

    Each program instance loops over multiple output tiles (persistent style)
    so that the grid size equals ``NUM_SMS`` rather than the total tile count.

    K-loop software pipelining: loads the next K-tile while computing the
    current one, hiding global memory latency behind arithmetic.

    Tile swizzling via ``GROUP_SIZE_M`` improves L2 locality by reordering
    the tile-ID to program-ID mapping.

    Args:
        A_ptr: Pointer to A of shape ``(M, K)``.
        B_ptr: Pointer to B of shape ``(K, N)``.
        C_ptr: Pointer to C of shape ``(M, N)`` (output, written in-place).
        M, N, K: Matrix dimensions.
        stride_am, stride_ak: Row and column strides of A.
        stride_bk, stride_bn: Row and column strides of B.
        stride_cm, stride_cn: Row and column strides of C.
        BLOCK_SIZE_M: Tile height (constexpr).
        BLOCK_SIZE_N: Tile width (constexpr).
        BLOCK_SIZE_K: Inner-loop tile depth (constexpr).
        GROUP_SIZE_M: Number of row-tiles to group for L2 swizzle (constexpr).
        NUM_SMS: Total number of SMs; controls persistent grid size (constexpr).
        ALLOW_TF32: Enable TF32 accumulation for fp32 inputs (constexpr).
    """
    pid = tl.program_id(0)

    # Total number of output tiles
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    # Persistent loop: round-robin tile assignment across SMs.
    for tile_id in range(pid, num_tiles, NUM_SMS):
        # Swizzled tile -> (tile_m, tile_n) mapping for L2 locality.
        num_tiles_in_group = GROUP_SIZE_M * num_tiles_n
        group_id = tile_id // num_tiles_in_group
        first_tile_m_in_group = group_id * GROUP_SIZE_M
        group_size_m_actual = tl.minimum(num_tiles_m - first_tile_m_in_group, GROUP_SIZE_M)
        tile_m = first_tile_m_in_group + ((tile_id % num_tiles_in_group) % group_size_m_actual)
        tile_n = (tile_id % num_tiles_in_group) // group_size_m_actual

        # Offsets for this tile
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Accumulator in fp32
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # ---- K-loop with software pipelining (double-buffer) ----
        # Prefetch the first A/B tile
        offs_k_0 = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs_0 = A_ptr + offs_m[:, None] * stride_am + offs_k_0[None, :] * stride_ak
        b_ptrs_0 = B_ptr + offs_k_0[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_a_0 = (offs_m[:, None] < M) & (offs_k_0[None, :] < K)
        mask_b_0 = (offs_k_0[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs_0, mask=mask_a_0, other=0.0)
        b = tl.load(b_ptrs_0, mask=mask_b_0, other=0.0)

        num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)

        for k_iter in range(0, num_k_iters):
            # Compute on the current tile
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)

            # Prefetch the NEXT tile (if not last iteration)
            next_k_start = (k_iter + 1) * BLOCK_SIZE_K
            if next_k_start < K:
                offs_k_next = next_k_start + tl.arange(0, BLOCK_SIZE_K)
                a_ptrs_next = A_ptr + offs_m[:, None] * stride_am + offs_k_next[None, :] * stride_ak
                b_ptrs_next = B_ptr + offs_k_next[:, None] * stride_bk + offs_n[None, :] * stride_bn
                mask_a_next = (offs_m[:, None] < M) & (offs_k_next[None, :] < K)
                mask_b_next = (offs_k_next[:, None] < K) & (offs_n[None, :] < N)
                a = tl.load(a_ptrs_next, mask=mask_a_next, other=0.0, eviction_policy="evict_last")
                b = tl.load(b_ptrs_next, mask=mask_b_next, other=0.0, eviction_policy="evict_last")

        # Store C tile
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask_c)


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

def gemm(
    A,
    B,
    C=None,
    *,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 32,
    GROUP_SIZE_M: int = 8,
    NUM_SMS: int | None = None,
    allow_tf32: bool = True,
):
    """Launch the persistent GEMM kernel: ``C = A @ B``.

    Args:
        A: Input tensor of shape ``(M, K)`` on GPU.
        B: Input tensor of shape ``(K, N)`` on GPU.
        C: Optional output tensor of shape ``(M, N)``.  If ``None``, a new
            tensor is allocated on the same device as *A*.
        BLOCK_SIZE_M: Tile height (default 128).
        BLOCK_SIZE_N: Tile width (default 128).
        BLOCK_SIZE_K: Inner-loop depth (default 32).
        GROUP_SIZE_M: Swizzle group size (default 8).
        NUM_SMS: Persistent grid size.  Defaults to the device SM count.
        allow_tf32: Enable TF32 for fp32 accumulation (default True).

    Returns:
        Output tensor C of shape ``(M, N)``.
    """
    import torch

    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2-D tensors"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: A is {M}x{K}, B is {K2}x{N}"

    if C is None:
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    else:
        assert C.shape == (M, N), f"C shape mismatch: expected ({M}, {N}), got {C.shape}"

    if NUM_SMS is None:
        NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

    # Adaptive GROUP_SIZE_M based on M/N ratio
    num_tiles_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    if num_tiles_m <= 4:
        effective_group_size = min(GROUP_SIZE_M, num_tiles_m)
    else:
        effective_group_size = GROUP_SIZE_M

    grid = (NUM_SMS,)

    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=effective_group_size,
        NUM_SMS=NUM_SMS,
        ALLOW_TF32=allow_tf32,
    )

    return C
