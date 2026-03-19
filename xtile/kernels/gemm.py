"""xtile.kernels.gemm - Standard Triton GEMM kernel.

Provides a persistent-style GEMM (C = A @ B) that serves as the baseline
compute kernel for all communication-overlap patterns.  The kernel uses
tile-level swizzling (GROUP_SIZE_M) and supports persistent launch via
NUM_SMS.

The host-side :func:`gemm` launcher sets up the grid, allocates the output
tensor (if needed), and invokes the kernel.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton JIT kernel
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
):
    """Persistent tile-swizzled GEMM kernel: C = A @ B.

    Each program instance loops over multiple output tiles (persistent style)
    so that the grid size equals ``NUM_SMS`` rather than the total tile count.
    Within each iteration the standard tiled matmul accumulation is performed
    in float32 before down-casting to the output dtype.

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
    """
    pid = tl.program_id(0)

    # Total number of output tiles
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    # Persistent loop: round-robin tile assignment across SMs.
    # This provides better load balancing than block-partition because
    # boundary tiles (with masking overhead) are distributed evenly.
    for tile_id in range(pid, num_tiles, NUM_SMS):
        # Swizzled tile -> (tile_m, tile_n) mapping for L2 locality.
        # GROUP_SIZE_M adjacent row-tiles are grouped together so that
        # they share B columns in the L2 cache.
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

        # Inner loop over K
        for k_start in range(0, K, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

            # Load A tile: (BLOCK_SIZE_M, BLOCK_SIZE_K)
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            # Load B tile: (BLOCK_SIZE_K, BLOCK_SIZE_N)
            b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            # Accumulate: use 3-operand form for fused multiply-add
            acc = tl.dot(a, b, acc, allow_tf32=True)

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
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
    )

    return C
