"""tncc.kernels.gemm - Optimized Triton GEMM kernel.

Provides a persistent-style GEMM (C = A @ B) aligned with the Iris
persistent_gemm pattern:

- Split K-loop: mask-free main loop + masked remainder (EVEN_K)
- Compiler hints: tl.max_contiguous + tl.multiple_of for vector loads
- Incremental pointer advancement (no recomputation)
- Modular index wrapping (% M, % N) eliminates M/N boundary masks in K-loop
- Tile-level swizzling (GROUP_SIZE_M) for L2 locality
- Persistent launch via NUM_SMS
- Auto-select block sizes based on M/N/K

The host-side :func:`gemm` launcher sets up the grid, allocates the output
tensor (if needed), and invokes the kernel.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block size auto-selection
# ---------------------------------------------------------------------------

def _select_config(M: int, N: int, K: int) -> tuple[int, int, int, int, int]:
    """Select ``(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages)``.

    Heuristic based on H100 PCIe benchmarking:
    - 128x128x64 remains the best large-matrix tile family.
    - `num_stages=4` is the most stable default across fp16/bf16 official
      `bench_gemm.py` measurements, even though explicit overrides remain
      available for local experimentation.
    - Small matrices use 64x64x32 to avoid excessive over-tiling.
    """
    if M <= 512 or N <= 512:
        return 64, 64, 32, 4, 4

    return 128, 128, 64, 4, 4


# Module-level cache for device SM count (avoids repeated CUDA API calls)
_device_sm_count: dict[int, int] = {}


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
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr = True,
):
    """Persistent tile-swizzled GEMM kernel: C = A @ B.

    Optimized K-loop with Iris-style mask elimination:
    - Modular index wrapping (% M, % N) removes M/N boundary masks from loads
    - EVEN_K splits K-loop into mask-free main + masked remainder
    - Compiler hints (max_contiguous, multiple_of) enable wide vector loads
    - Incremental pointer advancement minimizes address arithmetic

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
        EVEN_K: True when K % BLOCK_SIZE_K == 0 (constexpr).
        ALLOW_TF32: Enable TF32 accumulation for fp32 inputs (constexpr).
    """
    # Compiler hints: strides are always positive for standard tensors
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

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
        first_tile_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_tiles_m - first_tile_m, GROUP_SIZE_M)
        tile_m = first_tile_m + ((tile_id % num_tiles_in_group) % group_size_m)
        tile_n = (tile_id % num_tiles_in_group) // group_size_m

        # Index offsets with modular wrapping — all indices land in [0, M)
        # and [0, N), eliminating M/N boundary masks from the entire K-loop.
        rm = (tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        rk = tl.arange(0, BLOCK_SIZE_K)

        # Base pointers — advanced incrementally each K iteration
        A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # Accumulator in fp32
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # ---- Split K-loop: main iterations without any masks ----
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        for k in range(0, loop_k):
            a = tl.load(A_BASE)
            b = tl.load(B_BASE)
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        # ---- Remainder: K-boundary mask only (no M/N masks) ----
        if not EVEN_K:
            rk_rem = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_REM = A_ptr + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
            B_REM = B_ptr + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0)
            b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0)
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)

        # ---- Store C tile (M/N boundary mask needed only here) ----
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
    BLOCK_SIZE_M: int | None = None,
    BLOCK_SIZE_N: int | None = None,
    BLOCK_SIZE_K: int | None = None,
    GROUP_SIZE_M: int = 8,
    NUM_SMS: int | None = None,
    allow_tf32: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """Launch the persistent GEMM kernel: ``C = A @ B``.

    When block sizes are not specified, an auto-tuning heuristic selects
    optimal values based on the matrix dimensions.

    Args:
        A: Input tensor of shape ``(M, K)`` on GPU.
        B: Input tensor of shape ``(K, N)`` on GPU.
        C: Optional output tensor of shape ``(M, N)``.  If ``None``, a new
            tensor is allocated on the same device as *A*.
        BLOCK_SIZE_M: Tile height (default: auto-select).
        BLOCK_SIZE_N: Tile width (default: auto-select).
        BLOCK_SIZE_K: Inner-loop depth (default: auto-select).
        GROUP_SIZE_M: Swizzle group size (default 8).
        NUM_SMS: Persistent grid size.  Defaults to the device SM count.
        allow_tf32: Enable TF32 for fp32 accumulation (default True).
        num_warps: Optional override for Triton launch warps.
        num_stages: Optional override for Triton software pipeline depth.

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

    # Auto-select block sizes and launch parameters if not specified.
    if (
        BLOCK_SIZE_M is None
        or BLOCK_SIZE_N is None
        or BLOCK_SIZE_K is None
        or num_warps is None
        or num_stages is None
    ):
        selected_BM, selected_BN, selected_BK, selected_warps, selected_stages = (
            _select_config(M, N, K)
        )
        if BLOCK_SIZE_M is None:
            BLOCK_SIZE_M = selected_BM
        if BLOCK_SIZE_N is None:
            BLOCK_SIZE_N = selected_BN
        if BLOCK_SIZE_K is None:
            BLOCK_SIZE_K = selected_BK
        if num_warps is None:
            num_warps = selected_warps
        if num_stages is None:
            num_stages = selected_stages

    if NUM_SMS is None:
        dev_idx = A.device.index
        if dev_idx not in _device_sm_count:
            _device_sm_count[dev_idx] = torch.cuda.get_device_properties(dev_idx).multi_processor_count
        NUM_SMS = _device_sm_count[dev_idx]

    # EVEN_K: when K is divisible by BLOCK_SIZE_K, no remainder mask needed
    EVEN_K = (K % BLOCK_SIZE_K == 0)

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
        EVEN_K=EVEN_K,
        ALLOW_TF32=allow_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return C
