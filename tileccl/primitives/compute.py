# SPDX-License-Identifier: Apache-2.0
"""
tileccl.primitives.compute - Compute tile primitives.

Thin wrappers around Triton intrinsics for tile-level computation.
All functions are decorated with @triton.jit for full compiler visibility.
"""

import triton
import triton.language as tl


@triton.jit
def tile_dot(a, b, acc):
    """Tile matrix multiply-accumulate: acc += a @ b.

    Wraps ``tl.dot`` to perform a tile-level matrix multiplication and
    accumulate the result into *acc*.  Use this as the inner-loop building
    block for GEMM-style kernels.

    Args:
        a: Left-hand tile of shape ``(M, K)``.
        b: Right-hand tile of shape ``(K, N)``.
        acc: Accumulator tile of shape ``(M, N)`` that receives ``a @ b``.

    Returns:
        Updated accumulator tile ``acc + a @ b``.
    """
    return tl.dot(a, b, acc)


@triton.jit
def tile_reduce(tile, axis):
    """Reduce a tile along an axis using summation.

    Wraps ``tl.sum`` to collapse one dimension of a tile.  Use this for
    row-wise or column-wise reductions inside fused kernels (e.g. softmax
    normalization, variance computation).

    Args:
        tile: Input tile to reduce.
        axis: Axis along which to reduce (0 = rows, 1 = columns).

    Returns:
        Reduced tile with the specified axis collapsed.
    """
    return tl.sum(tile, axis=axis)


@triton.jit
def tile_reduce_max(tile, axis):
    """Reduce a tile along an axis using max.

    Wraps ``tl.max`` to find the maximum value along one dimension.  Useful
    for numerically-stable softmax (subtract max before exp).

    Args:
        tile: Input tile to reduce.
        axis: Axis along which to take the maximum.

    Returns:
        Reduced tile containing per-row or per-column maxima.
    """
    return tl.max(tile, axis=axis)


@triton.jit
def tile_reduce_min(tile, axis):
    """Reduce a tile along an axis using min.

    Wraps ``tl.min`` to find the minimum value along one dimension.

    Args:
        tile: Input tile to reduce.
        axis: Axis along which to take the minimum.

    Returns:
        Reduced tile containing per-row or per-column minima.
    """
    return tl.min(tile, axis=axis)


@triton.jit
def tile_elementwise(tile, op: tl.constexpr):
    """Apply an elementwise operation to a tile.

    Dispatches to Triton elementwise intrinsics based on the *op* string.
    Supported operations:

    - ``"exp"``  -- element-wise exponential
    - ``"log"``  -- element-wise natural logarithm
    - ``"abs"``  -- element-wise absolute value
    - ``"neg"``  -- element-wise negation
    - ``"sqrt"`` -- element-wise square root
    - ``"relu"`` -- element-wise ReLU (max(x, 0))
    - ``"sigmoid"`` -- element-wise sigmoid (1 / (1 + exp(-x)))

    Use this when you need a single element-wise transform fused into a
    larger tile pipeline without breaking the compiler's optimization scope.

    Args:
        tile: Input tile.
        op: Operation name as a ``tl.constexpr`` string.

    Returns:
        Tile with the operation applied element-wise.
    """
    if op == "exp":
        return tl.exp(tile)
    elif op == "log":
        return tl.log(tile)
    elif op == "abs":
        return tl.abs(tile)
    elif op == "neg":
        return -tile
    elif op == "sqrt":
        return tl.sqrt(tile)
    elif op == "relu":
        return tl.maximum(tile, 0.0)
    elif op == "sigmoid":
        return 1.0 / (1.0 + tl.exp(-tile))
    else:
        return tile


@triton.jit
def tile_cast(tile, dtype: tl.constexpr):
    """Cast a tile to a different data type.

    Wraps ``tile.to(dtype)`` for explicit dtype conversion within a JIT
    context.  Use this to down-cast accumulators (e.g. fp32 -> fp16) before
    storing or to up-cast inputs before computation.

    Args:
        tile: Input tile.
        dtype: Target Triton dtype (e.g. ``tl.float16``, ``tl.bfloat16``).

    Returns:
        Tile cast to the requested dtype.
    """
    return tile.to(dtype)


@triton.jit
def tile_zeros(shape_m: tl.constexpr, shape_n: tl.constexpr, dtype: tl.constexpr):
    """Create a zero-initialized tile.

    Returns a tile filled with zeros.  Typically used to initialize
    accumulators before a reduction or GEMM loop.

    Args:
        shape_m: Number of rows (must be ``tl.constexpr``).
        shape_n: Number of columns (must be ``tl.constexpr``).
        dtype: Triton dtype for the tile elements.

    Returns:
        A ``(shape_m, shape_n)`` tile of zeros with the given dtype.
    """
    return tl.zeros((shape_m, shape_n), dtype=dtype)


@triton.jit
def tile_full(shape_m: tl.constexpr, shape_n: tl.constexpr, value, dtype: tl.constexpr):
    """Create a constant-filled tile.

    Returns a tile where every element equals *value*.  Useful for
    initializing mask tiles, bias tiles, or sentinel-valued accumulators
    (e.g. ``-inf`` for max reduction).

    Args:
        shape_m: Number of rows (must be ``tl.constexpr``).
        shape_n: Number of columns (must be ``tl.constexpr``).
        value: Scalar fill value.
        dtype: Triton dtype for the tile elements.

    Returns:
        A ``(shape_m, shape_n)`` tile filled with *value*.
    """
    return tl.full((shape_m, shape_n), value, dtype=dtype)
