"""
tncc.primitives.memory - Memory tile primitives.

Thin wrappers around Triton load/store intrinsics for tile-level memory
access.  All functions are decorated with @triton.jit for full compiler
visibility.
"""

import triton
import triton.language as tl


@triton.jit
def tile_load(
    ptr,
    offsets,
    mask=None,
    other=0.0,
    cache_modifier: tl.constexpr = "",
):
    """Load a tile from global memory into registers.

    Wraps ``tl.load`` with support for masking and optional cache modifiers.
    Use this for all global-memory reads within a tile pipeline.

    When *mask* is provided, out-of-bounds lanes receive the *other* value,
    which prevents illegal memory accesses at tile boundaries.

    Args:
        ptr: Base pointer to the start of the memory region.
        offsets: Offset tensor describing which elements to load.
        mask: Optional boolean mask; ``True`` lanes are loaded, ``False``
            lanes receive *other*.  Pass ``None`` when the tile is
            guaranteed to be fully in-bounds.
        other: Default value for masked-out lanes (default ``0.0``).
        cache_modifier: Optional cache hint string.  Pass ``".cg"`` for
            cache-global (bypass L1), ``".ca"`` for cache-all (default
            hardware behaviour), or ``""`` to let the compiler decide.

    Returns:
        Tile loaded from ``ptr + offsets``.
    """
    addr = ptr + offsets
    if mask is not None:
        if cache_modifier == ".cg":
            return tl.load(addr, mask=mask, other=other, cache_modifier=".cg")
        elif cache_modifier == ".ca":
            return tl.load(addr, mask=mask, other=other, cache_modifier=".ca")
        else:
            return tl.load(addr, mask=mask, other=other)
    else:
        if cache_modifier == ".cg":
            return tl.load(addr, cache_modifier=".cg")
        elif cache_modifier == ".ca":
            return tl.load(addr, cache_modifier=".ca")
        else:
            return tl.load(addr)


@triton.jit
def tile_store(
    ptr,
    value,
    offsets,
    mask=None,
    cache_modifier: tl.constexpr = "",
):
    """Store a tile from registers to global memory.

    Wraps ``tl.store`` with support for masking and optional cache modifiers.
    Use this for all global-memory writes within a tile pipeline.

    Args:
        ptr: Base pointer to the start of the destination memory region.
        value: Tile of values to write.
        offsets: Offset tensor describing which elements to store.
        mask: Optional boolean mask; only ``True`` lanes are written.
            Pass ``None`` when the tile is guaranteed to be fully in-bounds.
        cache_modifier: Optional cache hint.  Pass ``".cs"`` for
            cache-streaming (write-back, evict first), ``".wb"`` for
            write-back, or ``""`` to let the compiler decide.

    Returns:
        None.
    """
    addr = ptr + offsets
    if mask is not None:
        if cache_modifier == ".cs":
            tl.store(addr, value, mask=mask, cache_modifier=".cs")
        elif cache_modifier == ".wb":
            tl.store(addr, value, mask=mask, cache_modifier=".wb")
        else:
            tl.store(addr, value, mask=mask)
    else:
        if cache_modifier == ".cs":
            tl.store(addr, value, cache_modifier=".cs")
        elif cache_modifier == ".wb":
            tl.store(addr, value, cache_modifier=".wb")
        else:
            tl.store(addr, value)


@triton.jit
def tile_copy(src_ptr, dst_ptr, offsets, mask=None):
    """Copy a tile from one memory location to another within the same GPU.

    Performs a load from *src_ptr* followed by a store to *dst_ptr* using
    the same *offsets* and *mask*.  This is a device-side memcpy at tile
    granularity -- useful for staging data between global memory regions
    (e.g. double-buffering, workspace management).

    Args:
        src_ptr: Base pointer to the source memory region.
        dst_ptr: Base pointer to the destination memory region.
        offsets: Offset tensor for addressing both source and destination.
        mask: Optional boolean mask applied to both load and store.

    Returns:
        None.
    """
    if mask is not None:
        data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, data, mask=mask)
    else:
        data = tl.load(src_ptr + offsets)
        tl.store(dst_ptr + offsets, data)


@triton.jit
def make_block_offsets(
    base_row,
    base_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_row,
    stride_col,
):
    """Create a 2D block offset pattern for tile addressing.

    Computes a ``(BLOCK_M, BLOCK_N)`` tensor of byte offsets so that::

        ptr + make_block_offsets(r, c, M, N, sr, sc)

    addresses the sub-matrix starting at row *base_row*, column *base_col*
    in a row-major matrix with strides *(stride_row, stride_col)*.

    This is the canonical way to turn a block program-id into a set of
    pointers for :func:`tile_load` / :func:`tile_store`.

    Args:
        base_row: Starting row index for this tile.
        base_col: Starting column index for this tile.
        BLOCK_M: Tile height (constexpr).
        BLOCK_N: Tile width (constexpr).
        stride_row: Stride (in elements) between consecutive rows.
        stride_col: Stride (in elements) between consecutive columns.

    Returns:
        A ``(BLOCK_M, BLOCK_N)`` offset tensor suitable for pointer
        arithmetic.
    """
    row_offsets = base_row + tl.arange(0, BLOCK_M)
    col_offsets = base_col + tl.arange(0, BLOCK_N)
    return row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
