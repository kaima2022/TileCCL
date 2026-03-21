"""
xtile.patterns._helpers - Shared kernel utilities for overlap patterns.

Provides @triton.jit device-side functions used by multiple overlap patterns.
Extracting common scatter logic here ensures a single source of truth for
offset computation and pointer translation semantics.

All functions are @triton.jit and are inlined by the Triton compiler into
the calling kernel -- there is zero call overhead.
"""

import triton
import triton.language as tl

from xtile.memory.translation import translate_ptr


@triton.jit
def scatter_tile_to_peer(
    C_ptr,
    tile_data,
    offs_m,
    offs_n,
    rank,
    peer,
    heap_bases,
    src_col_offset,
    valid_cols,
    dst_leading_dim,
    dst_col_offset,
    mask,
    CACHE_MODIFIER: tl.constexpr = ".wt",
):
    """Scatter a computed tile to a peer GPU via symmetric-heap translation.

    Translates the local output pointer to the peer's address space using
    :func:`~xtile.memory.translation.translate_ptr`, then stores the tile
    at the explicit destination layout described by the host-side
    execution contract.

    Parameters
    ----------
    C_ptr :
        Base pointer to the local output buffer (typed Triton pointer,
        must reside within the caller's symmetric heap).
    tile_data :
        Tile values to scatter (in registers, shape ``(BLOCK_M, BLOCK_N)``).
    offs_m :
        Row offsets for the tile, shape ``(BLOCK_M,)``.
    offs_n :
        Column offsets for the tile, shape ``(BLOCK_N,)``.
    rank :
        Caller's rank (the rank that owns *C_ptr*).
    peer :
        Target peer rank to scatter to.
    heap_bases :
        Pointer to the ``[world_size]`` int64 tensor of per-rank
        symmetric-heap base addresses (as mapped into the caller's
        address space).
    src_col_offset :
        Starting column in the local source buffer that is allowed to
        participate in the scatter.
    valid_cols :
        Number of valid source columns to scatter.
    dst_leading_dim :
        Row stride, in elements, of the destination layout on the peer.
    dst_col_offset :
        Starting destination column offset on the peer.
    mask :
        Boolean mask for boundary tiles, shape ``(BLOCK_M, BLOCK_N)``.
    CACHE_MODIFIER : tl.constexpr
        Cache modifier for the remote store.  Use ``".wt"`` for
        write-through (avoids L2 pollution), ``""`` for default.
    """
    # Translate local pointer to peer's heap region (via peer access).
    # The result is a typed pointer in the caller's address space that
    # maps to the equivalent location in peer's symmetric heap.
    remote_C = translate_ptr(C_ptr, rank, peer, heap_bases)

    # Convert the source columns into destination-local columns while
    # masking out anything outside the explicitly declared scatter span.
    col_mask = (offs_n >= src_col_offset) & (offs_n < src_col_offset + valid_cols)
    safe_local_cols = tl.where(col_mask, offs_n - src_col_offset, 0)
    offsets = offs_m[:, None] * dst_leading_dim + (dst_col_offset + safe_local_cols[None, :])
    final_mask = mask & col_mask[None, :]

    if CACHE_MODIFIER == ".wt":
        tl.store(remote_C + offsets, tile_data, mask=final_mask, cache_modifier=".wt")
    else:
        tl.store(remote_C + offsets, tile_data, mask=final_mask)
