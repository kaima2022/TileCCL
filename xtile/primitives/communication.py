"""
xtile.primitives.communication - Communication tile primitives.

Provides device-side routines for cross-GPU tile transfer using symmetric
memory and pointer translation.  All functions are decorated with
@triton.jit for full compiler visibility.

Two categories of operations are exposed:

* **Value-based** (register <-> remote memory): fine-grained, a single
  tile is loaded into / stored from registers directly.
* **Pointer-based** (memory <-> memory): coarse-grained, data is copied
  between local and remote memory regions without transiting through
  registers explicitly.

All routines rely on :func:`xtile.memory.translation.translate_ptr` to
convert a local pointer + rank information into a remote-accessible pointer.
"""

import triton
import triton.language as tl

from xtile.memory.translation import translate_ptr


# -----------------------------------------------------------------------
# Value-based operations (register <-> remote memory)
# -----------------------------------------------------------------------


@triton.jit
def tile_remote_load(
    ptr,
    from_rank,
    to_rank,
    heap_bases,
    offsets,
    mask=None,
    other=0.0,
):
    """Load a tile from a remote GPU's memory into local registers.

    Translates *ptr* (which is expressed in *to_rank*'s address space) into
    an address accessible by *from_rank*, then performs a standard
    ``tl.load``.  The result lives in the calling thread's registers.

    This is a **value-based** operation: data flows from remote global
    memory directly into registers.  Use this for fine-grained reads of
    small tiles where the latency of a full DMA copy is not justified.

    Args:
        ptr: Base pointer in the **source** rank's address space.
        from_rank: Rank that owns the memory being read.
        to_rank: Rank that is performing the read (the caller).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers,
            shape ``(world_size,)``.
        offsets: Offset tensor for tile addressing.
        mask: Optional boolean mask for out-of-bounds protection.
        other: Fill value for masked-out lanes (default ``0.0``).

    Returns:
        Tile loaded from the remote rank's memory.
    """
    remote_ptr = translate_ptr(ptr, from_rank, to_rank, heap_bases)
    if mask is not None:
        return tl.load(remote_ptr + offsets, mask=mask, other=other)
    else:
        return tl.load(remote_ptr + offsets)


@triton.jit
def tile_remote_store(
    ptr,
    value,
    src_rank,
    dst_rank,
    heap_bases,
    offsets,
    mask=None,
):
    """Store a tile from local registers into a remote GPU's memory.

    Translates *ptr* (expressed in *dst_rank*'s address space) into an
    address accessible by *src_rank*, then performs a standard ``tl.store``.

    This is a **value-based** operation: data flows from local registers
    into remote global memory.  Use this for fine-grained writes of small
    tiles, e.g. writing partial-sum results to a peer's accumulation buffer.

    Args:
        ptr: Base pointer in the **destination** rank's address space.
        value: Tile of values to write.
        src_rank: Rank that is performing the write (the caller).
        dst_rank: Rank that owns the destination memory.
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for tile addressing.
        mask: Optional boolean mask; only ``True`` lanes are written.

    Returns:
        None.
    """
    remote_ptr = translate_ptr(ptr, dst_rank, src_rank, heap_bases)
    if mask is not None:
        tl.store(remote_ptr + offsets, value, mask=mask)
    else:
        tl.store(remote_ptr + offsets, value)


# -----------------------------------------------------------------------
# Pointer-based operations (memory <-> memory)
# -----------------------------------------------------------------------


@triton.jit
def tile_put(
    src_ptr,
    dst_ptr,
    src_rank,
    dst_rank,
    heap_bases,
    offsets,
    mask=None,
):
    """Copy a tile from local memory to remote memory (push / put).

    Reads from *src_ptr* in the caller's local memory, translates *dst_ptr*
    to a remote-accessible address, and writes the data there.  This is
    analogous to an RDMA PUT or ``shmem_put``.

    This is a **pointer-based** operation: both source and destination are
    in global memory (local and remote respectively).  Use this for bulk
    tile transfers where throughput matters more than register residency.

    Args:
        src_ptr: Base pointer in the **local** (caller) rank's memory.
        dst_ptr: Base pointer in the **remote** rank's address space.
        src_rank: Rank performing the operation (the caller).
        dst_rank: Rank that owns the destination memory.
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for tile addressing.
        mask: Optional boolean mask for partial-tile transfers.

    Returns:
        None.
    """
    # Load from local memory
    if mask is not None:
        data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    else:
        data = tl.load(src_ptr + offsets)

    # Translate destination pointer to remote-accessible address
    remote_dst = translate_ptr(dst_ptr, dst_rank, src_rank, heap_bases)

    # Store to remote memory
    if mask is not None:
        tl.store(remote_dst + offsets, data, mask=mask)
    else:
        tl.store(remote_dst + offsets, data)


@triton.jit
def tile_get(
    dst_ptr,
    src_ptr,
    from_rank,
    to_rank,
    heap_bases,
    offsets,
    mask=None,
):
    """Copy a tile from remote memory to local memory (pull / get).

    Translates *src_ptr* to a remote-accessible address, reads from it,
    and writes into *dst_ptr* in the caller's local memory.  This is
    analogous to an RDMA GET or ``shmem_get``.

    This is a **pointer-based** operation: both source and destination are
    in global memory (remote and local respectively).  Use this for bulk
    tile fetches where throughput matters more than register residency.

    Args:
        dst_ptr: Base pointer in the **local** (caller) rank's memory.
        src_ptr: Base pointer in the **remote** rank's address space.
        from_rank: Rank that owns the source memory.
        to_rank: Rank performing the operation (the caller).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for tile addressing.
        mask: Optional boolean mask for partial-tile transfers.

    Returns:
        None.
    """
    # Translate source pointer to remote-accessible address
    remote_src = translate_ptr(src_ptr, from_rank, to_rank, heap_bases)

    # Load from remote memory
    if mask is not None:
        data = tl.load(remote_src + offsets, mask=mask, other=0.0)
    else:
        data = tl.load(remote_src + offsets)

    # Store to local memory
    if mask is not None:
        tl.store(dst_ptr + offsets, data, mask=mask)
    else:
        tl.store(dst_ptr + offsets, data)
