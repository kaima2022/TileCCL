# SPDX-License-Identifier: Apache-2.0
"""
tncc.memory.translation - Pointer translation engine.

Provides both **device-side** (``@triton.jit``) and **host-side** pointer
translation utilities.

Device-side
-----------
The Triton JIT functions in this module are the most performance-critical
code in TNCC.  ``translate_ptr`` converts a local device pointer into a
remote device pointer using the ``heap_bases`` table, enabling zero-copy
cross-GPU memory access via IPC.

Host-side
---------
:class:`PointerTranslator` wraps a ``heap_bases`` tensor and exposes a
Pythonic ``translate`` / ``validate`` interface for debugging and testing.

Reference
---------
*   Iris paper, Listing 1 -- ``translate`` function.
*   Iris ``__translate`` implementation (Python host-side + Triton device-side).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# ======================================================================
# Device-side functions (all @triton.jit)
# ======================================================================

@triton.jit
def translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT: tl.constexpr = 0):
    """Core pointer translation -- converts a local ptr to a remote ptr.

    This is the most critical function in TNCC.  It enables zero-copy
    remote memory access via IPC by computing:

        offset     = ptr - heap_bases[from_rank]
        remote_ptr = heap_bases[to_rank] + offset

    Implementation matches Iris Listing 1: exactly 5 core instructions
    (2 loads, 1 subtract, 1 add, 1 cast).  Pointer arithmetic goes
    through ``tl.pointer_type(tl.int8)`` for correctness.

    Parameters
    ----------
    ptr :
        Device pointer (any Triton pointer type) within *from_rank*'s heap.
    from_rank :
        Rank that owns *ptr*.
    to_rank :
        Target rank whose address space we translate into.
    heap_bases :
        Pointer to a ``[world_size]`` int64 tensor containing each rank's
        heap base address **as mapped into this rank's address space**.
    HINT : tl.constexpr
        Vectorization hint (number of contiguous elements).  When > 0,
        applies ``tl.max_contiguous(tl.multiple_of(...))`` to the
        translated pointer, enabling better vectorized loads/stores.
        Use ``BLOCK_SIZE`` for contiguous tile accesses.

    Returns
    -------
    Translated pointer of the same type as *ptr*.
    """
    # 1. Load the source rank's heap base address.
    from_base = tl.load(heap_bases + from_rank)

    # 2. Load the destination rank's heap base address.
    to_base = tl.load(heap_bases + to_rank)

    # 3. Compute byte offset within the source heap.
    # Use uint64 to avoid sign issues with high 64-bit addresses.
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base

    # 4. Compute the remote pointer via byte-level pointer arithmetic.
    # Cast the integer base to a byte pointer, add offset, then cast
    # to the original pointer type.  This produces correct pointer
    # semantics across backends (NVIDIA PTX / AMD LLVM).
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_byte = to_base_byte + offset

    # 5. Cast byte pointer back to the original pointer type.
    translated_ptr = tl.cast(translated_byte, ptr.dtype)

    # Optional vectorization hint for the compiler.
    if HINT > 0:
        translated_ptr = tl.max_contiguous(
            tl.multiple_of(translated_ptr, HINT), HINT
        )
    return translated_ptr


# -----------------------------------------------------------------------
# Convenience: remote_load / remote_store (scalar / 1-D)
# -----------------------------------------------------------------------

@triton.jit
def remote_load(
    ptr,
    from_rank,
    to_rank,
    heap_bases,
    mask=None,
    other=0.0,
    HINT: tl.constexpr = 0,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Load from a remote rank's heap.

    Translates *ptr* (which lives in *from_rank*'s address space) to the
    equivalent pointer in *to_rank*'s address space, then issues a
    ``tl.load``.

    Parameters
    ----------
    ptr :
        Source pointer within *from_rank*'s heap.
    from_rank :
        The rank that owns the original pointer.
    to_rank :
        The rank whose memory we actually read from.
    heap_bases :
        Pointer to the ``[world_size]`` int64 heap-bases tensor.
    mask :
        Optional boolean mask for the load (same semantics as ``tl.load``).
    other :
        Default value for masked-out lanes.
    HINT : tl.constexpr
        Vectorization hint for :func:`translate_ptr`.
    CACHE_MODIFIER : tl.constexpr
        Cache modifier for the load.  Use ``".cg"`` (cache-global, bypass
        L1) for non-reused remote data, ``""`` for default.

    Returns
    -------
    Loaded value(s).
    """
    remote_ptr = translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT=HINT)
    if CACHE_MODIFIER == ".cg":
        return tl.load(remote_ptr, mask=mask, other=other, cache_modifier=".cg")
    elif CACHE_MODIFIER == ".ca":
        return tl.load(remote_ptr, mask=mask, other=other, cache_modifier=".ca")
    else:
        return tl.load(remote_ptr, mask=mask, other=other)


@triton.jit
def remote_store(
    ptr,
    value,
    src_rank,
    dst_rank,
    heap_bases,
    mask=None,
    HINT: tl.constexpr = 0,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Store to a remote rank's heap.

    Translates *ptr* (in *src_rank*'s address space) to *dst_rank*'s space,
    then issues a ``tl.store``.

    Parameters
    ----------
    ptr :
        Destination pointer within *src_rank*'s heap.
    value :
        Value(s) to store.
    src_rank :
        Rank that owns the original pointer.
    dst_rank :
        Rank whose memory we write to.
    heap_bases :
        Pointer to the ``[world_size]`` int64 heap-bases tensor.
    mask :
        Optional boolean mask (same semantics as ``tl.store``).
    HINT : tl.constexpr
        Vectorization hint for :func:`translate_ptr`.
    CACHE_MODIFIER : tl.constexpr
        Cache modifier for the store.  Use ``".wt"`` (write-through,
        bypass L2 pollution) for remote writes, ``""`` for default.
    """
    remote_ptr = translate_ptr(ptr, src_rank, dst_rank, heap_bases, HINT=HINT)
    if CACHE_MODIFIER == ".wt":
        tl.store(remote_ptr, value, mask=mask, cache_modifier=".wt")
    elif CACHE_MODIFIER == ".cs":
        tl.store(remote_ptr, value, mask=mask, cache_modifier=".cs")
    else:
        tl.store(remote_ptr, value, mask=mask)


# -----------------------------------------------------------------------
# Block-level convenience: remote_load_block / remote_store_block
# -----------------------------------------------------------------------

@triton.jit
def remote_load_block(
    ptr,
    from_rank,
    to_rank,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
    mask=None,
    other=0.0,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Load a contiguous block from a remote rank's heap.

    Constructs a range of offsets ``[0, BLOCK_SIZE)`` relative to *ptr*,
    translates the base, and loads the entire block.

    Parameters
    ----------
    ptr :
        Base pointer within *from_rank*'s heap.
    from_rank :
        Rank that owns *ptr*.
    to_rank :
        Rank whose memory to read.
    heap_bases :
        Pointer to the ``[world_size]`` int64 heap-bases tensor.
    BLOCK_SIZE : tl.constexpr
        Number of elements to load.
    mask :
        Optional boolean mask of shape ``(BLOCK_SIZE,)``.
    other :
        Default value for masked-out positions.
    CACHE_MODIFIER : tl.constexpr
        Cache modifier: ``".cg"`` for non-reused remote reads, ``""`` default.

    Returns
    -------
    Loaded block (1-D tensor of *BLOCK_SIZE* elements).
    """
    remote_base = translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT=BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    if CACHE_MODIFIER == ".cg":
        return tl.load(remote_base + offsets, mask=mask, other=other, cache_modifier=".cg")
    else:
        return tl.load(remote_base + offsets, mask=mask, other=other)


@triton.jit
def remote_store_block(
    ptr,
    value,
    src_rank,
    dst_rank,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
    mask=None,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Store a contiguous block into a remote rank's heap.

    Parameters
    ----------
    ptr :
        Base destination pointer within *src_rank*'s heap.
    value :
        Block of values to store (1-D tensor of *BLOCK_SIZE* elements).
    src_rank :
        Rank that owns the original pointer.
    dst_rank :
        Rank whose memory to write.
    heap_bases :
        Pointer to the ``[world_size]`` int64 heap-bases tensor.
    BLOCK_SIZE : tl.constexpr
        Number of elements in the block.
    mask :
        Optional boolean mask of shape ``(BLOCK_SIZE,)``.
    CACHE_MODIFIER : tl.constexpr
        Cache modifier: ``".wt"`` for write-through, ``""`` default.
    """
    remote_base = translate_ptr(ptr, src_rank, dst_rank, heap_bases, HINT=BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    if CACHE_MODIFIER == ".wt":
        tl.store(remote_base + offsets, value, mask=mask, cache_modifier=".wt")
    else:
        tl.store(remote_base + offsets, value, mask=mask)


# ======================================================================
# Host-side pointer translation
# ======================================================================

class PointerTranslator:
    """Host-side pointer translation helper.

    Wraps a ``heap_bases`` tensor (shape ``[world_size]``, dtype int64) and
    provides convenient ``translate`` / ``validate`` methods intended for
    debugging and testing.  For production device-side translation use the
    ``@triton.jit`` functions above.

    Parameters
    ----------
    heap_bases : torch.Tensor
        Shape ``(world_size,)``, dtype ``torch.int64``.  Each entry is the
        base address of the corresponding rank's heap **as mapped into this
        rank's address space**.
    heap_size : int
        Size of each rank's heap in bytes.
    local_rank : int
        The rank of the current process.
    """

    def __init__(
        self,
        heap_bases: torch.Tensor,
        heap_size: int,
        local_rank: int,
    ) -> None:
        if heap_bases.ndim != 1:
            raise ValueError(
                f"heap_bases must be 1-D, got shape {tuple(heap_bases.shape)}"
            )
        if heap_bases.dtype != torch.int64:
            raise ValueError(
                f"heap_bases must be int64, got {heap_bases.dtype}"
            )
        self._bases: torch.Tensor = heap_bases.cpu()
        self._heap_size: int = heap_size
        self._local_rank: int = local_rank
        self._world_size: int = int(heap_bases.shape[0])

    # ---------------------------------------------------------------- query

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return self._world_size

    @property
    def heap_size(self) -> int:
        """Per-rank heap size in bytes."""
        return self._heap_size

    @property
    def local_rank(self) -> int:
        """This process's rank."""
        return self._local_rank

    def base(self, rank: int) -> int:
        """Return the heap base address for *rank*."""
        self._check_rank(rank)
        return int(self._bases[rank].item())

    # ---------------------------------------------------------------- translate

    def translate(self, ptr: int, from_rank: int, to_rank: int) -> int:
        """Translate a pointer from one rank's heap to another.

        Parameters
        ----------
        ptr : int
            Device pointer within *from_rank*'s heap.
        from_rank : int
            Source rank.
        to_rank : int
            Destination rank.

        Returns
        -------
        int
            The equivalent pointer in *to_rank*'s address space.

        Raises
        ------
        ValueError
            If *ptr* is outside *from_rank*'s heap or ranks are invalid.
        """
        self._check_rank(from_rank)
        self._check_rank(to_rank)

        from_base = int(self._bases[from_rank].item())
        to_base = int(self._bases[to_rank].item())

        offset = ptr - from_base
        if offset < 0 or offset >= self._heap_size:
            raise ValueError(
                f"Pointer 0x{ptr:x} is outside rank {from_rank}'s heap "
                f"[0x{from_base:x}, 0x{from_base + self._heap_size:x})"
            )
        return to_base + offset

    def get_offset(self, ptr: int, rank: Optional[int] = None) -> int:
        """Return the byte offset of *ptr* within *rank*'s heap.

        Parameters
        ----------
        ptr : int
            Device pointer.
        rank : int, optional
            Which rank's heap to measure against.  Defaults to
            ``local_rank``.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If *ptr* is outside the heap.
        """
        if rank is None:
            rank = self._local_rank
        self._check_rank(rank)
        base = int(self._bases[rank].item())
        offset = ptr - base
        if offset < 0 or offset >= self._heap_size:
            raise ValueError(
                f"Pointer 0x{ptr:x} is outside rank {rank}'s heap "
                f"[0x{base:x}, 0x{base + self._heap_size:x})"
            )
        return offset

    # ---------------------------------------------------------------- validate

    def validate(self, ptr: int, rank: Optional[int] = None) -> bool:
        """Check whether *ptr* lies within *rank*'s heap.

        Parameters
        ----------
        ptr : int
            Device pointer to validate.
        rank : int, optional
            Rank to check against.  Defaults to ``local_rank``.

        Returns
        -------
        bool
            ``True`` if the pointer is inside the heap, ``False`` otherwise.
        """
        if rank is None:
            rank = self._local_rank
        try:
            self._check_rank(rank)
        except ValueError:
            return False
        base = int(self._bases[rank].item())
        offset = ptr - base
        return 0 <= offset < self._heap_size

    # ---------------------------------------------------------------- internal

    def _check_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self._world_size:
            raise ValueError(
                f"rank={rank} out of range [0, {self._world_size})"
            )

    def __repr__(self) -> str:
        return (
            f"PointerTranslator(world_size={self._world_size}, "
            f"heap_size={self._heap_size}, local_rank={self._local_rank})"
        )
