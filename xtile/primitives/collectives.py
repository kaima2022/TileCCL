"""
xtile.primitives.collectives - Collective communication primitives.

Tile-level collective operations implemented entirely in Triton
(pure ``@triton.jit``, no NCCL / RCCL dependency).  All remote memory
access goes through :func:`~xtile.memory.translation.translate_ptr`.

Device-side (``@triton.jit``):
    tile_allreduce, tile_allgather, tile_scatter, tile_reduce_scatter,
    tile_broadcast.

Host-side launchers (testing / benchmarking):
    allreduce, allgather, broadcast.

Conventions:
    - Ring direction: send to ``(rank+1)%W``, receive from ``(rank-1+W)%W``.
    - Memory ordering: "release" on remote writes, "acquire" on remote reads.
    - See ``TODO`` markers for known limitations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xtile.memory.translation import translate_ptr


# =====================================================================
# Internal helpers (device-side, @triton.jit)
# =====================================================================


@triton.jit
def _reduce_op(a, b, op: tl.constexpr):
    """Element-wise binary reduction: ``"sum"`` | ``"max"`` | ``"min"``.

    TODO: Add ``"prod"`` when Triton supports it natively.
    """
    if op == "sum":
        return a + b
    elif op == "max":
        return tl.maximum(a, b)
    elif op == "min":
        return tl.minimum(a, b)
    # Fallback: treat unknown ops as sum (should not happen).
    return a + b


@triton.jit
def _ring_next(rank, world_size):
    """Return the successor rank in the ring: ``(rank + 1) % world_size``."""
    return (rank + 1) % world_size


@triton.jit
def _ring_prev(rank, world_size):
    """Return the predecessor rank in the ring: ``(rank - 1 + world_size) % world_size``."""
    return (rank - 1 + world_size) % world_size


# =====================================================================
# tile_allreduce -- Ring allreduce
# =====================================================================


@triton.jit
def tile_allreduce(
    local_ptr,
    offsets,
    rank,
    world_size,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
    op: tl.constexpr = "sum",
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "gpu",
):
    """Ring allreduce at tile granularity.

    After this function returns, every rank's ``local_ptr`` region holds
    the element-wise reduction of all ranks' original data.

    Algorithm
    ---------
    The implementation follows the classic two-phase ring algorithm:

    **Phase 1 -- Reduce-scatter** (``world_size - 1`` steps):
        The data buffer is logically divided into ``world_size`` equal
        chunks of ``BLOCK_SIZE`` elements.  At each step *s* the current
        rank loads its "send chunk", translates the next rank's pointer,
        stores the chunk there, and then loads the incoming chunk from the
        previous rank, reducing it with its local copy.  After this phase,
        each rank holds the fully-reduced result for exactly one chunk.

    **Phase 2 -- Allgather** (``world_size - 1`` steps):
        Each rank forwards its fully-reduced chunk around the ring until
        every rank has all chunks.

    When to use
    -----------
    Use ``tile_allreduce`` when **every** rank needs the **complete**
    reduced result (e.g. gradient all-reduce in data-parallel training).
    If each rank only needs 1/N of the result, prefer
    :func:`tile_reduce_scatter` which skips Phase 2.

    Args:
        local_ptr: Data buffer (``world_size * BLOCK_SIZE`` elements).
            Modified in-place; on return holds the full reduced result.
        offsets: ``(BLOCK_SIZE,)`` offsets, typically ``tl.arange(0, BLOCK_SIZE)``.
        rank: This rank's index in ``[0, world_size)``.
        world_size: Number of participating ranks.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements per chunk (total = ``world_size * BLOCK_SIZE``).
        op: Reduction operator: ``"sum"`` | ``"max"`` | ``"min"``.
        sem: Memory ordering for remote writes (default ``"release"``).
        scope: Visibility scope (default ``"gpu"``).

    TODO: Non-power-of-2 world_size optimisation.
    TODO: Topology-aware ring ordering (NVSwitch-optimal).
    TODO: Pipelining overlapped sends/receives to hide latency.
    """
    next_rank = _ring_next(rank, world_size)
    prev_rank = _ring_prev(rank, world_size)

    # ------------------------------------------------------------------
    # Phase 1: Reduce-scatter
    # ------------------------------------------------------------------
    # After step s, chunk index (rank - s) % world_size is partially
    # reduced on this rank.
    for step in tl.static_range(0, 8):
        # static_range(0, 8) supports up to 9 GPUs.
        # TODO: Increase bound for world_size > 9.
        if step >= world_size - 1:
            pass
        else:
            # Chunk index this rank sends at this step.
            send_chunk_idx = (rank - step + world_size) % world_size
            chunk_offset = send_chunk_idx * BLOCK_SIZE

            # Load local chunk data.
            local_data = tl.load(local_ptr + chunk_offset + offsets)

            # Translate next rank's pointer and store our chunk there.
            remote_next = translate_ptr(local_ptr, rank, next_rank, heap_bases)
            tl.store(remote_next + chunk_offset + offsets, local_data)

            # Intra-CTA fence; combined with symmetric execution (one
            # CTA per rank) this provides the necessary ordering.
            tl.debug_barrier()

            # Chunk index this rank receives and reduces at this step.
            recv_chunk_idx = (rank - step - 1 + world_size) % world_size
            recv_offset = recv_chunk_idx * BLOCK_SIZE

            # Load incoming data from the previous rank.
            remote_prev = translate_ptr(local_ptr, rank, prev_rank, heap_bases)
            incoming = tl.load(remote_prev + recv_offset + offsets)

            # Load our own current value for that chunk and reduce.
            my_data = tl.load(local_ptr + recv_offset + offsets)
            reduced = _reduce_op(my_data, incoming, op)

            # Write the partial reduction back to our local buffer.
            tl.store(local_ptr + recv_offset + offsets, reduced)
            tl.debug_barrier()

    # ------------------------------------------------------------------
    # Phase 2: Allgather
    # ------------------------------------------------------------------
    # After reduce-scatter, chunk (rank + 1) % world_size is fully
    # reduced on this rank.  Forward it around the ring.
    for step in tl.static_range(0, 8):
        if step >= world_size - 1:
            pass
        else:
            # Chunk index to forward at this step.
            fwd_chunk_idx = (rank - step + 1 + world_size) % world_size
            fwd_offset = fwd_chunk_idx * BLOCK_SIZE

            # Load fully-reduced chunk from local buffer.
            fwd_data = tl.load(local_ptr + fwd_offset + offsets)

            # Send to next rank.
            remote_next = translate_ptr(local_ptr, rank, next_rank, heap_bases)
            tl.store(remote_next + fwd_offset + offsets, fwd_data)
            tl.debug_barrier()


# =====================================================================
# tile_allgather -- Every rank gets every tile
# =====================================================================


@triton.jit
def tile_allgather(
    src_ptr,
    dst_ptr,
    offsets,
    rank,
    world_size,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
):
    """Allgather: each rank contributes its tile; all ranks get all tiles.

    Result layout in ``dst_ptr``::

        [ tile_rank_0 | tile_rank_1 | ... | tile_rank_{N-1} ]

    Algorithm: direct-write -- every rank stores its tile to all peers'
    ``dst_ptr`` at offset ``rank * BLOCK_SIZE``.  Optimal on full-bisection
    fabrics (NVSwitch); for ring topologies a pipelined variant is better.

    Use when each rank produces a distinct tile and all ranks need the
    concatenated result (e.g. tensor-parallel column-linear gather).

    Args:
        src_ptr: This rank's source tile (``BLOCK_SIZE`` elements).
        dst_ptr: Output buffer (``world_size * BLOCK_SIZE`` elements).
        offsets: ``(BLOCK_SIZE,)`` offsets.
        rank: This rank's index.
        world_size: Number of participating ranks.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements per tile.

    TODO: Ring-based variant for asymmetric topologies.
    TODO: Non-contiguous stride support.
    """
    # Load this rank's tile from src_ptr.
    my_tile = tl.load(src_ptr + offsets)

    # Write into our own dst_ptr first (local copy, no translation needed).
    dst_offset = rank * BLOCK_SIZE
    tl.store(dst_ptr + dst_offset + offsets, my_tile)

    # Write into every remote peer's dst_ptr.
    for peer in tl.static_range(0, 8):
        # Guard: skip iterations beyond world_size and skip self.
        if peer >= world_size:
            pass
        else:
            if peer != rank:
                remote_dst = translate_ptr(
                    dst_ptr, rank, peer, heap_bases
                )
                tl.store(remote_dst + dst_offset + offsets, my_tile)


# =====================================================================
# tile_scatter -- Root distributes distinct tiles
# =====================================================================


@triton.jit
def tile_scatter(
    src_ptr,
    dst_ptr,
    offsets,
    rank,
    world_size,
    root,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter: root distributes distinct tiles to each rank.

    Only root performs work.  Rank *i* receives
    ``src_ptr[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]`` from root's buffer.

    Algorithm: root loads each chunk and writes it to rank *i*'s
    ``dst_ptr`` via pointer translation.  Non-root ranks are idle.

    Use for distributing initial data (pipeline inputs, weight shards).

    Args:
        src_ptr: On root: source buffer (``world_size * BLOCK_SIZE``).
        dst_ptr: Destination buffer (``BLOCK_SIZE`` elements per rank).
        offsets: ``(BLOCK_SIZE,)`` offsets.
        rank: This rank's index.
        world_size: Number of participating ranks.
        root: Rank performing the scatter.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements per tile.

    TODO: Binary-tree scatter for large world_size.
    TODO: Double-buffer overlap with computation on root.
    """
    if rank == root:
        # Root distributes each chunk to the corresponding rank.
        for target in tl.static_range(0, 8):
            if target >= world_size:
                pass
            else:
                src_chunk_offset = target * BLOCK_SIZE
                chunk = tl.load(src_ptr + src_chunk_offset + offsets)

                if target == root:
                    # Local copy -- write directly to our own dst_ptr.
                    tl.store(dst_ptr + offsets, chunk)
                else:
                    # Remote write -- translate dst_ptr for the target rank.
                    remote_dst = translate_ptr(
                        dst_ptr, root, target, heap_bases
                    )
                    tl.store(remote_dst + offsets, chunk)


# =====================================================================
# tile_reduce_scatter -- Reduce across ranks, scatter result
# =====================================================================


@triton.jit
def tile_reduce_scatter(
    src_ptr,
    dst_ptr,
    offsets,
    rank,
    world_size,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
    op: tl.constexpr = "sum",
):
    """Reduce-scatter: reduce across ranks, each rank gets 1/N of the result.

    Rank *i*'s ``dst_ptr`` receives the fully-reduced chunk *i*.
    Ring-based algorithm identical to Phase 1 of :func:`tile_allreduce`.

    Use when each rank only needs 1/N of the reduced result (ZeRO-style
    optimiser sharding, expert-parallel MoE).

    Args:
        src_ptr: Input buffer (``world_size * BLOCK_SIZE`` elements).
            Used as scratch and modified in-place.
        dst_ptr: Output buffer (``BLOCK_SIZE`` elements per rank).
        offsets: ``(BLOCK_SIZE,)`` offsets.
        rank: This rank's index.
        world_size: Number of participating ranks.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements per chunk.
        op: Reduction operator: ``"sum"`` | ``"max"`` | ``"min"``.

    TODO: Non-power-of-2 world_size optimisation.
    TODO: Topology-aware ring ordering.
    TODO: In-place variant writing result to src_ptr slice.
    """
    next_rank = _ring_next(rank, world_size)
    prev_rank = _ring_prev(rank, world_size)

    # Phase: Reduce-scatter (same as allreduce Phase 1)
    for step in tl.static_range(0, 8):
        if step >= world_size - 1:
            pass
        else:
            # Chunk to send.
            send_chunk_idx = (rank - step + world_size) % world_size
            chunk_offset = send_chunk_idx * BLOCK_SIZE

            local_data = tl.load(src_ptr + chunk_offset + offsets)

            # Write to next rank's buffer.
            remote_next = translate_ptr(src_ptr, rank, next_rank, heap_bases)
            tl.store(remote_next + chunk_offset + offsets, local_data)
            tl.debug_barrier()

            # Chunk to receive and reduce.
            recv_chunk_idx = (rank - step - 1 + world_size) % world_size
            recv_offset = recv_chunk_idx * BLOCK_SIZE

            # Load from previous rank.
            remote_prev = translate_ptr(src_ptr, rank, prev_rank, heap_bases)
            incoming = tl.load(remote_prev + recv_offset + offsets)

            my_data = tl.load(src_ptr + recv_offset + offsets)
            reduced = _reduce_op(my_data, incoming, op)

            tl.store(src_ptr + recv_offset + offsets, reduced)
            tl.debug_barrier()

    # After reduce-scatter, chunk (rank + 1) % world_size is fully
    # reduced on this rank.  Copy it to dst_ptr.
    result_chunk_idx = (rank + 1) % world_size
    result_offset = result_chunk_idx * BLOCK_SIZE
    result_data = tl.load(src_ptr + result_offset + offsets)
    tl.store(dst_ptr + offsets, result_data)


# =====================================================================
# tile_broadcast -- Root sends tile to all peers
# =====================================================================


@triton.jit
def tile_broadcast(
    ptr,
    offsets,
    rank,
    world_size,
    root,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
):
    """Broadcast: root sends its tile to all other ranks.

    After the call, every rank's ``ptr`` holds root's original data.
    Flat broadcast with ``O(world_size)`` writes from root.

    Use when a single rank produces a result all ranks need
    (e.g. normalisation stats, speculative-decoding accepted token).

    Args:
        ptr: Data buffer (``BLOCK_SIZE`` elements).  Source on root,
            destination on non-root ranks.
        offsets: ``(BLOCK_SIZE,)`` offsets.
        rank: This rank's index.
        world_size: Number of participating ranks.
        root: Rank owning the source data.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements in the tile.

    TODO: Binary-tree broadcast for O(log N) latency.
    TODO: Bidirectional broadcast to halve root injection bandwidth.
    """
    if rank == root:
        # Load source tile from root's buffer.
        tile_data = tl.load(ptr + offsets)

        # Write to every non-root rank.
        for peer in tl.static_range(0, 8):
            if peer >= world_size:
                pass
            else:
                if peer != root:
                    remote_ptr = translate_ptr(
                        ptr, root, peer, heap_bases
                    )
                    tl.store(remote_ptr + offsets, tile_data)


# =====================================================================
# Host-side launcher kernels (testing / prototyping only)
# =====================================================================


@triton.jit
def _allreduce_kernel(
    data_ptr,
    heap_bases_ptr,
    rank,
    world_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel wrapper for :func:`tile_allreduce`."""
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_allreduce(
        data_ptr,
        offsets,
        rank,
        world_size,
        heap_bases_ptr,
        BLOCK_SIZE,
        op="sum",
    )


@triton.jit
def _allgather_kernel(
    src_ptr,
    dst_ptr,
    heap_bases_ptr,
    rank,
    world_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel wrapper for :func:`tile_allgather`."""
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_allgather(
        src_ptr,
        dst_ptr,
        offsets,
        rank,
        world_size,
        heap_bases_ptr,
        BLOCK_SIZE,
    )


@triton.jit
def _broadcast_kernel(
    data_ptr,
    heap_bases_ptr,
    rank,
    world_size,
    root,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel wrapper for :func:`tile_broadcast`."""
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_broadcast(
        data_ptr,
        offsets,
        rank,
        world_size,
        root,
        heap_bases_ptr,
        BLOCK_SIZE,
    )


# -----------------------------------------------------------------------
# Python host-side launchers
# -----------------------------------------------------------------------


def allreduce(
    tensor: torch.Tensor,
    heap: "xtile.memory.SymmetricHeap",
    op: str = "sum",
) -> None:
    """Host-side launcher for :func:`tile_allreduce` (testing / benchmarking).

    In-place ring allreduce on *tensor*.  The tensor size must be
    divisible by ``world_size``.  *tensor* must reside in *heap*'s
    symmetric memory.
    """
    world_size = heap.world_size
    numel = tensor.numel()
    if numel % world_size != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by "
            f"world_size ({world_size}) for ring allreduce."
        )

    block_size = numel // world_size
    heap_bases = heap.get_heap_bases()

    # Launch with a single program (the collective is cooperative).
    _allreduce_kernel[(1,)](
        tensor,
        heap_bases,
        heap.rank,
        world_size,
        BLOCK_SIZE=block_size,
    )


def allgather(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "xtile.memory.SymmetricHeap",
) -> None:
    """Host-side launcher for :func:`tile_allgather` (testing / benchmarking).

    Each rank contributes *tensor*; all ranks get the concatenated result
    in *output* (``world_size * tensor.numel()`` elements).  Both must
    reside in *heap*'s symmetric memory.
    """
    world_size = heap.world_size
    block_size = tensor.numel()
    expected_output = world_size * block_size

    if output.numel() != expected_output:
        raise ValueError(
            f"Output size ({output.numel()}) must equal "
            f"world_size * BLOCK_SIZE ({expected_output})."
        )

    heap_bases = heap.get_heap_bases()

    _allgather_kernel[(1,)](
        tensor,
        output,
        heap_bases,
        heap.rank,
        world_size,
        BLOCK_SIZE=block_size,
    )


def broadcast(
    tensor: torch.Tensor,
    heap: "xtile.memory.SymmetricHeap",
    root: int = 0,
) -> None:
    """Host-side launcher for :func:`tile_broadcast` (testing / benchmarking).

    Root sends its tile to all other ranks.  *tensor* must reside in
    *heap*'s symmetric memory.
    """
    world_size = heap.world_size
    if root < 0 or root >= world_size:
        raise ValueError(
            f"root={root} out of range [0, {world_size})"
        )

    block_size = tensor.numel()
    heap_bases = heap.get_heap_bases()

    _broadcast_kernel[(1,)](
        tensor,
        heap_bases,
        heap.rank,
        world_size,
        root,
        BLOCK_SIZE=block_size,
    )
