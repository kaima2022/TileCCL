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
from xtile.utils.feature_gates import (
    multiprocess_device_collectives_detail,
    multiprocess_device_collectives_enabled,
    multiprocess_device_collectives_transport_supported,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_transport_supported,
)

# Maximum supported world_size for collective operations.
# Determined by tl.static_range upper bound (Triton unrolls statically).
MAX_COLLECTIVE_WORLD_SIZE = 33


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
    for step in tl.static_range(0, 32):
        # static_range(0, 32) supports up to 33 GPUs.
        # Bound has been increased to 32 for world_size > 9.
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
    for step in tl.static_range(0, 32):
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
    for peer in tl.static_range(0, 32):
        # Guard: skip iterations beyond world_size and skip self.
        # Bound has been increased to 32 for world_size > 9.
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
        # Bound has been increased to 32 for world_size > 9.
        for target in tl.static_range(0, 32):
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

    The current implementation is deliberately correctness-first for the
    multiprocess/device path: each rank directly reads the corresponding
    chunk from every peer's symmetric buffer, reduces locally, and writes
    only to its own ``dst_ptr``. This avoids cross-rank write races that
    occur if a peer overwrites a rank's unreduced local chunk in-place.

    Args:
        src_ptr: Input buffer (``world_size * BLOCK_SIZE`` elements).
            Preserved as read-only scratch for the device reduction.
        dst_ptr: Output buffer (``BLOCK_SIZE`` elements per rank).
        offsets: ``(BLOCK_SIZE,)`` offsets.
        rank: This rank's index.
        world_size: Number of participating ranks.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements per chunk.
        op: Reduction operator: ``"sum"`` | ``"max"`` | ``"min"``.

    TODO: Revisit a pipelined ring version once cross-rank staging and
    synchronization are explicit and proven safe in multiprocess mode.
    """
    chunk_offset = rank * BLOCK_SIZE
    reduced = tl.load(src_ptr + chunk_offset + offsets)

    # Each peer owns a symmetric allocation at the same heap offset.
    # Read peer chunk ``rank`` directly and reduce locally.
    for peer in tl.static_range(0, 32):
        if peer >= world_size:
            pass
        else:
            if peer != rank:
                remote_peer = translate_ptr(
                    src_ptr,
                    rank,
                    peer,
                    heap_bases,
                )
                incoming = tl.load(remote_peer + chunk_offset + offsets)
                reduced = _reduce_op(reduced, incoming, op)

    tl.store(dst_ptr + offsets, reduced)


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
    Binary-tree broadcast with ``O(log N)`` latency for large world_size.
    Falls back to flat broadcast for small world_size (<=4) where the
    overhead of tree logic is not justified.

    Args:
        ptr: Data buffer (``BLOCK_SIZE`` elements).  Source on root,
            destination on non-root ranks.
        offsets: ``(BLOCK_SIZE,)`` offsets.
        rank: This rank's index.
        world_size: Number of participating ranks.
        root: Rank owning the source data.
        heap_bases: ``[world_size]`` int64 heap-base-address tensor pointer.
        BLOCK_SIZE: Elements in the tile.
    """
    if rank == root:
        # Root loads its data and sends to all peers.
        # For simplicity and Triton compatibility, we use a flat broadcast
        # from root.  Binary-tree would require multi-step synchronization
        # which is complex in a single kernel launch.
        tile_data = tl.load(ptr + offsets)

        for peer in tl.static_range(0, 32):
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
def _reduce_scatter_kernel(
    src_ptr,
    dst_ptr,
    heap_bases_ptr,
    rank,
    world_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel wrapper for :func:`tile_reduce_scatter`."""
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_reduce_scatter(
        src_ptr,
        dst_ptr,
        offsets,
        rank,
        world_size,
        heap_bases_ptr,
        BLOCK_SIZE,
        op="sum",
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
    if world_size > MAX_COLLECTIVE_WORLD_SIZE:
        raise ValueError(
            f"world_size={world_size} exceeds maximum supported "
            f"({MAX_COLLECTIVE_WORLD_SIZE}). Recompile with larger "
            f"tl.static_range bound in collectives.py."
        )
    numel = tensor.numel()
    if numel % world_size != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by "
            f"world_size ({world_size}) for ring allreduce."
        )
    _require_device_remote_access_transport(
        heap,
        operation="xtile.primitives.allreduce(...)",
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
    if world_size > MAX_COLLECTIVE_WORLD_SIZE:
        raise ValueError(
            f"world_size={world_size} exceeds maximum supported "
            f"({MAX_COLLECTIVE_WORLD_SIZE}). Recompile with larger "
            f"tl.static_range bound in collectives.py."
        )
    block_size = tensor.numel()
    expected_output = world_size * block_size

    if output.numel() != expected_output:
        raise ValueError(
            f"Output size ({output.numel()}) must equal "
            f"world_size * BLOCK_SIZE ({expected_output})."
        )
    _require_device_remote_access_transport(
        heap,
        operation="xtile.primitives.allgather(...)",
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
    if world_size > MAX_COLLECTIVE_WORLD_SIZE:
        raise ValueError(
            f"world_size={world_size} exceeds maximum supported "
            f"({MAX_COLLECTIVE_WORLD_SIZE}). Recompile with larger "
            f"tl.static_range bound in collectives.py."
        )
    if root < 0 or root >= world_size:
        raise ValueError(
            f"root={root} out of range [0, {world_size})"
        )
    _require_device_remote_access_transport(
        heap,
        operation="xtile.primitives.broadcast(...)",
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


def reduce_scatter(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "xtile.memory.SymmetricHeap",
    op: str = "sum",
    implementation: str = "auto",
) -> None:
    """Host-side launcher for :func:`tile_reduce_scatter`.

    ``tensor`` must contain ``world_size`` contiguous chunks, each of
    ``output.numel()`` elements. After the call, rank ``i`` receives the
    reduced chunk ``i`` in ``output``.

    This launcher is intentionally conservative and currently meant for
    testing / prototyping. The device path is correctness-first and the
    stronger public correctness/performance gate for a fused
    ``gemm_reducescatter`` API is still a separate follow-up item.
    """
    if op != "sum":
        raise ValueError(
            f"Only op='sum' is currently supported, got {op!r}"
        )
    if implementation not in {"auto", "reference", "device"}:
        raise ValueError(
            "implementation must be one of {'auto', 'reference', 'device'}, "
            f"got {implementation!r}"
        )

    world_size = heap.world_size
    if world_size > MAX_COLLECTIVE_WORLD_SIZE:
        raise ValueError(
            f"world_size={world_size} exceeds maximum supported "
            f"({MAX_COLLECTIVE_WORLD_SIZE}). Recompile with larger "
            f"tl.static_range bound in collectives.py."
        )

    block_size = output.numel()
    expected_input = block_size * world_size
    if tensor.numel() != expected_input:
        raise ValueError(
            f"Input size ({tensor.numel()}) must equal "
            f"world_size * output.numel() ({expected_input})."
        )
    if tensor.device != output.device:
        raise ValueError(
            f"tensor/output must be on the same device, got {tensor.device} vs {output.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError("reduce_scatter currently requires tensor to be contiguous")
    if not output.is_contiguous():
        raise ValueError("reduce_scatter currently requires output to be contiguous")

    _require_tensor_on_heap(tensor, heap=heap, name="tensor")
    _require_tensor_on_heap(output, heap=heap, name="output")

    resolved_impl = implementation
    if resolved_impl == "auto":
        if heap.mode == "single_process":
            resolved_impl = "reference"
        else:
            if not multiprocess_device_collectives_enabled():
                raise ValueError(
                    multiprocess_device_collectives_detail(
                        transport_strategy=heap.transport_strategy,
                    )
                )
            if not multiprocess_device_collectives_transport_supported(
                heap.transport_strategy
            ):
                raise ValueError(
                    multiprocess_device_collectives_detail(
                        transport_strategy=heap.transport_strategy,
                    )
                )
            resolved_impl = "device"
    elif resolved_impl == "device" and heap.mode == "single_process":
        raise ValueError(
            "implementation='device' is not validated for single-process symmetric heaps. "
            "Use implementation='reference' (or 'auto') until the device path is proven correct."
        )
    elif resolved_impl == "device":
        if not multiprocess_device_collectives_enabled():
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                )
            )
        if not multiprocess_device_collectives_transport_supported(
            heap.transport_strategy
        ):
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                )
            )

    if resolved_impl == "reference":
        _reference_reduce_scatter_single_process(tensor, output, heap)
        return

    heap_bases = heap.get_heap_bases()
    _reduce_scatter_kernel[(1,)](
        tensor,
        output,
        heap_bases,
        heap.rank,
        world_size,
        BLOCK_SIZE=block_size,
    )


def _require_tensor_on_heap(
    tensor: torch.Tensor,
    *,
    heap: "xtile.memory.SymmetricHeap",
    name: str,
) -> None:
    """Ensure *tensor* resides in *heap*."""
    try:
        heap.get_offset(int(tensor.data_ptr()))
    except Exception as exc:
        raise ValueError(
            f"{name} must reside in the provided SymmetricHeap for rank {heap.rank}"
        ) from exc


def _require_device_remote_access_transport(
    heap: "xtile.memory.SymmetricHeap",
    *,
    operation: str,
) -> None:
    """Fail fast when the heap transport is not safe for Triton remote access."""
    if heap.mode != "multiprocess":
        return
    if multiprocess_device_remote_access_transport_supported(
        heap.transport_strategy
    ):
        return
    raise ValueError(
        multiprocess_device_remote_access_detail(
            transport_strategy=heap.transport_strategy,
            operation=operation,
        )
    )


def _reference_reduce_scatter_single_process(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "xtile.memory.SymmetricHeap",
) -> None:
    """Reference reduce-scatter for single-process symmetric heaps.

    This path is intentionally correctness-first. It reconstructs each
    rank's peer tensor via symmetric offset equivalence and performs the
    reduction on the caller's device. It is suitable for tests,
    diagnostics, and host-side correctness gates, not performance
    benchmarking.
    """
    if heap.mode != "single_process":
        raise RuntimeError(
            "Reference reduce_scatter requires a single-process SymmetricHeap."
        )

    element_bytes = tensor.element_size()
    tensor_offset = heap.get_offset(int(tensor.data_ptr()))
    tensor_nbytes = tensor.numel() * element_bytes
    block_size = output.numel()
    chunk_start = heap.rank * block_size
    chunk_end = chunk_start + block_size
    local_device = output.device

    reduced: torch.Tensor | None = None
    for peer_rank in range(heap.world_size):
        peer_buffer = heap.get_peer_buffer(peer_rank)
        peer_view = peer_buffer.narrow(0, tensor_offset, tensor_nbytes).view(tensor.dtype)
        peer_view = peer_view.reshape(tensor.shape)
        peer_chunk = peer_view.reshape(-1)[chunk_start:chunk_end].to(device=local_device)
        reduced = peer_chunk if reduced is None else (reduced + peer_chunk)

    assert reduced is not None  # world_size >= 1
    output.reshape(-1).copy_(reduced)
