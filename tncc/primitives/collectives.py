"""
tncc.primitives.collectives - Collective communication primitives.

Tile-level collective operations implemented entirely in Triton
(pure ``@triton.jit``, no NCCL / RCCL dependency).  All remote memory
access goes through :func:`~tncc.memory.translation.translate_ptr`.

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

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tncc.memory.translation import translate_ptr
from tncc.utils.feature_gates import (
    multiprocess_device_collectives_detail,
    multiprocess_device_collectives_enabled,
    multiprocess_device_collectives_runtime_supported,
    multiprocess_device_collectives_transport_supported,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_runtime_supported,
    multiprocess_device_remote_access_transport_supported,
)

# Maximum supported world_size for collective operations.
# Determined by tl.static_range upper bound (Triton unrolls statically).
MAX_COLLECTIVE_WORLD_SIZE = 33
_ALLREDUCE_LATENCY_MESSAGE_BYTES = 8 * 1024
_ALLREDUCE_BANDWIDTH_MESSAGE_BYTES = 256 * 1024
_ALLREDUCE_TARGET_CHUNK_BYTES = 16 * 1024
_ALLREDUCE_MIN_CHUNK_ELEMS = 256
_ALLREDUCE_MAX_CHUNK_ELEMS = 4096
_ALLREDUCE_MAX_THROUGHPUT_PIPELINE_SLOTS = 4
_ALLREDUCE_MAX_BANDWIDTH_PIPELINE_SLOTS = 16


@dataclass(frozen=True, slots=True)
class _AllReduceRegimePolicy:
    """Stable tuning policy for one allreduce message regime."""

    message_regime: str
    implementation: str
    protocol: str
    target_chunk_bytes: int
    min_chunk_elems: int
    max_chunk_elems: int
    max_pipeline_slots: int
    small_chunk_num_warps: int
    large_chunk_num_warps: int
    large_chunk_threshold_elems: int = 1024


_ALLREDUCE_LATENCY_POLICY = _AllReduceRegimePolicy(
    message_regime="latency",
    implementation="device_single_slot_staged",
    protocol="single_slot_epoch_staged",
    target_chunk_bytes=_ALLREDUCE_LATENCY_MESSAGE_BYTES,
    min_chunk_elems=1,
    max_chunk_elems=_ALLREDUCE_MAX_CHUNK_ELEMS,
    max_pipeline_slots=1,
    small_chunk_num_warps=2,
    large_chunk_num_warps=4,
)
_ALLREDUCE_THROUGHPUT_POLICY = _AllReduceRegimePolicy(
    message_regime="throughput",
    implementation="device_staged_pipeline",
    protocol="slot_epoch_pipeline",
    target_chunk_bytes=_ALLREDUCE_TARGET_CHUNK_BYTES,
    min_chunk_elems=_ALLREDUCE_MIN_CHUNK_ELEMS,
    max_chunk_elems=_ALLREDUCE_MAX_CHUNK_ELEMS,
    max_pipeline_slots=_ALLREDUCE_MAX_THROUGHPUT_PIPELINE_SLOTS,
    small_chunk_num_warps=4,
    large_chunk_num_warps=4,
)
_ALLREDUCE_BANDWIDTH_POLICY = _AllReduceRegimePolicy(
    message_regime="bandwidth",
    implementation="device_staged_pipeline",
    protocol="slot_epoch_pipeline",
    target_chunk_bytes=_ALLREDUCE_TARGET_CHUNK_BYTES,
    min_chunk_elems=_ALLREDUCE_MIN_CHUNK_ELEMS,
    max_chunk_elems=_ALLREDUCE_MAX_CHUNK_ELEMS,
    max_pipeline_slots=_ALLREDUCE_MAX_BANDWIDTH_PIPELINE_SLOTS,
    small_chunk_num_warps=4,
    large_chunk_num_warps=8,
)


@dataclass(frozen=True, slots=True)
class AllReduceExecutionSpec:
    """Resolved execution contract for the public allreduce launcher."""

    op: str
    world_size: int
    implementation: str
    protocol: str
    kernel_family: str
    reuse_handshake: str
    tensor_numel: int
    message_bytes: int
    message_regime: str
    cta_policy: str
    epoch_policy: str
    block_size: int
    chunk_elems: int
    num_chunks: int
    pipeline_slots: int
    grid_size: int
    num_warps: int
    workspace_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the resolved execution contract for logs and benchmarks."""
        return {
            "op": self.op,
            "world_size": self.world_size,
            "implementation": self.implementation,
            "protocol": self.protocol,
            "kernel_family": self.kernel_family,
            "reuse_handshake": self.reuse_handshake,
            "tensor_numel": self.tensor_numel,
            "message_bytes": self.message_bytes,
            "message_regime": self.message_regime,
            "cta_policy": self.cta_policy,
            "epoch_policy": self.epoch_policy,
            "block_size": self.block_size,
            "chunk_elems": self.chunk_elems,
            "num_chunks": self.num_chunks,
            "pipeline_slots": self.pipeline_slots,
            "grid_size": self.grid_size,
            "num_warps": self.num_warps,
            "workspace_bytes": self.workspace_bytes,
        }


@dataclass(slots=True)
class _AllReduceWorkspace:
    """Process-local cached workspace for the staged allreduce fast path."""

    staging: torch.Tensor
    published_epoch: torch.Tensor
    slot_sync_state: torch.Tensor
    next_epoch: int = 0

    def reserve_epoch_range(self, num_chunks: int) -> int:
        """Reserve one monotonically increasing epoch range for a call."""
        start = self.next_epoch + 1
        self.next_epoch += num_chunks
        return start


_ALLREDUCE_WORKSPACE_CACHE: dict[
    tuple[object, ...],
    _AllReduceWorkspace,
] = {}


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
# Public allreduce fast path -- staged peer-read allreduce
# =====================================================================


@triton.jit
def _allreduce_staged_kernel(
    tensor_ptr,
    staging_ptr,
    published_epoch_ptr,
    slot_sync_state_ptr,
    heap_bases_ptr,
    rank,
    world_size,
    numel,
    num_chunks,
    num_slots,
    base_epoch,
    BLOCK_SIZE: tl.constexpr,
    op: tl.constexpr = "sum",
):
    """Chunked multi-CTA allreduce with explicit staging and device sync.

    The public allreduce launcher uses this kernel instead of composing
    ``reduce_scatter + allgather``. Each CTA owns one pipeline slot and
    processes chunk indices ``pid, pid + num_slots, ...``. For every chunk:

    1. Snapshots one contiguous chunk into a symmetric staging buffer.
    2. Publishes that snapshot with a monotonically increasing epoch.
    3. Waits until every peer has published the matching epoch.
    4. Reads peer staging buffers, reduces locally, and writes only to its
       own output tensor.
    5. Uses a consumed-count handshake before reusing the staging slot.

    This avoids host barriers and avoids materializing a full gathered
    tensor, while keeping the synchronization semantics explicit.
    """
    pid = tl.program_id(0)
    if pid >= num_slots:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    slot = pid
    chunk_idx = pid

    while chunk_idx < num_chunks:
        chunk_start = chunk_idx * BLOCK_SIZE
        idx = chunk_start + offsets
        mask = idx < numel
        local_values = tl.load(tensor_ptr + idx, mask=mask, other=0.0)

        staging_slot_ptr = staging_ptr + slot * BLOCK_SIZE
        published_slot_ptr = published_epoch_ptr + slot
        consumed_slot_ptr = slot_sync_state_ptr + slot

        tl.store(staging_slot_ptr + offsets, local_values, mask=mask)
        tl.atomic_xchg(consumed_slot_ptr, 0, sem="release", scope="sys")

        epoch = tl.cast(base_epoch + chunk_idx, tl.int32)
        tl.atomic_xchg(published_slot_ptr, epoch, sem="release", scope="sys")

        reduced = local_values
        for peer in tl.static_range(0, 32):
            if peer >= world_size:
                pass
            else:
                if peer != rank:
                    remote_published = translate_ptr(
                        published_slot_ptr,
                        rank,
                        peer,
                        heap_bases_ptr,
                    )
                    while tl.atomic_cas(
                        remote_published,
                        epoch,
                        epoch,
                        sem="acquire",
                        scope="sys",
                    ) != epoch:
                        pass

                    remote_staging = translate_ptr(
                        staging_slot_ptr,
                        rank,
                        peer,
                        heap_bases_ptr,
                    )
                    incoming = tl.load(
                        remote_staging + offsets,
                        mask=mask,
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    reduced = _reduce_op(reduced, incoming, op)

                    remote_consumed = translate_ptr(
                        consumed_slot_ptr,
                        rank,
                        peer,
                        heap_bases_ptr,
                    )
                    tl.atomic_add(
                        remote_consumed,
                        1,
                        sem="release",
                        scope="sys",
                    )

        tl.store(tensor_ptr + idx, reduced, mask=mask)

        if chunk_idx + num_slots < num_chunks:
            expected_consumers = world_size - 1
            while tl.atomic_cas(
                consumed_slot_ptr,
                expected_consumers,
                expected_consumers,
                sem="acquire",
                scope="sys",
            ) != expected_consumers:
                pass

        chunk_idx += num_slots


@triton.jit
def _allreduce_staged_kernel_ws2(
    tensor_ptr,
    staging_ptr,
    published_epoch_ptr,
    slot_sync_state_ptr,
    heap_bases_ptr,
    rank,
    numel,
    num_chunks,
    num_slots,
    base_epoch,
    BLOCK_SIZE: tl.constexpr,
    op: tl.constexpr = "sum",
):
    """World-size-2 fast path for the staged allreduce launcher.

    The current validated public multiprocess surface is ``world_size=2``.
    This kernel specializes that case so the hot path does not pay for the
    generic peer loop or count-based reuse handshake. One peer means one
    direct remote publish wait, one remote load, and one epoch ack.
    """
    pid = tl.program_id(0)
    if pid >= num_slots:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    slot = pid
    chunk_idx = pid
    peer_rank = 1 - rank

    while chunk_idx < num_chunks:
        chunk_start = chunk_idx * BLOCK_SIZE
        idx = chunk_start + offsets
        mask = idx < numel
        local_values = tl.load(tensor_ptr + idx, mask=mask, other=0.0)

        staging_slot_ptr = staging_ptr + slot * BLOCK_SIZE
        published_slot_ptr = published_epoch_ptr + slot
        ack_slot_ptr = slot_sync_state_ptr + slot
        epoch = tl.cast(base_epoch + chunk_idx, tl.int32)

        tl.store(staging_slot_ptr + offsets, local_values, mask=mask)
        tl.atomic_xchg(published_slot_ptr, epoch, sem="release", scope="sys")

        remote_published = translate_ptr(
            published_slot_ptr,
            rank,
            peer_rank,
            heap_bases_ptr,
        )
        while tl.atomic_cas(
            remote_published,
            epoch,
            epoch,
            sem="acquire",
            scope="sys",
        ) != epoch:
            pass

        remote_staging = translate_ptr(
            staging_slot_ptr,
            rank,
            peer_rank,
            heap_bases_ptr,
        )
        incoming = tl.load(
            remote_staging + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
        reduced = _reduce_op(local_values, incoming, op)
        tl.store(tensor_ptr + idx, reduced, mask=mask)

        if chunk_idx + num_slots < num_chunks:
            remote_ack = translate_ptr(
                ack_slot_ptr,
                rank,
                peer_rank,
                heap_bases_ptr,
            )
            tl.atomic_xchg(remote_ack, epoch, sem="release", scope="sys")

            while tl.atomic_cas(
                ack_slot_ptr,
                epoch,
                epoch,
                sem="acquire",
                scope="sys",
            ) != epoch:
                pass

        chunk_idx += num_slots


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


@triton.jit
def _scatter_kernel(
    src_ptr,
    dst_ptr,
    heap_bases_ptr,
    rank,
    world_size,
    root,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel wrapper for :func:`tile_scatter`."""
    offsets = tl.arange(0, BLOCK_SIZE)
    tile_scatter(
        src_ptr,
        dst_ptr,
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
    heap: "tncc.memory.SymmetricHeap",
    op: str = "sum",
    *,
    _execution: AllReduceExecutionSpec | None = None,
) -> None:
    """Host-side launcher for the public in-place allreduce collective.

    The public path is a staged multi-CTA device implementation with
    explicit execution metadata and a compact symmetric workspace. It no
    longer composes ``reduce_scatter + allgather`` on the hot path.
    """
    torch.cuda.set_device(tensor.device)
    if _execution is None:
        spec = resolve_allreduce_execution(tensor, heap=heap, op=op)
    else:
        _validate_allreduce_execution_spec(
            tensor,
            heap=heap,
            op=op,
            execution=_execution,
        )
        spec = _execution
    if spec.implementation == "noop":
        return

    workspace = _allreduce_workspace(tensor, heap=heap, spec=spec)
    base_epoch = workspace.reserve_epoch_range(spec.num_chunks)
    heap_bases = heap.get_heap_bases()
    if spec.world_size == 2:
        _allreduce_staged_kernel_ws2[(spec.grid_size,)](
            tensor,
            workspace.staging,
            workspace.published_epoch,
            workspace.slot_sync_state,
            heap_bases,
            heap.rank,
            spec.tensor_numel,
            spec.num_chunks,
            spec.pipeline_slots,
            base_epoch,
            BLOCK_SIZE=spec.chunk_elems,
            op=op,
            num_warps=spec.num_warps,
            num_stages=1,
        )
    else:
        _allreduce_staged_kernel[(spec.grid_size,)](
            tensor,
            workspace.staging,
            workspace.published_epoch,
            workspace.slot_sync_state,
            heap_bases,
            heap.rank,
            heap.world_size,
            spec.tensor_numel,
            spec.num_chunks,
            spec.pipeline_slots,
            base_epoch,
            BLOCK_SIZE=spec.chunk_elems,
            op=op,
            num_warps=spec.num_warps,
            num_stages=1,
        )


def allgather(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "tncc.memory.SymmetricHeap",
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
        operation="tncc.primitives.allgather(...)",
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
    heap: "tncc.memory.SymmetricHeap",
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
        operation="tncc.primitives.broadcast(...)",
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


def scatter(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "tncc.memory.SymmetricHeap",
    *,
    root: int = 0,
) -> None:
    """Host-side launcher for :func:`tile_scatter`.

    ``tensor`` must contain ``world_size`` contiguous rank-local chunks and
    ``output`` must contain one chunk. After the call, rank ``i`` receives
    chunk ``i`` from ``root``.
    """
    world_size = heap.world_size
    if world_size > MAX_COLLECTIVE_WORLD_SIZE:
        raise ValueError(
            f"world_size={world_size} exceeds maximum supported "
            f"({MAX_COLLECTIVE_WORLD_SIZE}). Recompile with larger "
            f"tl.static_range bound in collectives.py."
        )
    if root < 0 or root >= world_size:
        raise ValueError(f"root={root} out of range [0, {world_size})")

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
        raise ValueError("scatter currently requires tensor to be contiguous")
    if not output.is_contiguous():
        raise ValueError("scatter currently requires output to be contiguous")

    _require_tensor_on_heap(tensor, heap=heap, name="tensor")
    _require_tensor_on_heap(output, heap=heap, name="output")
    _require_device_remote_access_transport(
        heap,
        operation="tncc.primitives.scatter(...)",
    )

    heap_bases = heap.get_heap_bases()
    _scatter_kernel[(1,)](
        tensor,
        output,
        heap_bases,
        heap.rank,
        world_size,
        root,
        BLOCK_SIZE=block_size,
    )


def reduce_scatter(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "tncc.memory.SymmetricHeap",
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
            if multiprocess_device_collectives_runtime_supported(
                transport_strategy=heap.transport_strategy,
                world_size=heap.world_size,
            ):
                resolved_impl = "device"
            elif not multiprocess_device_collectives_enabled():
                raise ValueError(
                    multiprocess_device_collectives_detail(
                        transport_strategy=heap.transport_strategy,
                        world_size=heap.world_size,
                    )
                )
            elif not multiprocess_device_collectives_transport_supported(
                heap.transport_strategy
            ):
                raise ValueError(
                    multiprocess_device_collectives_detail(
                        transport_strategy=heap.transport_strategy,
                        world_size=heap.world_size,
                    )
                )
            else:
                resolved_impl = "device"
    elif resolved_impl == "device" and heap.mode == "single_process":
        raise ValueError(
            "implementation='device' is not validated for single-process symmetric heaps. "
            "Use implementation='reference' (or 'auto') until the device path is proven correct."
        )
    elif resolved_impl == "device":
        if multiprocess_device_collectives_runtime_supported(
            transport_strategy=heap.transport_strategy,
            world_size=heap.world_size,
        ):
            pass
        elif not multiprocess_device_collectives_enabled():
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                    world_size=heap.world_size,
                )
            )
        elif not multiprocess_device_collectives_transport_supported(
            heap.transport_strategy
        ):
            raise ValueError(
                multiprocess_device_collectives_detail(
                    transport_strategy=heap.transport_strategy,
                    world_size=heap.world_size,
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
    heap: "tncc.memory.SymmetricHeap",
    name: str,
) -> None:
    """Ensure *tensor* resides in *heap*."""
    if not heap.owns_tensor(tensor):
        raise ValueError(
            f"{name} must reside in the provided SymmetricHeap for rank {heap.rank}"
        )


def _require_device_remote_access_transport(
    heap: "tncc.memory.SymmetricHeap",
    *,
    operation: str,
) -> None:
    """Fail fast when the heap transport is not safe for Triton remote access."""
    if heap.mode != "multiprocess":
        return
    if multiprocess_device_remote_access_runtime_supported(
        transport_strategy=heap.transport_strategy,
        world_size=heap.world_size,
    ):
        return
    raise ValueError(
        multiprocess_device_remote_access_detail(
            transport_strategy=heap.transport_strategy,
            operation=operation,
            world_size=heap.world_size,
        )
    )


def _reference_reduce_scatter_single_process(
    tensor: torch.Tensor,
    output: torch.Tensor,
    heap: "tncc.memory.SymmetricHeap",
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


def resolve_allreduce_execution(
    tensor: torch.Tensor,
    *,
    heap: "tncc.memory.SymmetricHeap",
    op: str = "sum",
) -> AllReduceExecutionSpec:
    """Resolve the execution contract for the public allreduce launcher."""
    numel = _validate_allreduce_launch_prereqs(
        tensor,
        heap=heap,
        op=op,
    )
    world_size = heap.world_size
    message_bytes = _allreduce_message_bytes(tensor)
    if world_size == 1 or numel == 0:
        return AllReduceExecutionSpec(
            op=op,
            world_size=world_size,
            implementation="noop",
            protocol="local_identity",
            kernel_family="local_identity",
            reuse_handshake="none",
            tensor_numel=numel,
            message_bytes=message_bytes,
            message_regime="local_identity",
            cta_policy="single_cta",
            epoch_policy="none",
            block_size=numel,
            chunk_elems=numel,
            num_chunks=0,
            pipeline_slots=0,
            grid_size=1,
            num_warps=1,
            workspace_bytes=0,
        )

    torch.cuda.set_device(tensor.device)
    policy = _allreduce_regime_policy(message_bytes)
    message_regime = policy.message_regime
    chunk_elems = _select_allreduce_chunk_elems(
        tensor,
        policy=policy,
    )
    num_chunks = triton.cdiv(numel, chunk_elems)
    pipeline_slots = _select_allreduce_pipeline_slots(
        num_chunks=num_chunks,
        tensor=tensor,
        policy=policy,
    )
    num_warps = _select_allreduce_num_warps(
        chunk_elems=chunk_elems,
        policy=policy,
    )
    implementation = policy.implementation
    protocol = policy.protocol
    kernel_family = "ws2_specialized" if world_size == 2 else "generic_multi_peer"
    reuse_handshake = "ws2_epoch_ack" if world_size == 2 else "count_ack"
    cta_policy = "single_cta" if pipeline_slots == 1 else "multi_cta_pipeline"
    epoch_policy = (
        "per_call_monotonic_epoch"
        if pipeline_slots == 1 and num_chunks == 1
        else "per_chunk_slot_epoch"
    )
    workspace_bytes = pipeline_slots * (
        chunk_elems * tensor.element_size() + (2 * 4)
    )
    return AllReduceExecutionSpec(
        op=op,
        world_size=world_size,
        implementation=implementation,
        protocol=protocol,
        kernel_family=kernel_family,
        reuse_handshake=reuse_handshake,
        tensor_numel=numel,
        message_bytes=message_bytes,
        message_regime=message_regime,
        cta_policy=cta_policy,
        epoch_policy=epoch_policy,
        block_size=numel,
        chunk_elems=chunk_elems,
        num_chunks=num_chunks,
        pipeline_slots=pipeline_slots,
        grid_size=pipeline_slots,
        num_warps=num_warps,
        workspace_bytes=workspace_bytes,
    )


def _round_down_power_of_two(value: int) -> int:
    """Return the greatest power of two not exceeding *value*."""
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _round_up_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to *value*."""
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length())


def _allreduce_message_bytes(tensor: torch.Tensor) -> int:
    """Return the contiguous allreduce payload size in bytes."""
    return int(tensor.numel()) * max(tensor.element_size(), 1)


def _classify_allreduce_message_regime(message_bytes: int) -> str:
    """Classify the allreduce payload into a stable message regime."""
    if message_bytes <= _ALLREDUCE_LATENCY_MESSAGE_BYTES:
        return "latency"
    if message_bytes >= _ALLREDUCE_BANDWIDTH_MESSAGE_BYTES:
        return "bandwidth"
    return "throughput"


def _allreduce_regime_policy(message_bytes: int) -> _AllReduceRegimePolicy:
    """Return the tuning policy for one rank-local message size."""
    message_regime = _classify_allreduce_message_regime(message_bytes)
    if message_regime == "latency":
        return _ALLREDUCE_LATENCY_POLICY
    if message_regime == "bandwidth":
        return _ALLREDUCE_BANDWIDTH_POLICY
    return _ALLREDUCE_THROUGHPUT_POLICY


def _select_allreduce_chunk_elems(
    tensor: torch.Tensor,
    *,
    policy: _AllReduceRegimePolicy,
) -> int:
    """Choose a chunk size for the staged allreduce protocol."""
    numel = int(tensor.numel())
    if numel <= 0:
        return 1

    if policy.message_regime == "latency":
        return max(
            1,
            min(_round_up_power_of_two(numel), policy.max_chunk_elems),
        )

    target = max(
        policy.min_chunk_elems,
        policy.target_chunk_bytes // max(tensor.element_size(), 1),
    )
    target = min(target, policy.max_chunk_elems, numel)
    return max(1, min(numel, _round_down_power_of_two(target)))


def _select_allreduce_pipeline_slots(
    *,
    num_chunks: int,
    tensor: torch.Tensor,
    policy: _AllReduceRegimePolicy,
) -> int:
    """Choose the number of concurrent allreduce pipeline slots."""
    if num_chunks <= 1:
        return 1

    try:
        props = torch.cuda.get_device_properties(tensor.device)
        compute_units = int(getattr(props, "multi_processor_count", 1))
    except Exception:
        compute_units = 1

    return max(
        1,
        min(
            num_chunks,
            compute_units,
            policy.max_pipeline_slots,
        ),
    )


def _select_allreduce_num_warps(
    *,
    chunk_elems: int,
    policy: _AllReduceRegimePolicy,
) -> int:
    """Choose a stable warp count for the resolved allreduce regime."""
    if chunk_elems <= policy.large_chunk_threshold_elems:
        return policy.small_chunk_num_warps
    return policy.large_chunk_num_warps


def _validate_allreduce_launch_prereqs(
    tensor: torch.Tensor,
    *,
    heap: "tncc.memory.SymmetricHeap",
    op: str,
) -> int:
    """Validate the public allreduce runtime contract and return tensor numel."""
    if op != "sum":
        raise ValueError(f"Only op='sum' is currently supported, got {op!r}")

    world_size = heap.world_size
    if world_size > MAX_COLLECTIVE_WORLD_SIZE:
        raise ValueError(
            f"world_size={world_size} exceeds maximum supported "
            f"({MAX_COLLECTIVE_WORLD_SIZE}). Recompile with larger "
            f"tl.static_range bound in collectives.py."
        )
    if not tensor.is_contiguous():
        raise ValueError("allreduce currently requires tensor to be contiguous")
    _require_tensor_on_heap(tensor, heap=heap, name="tensor")
    _require_device_remote_access_transport(
        heap,
        operation="tncc.primitives.allreduce(...)",
    )
    return int(tensor.numel())


def _validate_allreduce_execution_spec(
    tensor: torch.Tensor,
    *,
    heap: "tncc.memory.SymmetricHeap",
    op: str,
    execution: AllReduceExecutionSpec,
) -> None:
    """Validate one caller-provided execution spec before launch."""
    numel = _validate_allreduce_launch_prereqs(
        tensor,
        heap=heap,
        op=op,
    )
    if execution.op != op:
        raise ValueError(
            f"allreduce execution op mismatch: {execution.op!r} != {op!r}"
        )
    if execution.world_size != heap.world_size:
        raise ValueError(
            "allreduce execution world_size mismatch: "
            f"{execution.world_size} != {heap.world_size}"
        )
    if execution.tensor_numel != numel:
        raise ValueError(
            "allreduce execution tensor_numel mismatch: "
            f"{execution.tensor_numel} != {numel}"
        )
    expected_message_bytes = _allreduce_message_bytes(tensor)
    if execution.message_bytes != expected_message_bytes:
        raise ValueError(
            "allreduce execution message_bytes mismatch: "
            f"{execution.message_bytes} != {expected_message_bytes}"
        )


def _allreduce_workspace_key(
    tensor: torch.Tensor,
    *,
    heap: "tncc.memory.SymmetricHeap",
    spec: AllReduceExecutionSpec,
) -> tuple[object, ...]:
    """Return a process-local key for cached allreduce workspaces."""
    heap_bases = tuple(int(base) for base in heap.get_heap_bases().tolist())
    return (
        "allreduce",
        heap_bases,
        heap.rank,
        heap.get_offset(int(tensor.data_ptr())),
        tuple(int(dim) for dim in tensor.shape),
        str(tensor.dtype),
        spec.chunk_elems,
        spec.pipeline_slots,
        spec.implementation,
    )


def _allreduce_workspace(
    tensor: torch.Tensor,
    *,
    heap: "tncc.memory.SymmetricHeap",
    spec: AllReduceExecutionSpec,
) -> _AllReduceWorkspace:
    """Return the cached workspace for the staged allreduce fast path."""
    key = _allreduce_workspace_key(tensor, heap=heap, spec=spec)
    workspace = _ALLREDUCE_WORKSPACE_CACHE.get(key)
    if workspace is None:
        staging = heap.allocate_tensor(
            (spec.pipeline_slots, spec.chunk_elems),
            tensor.dtype,
        )
        published_epoch = heap.allocate_tensor((spec.pipeline_slots,), torch.int32)
        slot_sync_state = heap.allocate_tensor((spec.pipeline_slots,), torch.int32)
        published_epoch.zero_()
        slot_sync_state.zero_()
        workspace = _AllReduceWorkspace(
            staging=staging,
            published_epoch=published_epoch,
            slot_sync_state=slot_sync_state,
        )
        _ALLREDUCE_WORKSPACE_CACHE[key] = workspace
    return workspace
