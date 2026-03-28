"""
tncc.sync.primitives - Synchronization primitives.

Provides device-side atomic operations, signal/wait primitives, and barriers
for coordinating tile-level communication across GPUs.  All functions are
decorated with ``@triton.jit`` for full compiler visibility.

Two categories of primitives
-----------------------------

**Remote atomics** (``tile_atomic_*``)
    Operate on memory that may belong to a different GPU.  These require
    ``target_rank``, ``caller_rank``, and ``heap_bases`` parameters so the
    pointer can be translated via :func:`~tncc.memory.translation.translate_ptr`.
    When ``target_rank == caller_rank``, translation is an identity -- the
    pointer is used as-is -- so these functions are safe for local use too.

**Local signal primitives** (``tile_signal``, ``tile_wait``, etc.)
    Operate on memory local to the current GPU (e.g. a lock tensor shared
    between compute and comm workers within the same kernel launch).  No
    pointer translation is performed.  These are the building blocks for
    TileLink-style producer-consumer synchronization.

Memory ordering
---------------
Follows the C++ memory model (NOT OpenSHMEM quiet/wait_until):

* ``"relaxed"`` -- no ordering guarantees beyond atomicity.
* ``"acquire"`` -- subsequent reads/writes cannot be reordered before this op.
* ``"release"`` -- preceding reads/writes cannot be reordered after this op.
* ``"acq_rel"`` -- combined acquire + release semantics.

Memory scope
------------
Controls the visibility domain of the atomic:

* ``"block"`` -- visible within the same CTA / thread-block.
* ``"gpu"``   -- visible across all CTAs on the same GPU.
* ``"sys"``   -- system scope, visible across GPUs and host (multi-node).
"""

import triton
import triton.language as tl

from tncc.memory.translation import translate_ptr


# =====================================================================
# Remote atomic operations
# =====================================================================
#
# All remote atomics follow the same pattern:
#   1. Translate ptr from target_rank's address space to caller_rank's.
#   2. Apply offset.
#   3. Issue the hardware atomic with the specified sem/scope.
#
# When target_rank == caller_rank, translate_ptr returns the original
# pointer unchanged, so these functions degrade to efficient local
# atomics with zero overhead (the two tl.load + sub + add cancel out).


@triton.jit
def tile_atomic_add(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "relaxed",
    scope: tl.constexpr = "gpu",
):
    """Atomic add to (possibly remote) memory.

    Atomically performs ``*addr += value`` for each lane.

    Memory ordering:
        - ``"relaxed"``: No ordering; use when the atomic is only for
          accumulation and no synchronization is implied.
        - ``"release"``: Use when this add publishes data that another
          thread will acquire.
        - ``"acq_rel"``: Use when the add both publishes and consumes.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: Value(s) to add atomically.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the addition.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_add(addr, value, sem=sem, scope=scope)


@triton.jit
def tile_atomic_cas(
    ptr,
    expected,
    desired,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "acq_rel",
    scope: tl.constexpr = "gpu",
):
    """Atomic compare-and-swap on (possibly remote) memory.

    For each lane, atomically: if ``*addr == expected`` then set
    ``*addr = desired``.  Returns the value that was in memory before the
    operation (which equals *expected* on success).

    Memory ordering:
        - ``"acq_rel"`` (default): The standard choice for CAS-based
          synchronization -- acquires on read, releases on successful write.
        - ``"acquire"``: Use in a spin-loop where you only need to observe
          the latest value.
        - ``"relaxed"``: Rarely appropriate for CAS.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        expected: Expected current value.
        desired: Value to write if *expected* matches.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address (== *expected* if the swap succeeded).
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_cas(addr, expected, desired, sem=sem, scope=scope)


@triton.jit
def tile_atomic_xchg(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "gpu",
):
    """Atomic exchange on (possibly remote) memory.

    Atomically sets ``*addr = value`` and returns the old value.

    Memory ordering:
        - ``"release"`` (default): Ensures all prior writes are visible
          before the exchange takes effect.  The canonical choice when
          using xchg to publish data (e.g. signalling).
        - ``"acq_rel"``: Needed when the returned old value is used to
          make control-flow decisions that guard subsequent reads.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: New value to write.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the exchange.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_xchg(addr, value, sem=sem, scope=scope)


@triton.jit
def tile_atomic_min(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "relaxed",
    scope: tl.constexpr = "gpu",
):
    """Atomic min on (possibly remote) memory.

    Atomically performs ``*addr = min(*addr, value)`` for each lane.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: Value(s) to compare.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the min operation.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_min(addr, value, sem=sem, scope=scope)


@triton.jit
def tile_atomic_max(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "relaxed",
    scope: tl.constexpr = "gpu",
):
    """Atomic max on (possibly remote) memory.

    Atomically performs ``*addr = max(*addr, value)`` for each lane.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: Value(s) to compare.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the max operation.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_max(addr, value, sem=sem, scope=scope)


@triton.jit
def tile_atomic_and(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "relaxed",
    scope: tl.constexpr = "gpu",
):
    """Atomic bitwise AND on (possibly remote) memory.

    Atomically performs ``*addr &= value`` for each lane.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: Bitmask value(s) to AND.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the AND operation.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_and(addr, value, sem=sem, scope=scope)


@triton.jit
def tile_atomic_or(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "relaxed",
    scope: tl.constexpr = "gpu",
):
    """Atomic bitwise OR on (possibly remote) memory.

    Atomically performs ``*addr |= value`` for each lane.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: Bitmask value(s) to OR.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the OR operation.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_or(addr, value, sem=sem, scope=scope)


@triton.jit
def tile_atomic_xor(
    ptr,
    value,
    target_rank,
    caller_rank,
    heap_bases,
    offsets,
    sem: tl.constexpr = "relaxed",
    scope: tl.constexpr = "gpu",
):
    """Atomic bitwise XOR on (possibly remote) memory.

    Atomically performs ``*addr ^= value`` for each lane.

    Args:
        ptr: Base pointer (in *target_rank*'s address space).
        value: Bitmask value(s) to XOR.
        target_rank: Rank that owns the target memory.
        caller_rank: Rank executing this operation (the calling GPU).
        heap_bases: Int64 tensor of per-rank symmetric-heap base pointers.
        offsets: Offset tensor for addressing.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old value at each address before the XOR operation.
    """
    addr = translate_ptr(ptr, target_rank, caller_rank, heap_bases) + offsets
    return tl.atomic_xor(addr, value, sem=sem, scope=scope)


# =====================================================================
# Local signal primitives (TileLink-style producer-consumer)
# =====================================================================
#
# These primitives operate on LOCAL memory only (e.g. a lock tensor
# allocated on the current GPU).  They do NOT perform pointer translation.
# Use them for intra-kernel synchronization between compute and comm
# workers (WG specialization, producer-consumer patterns).
#
# For cross-GPU signalling, translate the locks pointer first via
# translate_ptr, then use tile_signal/tile_wait on the translated pointer.


@triton.jit
def tile_signal(
    locks,
    tile_id,
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "gpu",
):
    """Signal that a tile is ready for consumption.

    Atomically sets ``locks[tile_id] = 1`` with release semantics so that
    all preceding writes (the tile data) are visible to any thread that
    subsequently acquires this lock via :func:`tile_wait`.

    This implements the **producer** side of TileLink-style signalling.
    This is a LOCAL operation -- the *locks* pointer must already be in
    the caller's address space.  For cross-GPU signals, translate the
    pointer first.

    Memory ordering:
        - ``"release"`` (default): Ensures the tile data written before
          this call is visible to the consumer that does an acquire-load
          on the same lock.  Do not weaken to ``"relaxed"`` unless you
          have an external fence.

    Args:
        locks: Pointer to the lock array (must be in caller's address space).
        tile_id: Index of the tile to signal (used as offset into *locks*).
        sem: Memory ordering semantic.
        scope: Memory visibility scope.
    """
    tl.atomic_xchg(locks + tile_id, 1, sem=sem, scope=scope)


@triton.jit
def tile_wait(
    locks,
    tile_id,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "gpu",
):
    """Wait for a tile to become ready, then consume the signal.

    Spins on ``locks[tile_id]`` using atomic CAS until the value is ``1``,
    then atomically resets it to ``0``.  The acquire semantics ensure that
    all writes made by the producer (before its release-signal) are visible
    after this function returns.

    This implements the **consumer** side of TileLink-style signalling.
    This is a LOCAL operation -- the *locks* pointer must already be in
    the caller's address space.

    Memory ordering:
        - ``"acquire"`` (default): Ensures that reads of the tile data
          after this call see the values written by the producer before
          its release-signal.  Do not weaken to ``"relaxed"``.

    Args:
        locks: Pointer to the lock array (must be in caller's address space).
        tile_id: Index of the tile to wait on.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.
    """
    while tl.atomic_cas(locks + tile_id, 1, 0, sem=sem, scope=scope) != 1:
        pass


@triton.jit
def tile_signal_add(
    signals,
    tile_id,
    value=1,
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "gpu",
):
    """Increment a signal counter to indicate progress.

    Atomically adds *value* to ``signals[tile_id]`` with release semantics.
    Multiple producers can each increment the counter; a consumer uses
    :func:`tile_wait_ge` to wait until the counter reaches a threshold.

    This is the counting generalisation of :func:`tile_signal` -- use it
    when a tile depends on contributions from multiple producers (e.g.
    reduce-scatter where each rank contributes one partial sum).

    This is a LOCAL operation.

    Args:
        signals: Pointer to the signal-counter array (caller's address space).
        tile_id: Index of the signal counter to increment.
        value: Amount to add (default ``1``).
        sem: Memory ordering semantic.
        scope: Memory visibility scope.

    Returns:
        The old counter value before the addition.
    """
    return tl.atomic_add(signals + tile_id, value, sem=sem, scope=scope)


@triton.jit
def tile_wait_ge(
    signals,
    tile_id,
    threshold,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "gpu",
):
    """Wait until a signal counter reaches or exceeds a threshold.

    Spins on ``signals[tile_id]`` using a zero-add polling loop.  Once the
    counter ``>= threshold``, the function returns.  The counter is **not**
    reset -- the caller is responsible for lifecycle management.

    Use this with :func:`tile_signal_add` for multi-producer patterns.
    This is a LOCAL operation.

    Args:
        signals: Pointer to the signal-counter array (caller's address space).
        tile_id: Index of the signal counter to poll.
        threshold: Minimum counter value required to proceed.
        sem: Memory ordering semantic.
        scope: Memory visibility scope.
    """
    while tl.atomic_add(signals + tile_id, 0, sem=sem, scope=scope) < threshold:
        pass


# =====================================================================
# Barrier
# =====================================================================


@triton.jit
def tile_barrier(
    barriers,
    rank,
    world_size,
    sem: tl.constexpr = "acq_rel",
    scope: tl.constexpr = "gpu",
):
    """Device-side barrier across *world_size* participants.

    Each participant atomically increments ``barriers[0]`` and then spins
    until the counter reaches *world_size*.  This is a **single-use**
    barrier -- for repeated barriers, use separate slots or reset between
    rounds.

    Implementation:
        Arrival uses ``atomic_add(..., 1)`` with release semantics.
        Polling uses ``atomic_add(..., 0)`` with acquire semantics
        (a read that does not modify the counter).

    Args:
        barriers: Pointer to the barrier counter (must be initialised to 0).
        rank: Caller's rank (reserved for future asymmetric-barrier extensions).
        world_size: Total number of participants.
        sem: Memory ordering semantic for both arrival and polling.
        scope: Memory visibility scope.
    """
    # Arrive: increment barrier counter (release prior writes)
    tl.atomic_add(barriers, 1, sem="release", scope=scope)

    # Wait: spin until all participants have arrived (acquire their writes)
    while tl.atomic_add(barriers, 0, sem="acquire", scope=scope) < world_size:
        pass
