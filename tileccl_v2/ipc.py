# SPDX-License-Identifier: Apache-2.0
"""
tileccl_v2.ipc — Device-side IPC primitives.

Provides Triton JIT functions for cross-GPU communication:
- translate_ptr: Iris-style 5-instruction pointer translation
- tile_signal / tile_wait: Producer-consumer synchronization with timeout
- remote_load / remote_store: Convenience wrappers

Reference: TileCCL-spike translate_ptr + Iris paper Listing 1.
"""

from __future__ import annotations

import triton
import triton.language as tl


# ======================================================================
# Pointer translation (Iris-style, 5 core instructions)
# ======================================================================

@triton.jit
def translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT: tl.constexpr = 0):
    """Translate a local pointer to a remote pointer via IPC.

    Computes: offset = ptr - heap_bases[from_rank]
              remote_ptr = heap_bases[to_rank] + offset

    Exactly 5 core instructions: 2 loads, 1 subtract, 1 add, 1 cast.

    SAFETY: Caller must ensure ptr is within [heap_bases[from_rank],
    heap_bases[from_rank] + heap_size). If ptr < from_base, unsigned
    subtraction wraps around producing a huge offset → segfault.

    Parameters
    ----------
    ptr : pointer
        Device pointer within from_rank's heap.
    from_rank : int
        Rank that owns ptr.
    to_rank : int
        Target rank to translate into.
    heap_bases : pointer to int64
        [world_size] tensor of heap base addresses.
    HINT : tl.constexpr
        Vectorization hint (BLOCK_SIZE for contiguous accesses).
    """
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)

    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base

    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_byte = to_base_byte + offset
    translated_ptr = tl.cast(translated_byte, ptr.dtype)

    if HINT > 0:
        translated_ptr = tl.max_contiguous(
            tl.multiple_of(translated_ptr, HINT), HINT
        )
    return translated_ptr


# ======================================================================
# Remote load / store
# ======================================================================

@triton.jit
def remote_load(
    ptr, from_rank, to_rank, heap_bases,
    mask=None, other=0.0,
    HINT: tl.constexpr = 0,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Load from a remote rank's heap via pointer translation."""
    remote_ptr = translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT=HINT)
    if CACHE_MODIFIER == ".cg":
        return tl.load(remote_ptr, mask=mask, other=other, cache_modifier=".cg")
    else:
        return tl.load(remote_ptr, mask=mask, other=other)


@triton.jit
def remote_store(
    ptr, value, src_rank, dst_rank, heap_bases,
    mask=None,
    HINT: tl.constexpr = 0,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Store to a remote rank's heap via pointer translation."""
    remote_ptr = translate_ptr(ptr, src_rank, dst_rank, heap_bases, HINT=HINT)
    if CACHE_MODIFIER == ".wt":
        tl.store(remote_ptr, value, mask=mask, cache_modifier=".wt")
    else:
        tl.store(remote_ptr, value, mask=mask)


@triton.jit
def remote_load_block(
    ptr, from_rank, to_rank, heap_bases,
    BLOCK_SIZE: tl.constexpr,
    mask=None, other=0.0,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Load a contiguous block from a remote rank's heap."""
    remote_base = translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT=BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    if CACHE_MODIFIER == ".cg":
        return tl.load(remote_base + offsets, mask=mask, other=other, cache_modifier=".cg")
    else:
        return tl.load(remote_base + offsets, mask=mask, other=other)


@triton.jit
def remote_store_block(
    ptr, value, src_rank, dst_rank, heap_bases,
    BLOCK_SIZE: tl.constexpr,
    mask=None,
    CACHE_MODIFIER: tl.constexpr = "",
):
    """Store a contiguous block to a remote rank's heap."""
    remote_base = translate_ptr(ptr, src_rank, dst_rank, heap_bases, HINT=BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    if CACHE_MODIFIER == ".wt":
        tl.store(remote_base + offsets, value, mask=mask, cache_modifier=".wt")
    else:
        tl.store(remote_base + offsets, value, mask=mask)


# ======================================================================
# Signal / Wait primitives
# ======================================================================

@triton.jit
def tile_signal(
    locks, tile_id,
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "sys",
):
    """Signal that a tile is ready (producer side).

    Atomically sets locks[tile_id] = 1 with release semantics,
    ensuring all prior writes (tile data) are visible to the consumer.

    Uses scope="sys" by default for cross-GPU visibility (PCIe).
    """
    tl.atomic_xchg(locks + tile_id, 1, sem=sem, scope=scope)


@triton.jit
def tile_wait(
    locks, tile_id,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "sys",
):
    """Wait for a tile to become ready (consumer side).

    Spins on locks[tile_id] via atomic CAS until the lock is set to 1.
    Blocks indefinitely. Returns after consuming the signal (lock reset to 0).

    Uses scope="sys" for cross-GPU visibility.

    See tile_try_wait for a bounded variant with timeout.
    """
    while tl.atomic_cas(locks + tile_id, 1, 0, sem=sem, scope=scope) != 1:
        pass


@triton.jit
def tile_try_wait(
    locks, tile_id,
    MAX_SPINS: tl.constexpr = 1024,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "sys",
):
    """Bounded wait for a tile signal with timeout.

    Spins for at most MAX_SPINS CAS probes. Returns 1 on success
    (signal consumed, lock reset to 0); returns 0 on timeout.

    MAX_SPINS is compile-time unrolled via tl.static_range, so keep
    it reasonable (≤4096) to avoid slow compilation.
    """
    for _ in tl.static_range(MAX_SPINS):
        if tl.atomic_cas(locks + tile_id, 1, 0, sem=sem, scope=scope) == 1:
            return 1
    return 0


@triton.jit
def tile_poll(
    locks, tile_id,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "sys",
):
    """Poll for a tile to become ready without consuming the signal.

    Read-only check: spins on CAS(1,1) which reads without clearing.
    Does NOT reset the flag, so multiple consumers can poll the same tile.
    Blocks indefinitely.

    Use this in fused kernels where comm CTA polls tiles signaled by compute CTA.

    See tile_try_poll for a bounded variant with timeout.
    """
    while tl.atomic_cas(locks + tile_id, 1, 1, sem=sem, scope=scope) != 1:
        pass


@triton.jit
def tile_try_poll(
    locks, tile_id,
    MAX_SPINS: tl.constexpr = 1024,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "sys",
):
    """Bounded poll for a tile signal with timeout, non-consuming.

    Like tile_try_wait but uses CAS(1,1): reads without clearing the flag.
    Returns 1 on success, 0 on timeout.
    """
    for _ in tl.static_range(MAX_SPINS):
        if tl.atomic_cas(locks + tile_id, 1, 1, sem=sem, scope=scope) == 1:
            return 1
    return 0


@triton.jit
def tile_signal_add(
    signals, tile_id, value=1,
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "sys",
):
    """Increment a signal counter (multi-producer pattern).

    Used for reduce-scatter where multiple ranks contribute partial sums.
    """
    return tl.atomic_add(signals + tile_id, value, sem=sem, scope=scope)


@triton.jit
def tile_wait_ge(
    signals, tile_id, threshold,
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "sys",
):
    """Wait until signal counter >= threshold. Blocks indefinitely.

    Spins on atomic_add(0) to read counter without modifying.
    """
    while tl.atomic_add(signals + tile_id, 0, sem=sem, scope=scope) < threshold:
        pass
