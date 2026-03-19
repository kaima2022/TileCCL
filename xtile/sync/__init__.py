"""
xtile.sync - Synchronization layer.

Exports all synchronization primitives:

- **Atomics**: add, cas, xchg, min, max, and/or/xor with memory ordering.
- **Signals**: TileLink-style producer-consumer signalling (signal/wait).
- **Barriers**: Device-side barrier across multiple ranks.
"""

from xtile.sync.primitives import (
    tile_atomic_add,
    tile_atomic_and,
    tile_atomic_cas,
    tile_atomic_max,
    tile_atomic_min,
    tile_atomic_or,
    tile_atomic_xchg,
    tile_atomic_xor,
    tile_barrier,
    tile_signal,
    tile_signal_add,
    tile_wait,
    tile_wait_ge,
)

__all__ = [
    # atomic operations
    "tile_atomic_add",
    "tile_atomic_cas",
    "tile_atomic_xchg",
    "tile_atomic_min",
    "tile_atomic_max",
    "tile_atomic_and",
    "tile_atomic_or",
    "tile_atomic_xor",
    # signal primitives
    "tile_signal",
    "tile_wait",
    "tile_signal_add",
    "tile_wait_ge",
    # barrier
    "tile_barrier",
]
