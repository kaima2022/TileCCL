"""
xtile.primitives - Core primitive layer.

Exports all public primitives from the four sub-modules:

- **compute**: Tile-level arithmetic (dot, reduce, elementwise, cast, fill).
- **memory**: Tile-level memory access (load, store, copy, offset generation).
- **communication**: Cross-GPU tile transfer (remote load/store, put/get).
- **collectives**: Multi-GPU collective operations (allreduce, allgather,
  broadcast, scatter, reduce_scatter).
"""

from xtile.primitives.compute import (
    tile_cast,
    tile_dot,
    tile_elementwise,
    tile_full,
    tile_reduce,
    tile_reduce_max,
    tile_reduce_min,
    tile_zeros,
)
from xtile.primitives.memory import (
    make_block_offsets,
    tile_copy,
    tile_load,
    tile_store,
)
from xtile.primitives.communication import (
    tile_get,
    tile_put,
    tile_remote_load,
    tile_remote_store,
)
from xtile.primitives.collectives import (
    tile_allreduce,
    tile_allgather,
    tile_scatter,
    tile_reduce_scatter,
    tile_broadcast,
    allreduce,
    allgather,
    broadcast,
    scatter,
    reduce_scatter,
)

__all__ = [
    # compute
    "tile_dot",
    "tile_reduce",
    "tile_reduce_max",
    "tile_reduce_min",
    "tile_elementwise",
    "tile_cast",
    "tile_zeros",
    "tile_full",
    # memory
    "tile_load",
    "tile_store",
    "tile_copy",
    "make_block_offsets",
    # communication
    "tile_remote_load",
    "tile_remote_store",
    "tile_put",
    "tile_get",
    # collectives (device-side)
    "tile_allreduce",
    "tile_allgather",
    "tile_scatter",
    "tile_reduce_scatter",
    "tile_broadcast",
    # collectives (host-side launchers)
    "allreduce",
    "allgather",
    "broadcast",
    "scatter",
    "reduce_scatter",
]
