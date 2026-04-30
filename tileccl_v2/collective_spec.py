# SPDX-License-Identifier: Apache-2.0
"""Minimal collective semantics for TileCCL proof planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from tileccl_v2.tile_group import TileGroup


class CollectiveKind(Enum):
    """Collective kinds covered by the current TileCCL proof runtime."""

    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass(frozen=True)
class CollectiveSpec:
    """Small semantic contract shared by AG/RS proof integrations.

    This deliberately does not resurrect the old PlanCompiler API. It only
    captures the semantics needed by the current fused proofs: whether rows
    must be split by owner and how to validate group ownership.
    """

    kind: CollectiveKind
    world_size: int = 2
    row_axis: int = 0

    @property
    def requires_row_owner_split(self) -> bool:
        return self.kind is CollectiveKind.REDUCE_SCATTER

    def shard_rows(self, M: int) -> int:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if M % self.world_size != 0:
            raise ValueError(f"M={M} must be divisible by world_size={self.world_size}")
        return M // self.world_size

    def row_split_boundaries(self, M: int) -> list[int]:
        """Return row boundaries that TileGroups must not cross."""
        if not self.requires_row_owner_split:
            return []
        shard_m = self.shard_rows(M)
        return [shard_m * rank for rank in range(1, self.world_size)]

    def owner_for_row(self, row: int, M: int) -> int:
        """Return the row-shard owner rank for ReduceScatter."""
        if not self.requires_row_owner_split:
            raise ValueError(f"{self.kind.value} does not have row owners")
        shard_m = self.shard_rows(M)
        if row < 0 or row >= M:
            raise ValueError(f"row={row} outside [0, {M})")
        return row // shard_m

    def validate_tile_groups(self, M: int, groups: Sequence[TileGroup]) -> None:
        """Validate groups against this collective's ownership constraints."""
        if not self.requires_row_owner_split:
            return
        for row_start, row_end, _ in groups:
            if row_start >= row_end:
                raise ValueError(f"invalid TileGroup rows {row_start}-{row_end}")
            start_owner = self.owner_for_row(row_start, M)
            end_owner = self.owner_for_row(row_end - 1, M)
            if start_owner != end_owner:
                raise ValueError(
                    f"TileGroup rows {row_start}-{row_end} cross row owner boundary"
                )


def all_gather_spec(*, world_size: int = 2) -> CollectiveSpec:
    return CollectiveSpec(CollectiveKind.ALL_GATHER, world_size=world_size)


def reduce_scatter_spec(*, world_size: int = 2) -> CollectiveSpec:
    return CollectiveSpec(CollectiveKind.REDUCE_SCATTER, world_size=world_size)
