# SPDX-License-Identifier: Apache-2.0
"""TileGroup planning utilities.

This module is the first shared planning layer extracted from the fused AG/RS
proofs. It intentionally stays backend-neutral: it builds row-aligned
TileGroups and the Triton swizzle group table, but it does not own any kernel
logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_GROUP_M = 8
DEFAULT_TAU_MIN_BYTES = 262144

TileGroup = tuple[int, int, int]


@dataclass(frozen=True)
class TileGroupPlan:
    """TileGroup metadata consumed by fused proof kernels."""

    M: int
    N: int
    tile_bytes: int
    target_group_tiles: int | None
    max_groups: int
    groups: list[TileGroup]
    group_table: torch.Tensor
    block_m: int = DEFAULT_BLOCK_M
    block_n: int = DEFAULT_BLOCK_N
    group_m: int = DEFAULT_GROUP_M
    tau_min_bytes: int = DEFAULT_TAU_MIN_BYTES
    p0_min_tiles: int = 1

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def max_rb(self) -> int:
        return max(
            math.ceil((row_end - row_start) / self.block_m)
            for row_start, row_end, _ in self.groups
        )

    @property
    def max_cb(self) -> int:
        return math.ceil(self.N / self.block_n)

    @property
    def max_group_elems(self) -> int:
        return max(
            (row_end - row_start) * self.N for row_start, row_end, _ in self.groups
        )

    @property
    def min_group_tiles(self) -> int:
        return min(tiles for _, _, tiles in self.groups)

    @property
    def max_group_tiles(self) -> int:
        return max(tiles for _, _, tiles in self.groups)

    @property
    def avg_group_tiles(self) -> float:
        return sum(tiles for _, _, tiles in self.groups) / len(self.groups)

    def row_starts(self) -> list[int]:
        return [row_start for row_start, _, _ in self.groups]

    def row_ends(self) -> list[int]:
        return [min(row_end, self.M) for _, row_end, _ in self.groups]

    def tile_counts(self) -> list[int]:
        return [tiles for _, _, tiles in self.groups]


def build_tile_groups(
    M: int,
    N: int,
    tile_bytes: int,
    *,
    target_group_tiles: int | None = None,
    max_groups: int = 64,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    group_m: int = DEFAULT_GROUP_M,
    tau_min_bytes: int = DEFAULT_TAU_MIN_BYTES,
) -> list[TileGroup]:
    """Build wave-aligned TileGroups.

    Returns ``(row_start, row_end, tile_count)`` tuples. P0 enforces the
    minimum P2P saturation size, while P1 aligns group boundaries to the
    persistent GEMM swizzle wave.
    """
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive")
    if tile_bytes <= 0:
        raise ValueError("tile_bytes must be positive")
    if block_m <= 0 or block_n <= 0 or group_m <= 0:
        raise ValueError("block_m, block_n, and group_m must be positive")
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")

    n_tile_rows = math.ceil(M / block_m)
    n_tile_cols = math.ceil(N / block_n)
    p0_min_tiles = max(1, math.ceil(tau_min_bytes / tile_bytes))
    g_min_tiles = max(p0_min_tiles, target_group_tiles or p0_min_tiles)

    waves = list(range(0, n_tile_rows + 1, group_m))
    if waves[-1] != n_tile_rows:
        waves.append(n_tile_rows)

    groups: list[TileGroup] = []
    start_wave = 0
    buffered_tiles = 0
    for idx in range(len(waves) - 1):
        wave_tile_rows = waves[idx + 1] - waves[idx]
        buffered_tiles += wave_tile_rows * n_tile_cols
        if buffered_tiles >= g_min_tiles:
            groups.append(
                (start_wave * block_m, waves[idx + 1] * block_m, buffered_tiles)
            )
            start_wave = waves[idx + 1]
            buffered_tiles = 0

    if buffered_tiles > 0:
        if groups:
            first, _, tiles = groups[-1]
            groups[-1] = (first, M, tiles + buffered_tiles)
        else:
            groups.append((0, M, buffered_tiles))

    if len(groups) > max_groups:
        merge_factor = math.ceil(len(groups) / max_groups)
        merged: list[TileGroup] = []
        for idx in range(0, len(groups), merge_factor):
            chunk = groups[idx : idx + merge_factor]
            merged.append(
                (chunk[0][0], min(chunk[-1][1], M), sum(group[2] for group in chunk))
            )
        groups = merged

    return groups


def split_groups_at_row_boundaries(
    M: int,
    N: int,
    groups: Sequence[TileGroup],
    row_boundaries: Sequence[int],
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> list[TileGroup]:
    """Split groups so no group crosses a row ownership boundary."""
    n_tile_cols = math.ceil(N / block_n)
    boundaries = sorted(
        {0, M, *(boundary for boundary in row_boundaries if 0 < boundary < M)}
    )
    out: list[TileGroup] = []

    for row_start, raw_row_end, _ in groups:
        row_end = min(raw_row_end, M)
        if row_start >= row_end:
            continue
        cursor = row_start
        while cursor < row_end:
            next_boundary = next(boundary for boundary in boundaries if boundary > cursor)
            split_end = min(row_end, next_boundary)
            tile_rows = math.ceil((split_end - cursor) / block_m)
            out.append((cursor, split_end, tile_rows * n_tile_cols))
            cursor = split_end

    return out


def split_groups_at_row_shards(
    M: int,
    N: int,
    groups: Sequence[TileGroup],
    *,
    num_shards: int,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> list[TileGroup]:
    """Split TileGroups at equal row-shard boundaries."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if M % num_shards != 0:
        raise ValueError(f"M={M} must be divisible by num_shards={num_shards}")

    shard_m = M // num_shards
    boundaries = [shard_m * shard for shard in range(1, num_shards)]
    return split_groups_at_row_boundaries(
        M,
        N,
        groups,
        boundaries,
        block_m=block_m,
        block_n=block_n,
    )


def build_group_table(
    M: int,
    N: int,
    groups: Sequence[TileGroup],
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    group_m: int = DEFAULT_GROUP_M,
) -> torch.Tensor:
    """Build a Triton tile-id to TileGroup-id table."""
    n_tile_rows = math.ceil(M / block_m)
    n_tile_cols = math.ceil(N / block_n)
    total_tiles = n_tile_rows * n_tile_cols
    table = torch.zeros(total_tiles, dtype=torch.int32)
    tiles_per_swizzle_group = group_m * n_tile_cols

    for tile_id in range(total_tiles):
        swizzle_group = tile_id // tiles_per_swizzle_group
        first_tile_m = swizzle_group * group_m
        group_size_m = min(n_tile_rows - first_tile_m, group_m)
        tile_m = first_tile_m + ((tile_id % tiles_per_swizzle_group) % group_size_m)
        tile_row = tile_m * block_m
        for gid, (row_start, row_end, _) in enumerate(groups):
            if row_start <= tile_row < row_end:
                table[tile_id] = gid
                break

    return table


def build_tile_group_plan(
    M: int,
    N: int,
    tile_bytes: int,
    *,
    target_group_tiles: int | None = None,
    max_groups: int = 64,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    group_m: int = DEFAULT_GROUP_M,
    tau_min_bytes: int = DEFAULT_TAU_MIN_BYTES,
    row_split_boundaries: Sequence[int] = (),
) -> TileGroupPlan:
    """Build TileGroup metadata and the swizzle group table."""
    groups = build_tile_groups(
        M,
        N,
        tile_bytes,
        target_group_tiles=target_group_tiles,
        max_groups=max_groups,
        block_m=block_m,
        block_n=block_n,
        group_m=group_m,
        tau_min_bytes=tau_min_bytes,
    )
    if row_split_boundaries:
        groups = split_groups_at_row_boundaries(
            M,
            N,
            groups,
            row_split_boundaries,
            block_m=block_m,
            block_n=block_n,
        )
    group_table = build_group_table(
        M,
        N,
        groups,
        block_m=block_m,
        block_n=block_n,
        group_m=group_m,
    )
    p0_min_tiles = max(1, math.ceil(tau_min_bytes / tile_bytes))

    return TileGroupPlan(
        M=M,
        N=N,
        tile_bytes=tile_bytes,
        target_group_tiles=target_group_tiles,
        max_groups=max_groups,
        groups=groups,
        group_table=group_table,
        block_m=block_m,
        block_n=block_n,
        group_m=group_m,
        tau_min_bytes=tau_min_bytes,
        p0_min_tiles=p0_min_tiles,
    )
