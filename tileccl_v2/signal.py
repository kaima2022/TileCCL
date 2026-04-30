# SPDX-License-Identifier: Apache-2.0
"""Host-side TileGroup signal layout helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from tileccl_v2.heap import SymmetricHeap


@dataclass
class TileGroupSignalState:
    """Signal, barrier, and trace tensors for TileGroup-ready kernels."""

    counter: torch.Tensor
    barrier: torch.Tensor
    trace_comm_counter: torch.Tensor
    trace_start_clock: torch.Tensor
    trace_ready_clock: torch.Tensor
    trace_done_clock: torch.Tensor
    extra_counters: dict[str, torch.Tensor] = field(default_factory=dict)

    def extra(self, name: str) -> torch.Tensor:
        try:
            return self.extra_counters[name]
        except KeyError as exc:
            raise KeyError(f"unknown TileGroup signal counter: {name}") from exc

    def reset(self) -> None:
        self.counter.zero_()
        self.barrier.zero_()
        for tensor in self.extra_counters.values():
            tensor.zero_()
        self.trace_comm_counter.zero_()
        self.trace_start_clock.zero_()
        self.trace_ready_clock.zero_()
        self.trace_done_clock.zero_()


def allocate_tile_group_signals(
    *,
    heap: SymmetricHeap,
    n_groups: int,
    device: torch.device,
    extra_counters: Iterable[str] = (),
) -> TileGroupSignalState:
    """Allocate TileGroup signal tensors for a fused proof state."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")

    extra = {
        name: heap.allocate_tensor((n_groups,), dtype=torch.int32)
        for name in extra_counters
    }
    return TileGroupSignalState(
        counter=heap.allocate_tensor((n_groups,), dtype=torch.int32),
        barrier=heap.allocate_tensor((n_groups,), dtype=torch.int32),
        trace_comm_counter=torch.empty((n_groups,), dtype=torch.int32, device=device),
        trace_start_clock=torch.empty((1,), dtype=torch.int64, device=device),
        trace_ready_clock=torch.empty((n_groups,), dtype=torch.int64, device=device),
        trace_done_clock=torch.empty((n_groups,), dtype=torch.int64, device=device),
        extra_counters=extra,
    )
