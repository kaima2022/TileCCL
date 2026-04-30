# SPDX-License-Identifier: Apache-2.0
"""Workgroup specialization planning utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WGPlan:
    """Compute/comm WG allocation and per-group task geometry."""

    total_sms: int
    num_comm_wgs: int
    max_group_elems: int
    copy_elems: int
    reduce_elems: int | None = None
    packed_values_per_word: int = 4

    @property
    def num_comp_wgs(self) -> int:
        return self.total_sms - self.num_comm_wgs

    @property
    def total_wgs(self) -> int:
        return self.num_comp_wgs + self.num_comm_wgs

    @property
    def grid(self) -> tuple[int]:
        return (self.total_wgs,)

    @property
    def copy_qwords(self) -> int:
        return self.copy_elems // self.packed_values_per_word

    @property
    def copy_tasks_per_group(self) -> int:
        return math.ceil(self.max_group_elems / self.copy_elems)

    @property
    def reduce_tasks_per_group(self) -> int:
        if self.reduce_elems is None:
            return 0
        return math.ceil(self.max_group_elems / self.reduce_elems)

    @property
    def tasks_per_group(self) -> int:
        if self.reduce_elems is None:
            return self.copy_tasks_per_group
        return max(self.copy_tasks_per_group, self.reduce_tasks_per_group)


def build_wg_plan(
    *,
    total_sms: int,
    num_comm_wgs: int,
    max_group_elems: int,
    copy_elems: int,
    reduce_elems: int | None = None,
    packed_values_per_word: int = 4,
) -> WGPlan:
    """Validate and derive WG specialization parameters."""
    if total_sms <= 0:
        raise ValueError("total_sms must be positive")
    if num_comm_wgs < 0:
        raise ValueError("num_comm_wgs must be non-negative")
    if total_sms - num_comm_wgs <= 0:
        raise ValueError(f"num_comm_wgs={num_comm_wgs} leaves no compute WGs")
    if max_group_elems <= 0:
        raise ValueError("max_group_elems must be positive")
    if copy_elems <= 0:
        raise ValueError("copy_elems must be positive")
    if packed_values_per_word <= 0:
        raise ValueError("packed_values_per_word must be positive")
    if copy_elems % packed_values_per_word != 0:
        raise ValueError("copy_elems must be divisible by packed_values_per_word")
    if max_group_elems % packed_values_per_word != 0:
        raise ValueError("max_group_elems must be divisible by packed_values_per_word")
    if reduce_elems is not None and reduce_elems <= 0:
        raise ValueError("reduce_elems must be positive")

    return WGPlan(
        total_sms=total_sms,
        num_comm_wgs=num_comm_wgs,
        max_group_elems=max_group_elems,
        copy_elems=copy_elems,
        reduce_elems=reduce_elems,
        packed_values_per_word=packed_values_per_word,
    )
