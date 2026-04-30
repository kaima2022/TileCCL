# SPDX-License-Identifier: Apache-2.0
"""P2P transport planning helpers for fused TileCCL proofs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tileccl_v2.wg import WGPlan, build_wg_plan


PACKED_FP16_VALUES_PER_WORD = 4


class CommMode(Enum):
    """Device-side P2P communication direction."""

    PUSH = "push"
    PULL = "pull"


@dataclass(frozen=True)
class P2PTransportPlan:
    """Transport knobs shared by AG/RS fused kernels."""

    comm_mode: CommMode
    copy_elems: int
    reduce_elems: int | None = None
    packed_values_per_word: int = PACKED_FP16_VALUES_PER_WORD

    @property
    def push_mode(self) -> bool:
        return self.comm_mode is CommMode.PUSH

    @property
    def copy_qwords(self) -> int:
        return self.copy_elems // self.packed_values_per_word

    def build_wg_plan(
        self,
        *,
        total_sms: int,
        num_comm_wgs: int,
        max_group_elems: int,
    ) -> WGPlan:
        return build_wg_plan(
            total_sms=total_sms,
            num_comm_wgs=num_comm_wgs,
            max_group_elems=max_group_elems,
            copy_elems=self.copy_elems,
            reduce_elems=self.reduce_elems,
            packed_values_per_word=self.packed_values_per_word,
        )


def normalize_comm_mode(comm_mode: str | CommMode) -> CommMode:
    if isinstance(comm_mode, CommMode):
        return comm_mode
    try:
        return CommMode(comm_mode)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in CommMode)
        raise ValueError(f"unknown comm_mode={comm_mode!r}; expected one of {allowed}") from exc


def build_p2p_transport_plan(
    *,
    comm_mode: str | CommMode,
    copy_elems: int,
    reduce_elems: int | None = None,
    packed_values_per_word: int = PACKED_FP16_VALUES_PER_WORD,
) -> P2PTransportPlan:
    """Build transport metadata used to derive kernel launch constants."""
    if copy_elems <= 0:
        raise ValueError("copy_elems must be positive")
    if reduce_elems is not None and reduce_elems <= 0:
        raise ValueError("reduce_elems must be positive")
    if packed_values_per_word <= 0:
        raise ValueError("packed_values_per_word must be positive")

    return P2PTransportPlan(
        comm_mode=normalize_comm_mode(comm_mode),
        copy_elems=copy_elems,
        reduce_elems=reduce_elems,
        packed_values_per_word=packed_values_per_word,
    )
