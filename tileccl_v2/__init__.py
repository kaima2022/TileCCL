# SPDX-License-Identifier: Apache-2.0
"""TileCCL proof runtime.

The current package root is intentionally small: Gate 1/Gate 2 proof kernels
use symmetric heap IPC, Triton pointer translation, the calibrated cost model,
and timeline artifact helpers. Earlier broad library experiments are archived
under ``experiments/archive/legacy_tileccl_v2_library``.
"""

from tileccl_v2.cost_model import (
    CalibrationPoint,
    CostModelCalibrator,
    HardwareProfile,
    Interconnect,
    PipelineCost,
    TileCost,
    TileCostModel,
)
from tileccl_v2.collective_spec import (
    CollectiveKind,
    CollectiveSpec,
    all_gather_spec,
    reduce_scatter_spec,
)
from tileccl_v2.heap import SymmetricHeap
from tileccl_v2.ipc import (
    remote_load,
    remote_load_block,
    remote_store,
    remote_store_block,
    tile_poll,
    tile_signal,
    tile_signal_add,
    tile_try_poll,
    tile_try_wait,
    tile_wait,
    tile_wait_ge,
    translate_ptr,
)
from tileccl_v2.runtime.timeline import TimelineEvent, TimelineRecorder
from tileccl_v2.signal import TileGroupSignalState, allocate_tile_group_signals
from tileccl_v2.tile_group import (
    TileGroup,
    TileGroupPlan,
    build_group_table,
    build_tile_group_plan,
    build_tile_groups,
    split_groups_at_row_boundaries,
    split_groups_at_row_shards,
)
from tileccl_v2.transport import (
    CommMode,
    P2PTransportPlan,
    build_p2p_transport_plan,
    normalize_comm_mode,
)
from tileccl_v2.wg import WGPlan, build_wg_plan

__version__ = "2.3.0-framework-seed"

__all__ = [
    "CalibrationPoint",
    "CollectiveKind",
    "CollectiveSpec",
    "CommMode",
    "CostModelCalibrator",
    "HardwareProfile",
    "Interconnect",
    "P2PTransportPlan",
    "PipelineCost",
    "SymmetricHeap",
    "TileCost",
    "TileCostModel",
    "TileGroup",
    "TileGroupPlan",
    "TileGroupSignalState",
    "TimelineEvent",
    "TimelineRecorder",
    "WGPlan",
    "allocate_tile_group_signals",
    "all_gather_spec",
    "build_group_table",
    "build_p2p_transport_plan",
    "build_tile_group_plan",
    "build_tile_groups",
    "build_wg_plan",
    "normalize_comm_mode",
    "remote_load",
    "remote_load_block",
    "remote_store",
    "remote_store_block",
    "reduce_scatter_spec",
    "split_groups_at_row_boundaries",
    "split_groups_at_row_shards",
    "tile_poll",
    "tile_signal",
    "tile_signal_add",
    "tile_try_poll",
    "tile_try_wait",
    "tile_wait",
    "tile_wait_ge",
    "translate_ptr",
]
