# SPDX-License-Identifier: Apache-2.0
"""
tileccl_v2.cost_model — Tile-native α-β communication cost model.

Models tile transfer and computation costs for plan optimization.
The PlanCompiler uses this to assign priorities based on actual
hardware characteristics, and the TileSimulator uses it for
realistic timing predictions.

α-β model: latency = α + n_bytes × β
  α = startup/launch overhead (microseconds)
  β = per-byte transfer time (microseconds/byte) = 1/bandwidth

Hardware presets provided for common interconnects.
Calibration from measured data supported.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
}


class Interconnect(Enum):
    """Known GPU interconnect types with measured α-β parameters."""
    PCIE_GEN4 = auto()   # PCIe Gen4 x16 (~25 GB/s)
    PCIE_GEN5 = auto()   # PCIe Gen5 x16 (~50 GB/s)
    NVLINK_3 = auto()    # NVLink 3.0 (A100, ~300 GB/s per direction)
    NVLINK_4 = auto()    # NVLink 4.0 (H100, ~450 GB/s per direction)
    CUSTOM = auto()      # User-provided parameters


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware parameters for cost estimation.

    Two-regime comm model:
      predicted = max(min_comm_us, alpha_us + bytes × beta_us_per_byte)

    NCCL has a floor latency (~37µs on H100 PCIe) from kernel launch
    and protocol overhead that dominates small messages. The α-β term
    only matters once the message is large enough for bandwidth to bind.

    Attributes
    ----------
    alpha_us : float
        Startup latency per transfer in microseconds.
    beta_us_per_byte : float
        Per-byte transfer time in microseconds.
        Derived from bandwidth: beta = 1e6 / (bandwidth_gbps * 1e9 / 8).
    gemm_tflops_fp16 : float
        Peak FP16 GEMM throughput in TFLOPS.
    gemm_efficiency : float
        Fraction of peak GEMM throughput achievable (0.0-1.0).
        Accounts for memory bandwidth limits, occupancy, etc.
    min_comm_us : float
        NCCL floor latency in microseconds. The minimum time any
        comm operation takes regardless of message size.
        Set to 0.0 to disable (pure α-β model).
    name : str
        Human-readable name for this profile.
    """

    alpha_us: float
    beta_us_per_byte: float
    gemm_tflops_fp16: float
    gemm_efficiency: float = 0.7
    min_comm_us: float = 0.0
    name: str = "custom"

    @property
    def bandwidth_gb_s(self) -> float:
        """Effective bandwidth in GB/s."""
        if self.beta_us_per_byte <= 0:
            return float("inf")
        return 1.0 / (self.beta_us_per_byte * 1e-6) / 1e9

    @classmethod
    def from_interconnect(cls, interconnect: Interconnect) -> HardwareProfile:
        """Create a profile from a known interconnect type."""
        presets = {
            Interconnect.PCIE_GEN4: cls(
                alpha_us=5.0,
                beta_us_per_byte=1e6 / (25e9),  # 25 GB/s
                gemm_tflops_fp16=312.0,  # A100 tensor core peak
                gemm_efficiency=0.6,
                name="PCIe Gen4 (A100)",
            ),
            Interconnect.PCIE_GEN5: cls(
                alpha_us=3.0,
                beta_us_per_byte=1e6 / (50e9),  # 50 GB/s
                gemm_tflops_fp16=989.0,  # H100 tensor core peak
                gemm_efficiency=0.65,
                name="PCIe Gen5 (H100)",
            ),
            Interconnect.NVLINK_3: cls(
                alpha_us=1.5,
                beta_us_per_byte=1e6 / (300e9),  # 300 GB/s
                gemm_tflops_fp16=312.0,  # A100
                gemm_efficiency=0.7,
                name="NVLink 3.0 (A100)",
            ),
            Interconnect.NVLINK_4: cls(
                alpha_us=1.0,
                beta_us_per_byte=1e6 / (450e9),  # 450 GB/s
                gemm_tflops_fp16=989.0,  # H100
                gemm_efficiency=0.75,
                name="NVLink 4.0 (H100)",
            ),
        }
        if interconnect == Interconnect.CUSTOM:
            raise ValueError("Use HardwareProfile() directly for custom parameters")
        return presets[interconnect]


@dataclass
class TileCost:
    """Cost breakdown for a single tile operation.

    Attributes
    ----------
    comm_us : float
        Communication time in microseconds (α + size × β).
    compute_us : float
        Compute time in microseconds (2MNK / throughput).
    tile_bytes : int
        Tile size in bytes.
    overlap_possible : bool
        Whether comm can be overlapped with this tile's compute.
    """

    comm_us: float
    compute_us: float
    tile_bytes: int
    overlap_possible: bool = True

    @property
    def sequential_us(self) -> float:
        """Time if comm and compute run sequentially."""
        return self.comm_us + self.compute_us

    @property
    def overlapped_us(self) -> float:
        """Time if comm and compute fully overlap (bounded by max)."""
        return max(self.comm_us, self.compute_us)

    @property
    def compute_bound(self) -> bool:
        """True if compute takes longer than communication."""
        return self.compute_us >= self.comm_us


@dataclass
class PipelineCost:
    """Cost analysis for a full tile pipeline.

    Attributes
    ----------
    tile_costs : list of TileCost
        Per-tile cost breakdowns.
    sequential_us : float
        Total time with no overlap.
    pipelined_us : float
        Total time with tile-level overlap.
    overlap_ratio : float
        Fraction of communication hidden (0.0 = no overlap, 1.0 = full).
    bottleneck : str
        "compute" or "comm" — which limits pipelining.
    """

    tile_costs: list[TileCost]
    sequential_us: float
    pipelined_us: float
    overlap_ratio: float
    bottleneck: str

    @property
    def speedup(self) -> float:
        """Speedup from overlapping vs sequential."""
        if self.pipelined_us <= 0:
            return 1.0
        return self.sequential_us / self.pipelined_us

    def summary(self) -> str:
        return (
            f"Pipeline: {len(self.tile_costs)} tiles, "
            f"seq={self.sequential_us:.1f}µs, pipe={self.pipelined_us:.1f}µs, "
            f"speedup={self.speedup:.2f}x, overlap={self.overlap_ratio:.1%}, "
            f"bottleneck={self.bottleneck}"
        )


class TileCostModel:
    """Tile-native α-β cost model for communication and computation.

    Predicts per-tile costs for plan optimization and simulation.

    Parameters
    ----------
    profile : HardwareProfile
        Hardware characteristics for cost estimation.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        self.profile = profile

    def tile_comm_time(
        self,
        tile_M: int,
        tile_N: int,
        dtype: str = "float16",
    ) -> float:
        """Predict communication time for one tile transfer (microseconds).

        Two-regime model: max(min_comm_us, α + tile_bytes × β)
        The floor captures NCCL's fixed overhead for small messages.
        """
        dtype_bytes = DTYPE_BYTES.get(dtype, 2)
        tile_bytes = tile_M * tile_N * dtype_bytes
        ab_time = self.profile.alpha_us + tile_bytes * self.profile.beta_us_per_byte
        return max(self.profile.min_comm_us, ab_time)

    def tile_compute_time(
        self,
        tile_M: int,
        tile_N: int,
        tile_K: int,
        dtype: str = "float16",
    ) -> float:
        """Predict GEMM compute time for one tile (microseconds).

        GEMM FLOPs = 2 × M × N × K
        Time = FLOPs / (peak_tflops × efficiency × 1e6)
        """
        flops = 2.0 * tile_M * tile_N * tile_K
        effective_tflops = (
            self.profile.gemm_tflops_fp16 * self.profile.gemm_efficiency
        )
        if effective_tflops <= 0:
            return float("inf")
        return flops / (effective_tflops * 1e6)

    def tile_cost(
        self,
        tile_M: int,
        tile_N: int,
        tile_K: int,
        dtype: str = "float16",
    ) -> TileCost:
        """Full cost breakdown for one tile (comm + compute)."""
        dtype_bytes = DTYPE_BYTES.get(dtype, 2)
        return TileCost(
            comm_us=self.tile_comm_time(tile_M, tile_N, dtype),
            compute_us=self.tile_compute_time(tile_M, tile_N, tile_K, dtype),
            tile_bytes=tile_M * tile_N * dtype_bytes,
        )

    def pipeline_cost(
        self,
        n_tiles: int,
        tile_M: int,
        tile_N: int,
        tile_K: int,
        dtype: str = "float16",
        n_chunks: int = 1,
    ) -> PipelineCost:
        """Analyze cost of a tile pipeline (AG→GEMM or GEMM→RS).

        Models the steady-state pipeline where tiles overlap:
          - First tile: comm + compute (no overlap)
          - Middle tiles: max(comm, compute) each
          - Last tile: max(comm, compute)

        Parameters
        ----------
        n_tiles : int
            Total number of tiles in the pipeline.
        tile_M, tile_N, tile_K : int
            Tile dimensions for GEMM.
        dtype : str
            Data type.
        n_chunks : int
            Number of communication chunks per tile.
            More chunks = finer-grained overlap but more α overhead.
        """
        tc = self.tile_cost(tile_M, tile_N, tile_K, dtype)

        # Adjust comm for chunking (more α, same total bytes)
        if n_chunks > 1:
            chunk_bytes = tc.tile_bytes // n_chunks
            chunk_comm = (
                self.profile.alpha_us + chunk_bytes * self.profile.beta_us_per_byte
            )
            tc = TileCost(
                comm_us=chunk_comm * n_chunks,
                compute_us=tc.compute_us,
                tile_bytes=tc.tile_bytes,
            )

        tile_costs = [tc] * n_tiles
        sequential = tc.sequential_us * n_tiles

        if n_tiles <= 1:
            pipelined = tc.sequential_us
        else:
            # Pipeline: first tile has startup, rest overlap
            pipelined = tc.sequential_us + (n_tiles - 1) * tc.overlapped_us

        if sequential > 0:
            overlap_ratio = 1.0 - pipelined / sequential
        else:
            overlap_ratio = 0.0

        bottleneck = "compute" if tc.compute_bound else "comm"

        return PipelineCost(
            tile_costs=tile_costs,
            sequential_us=sequential,
            pipelined_us=pipelined,
            overlap_ratio=overlap_ratio,
            bottleneck=bottleneck,
        )

    def optimal_tile_size(
        self,
        M: int,
        N: int,
        K: int,
        dtype: str = "float16",
        candidates: Optional[Sequence[int]] = None,
    ) -> tuple[int, PipelineCost]:
        """Find the tile size that minimizes pipeline time.

        Larger tiles → fewer α overheads but less overlap opportunity.
        Smaller tiles → more overlap but more startup overhead.

        Returns (best_tile_size, best_cost).
        """
        if candidates is None:
            candidates = [32, 64, 128, 256, 512]

        best_size = candidates[0]
        best_cost = None

        for tile_size in candidates:
            if tile_size > M or tile_size > N:
                continue
            n_tiles_m = math.ceil(M / tile_size)
            n_tiles_n = math.ceil(N / tile_size)
            n_tiles = n_tiles_m * n_tiles_n

            cost = self.pipeline_cost(
                n_tiles=n_tiles,
                tile_M=tile_size,
                tile_N=tile_size,
                tile_K=K,
                dtype=dtype,
            )

            if best_cost is None or cost.pipelined_us < best_cost.pipelined_us:
                best_size = tile_size
                best_cost = cost

        return best_size, best_cost

    def allreduce_cost(
        self,
        M: int,
        N: int,
        K: int,
        world_size: int,
        tile_size: int = 128,
        dtype: str = "float16",
    ) -> dict[str, PipelineCost]:
        """Estimate AllReduce cost decomposed into RS + AG phases.

        Returns dict with 'reduce_scatter', 'allgather', and 'total' keys.
        """
        shard_rows = M // world_size
        n_tiles_m = math.ceil(shard_rows / tile_size)
        n_tiles_n = math.ceil(N / tile_size)
        n_tiles_per_rank = n_tiles_m * n_tiles_n

        # RS: each rank sends its partial tiles (world_size-1 transfers)
        rs_cost = self.pipeline_cost(
            n_tiles=n_tiles_per_rank * (world_size - 1),
            tile_M=tile_size,
            tile_N=tile_size,
            tile_K=K,
            dtype=dtype,
        )

        # AG: gather reduced shards
        ag_cost = self.pipeline_cost(
            n_tiles=n_tiles_per_rank * (world_size - 1),
            tile_M=tile_size,
            tile_N=tile_size,
            tile_K=K,
            dtype=dtype,
        )

        # Total: RS then AG (partially overlappable in practice)
        total_seq = rs_cost.sequential_us + ag_cost.sequential_us
        total_pipe = rs_cost.pipelined_us + ag_cost.pipelined_us
        total = PipelineCost(
            tile_costs=rs_cost.tile_costs + ag_cost.tile_costs,
            sequential_us=total_seq,
            pipelined_us=total_pipe,
            overlap_ratio=1.0 - total_pipe / total_seq if total_seq > 0 else 0.0,
            bottleneck=rs_cost.bottleneck,
        )

        return {
            "reduce_scatter": rs_cost,
            "allgather": ag_cost,
            "total": total,
        }


@dataclass
class CalibrationPoint:
    """A measured (tile_size, comm_time_us) pair for calibration."""
    tile_M: int
    tile_N: int
    dtype: str
    measured_comm_us: float
    measured_compute_us: Optional[float] = None


class CostModelCalibrator:
    """Calibrate α-β parameters from measured data.

    Supports two fitting modes:
    1. Standard: fits α + β*bytes on all points (may fail with NCCL data).
    2. Two-regime: detects the NCCL floor, fits β only on large-message
       points where bandwidth dominates.
    """

    @staticmethod
    def calibrate_comm(
        points: Sequence[CalibrationPoint],
        min_bytes_threshold: int = 0,
    ) -> tuple[float, float]:
        """Fit α and β from measured communication times.

        Parameters
        ----------
        points : Sequence[CalibrationPoint]
            Measured (tile_size, latency) pairs.
        min_bytes_threshold : int
            Only fit on data points with tile_bytes >= this value.
            Use to exclude small messages dominated by NCCL overhead.
            Default 0 uses all points (original behavior).

        Returns (alpha_us, beta_us_per_byte).
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 calibration points")

        # Filter by threshold
        filtered = []
        for p in points:
            x = p.tile_M * p.tile_N * DTYPE_BYTES.get(p.dtype, 2)
            if x >= min_bytes_threshold:
                filtered.append(p)

        if len(filtered) < 2:
            raise ValueError(
                f"Need at least 2 points with bytes >= {min_bytes_threshold}, "
                f"got {len(filtered)}"
            )

        # Linear regression: time = α + bytes × β
        n = len(filtered)
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0

        for p in filtered:
            x = p.tile_M * p.tile_N * DTYPE_BYTES.get(p.dtype, 2)
            y = p.measured_comm_us
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_xx += x * x

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            raise ValueError("Degenerate calibration data (all same size)")

        beta = (n * sum_xy - sum_x * sum_y) / denom
        alpha = (sum_y - beta * sum_x) / n

        # Ensure non-negative
        alpha = max(0.0, alpha)
        beta = max(0.0, beta)

        return alpha, beta

    @staticmethod
    def calibrate_two_regime(
        points: Sequence[CalibrationPoint],
    ) -> tuple[float, float, float]:
        """Fit a two-regime model: max(min_comm, α + β*bytes).

        NCCL latency has a characteristic U-shape: high for small messages
        (protocol overhead), dips to a minimum (~40µs on H100 PCIe at ~2MB),
        then increases linearly with size (bandwidth-bound).

        Strategy:
        1. Find the minimum-latency point (the regime transition).
        2. Fit α-β only on points ABOVE the minimum with increasing latency.
        3. The floor = measured minimum latency.

        Returns (alpha_us, beta_us_per_byte, min_comm_us).
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 calibration points for two-regime fit")

        # Sort by bytes
        sized = []
        for p in points:
            x = p.tile_M * p.tile_N * DTYPE_BYTES.get(p.dtype, 2)
            sized.append((x, p.measured_comm_us, p))
        sized.sort(key=lambda t: t[0])

        # Find the minimum-latency point (transition between regimes)
        min_idx = min(range(len(sized)), key=lambda i: sized[i][1])
        min_latency = sized[min_idx][1]
        min_bytes = sized[min_idx][0]

        # BW regime: points with bytes > min_bytes AND latency > min_latency
        bw_points = [
            t[2] for t in sized
            if t[0] > min_bytes and t[1] > min_latency
        ]

        if len(bw_points) >= 2:
            alpha, beta = CostModelCalibrator.calibrate_comm(bw_points)
        else:
            # Not enough BW-regime data — use the two largest points
            largest = [t[2] for t in sized[-2:]]
            alpha, beta = CostModelCalibrator.calibrate_comm(largest)

        return alpha, beta, min_latency

    @staticmethod
    def calibrate_from_measurements(
        points: Sequence[CalibrationPoint],
        base_profile: HardwareProfile,
        two_regime: bool = False,
    ) -> HardwareProfile:
        """Create a calibrated HardwareProfile from measurements.

        Parameters
        ----------
        points : Sequence[CalibrationPoint]
            Measured data points.
        base_profile : HardwareProfile
            Base profile for GEMM parameters.
        two_regime : bool
            If True, use two-regime fitting (detects NCCL floor).
        """
        if two_regime:
            alpha, beta, min_comm = CostModelCalibrator.calibrate_two_regime(points)
        else:
            alpha, beta = CostModelCalibrator.calibrate_comm(points)
            min_comm = 0.0

        return HardwareProfile(
            alpha_us=alpha,
            beta_us_per_byte=beta,
            gemm_tflops_fp16=base_profile.gemm_tflops_fp16,
            gemm_efficiency=base_profile.gemm_efficiency,
            min_comm_us=min_comm,
            name=f"{base_profile.name} (calibrated)",
        )
