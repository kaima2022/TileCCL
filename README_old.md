<div align="center">

<p>
  <img src="assets/logo.png" width="420" alt="TileCCL"/>
</p>

# TileCCL: Tile-Native Collective Communication Library

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="images/logo/xgs_logo_.png" alt="信工所Logo" width="180"/><br/>
        <strong>Institute of Information Engineering</strong><br/>
        Chinese Academy of Sciences<br/>
        <em>State Key Laboratory of Cyberspace Security Defense</em>
      </td>
      <td align="center" width="50%">
        <img src="images/logo/wdzs_logo.png" alt="微电子所Logo" width="180"/><br/>
        <strong>Institute of Microelectronics</strong><br/>
        Chinese Academy of Sciences<br/>
        <em>Artificial Intelligence Chip and System Research and Development Center</em>
      </td>
    </tr>
  </table>
</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg)]()
[![Triton](https://img.shields.io/badge/Triton-3.0%2B-purple.svg)]()

</div>

## Overview

TileCCL is a tile-native collective communication library for distributed GPU workloads. It targets the synchronization bubbles created by traditional tensor-level or chunk-level collectives, where downstream compute often waits for a large communication unit even when only a small part of the result is needed next.

Instead of treating a collective as a monolithic operation around a full tensor, TileCCL makes communication readiness follow the producer and consumer tile schedule. GEMM output can be exposed, grouped, transferred, and consumed as soon as the relevant work is ready, allowing communication to overlap with computation at a finer granularity.

## Architecture

<p align="center">
  <img src="assets/TileCCL-architecture-new.png" width="760" alt="Architecture"/>
</p>

GEMM produces tiles. The TileGroup Builder (driven by a CostModel) groups them into physically-sized units. Compute and communication workgroups then run concurrently in a single persistent kernel — compute produces and signals, communication polls barriers and pushes ready groups to peers via CUDA IPC.

- Does not modify the GEMM kernel. Adds one `atomic_add` in the epilogue.
- TileGroup = physics-driven grouping. P0 (P2P saturation) + P1 (wave alignment) + P2 (pipeline balance) determine group boundaries.
- Device-side P2P. Compute workgroups and communication workgroups run concurrently in a single persistent kernel. Ready TileGroups are pushed to peer GPUs immediately via CUDA IPC, without NCCL or NVSHMEM.
- Proven on two collectives. GEMM-output AllGather (1.34–1.54×) and GEMM→ReduceScatter (1.40–1.71×) on 2×H100 with same-backend controlled experiments.

## Data Flow

<p align="center">
  <img src="assets/TileCCL-dataflow-comparison.png" width="760" alt="Data Flow"/>
</p>

Three granularities compared: bulk tensor (one large transfer), per-tile (many tiny transfers), and TileGroup (tiles assembled into groups, then one transfer per group). TileGroup balances signal overhead against P2P efficiency.

## Signal-Communication Granularity

<p align="center">
  <img src="assets/TileCCL-spectrum.png" width="760" alt="Granularity Spectrum"/>
</p>

Existing systems occupy different points in the signal-vs-communication granularity space. TileCCL sits on the diagonal — signal and communication aligned at the same TileGroup granularity, determined by physical constraints rather than hardcoded or compiler-fixed.

## Preliminary Results

These are early proof results on 2x NVIDIA H100 PCIe GPUs with NVLink peer access. The fused proofs use the same Triton persistent GEMM backend across all variants; differences are limited to TileGroup signaling and device-side P2P communication.

### Gate 1: GEMM-output AllGather

| Shape MxNxK | S0 Bulk | S1 per tile | S2 TileCCL | S1 vs S0 | S0/S2 |
|:---|---:|---:|---:|:---|---:|
| 16384x4096x1024 | 1.224 ms | 1.160 ms | 0.797 ms | 1.06x faster | 1.54x faster |
| 8192x4096x2048 | 0.902 ms | 0.753 ms | 0.675 ms | 1.20x faster | 1.34x faster |
| 8192x4096x1024 | 0.706 ms | 0.630 ms | 0.460 ms | 1.12x faster | 1.54x faster |

### Gate 2: GEMM to ReduceScatter

| Shape MxNxK_total | S0 Bulk | S1 per tile | S2 TileCCL | S1 vs S0 | S0/S2 |
|:---|---:|---:|---:|:---|---:|
| 16384x4096x2048 | 1.646 ms | 1.083 ms | 0.963 ms | 1.52x faster | 1.71x faster |
| 8192x4096x4096 | 1.122 ms | 0.751 ms | 0.799 ms | 1.49x faster | 1.40x faster |
| 8192x4096x2048 | 0.934 ms | 0.619 ms | 0.610 ms | 1.51x faster | 1.53x faster |

## Repository Layout

```text
TileCCL/
├── tileccl_v2/              # Framework seed
│   ├── heap.py              # CUDA IPC symmetric heap
│   ├── ipc.py               # Pointer translation, signal/wait, P2P primitives
│   ├── tile_group.py        # TileGroup planning (P0+P1+P2 constraints)
│   ├── wg.py                # Compute/comm WG allocation
│   ├── collective_spec.py   # AG/RS semantics
│   ├── signal.py            # TileGroup signal tensor layout
│   ├── transport.py         # P2P transport plan (push/pull, packed copy)
│   ├── cost_model.py        # Cost model seed
│   └── runtime/timeline.py  # Timeline artifact recorder
├── assets/                  # Architecture & dataflow diagrams
└── images/logo/             # Institution logos
```

The proof experiments (`fused_ag_iris.py`, `fused_rs_iris.py`) live in a separate experimental repository and are not part of this framework seed.

## Installation

TileCCL requires Python 3.10+, PyTorch 2.4+, and Triton 3.0+.

```bash
pip install -e .
```

For development utilities:

```bash
pip install -e ".[dev,benchmark]"
```

## Quick Start

```python
from tileccl_v2 import SymmetricHeap, build_tile_group_plan, build_p2p_transport_plan

spec = reduce_scatter_spec(world_size=2)
plan = build_tile_group_plan(8192, 4096, 128*128*2)
transport = build_p2p_transport_plan(comm_mode="push", copy_elems=16384)
print(plan.n_groups, transport.push_mode)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
