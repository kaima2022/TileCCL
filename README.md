<div align="center">

<p>
  <img src="assets/logo.png" width="420" alt="TileCCL"/>
</p>

# TileCCL: Tile-Native Collective Communication Library

Tile-native communication and scheduling primitives for overlapping GPU compute
and cross-GPU data movement at TileGroup granularity.

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

TileCCL explores a tile-native collective communication design: instead of
waiting for a full tensor or chunk-level collective boundary, a producer kernel
exposes TileGroup readiness and a communication worker group moves the ready
TileGroup directly through peer-accessible GPU memory.

The current open-source seed is intentionally small. It contains the stable
runtime and planning pieces extracted from fused GEMM+AllGather and
GEMM+ReduceScatter proof experiments:

- `tileccl_v2/heap.py`: CUDA IPC symmetric heap allocation and peer mapping.
- `tileccl_v2/ipc.py`: Triton device-side pointer translation and signal/P2P primitives.
- `tileccl_v2/tile_group.py`: TileGroup planning, P0/P1 group boundaries, group table generation.
- `tileccl_v2/wg.py`: compute/communication workgroup allocation.
- `tileccl_v2/collective_spec.py`: minimal AllGather and ReduceScatter semantics.
- `tileccl_v2/signal.py`: TileGroup arrival/barrier/trace tensor layout.
- `tileccl_v2/transport.py`: push/pull and packed P2P transport planning.
- `tileccl_v2/cost_model.py`: cost-model seed for future P2 TileGroup calibration.
- `tileccl_v2/runtime/timeline.py`: timeline artifact recorder.

Older exploratory APIs remain in the repository for historical context, but the
active framework seed is `tileccl_v2/`.

## Motivation

Existing systems can often expose tile or wave-level progress inside a compute
kernel, while the communication layer still executes at tensor or coarse chunk
granularity. TileCCL inserts a TileGroup abstraction between a GEMM tile and a
full collective tensor:

```text
Traditional overlap:
  signal:  tile / wave progress
  comm:    full tensor or coarse chunk

TileCCL:
  signal:  per TileGroup
  comm:    per TileGroup
```

The goal is to align readiness signaling and communication at the same physical
granularity so that communication can begin as soon as a useful group of tiles
is ready, without forcing a full tensor synchronization point.

## Architecture

<p align="center">
  <img src="assets/TileCCL-architecture.png" width="760" alt="TileCCL Architecture"/>
</p>

The current design has four layers:

1. **Compute backend**: Triton/TileLang/CUTLASS style GEMM kernels. TileCCL
   does not change GEMM math; it adds TileGroup-ready signaling around the
   kernel.
2. **TileGroup planning**: chooses group boundaries using physical constraints:
   P2P saturation threshold, wave-aligned boundaries, and future cost-model
   calibration.
3. **Signal and transport**: arrival counters, barriers, pointer translation,
   packed P2P push/pull, and reduce task planning.
4. **Collective semantics**: minimal AllGather and ReduceScatter specs that
   constrain ownership and row-shard boundaries.

## Preliminary Results

These are early proof results on 2x NVIDIA H100 PCIe GPUs with NVLink peer
access. The fused proofs use the same Triton persistent GEMM backend across all
variants; differences are limited to TileGroup signaling and device-side P2P
communication.

Variants:

- `S0 Bulk`: GEMM followed by host-side bulk P2P copy/reduce.
- `S1 per-tile signal`: same GEMM plus per-tile arrival `atomic_add`; the last
  tile in a TileGroup flips the TileGroup barrier, followed by host-side
  transfer. This isolates signaling overhead.
- `S2 TileCCL`: same per-tile signal path plus device-side communication
  workgroups that move/reduce ready TileGroups.

### Gate 1: GEMM-output AllGather

| Shape MxNxK | S0 Bulk | S1 per-tile signal | S2 TileCCL | Speedup | Signal overhead |
|:---|---:|---:|---:|---:|---:|
| 16384x4096x1024 | 1.211 ms | 1.217 ms | 0.792 ms | 1.53x | +0.6% |
| 8192x4096x2048 | 0.887 ms | 0.895 ms | 0.632 ms | 1.40x | +0.8% |
| 8192x4096x1024 | 0.693 ms | 0.698 ms | 0.455 ms | 1.52x | +0.7% |

### Gate 2: GEMM to ReduceScatter

| Shape MxNxK_total | S0 Bulk | S1 per-tile signal | S2 TileCCL | Speedup | Signal overhead |
|:---|---:|---:|---:|---:|---:|
| 16384x4096x2048 | 1.618 ms | 1.630 ms | 0.961 ms | 1.68x | +0.7% |
| 8192x4096x4096 | 1.105 ms | 1.113 ms | 0.771 ms | 1.43x | +0.7% |
| 8192x4096x2048 | 0.905 ms | 0.921 ms | 0.585 ms | 1.55x | +1.7% |

Latest smoke reproduction after the framework-seed extraction:

| Proof | Shape | S0 Bulk | S1 per-tile signal | S2 TileCCL | Speedup |
|:---|:---|---:|---:|---:|---:|
| AG | 8192x4096x1024 | 0.712 ms | 0.715 ms | 0.464 ms | 1.54x |
| RS | 8192x4096x1024 | 0.938 ms | 0.947 ms | 0.616 ms | 1.52x |

## Repository Layout

```text
TileCCL/
├── tileccl_v2/              # Current proof runtime and framework seed
│   ├── heap.py              # CUDA IPC symmetric heap
│   ├── ipc.py               # Triton pointer translation and signal primitives
│   ├── tile_group.py        # TileGroup planning
│   ├── wg.py                # compute/comm WG planning
│   ├── collective_spec.py   # AG/RS semantic seed
│   ├── signal.py            # TileGroup signal tensor layout
│   ├── transport.py         # P2P transport plan
│   ├── cost_model.py        # Cost model seed
│   └── runtime/timeline.py  # Timeline artifacts
├── tileccl/                 # Earlier exploratory package
├── examples/                # Earlier examples
├── tests/                   # Earlier tests
├── assets/                  # Figures and project logo
└── images/logo/             # Institution logos used by this README
```

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

Minimal import check for the current framework seed:

```python
from tileccl_v2 import (
    SymmetricHeap,
    build_tile_group_plan,
    build_p2p_transport_plan,
    reduce_scatter_spec,
)

spec = reduce_scatter_spec(world_size=2)
plan = build_tile_group_plan(
    8192,
    4096,
    128 * 128 * 2,
    row_split_boundaries=spec.row_split_boundaries(8192),
)
transport = build_p2p_transport_plan(comm_mode="push", copy_elems=16384)

print(plan.n_groups, transport.push_mode)
```

## Status

TileCCL is pre-alpha research software. The current codebase is a framework
seed extracted from proof experiments, not yet a production collective library.
The next planned steps are:

- connect the cost model to TileGroup size selection;
- formalize device-side signal/barrier primitive contracts;
- add backend hooks for packed P2P copy/reduce;
- extend collective specs beyond AG/RS after the current path is stable.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
