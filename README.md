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




## Architecture



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



## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
