<p align="center">
  <img src="assets/logo.png" width="480" alt="TileCCL"/>
</p>


<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="images/logo/xgs_logo_.png" alt="Institute of Information Engineering logo" width="180"/><br/>
        <strong>Institute of Information Engineering</strong><br/>
        Chinese Academy of Sciences<br/>
        <em>State Key Laboratory of Cyberspace Security Defense</em>
      </td>
      <td align="center" width="50%">
        <img src="images/logo/wdzs_logo.png" alt="Institute of Microelectronics logo" width="180"/><br/>
        <strong>Institute of Microelectronics</strong><br/>
        Chinese Academy of Sciences<br/>
        <em>Artificial Intelligence Chip and System Research and Development Center</em>
      </td>
    </tr>
  </table>
</div>

# TileCCL: Tile-native Collective Communication Library

TileCCL is a tile-native collective communication library for expressing cross-GPU data movement, synchronization, and collective execution directly in Triton. It supports tile-level collectives, GEMM+collective operators, and overlap strategies for multi-GPU kernels where communication needs to remain explicit instead of being hidden behind a separate runtime boundary.

## TileCCL Architecture

<p align="center">
  <img src="assets/TileCCL-architecture.png" width="760" alt="TileCCL Architecture"/>
</p>


**User API** — TileCCL exposes high-level collective and GEMM+collective entry points such as `gemm_allscatter`, `gemm_allgather`, `gemm_reducescatter`, `allreduce`, `allgather`, and `reduce_scatter`. This keeps the programming surface at the operator level while leaving execution policy explicit.

**Execution Engine** — Between the API and the device-side kernels, TileCCL resolves layout contracts, applies plan-based execution, and dispatches an overlap strategy. This layer separates public operation semantics from pattern-specific execution choices.

**Tile-Native Primitives** — The core execution layer is built from Triton JIT primitives, so communication and synchronization remain visible inside the same programming model as compute.

- **Data Movement** — `tile_remote_load`, `tile_remote_store`, `tile_put`, and `tile_get` provide fine-grained and bulk tile transfer across peer-accessible memory.
- **Synchronization** — `tile_signal`, `tile_wait`, and remote atomic operations provide cross-tile coordination with explicit memory-ordering semantics.
- **Tile Collectives** — `tile_allreduce`, `tile_allgather`, `tile_reduce_scatter`, `tile_broadcast`, and `tile_scatter` implement collective algorithms directly at tile granularity.

**Symmetric Memory** — Heap allocation, peer mapping, and `translate_ptr` provide the address translation layer that lets tile-level kernels access peer memory through a shared symmetric-memory model.

**Hardware Substrate** — TileCCL runs on GPU backends and interconnect-capable peer-memory paths underneath the symmetric-memory layer, providing the transport foundation for tile-native communication.


## Repository Layout

```text
TileCCL/
├── tileccl/                 # Public library package
│   ├── memory/              # Symmetric heap, peer mapping, pointer translation
│   ├── primitives/          # Triton tile movement, synchronization, collectives
│   ├── patterns/            # Compute-communication overlap strategies
│   ├── kernels/             # GEMM and fused kernel entry points
│   └── ops.py               # Public operator-level APIs
├── tileccl_v2/              # TileGroup proof/runtime seed
│   ├── heap.py              # CUDA IPC symmetric heap
│   ├── ipc.py               # Pointer translation, signal/wait, P2P primitives
│   ├── tile_group.py        # TileGroup planning
│   ├── wg.py                # Compute/communication workgroup allocation
│   ├── collective_spec.py   # AllGather/ReduceScatter semantics
│   ├── signal.py            # TileGroup signal tensor layout
│   ├── transport.py         # P2P transport planning
│   ├── cost_model.py        # Tile-native communication cost model
│   └── runtime/timeline.py  # Timeline artifact recorder
├── examples/                # Single-process, multiprocess, benchmark examples
├── tests/                   # Smoke tests
├── assets/                  # Architecture and dataflow diagrams
└── images/logo/             # Institution logos
```



## Key Feature: TileGroup-Based Overlap

TileCCL targets synchronization bubbles created by traditional tensor-level or chunk-level collectives, where downstream compute often waits for a large communication unit even when only a small part of the result is needed next.

Instead of treating a collective as a monolithic operation around a full tensor, TileCCL makes communication readiness follow the producer and consumer tile schedule. GEMM output can be exposed, grouped, transferred, and consumed as soon as the relevant work is ready, allowing communication to overlap with computation at a finer granularity.

GEMM produces tiles. The TileGroup builder, driven by a cost model, groups them into physically sized units. Compute and communication workgroups can then run concurrently in a single persistent kernel: compute produces and signals, while communication polls barriers and pushes ready groups to peers via CUDA IPC.

- **Minimal GEMM intrusion** — TileGroup readiness can be exposed at the epilogue boundary with lightweight signaling.
- **Physics-driven grouping** — P2P saturation, wave alignment, and pipeline balance determine TileGroup boundaries.
- **Device-side P2P transport** — Ready TileGroups are pushed to peer GPUs via CUDA IPC without routing through NCCL or NVSHMEM collectives.

### Data Flow

<p align="center">
  <img src="assets/TileCCL-dataflow-comparison.png" width="760" alt="TileCCL Data Flow"/>
</p>

TileCCL compares three communication granularities: bulk tensor transfer, per-tile transfer, and TileGroup transfer. TileGroup sits between the extremes by assembling tiles into transfer-efficient groups while keeping readiness aligned with the compute schedule.

## Preliminary Results

These early proof results were collected on 2x NVIDIA H100 GPUs with NVLink peer access. The fused proofs use the same Triton persistent GEMM backend across all variants; differences are limited to TileGroup signaling and device-side P2P communication.

### Gate 1: GEMM-Output AllGather

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

## Install

TileCCL requires Python 3.10+, PyTorch >= 2.4, and Triton >= 3.0.

Install from source:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

For development with benchmark dependencies:

```bash
pip install -e ".[dev,benchmark]"
```

## Quick Start

Run the single-process example:

```bash
python examples/single_process.py
```

Run the distributed example:

```bash
torchrun --nproc_per_node=<num_gpus> examples/multiprocess.py
```

More entry points are available under `examples/`.

Minimal single-process usage:

```python
import torch
import tileccl

ctx = tileccl.init(heap_size=512 * 1024 * 1024)
A = ctx.randn(4096, 4096, dtype=torch.float16)
B = ctx.randn(4096, 8192, dtype=torch.float16)
C = ctx.zeros(4096, 8192, dtype=torch.float16)

tileccl.ops.gemm_allscatter(A, B, C, ctx=ctx)
```

TileGroup planning API example:

```python
from tileccl_v2 import (
    build_p2p_transport_plan,
    build_tile_group_plan,
    reduce_scatter_spec,
)

spec = reduce_scatter_spec(world_size=2)
plan = build_tile_group_plan(8192, 4096, 128 * 128 * 2)
transport = build_p2p_transport_plan(comm_mode="push", copy_elems=16384)

print(spec.kind, plan.n_groups, transport.push_mode)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
