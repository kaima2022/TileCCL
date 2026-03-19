# XTile

**Cross-platform tile communication library with full compiler visibility.**

XTile combines the best ideas from [Iris](https://github.com/ROCm/iris), TileScale, TileLink, and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) to provide a unified tile communication library for GPU-accelerated distributed computing.

## Key Features

- **Communication as First-Class Primitive** -- alongside compute and memory, not an afterthought
- **Full Compiler Visibility** -- pure Triton implementation, compiler can optimize across compute-communication boundaries
- **Hardware Portability** -- single API for NVIDIA (Hopper/Blackwell) and AMD (CDNA3/CDNA4)
- **Built-in Overlap Patterns** -- bulk-sync, fused sequential, producer-consumer, workgroup specialization
- **Auto Pattern Selection** -- automatically choose the best overlap strategy for your workload

## Architecture

```
User API          xtile.init / xtile.Tile / xtile.SymmetricHeap
Pattern Library   BulkSync / FusedSequential / ProducerConsumer / WGSpecialized
Core Primitives   compute (tile_dot) / memory (tile_load) / communication (tile_remote_store)
Synchronization   acquire/release semantics / tile_signal / tile_wait
Memory Mgmt       SymmetricHeap / Pointer Translation
HAL               NVIDIA (CUDA IPC, NVLink) / AMD (HIP IPC, Infinity Fabric)
```

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
import xtile

# Initialize
ctx = xtile.init(backend="auto")

# Create symmetric heap for multi-GPU communication
heap = xtile.SymmetricHeap(size=1 << 30, rank=ctx.rank, world_size=ctx.world_size)

# Use patterns for fused compute-communication
from xtile.patterns import auto_select
pattern = auto_select("gemm_allscatter", M=8192, N=8192, K=8192, world_size=ctx.world_size)
pattern.execute(A, B, C)
```

## Development

```bash
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Lint check
make bench         # Run benchmarks
```

## Status

**Phase 0** -- Infrastructure setup. See [development plan](XTile_开发计划.md) for the full roadmap.

## License

Apache 2.0
