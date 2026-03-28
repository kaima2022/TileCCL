# TNCC (Tile Native Collective Communication)

**Tile Native Collective Communication with full compiler visibility.**

TNCC (Tile Native Collective Communication) combines the best ideas from [Iris](https://github.com/ROCm/iris), TileScale, TileLink, and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) to provide a unified tile communication library for GPU-accelerated distributed computing.

## Key Features

- **Communication as First-Class Primitive** -- alongside compute and memory, not an afterthought
- **Full Compiler Visibility** -- pure Triton implementation, compiler can optimize across compute-communication boundaries
- **Hardware Portability** -- single API for NVIDIA (Hopper/Blackwell) and AMD (CDNA3/CDNA4)
- **Built-in Overlap Patterns** -- bulk-sync, fused sequential, producer-consumer, workgroup specialization
- **Auto Pattern Selection** -- automatically choose the best overlap strategy for your workload

## Architecture

```
User API          tncc.init / tncc.Tile / tncc.SymmetricHeap
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
import torch
import tncc

# Single GPU / distributed rank-local initialization
ctx = tncc.init(backend="auto", heap_size=1 << 30)

# Tensors now come directly from the attached symmetric heap
A = ctx.randn(8192, 8192, dtype=torch.float16)
B = ctx.randn(8192, 8192, dtype=torch.float16)
C = ctx.zeros(8192, 8192, dtype=torch.float16)

# Use patterns for fused compute-communication
from tncc.patterns import auto_select
pattern = auto_select("gemm_allscatter", M=8192, N=8192, K=8192, world_size=ctx.world_size, ctx=ctx)
pattern.execute(A, B, C)
```

For single-process multi-GPU experiments, use:

```python
contexts = tncc.init_local(world_size=2, heap_size=1 << 30)
ctx0 = contexts[0]
ctx1 = contexts[1]
```

Pattern benchmark now auto-sizes the symmetric heap from the tested shape:

```bash
tncc bench pattern --quick --warmup 2 --iters 5
tncc bench pattern --heap-size-mb 1024
```

## Development

```bash
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Lint check
make bench         # Run benchmarks
```

## Status

**Phase 7** -- Unified runtime context, benchmark hardening, and full-size pattern reruns.
See [CLAUDE.md](CLAUDE.md) for the current engineering status and measured baselines.

## License

Apache 2.0
