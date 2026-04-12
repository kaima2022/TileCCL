<p align="center">
  <img src="assets/logo.png" width="480" alt="TileCCL"/>
</p>

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

Minimal API usage:

```python
import torch
import tncc

ctx = tncc.init(heap_size=512 * 1024 * 1024)
A = ctx.randn(4096, 4096, dtype=torch.float16)
B = ctx.randn(4096, 8192, dtype=torch.float16)
C = ctx.zeros(4096, 8192, dtype=torch.float16)

tncc.ops.gemm_allscatter(A, B, C, ctx=ctx)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
