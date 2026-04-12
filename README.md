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

Requires Python 3.10+, PyTorch >= 2.4, and Triton >= 3.0.

```bash
pip install -e .
```

Optional benchmark dependencies:

```bash
pip install -e ".[benchmark]"
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

## Example: FusedSequential Pattern

```python
# Layer 1: user API
tncc.ops.gemm_allscatter(A, B, C, ctx=ctx, pattern="fused_sequential")
```

```python
# Layer 2: plan + execution contract
plan = build_gemm_allscatter_plan(
    A, B, C,
    ctx=ctx,
    pattern="fused_sequential",
)
plan.execute(A, B, C)

@dataclass(frozen=True, slots=True)
class PatternExecutionSpec:
    M: int
    K: int
    full_N: int
    local_N: int
    scatter_src_col_offset: int
    scatter_cols: int
    scatter_dst_leading_dim: int
    scatter_dst_col_offset: int
```

```python
# Layer 3: overlap pattern
spec = self.resolve_execution(A, B, C, ...)
self._fused_kernel[grid](
    A, B, C,
    heap_bases,
    M, N, K,
    spec.scatter_src_col_offset,
    spec.scatter_cols,
    spec.scatter_dst_leading_dim,
    spec.scatter_dst_col_offset,
    ...,
)
```

```python
# Layer 4: persistent fused kernel
pid = tl.program_id(0)
total_tiles = num_tiles_m * num_tiles_n

for tile_id in range(pid, total_tiles, NUM_SMS):
    tile_m = tile_id // num_tiles_n
    tile_n = tile_id % num_tiles_n

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, loop_k):
        ...
        acc = tl.dot(a, b, acc, allow_tf32=True)

    if not EVEN_K:
        ...
        acc = tl.dot(a, b, acc, allow_tf32=True)

    result = acc.to(C_ptr.dtype.element_ty)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result, mask=c_mask)

    for peer in range(world_size):
        if peer != rank:
            scatter_tile_to_peer(
                C_ptr, result, offs_m, offs_n,
                rank, peer, heap_bases,
                scatter_src_col_offset, scatter_cols,
                scatter_dst_leading_dim, scatter_dst_col_offset,
                c_mask,
            )
```

```python
# Layer 5: device primitive. Inspired by Iris.
remote_C = translate_ptr(C_ptr, rank, peer, heap_bases)
offsets = offs_m[:, None] * dst_leading_dim + (dst_col_offset + safe_local_cols[None, :])
tl.store(remote_C + offsets, tile_data, mask=final_mask, cache_modifier=".wt")
```

`gemm_allscatter` -> `build_*_plan` -> `PatternExecutionSpec` -> `FusedSequentialPattern` -> persistent kernel -> `translate_ptr` + remote `tl.store`

FusedSequential uses data-movement primitives directly; it does not need `tile_signal` / `tile_wait` in this path.

## Requirements

- NVIDIA GPUs with NVLink interconnect (verified on H100), or AMD GPUs with xGMI
- CUDA 12.x / ROCm 6.x, PyTorch >= 2.4, Triton >= 3.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
