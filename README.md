<p align="center">
  <img src="assets/logo.png" width="480" alt="TileCCL"/>
</p>

# TileCCL: Tile-native Collective Communication Library

TileCCL brings collective communication into the tile programming model as a first-class library surface. Instead of hiding communication behind opaque runtime calls between kernels, TileCCL expresses allreduce, allgather, reduce-scatter, and fused GEMM+collective flows as compiler-visible Triton programs where compute, communication, and synchronization operate at the same tile granularity within a single device-side program.

<p align="center">
  <img src="assets/TileCCL-architecture.png" width="760" alt="TileCCL Architecture"/>
</p>

## Key Features

- **Three primitive groups, one compiler view.** Data movement (`tile_put`, `tile_get`, `tile_remote_store`), synchronization (`tile_signal`, `tile_wait`, remote atomics), and collectives (`tile_allreduce`, `tile_allgather`, `tile_reduce_scatter`) — all `@triton.jit` functions the Triton compiler can see and optimize end-to-end.

- **symmetric memory.** A 5-instruction `translate_ptr` maps any local heap offset to a peer GPU's address space. No staging buffers, no memcpy, no host round-trips — kernels directly read and write remote memory over NVLink or xGMI.

- **Tile-granularity overlap.** Communication begins the moment a tile is produced, not after the entire matrix is computed. Within a single persistent kernel, compute and communication interleave at tile boundaries, hiding latency behind useful work.

- **No opaque runtime.** Ring allreduce, direct-write allgather, atomic reduce-scatter — all implemented as pure Triton programs. The compiler optimizes the full compute-communication graph as a single program.

- **Cross-vendor from single source.** The same Triton primitive code compiles for NVIDIA (CUDA / NVLink) and AMD (HIP / xGMI). Backend abstraction handles IPC, topology detection, and peer access without changing the kernel source.

- **Plan-based execution.** Build an execution plan once (`build_gemm_allscatter_plan`), then reuse it across iterations. Planning overhead — pattern selection, contract validation, workspace allocation — is amortized to near zero.

## TileCCL - Tile Primitive Groups

**Data Movement** — Two modes of cross-GPU tile transfer:

- *Value-based* (`tile_remote_load`, `tile_remote_store`): register-to-remote, fine-grained, ideal for small tiles.
- *Pointer-based* (`tile_put`, `tile_get`): memory-to-remote, RDMA-style, bulk throughput.

**Synchronization** — Tile-level coordination with explicit memory ordering:

- Producer-consumer: `tile_signal` / `tile_wait` (acquire-release semantics).
- Remote atomics: `tile_atomic_add`, `cas`, `xchg`, `min`, `max`, `and`, `or`, `xor` — all with configurable scope (`gpu` / `sys`).

**Tile Collectives** — Standard collective algorithms as pure Triton JIT code:
- `tile_allreduce`, `tile_allgather`, `tile_reduce_scatter`, `tile_broadcast`, `tile_scatter`.
- Implemented as ring protocols — fully visible to the compiler, no NCCL dependency.

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

    rm = offs_m % M
    rn = offs_n % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    loop_k = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        loop_k -= 1

    for k in range(0, loop_k):
        a = tl.load(A_BASE)
        b = tl.load(B_BASE)
        acc = tl.dot(a, b, acc, allow_tf32=True)
        A_BASE += BLOCK_K * stride_ak
        B_BASE += BLOCK_K * stride_bk

    if not EVEN_K:
        rk_rem = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
        A_REM = A_ptr + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
        B_REM = B_ptr + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0)
        b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0)
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
# Layer 5: device primitive. inspired by Iris.
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
