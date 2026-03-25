# XTile 现状流程：GEMM + AllScatter 全栈代码流

> 文档定位
> - 本文基于当前仓库源码、测试与结构化实验结果整理。
> - 目标是说明当前实现、已验证范围、已知边界与后续工作。
> - 原始文档保留不动；本文件作为当前可维护版本持续更新。

## 任务定义

当前讨论的问题是：4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
先算 C_local = A × B（本地 GEMM），然后把 C_local scatter 到所有其他 GPU。

M=8192, N=4608, K=36864, 4 GPUs.

当前仓库里长期共存着两套调用约定：

1. **correctness / E2E 路径**
   - 常见于 `tests/test_patterns/*.py`
   - 传入完整 `B(K, N)` 与完整 `C(M, N)`
   - scatter 只把“本 rank 负责的列分片”写到 peer 的完整输出缓冲区

2. **benchmark 路径**
   - 常见于 `tests/benchmarks/bench_patterns.py`
   - 传入 `B(K, N_per_rank)` 与 `C(M, N_per_rank)`
   - 现在这条路径已经通过显式 `full_N + b_layout + c_layout` contract 与 pattern host 侧对齐

因此，下面文档会明确区分“**当前 correctness 主路径**”和“**当前 benchmark 主路径**”，但两条路径现在已经开始共享同一套显式 execution contract。

---

## 第 1 层：用户代码

### 实际源码（方式 A：当前推荐用户入口，高层 API）

这部分直接贴当前 `xtile/ops.py` 里的真实入口实现。当前推荐用户入口仍是 `xtile.ops.gemm_allscatter(...)`，其内部已经升级成“build plan -> execute plan”的主链：

```python
def gemm_allscatter(
    A,
    B,
    C,
    *,
    ctx: xtile.XTileContext | None = None,
    full_N: int | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
    pattern: str | type[Pattern] | Pattern | None = "auto",
    hw_info: object | None = None,
    storage_kind: str = "symmetric",
) -> Any:
    """Run GEMM + all-scatter through the stable public full-buffer API.

    If ``b_layout`` / ``c_layout`` are omitted, the call is interpreted as
    the canonical public ``full/full`` contract. Expert sharded usage should
    prefer :func:`gemm_allscatter_sharded`.
    """
    plan = build_gemm_allscatter_plan(
        A,
        B,
        C,
        ctx=ctx,
        full_N=full_N,
        b_layout=b_layout,
        c_layout=c_layout,
        pattern=pattern,
        hw_info=hw_info,
        storage_kind=storage_kind,
    )
    return plan.execute(A, B, C)
```

### 实际源码（方式 B：当前 benchmark 主路径）

`tests/benchmarks/bench_patterns.py` 当前 benchmark 仍然走 shard-buffer 约定，但不再手工拼 contract + 手工调 pattern，而是也改成走统一的 plan builder 主链：

```python
N_per_rank = N // world_size
A = ctx.randn(M, K, dtype=torch.float16)
B = ctx.randn(K, N_per_rank, dtype=torch.float16)
C = ctx.zeros(M, N_per_rank, dtype=torch.float16)
ctx.barrier()
reference_plan = build_gemm_allscatter_plan(
    A,
    B,
    C,
    ctx=ctx,
    full_N=N,
    b_layout="shard",
    c_layout="shard",
)

plan = build_gemm_allscatter_plan(
    A,
    B,
    C,
    ctx=ctx,
    full_N=N,
    b_layout="shard",
    c_layout="shard",
    pattern=cls,
)
plan.execute(A, B, C)
```

需要特别注意：benchmark 路径已经**不再依赖 `B.shape[1]` 隐式猜 full-N**；并且 benchmark / CLI helper / 高层 op 现在已经开始共享同一条 `build_gemm_allscatter_plan(...)` 主链。

### 当前状态与边界

| 主题 | 当前状态 | 当前边界 |
|------|------|------|
| 用户入口 | `xtile.ops.gemm_allscatter(...)` 已可用 | 高层 API 命名与最终 op 集合仍可继续收敛 |
| 堆上张量分配 | `XTileContext` 已支持 `empty/zeros/randn()` | 已打通 |
| 后端检测 | `xtile.init()` 默认就是 `backend="auto"` | 已集成 |
| runtime ctx 主路径 | `xtile.init(..., heap=...)` / `heap_size=...` / `init_local(...)` 都可直接返回可运行 pattern 的真实 ctx | 主路径已统一 |
| benchmark / tests / CLI / runtime 一致性 | 已统一到 `XTileContext` | 已打通 |
| 高层 op API | `xtile.ops.gemm_allscatter(...)` / `xtile.ops.gemm_allgather(...)` / `xtile.ops.allgather(...)` / `xtile.ops.allreduce(...)` / `xtile.ops.reduce_scatter(...)` / `xtile.ops.gemm_reducescatter(...)` 已接入，并统一走显式 `plan` 主链 | 在当前 public multiprocess surface（`world_size=2 + ctypes_ipc`）上，上述入口已完成同级别 host contract、回归和结构化 benchmark 验收；更大 world size、更多 transport、stress 与性能门禁仍未完成 |
| correctness / benchmark 双路径 | 两者都存在，但都已开始通过显式 contract + plan builder 收敛 | 仍需把更多调用点统一迁到高层 op API，并继续弱化直接 `pattern.execute(...)` 的 public 角色 |

### 当前高层 op 家族与公共主链

当前第 1 层已经不再只有 `gemm_allscatter(...)` 一个入口，而是形成了同一套 host-side 公共形态：

- `gemm_allscatter(...)`：默认 public `full/full` contract，expert 路径可显式声明 `full/shard` 或 `shard/shard`
- `gemm_allgather(...)`：`A(M, K)` full、`B(K, N / world_size)` shard、`C(M, N)` full
- `gemm_reducescatter(...)`：`A(M, K)` 本 rank contribution、`B(K, N)` full、`C(M, N / world_size)` shard
- `allgather(...)`、`allreduce(...)`、`reduce_scatter(...)`：都已经有稳定高层 host contract

这几类入口现在都走 `build_*_plan(...) -> plan.execute(...)` 主链，而不是继续依赖调用方手工拼 `pattern.execute(...)`。当前真正保守处理的边界，不是“有没有 API”，而是 multiprocess 正式支持面仍只收口到 `world_size=2 + ctypes_ipc`。

---

## 第 2 层：Pattern 层（以 FusedSequential 为例）

### 实际源码：显式 execution contract

当前 pattern 层的关键变化不是 kernel 花样，而是 `xtile/patterns/contracts.py` 先把逻辑 shape/layout 解释清楚，再交给 pattern 执行。下面是当前真实源码摘录：

```python
@dataclass(frozen=True, slots=True)
class PatternExecutionSpec:
    """Canonical execution contract consumed by all overlap patterns."""

    M: int
    K: int
    full_N: int
    local_N: int
    rank: int
    world_size: int
    rhs: PatternTensorSpec
    output: PatternTensorSpec
    scatter_src_col_offset: int
    scatter_cols: int
    scatter_dst_leading_dim: int
    scatter_dst_col_offset: int
```

```python
if resolved_c_layout == "full":
    scatter_cols = resolved_full_N if world_size == 1 else shard_N
    scatter_src_col_offset = 0 if world_size == 1 else rank * shard_N
    scatter_dst_col_offset = 0 if world_size == 1 else rank * shard_N
    scatter_dst_leading_dim = resolved_full_N
else:
    scatter_cols = c_cols
    scatter_src_col_offset = 0
    scatter_dst_col_offset = 0
    scatter_dst_leading_dim = c_cols

return PatternExecutionSpec(
    M=M,
    K=K,
    full_N=resolved_full_N,
    local_N=b_cols,
    rank=rank,
    world_size=world_size,
    rhs=rhs_spec,
    output=output_spec,
    scatter_src_col_offset=scatter_src_col_offset,
    scatter_cols=scatter_cols,
    scatter_dst_leading_dim=scatter_dst_leading_dim,
    scatter_dst_col_offset=scatter_dst_col_offset,
)
```

### 实际源码：FusedSequentialPattern.execute

```python
def execute(
    self,
    A: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
    **kwargs: Any,
) -> None:
    """Run the fused GEMM + scatter kernel.

    Args:
        A: Input matrix of shape ``(M, K)``.
        B: Input matrix of shape ``(K, N)``.
        C: Output matrix of shape ``(M, N_local)`` -- written locally
           and simultaneously scattered to peers.
    """
    import torch

    spec = self.resolve_execution(
        A,
        B,
        C,
        spec=kwargs.get("spec"),
        full_N=kwargs.get("full_N"),
        b_layout=kwargs.get("b_layout"),
        c_layout=kwargs.get("c_layout"),
        storage_kind=kwargs.get("storage_kind", "symmetric"),
    )
    M, K = A.shape
    N = spec.local_N

    num_sms = self.NUM_SMS or self.ctx.backend.get_device_properties().compute_units

    num_tiles_m = triton.cdiv(M, self.BLOCK_M)
    num_tiles_n = triton.cdiv(N, self.BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    world_size = self.ctx.world_size
    heap_bases = self.ctx.heap_bases

    grid = (min(num_sms, total_tiles),)
    EVEN_K = (K % self.BLOCK_K == 0)
    self._fused_kernel[grid](
        A, B, C,
        heap_bases,
        M, N, K,
        spec.scatter_src_col_offset,
        spec.scatter_cols,
        spec.scatter_dst_leading_dim,
        spec.scatter_dst_col_offset,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        self.ctx.rank,
        world_size,
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        BLOCK_M=self.BLOCK_M,
        BLOCK_N=self.BLOCK_N,
        BLOCK_K=self.BLOCK_K,
        NUM_SMS=num_sms,
        EVEN_K=EVEN_K,
        num_warps=4,
        num_stages=4,
    )
```

### 实际源码：_fused_kernel 的关键片段

```python
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

### 当前实现要点

- Persistent kernel + round-robin 调度已经落地
- 每算完一个 tile 就立刻进入 scatter
- `scatter_tile_to_peer` 使用 `translate_ptr` + `tl.store`
- `heap_bases` 作为 kernel 参数直接传入
- full/shard 语义已经提升成显式 host-side contract，不再依赖 pattern 内部猜 `B.shape[1]`
- 当前仍未完全收敛的，是默认 public API 是否全部走 `xtile.ops.*`，以及是否继续保留多种输出 layout 模型

---

## 第 3 层：通信底层

### 实际 scatter_tile_to_peer 实现

```python
# xtile/patterns/_helpers.py

@triton.jit
def scatter_tile_to_peer(
    C_ptr, tile_data, offs_m, offs_n,
    rank, peer, heap_bases,
    src_col_offset, valid_cols,
    dst_leading_dim, dst_col_offset,
    mask,
    CACHE_MODIFIER: tl.constexpr = ".wt",  # 默认 write-through
):
    # 步骤 1：翻译指针到 peer 的地址空间
    remote_C = translate_ptr(C_ptr, rank, peer, heap_bases)

    # 步骤 2：根据显式 execution contract 计算源/目的列偏移
    col_mask = (offs_n >= src_col_offset) & (offs_n < src_col_offset + valid_cols)
    safe_local_cols = tl.where(col_mask, offs_n - src_col_offset, 0)
    offsets = offs_m[:, None] * dst_leading_dim + (dst_col_offset + safe_local_cols[None, :])
    final_mask = mask & col_mask[None, :]

    # 步骤 3：写入远端内存
    if CACHE_MODIFIER == ".wt":
        tl.store(remote_C + offsets, tile_data, mask=final_mask, cache_modifier=".wt")
    else:
        tl.store(remote_C + offsets, tile_data, mask=final_mask)
```

### 实际 translate_ptr 实现

```python
# xtile/memory/translation.py

@triton.jit
def translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT: tl.constexpr = 0):
    # 完全匹配 Iris Listing 1 的 5 条指令
    from_base = tl.load(heap_bases + from_rank).to(tl.uint64)
    to_base = tl.load(heap_bases + to_rank).to(tl.uint64)
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_byte = to_base_byte + offset
    translated = tl.cast(translated_byte, ptr.dtype)
    if HINT > 0:
        translated = tl.max_contiguous(tl.multiple_of(translated, HINT), HINT)
    return translated
```

### 当前实现要点

`translate_ptr + tl.store` 这条核心路径已经成立，编译器也确实全可见；现在 scatter helper 已经不再把 full-buffer / shard-buffer 语义写死在 device helper 里，而是显式消费 host-side contract 传下来的 offset / leading-dim 元数据。

当前还额外做了一个直接的工程优化：默认使用 `.wt` (write-through) cache modifier 减少 L2 污染。

---

## 第 4 层：内存建立

### 模式 A：单进程 — SymmetricHeap.create_all（推荐）

```python
# xtile/memory/symmetric_heap.py

@classmethod
def create_all(cls, size, world_size, backend):
    """单进程模式：一个进程管理所有 GPU。"""
    # 步骤 1：在每个 GPU 上分配连续内存
    buffers = []
    for rank in range(world_size):
        torch.cuda.set_device(rank)
        buf = torch.empty(size, dtype=torch.uint8, device=f"cuda:{rank}")
        buffers.append(buf)

    # 步骤 2：启用所有 GPU 对之间的 peer access
    for i in range(world_size):
        torch.cuda.set_device(i)
        for j in range(world_size):
            if i != j:
                backend.enable_peer_access(j)

    # 步骤 3：构建 heap_bases（所有 GPU 的堆基址）
    base_ptrs = [buf.data_ptr() for buf in buffers]
    heaps = []
    for rank in range(world_size):
        torch.cuda.set_device(rank)
        heap_bases = torch.tensor(base_ptrs, dtype=torch.int64, device=f"cuda:{rank}")
        heap = cls.__new__(cls)
        heap._buffer = buffers[rank]
        heap._heap_bases = heap_bases
        heap._rank = rank
        heap._world_size = world_size
        # ... 其他初始化
        heaps.append(heap)

    return heaps
```

### 模式 B：多进程 — _setup_multiprocess（device-safe auto path + forced diagnostics）

```python
# xtile/memory/symmetric_heap.py

def _setup_multiprocess(self):
    """Exchange heap pointers across ranks for multi-process mode.

    Auto mode now tries only device-safe transports. Other strategies are
    kept for forced diagnostics, not normal runtime fallback.
    """
    self._backend.init_ipc()
    forced_strategy = forced_multiprocess_transport()
    strategies = (
        [forced_strategy]
        if forced_strategy is not None
        else [
            "ctypes_ipc",
        ]
    )
    errors = []

    for strategy in strategies:
        try:
            if strategy == "ctypes_ipc":
                self._setup_multiprocess_ctypes_ipc()
            elif strategy == "pytorch_ipc":
                self._setup_multiprocess_pytorch_ipc()
            else:
                self._setup_multiprocess_peer_access_pointer_exchange()
            return
        except Exception as exc:
            errors.append((strategy, exc))
            if forced_strategy is not None:
                break

    raise RuntimeError(f"All multiprocess transport strategies failed: {errors}")
```

### IPC 诊断实测结果

跨进程测试（`torch.multiprocessing.spawn`，2× H100，`ptrace_scope=1`）当前真实状态如下：

| transport / 场景 | host-side IPC bring-up | 最小 Triton remote load/store | multiprocess reduce_scatter(device) | 当前定位 |
|------|------|------|------|------|
| `ctypes_ipc` | ✅ PASS | ✅ PASS | ✅ PASS | 当前唯一通过真实 device-side 矩阵的 multiprocess transport |
| `pytorch_ipc` | ✅ PASS | ❌ FAIL (`CUDA illegal memory access`) | ❌ FAIL (`CUDA illegal memory access`) | 只能保留为 forced diagnostic / host-side bring-up 对照 |
| `peer_access_pointer_exchange` | ❌ FAIL (`cudaMemcpy err=1`) | ❌ FAIL (`CUDA illegal memory access`) | ❌ FAIL | 只能保留为 forced diagnostic，不是有效 public transport |

真实命令与结果：

- `python -m tests.test_e2e._run_ipc_test`
  - 结果：`ALL MULTI-GPU IPC TESTS PASSED`
- `XTILE_FORCE_MULTIPROCESS_TRANSPORT=pytorch_ipc python -u -m tests.test_e2e._run_ipc_test`
  - 结果：`ALL MULTI-GPU IPC TESTS PASSED`
- `XTILE_FORCE_MULTIPROCESS_TRANSPORT=peer_access_pointer_exchange python -u -m tests.test_e2e._run_ipc_test`
  - 结果：失败，`cudaMemcpy err=1`
- `PYTHONPATH=. python -u tests/benchmarks/bench_triton_remote_access_multiprocess.py --warmup 2 --iters 5 --timeout-sec 60 --output-json docs/generated/triton_remote_access_multiprocess_matrix.json`
  - 结果：`12` 个 case 中 `6` 个通过、`6` 个失败；只有 `auto/ctypes_ipc` 在 `fp16/bf16/fp32` 上通过最小 Triton remote load/store 矩阵
- `XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 python -m tests.test_e2e._run_reduce_scatter_multiprocess`
  - 结果：`rank0=4.0 / rank1=6.0`，primitive 与高层 API 都与期望一致
- `python -u -m tests.test_e2e._run_triton_remote_access_multiprocess --dtype float32 --force-transport pytorch_ipc --operation both`
  - 结果：失败，`CUDA error: an illegal memory access was encountered`
- `python -u -m tests.test_e2e._run_triton_remote_access_multiprocess --dtype float32 --force-transport peer_access_pointer_exchange --operation both`
  - 结果：失败，`CUDA error: an illegal memory access was encountered`

这说明当前问题已经不再只是“heap / IPC bring-up 会不会崩”，而是更精确的：
- `pytorch_ipc` 在当前机器上 **host-side 可读**
- 但它 **不是 Triton device-side remote dereference 可用 transport**
- 所以它不能继续作为 XTile multiprocess device path 的自动 fallback

### 当前状态与边界

| 主题 | 当前状态 | 当前边界 |
|------|------|------|
| IPC 调用约定 | `ctypes` Structure by-value 修复已落地 | 调用约定已修正 |
| `ptrace_scope=1` 环境 | 当前真实环境下 `ctypes_ipc` 已复测通过 | 当前问题已不在 host-side bring-up，而在 device-side remote dereference 能否成立 |
| 多进程 auto path | 已收窄为 `ctypes_ipc`；`pytorch_ipc` / `peer_access_pointer_exchange` 仅保留 forced diagnostics | multiprocess 正式支持面仍偏窄 |
| 跨节点 IPC | 当前不支持 | 后续需 UCX / GDR |

**与 Iris 的区别**：不能再简单写成“Iris 仅使用 HIP IPC 单一路径”。当前 Iris 的主实现已经演进到 allocator + fd passing + DMA-BUF 映射；XTile 的特点也不应再表述成“三条 transport 同等可用”。更准确的表述是：XTile 显式实现了 `ctypes_ipc` / `pytorch_ipc` / `peer_access_pointer_exchange` 三条 bring-up 策略，并且已经通过真实矩阵确认只有 `ctypes_ipc` 可作为当前 multiprocess device path 的 auto transport。

### Iris 与 XTile 的实质差异、优劣与对齐方向

这里需要把“都能建立 symmetric memory”拆开看，二者今天其实代表两种不同风格：

| 维度 | Iris | XTile |
|------|------|------|
| 主抽象 | `SymmetricHeap` + allocator backend | `SymmetricHeap` + mode/fallback |
| 内存来源 | `TorchAllocator` / `VMemAllocator` 可切换 | 当前主要是 `torch.empty(uint8)` bump allocator |
| 远端映射主路径 | FD passing + DMA-BUF export/import + VA map/access | auto=`ctypes_ipc`；`pytorch_ipc` / `peer_access_pointer_exchange` 仅保留 forced diagnostics |
| 地址空间观念 | 更像“统一保留虚拟地址区间，再把 peer segment 映射进去” | 更像“先拿到能访问的远端 base，再暴露给 Triton 做 translate_ptr” |
| 多进程一致性 | 更强，路径更像 canonical import/map | 更务实，优先保证当前环境能跑起来 |
| 单进程 bring-up | 不如 XTile 直接 | 很直接，`create_all()` 非常适合单节点 benchmark |
| allocator 可扩展性 | 更强 | 目前较弱 |
| NVIDIA 当前可用性 | 本仓库语境下不是主战场 | 当前更强，尤其在 H100 bring-up 上更直接可用 |

**差异本质**：

1. **Iris 更像“一个 canonical 的内存映射体系”**
   - allocator 是一等对象
   - FD / DMA-BUF / map/access 是主路径的一部分
   - 更适合把“跨进程、跨设备、连续 VA 语义”长期做深

2. **XTile 更像“一个务实可用的 bring-up 体系”**
   - 同一接口下先把 transport 分层实现清楚
   - auto path 只保留当前真实通过 device-side 矩阵的 `ctypes_ipc`
   - 其他 transport 继续保留为 forced diagnostics / bring-up 对照
   - 对当前 NVIDIA bring-up、实验收口和风险控制更友好

**适用性判断**

- 如果问“**长期架构形态**”，Iris 这一套 allocator + import/map 更好，原因是语义更统一、可扩展性更强、未来更容易把多进程 / 多节点 / 多 allocator 做成同一个 canonical path。
- 如果问“**当前这台 H100 机器上谁更实用**”，XTile 现在更合适，因为它已经覆盖了当前环境里的关键约束，尤其是在 `ptrace_scope=1` 这类限制下仍能跑通。

**XTile 应该如何向 Iris 对齐，而不是简单照抄？**

总纲不变，但详细动作已经并入下文“当前差距与下一步计划”。核心原则只有三条：

1. 保留 XTile 当前“多 transport 显式实现 + 受控强制诊断”的工程优势，但 auto path 继续只走已验证的 device-safe transport。
2. 吸收 Iris 的 allocator-first / export-import-map canonical path，把 allocator、segment metadata、FD/DMA-BUF 映射能力做成底层后端抽象。
3. 对上统一为稳定 public contract：逻辑 shape、layout metadata、heap ownership、external import 语义不能再跟具体 fallback 路径绑死。

### 当前底座已落地的部分

当前第 4 层已经不再只是一个“能分配对称堆、能交换 base pointer”的 bring-up 原型，而是进入了 allocator-first 的第一阶段：

- `xtile.memory.allocators` 已建立 allocator backend 边界，默认 backend 为 `torch_bump`
- `SymmetricHeap` 已公开 segment metadata、peer export/import metadata、segment-scoped peer catalog
- `heap_bases` 已从 primary-segment import catalog 派生，而不是继续依赖平铺记录的隐式顺序
- `import_external_tensor(...)` / `as_symmetric(...)` 与 `XTileContext.as_symmetric(...)` 已接入，当前 external import 语义明确是 copy-based
- `xtile.describe_runtime_support(ctx)` / `ctx.support_matrix()` / `xtile support --json` 已形成统一状态出口
- `figures/data/*.json`、`scripts/plot_figures.py`、`docs/generated/benchmark_runtime_summary.md` 已开始共享同一套结构化数据源

这部分要写准确：它说明 XTile 的 allocator-first v1 已经落地，但**不等于** canonical `allocator + export/import-map/access` backend 已经完成。

---

## 第 5 层：硬件执行

```
  GPU0 上的 warp 执行 tl.store(translated_ptr, tile_data, cache_modifier=".wt")
    │
    ▼
translated_ptr 指向 GPU1 的 HBM 地址
    │
    ▼
H100 NVLink 硬件自动路由：
  GPU0 SM → tl.store → L2 miss → NVLink (NV12, 300 GB/s) → GPU1 HBM

P2P benchmark 最新 canonical serial rerun 显示 best read ≈ 248.74 GB/s、best write ≈ 248.43 GB/s；scatter 更相关的是写带宽，量级约 82.8%–82.9% 峰值。
```

### 当前硬件与性能边界

| 主题 | 当前状态 | 当前边界 |
|------|------|------|
| 硬件验证范围 | 仅在 NVIDIA H100 上实测 | AMD 待硬件验证，代码路径已就绪 |
| P2P 带宽 | 当前约 82.7%–82.9% 峰值 | 距离更高目标仍受 NVLink 协议开销与 Triton/PTX 路径限制 |

---

## 附加：Auto-Select 引擎

### 实际实现

```python
# xtile/patterns/auto_select.py

def auto_select(op, M, N, K, world_size, ctx=None):
    sm_count, bw = _detect_hardware_info()
    bw_scale = bw / 300.0

    n_per_rank = N // world_size
    total_tiles_128 = ((M + 127) // 128) * ((n_per_rank + 127) // 128)
    flops = 2 * M * n_per_rank * K
    scatter_bytes = M * n_per_rank * 2
    compute_intensity = flops / max(scatter_bytes, 1)

    if M < 256:
        return BulkSyncPattern
    elif n_per_rank < 1024 and K > int(12288 * bw_scale):
        return FusedSequentialPattern
    elif n_per_rank < 2048 and K > int(6144 * bw_scale):
        return ProducerConsumerPattern
    elif total_tiles_128 >= sm_count and K > int(4096 * bw_scale):
        return WGSpecializedPattern
    elif N > 4096 and K > int(8192 * bw_scale):
        return WGSpecializedPattern
    elif compute_intensity > 256 and total_tiles_128 > 16:
        return FusedSequentialPattern
    else:
        return BulkSyncPattern
```

### 当前实现特征

当前实现需要明确补两点：
- `auto_select(...)` 当前识别的 op 名包括 `gemm_allscatter`、`gemm_allgather`、`gemm_reducescatter`，不是 `gemm_scatter`；其中 `gemm_reducescatter(...)` 现已具备稳定 host-side public contract，但当前并不走 Triton fused pattern auto-select，而是“local GEMM materialize + packed reduce_scatter”主链。
- 对本例 `N=4608, world_size=4`，`n_per_rank = 1152`，不会命中 `fused_sequential` 的 `< 1024` 分支，而会更接近 `producer_consumer` 分支。
- 当前实现还带 `compute_intensity` 与 `N > 4096` 的补充分支，不只依赖少量固定规则。

---

## 当前差距与下一步计划

前文已经把当前用户层、pattern contract、通信 helper、对称堆建立方式和硬件执行主链直接展示出来。本节不再重复“已实现清单”，只保留当前仍未收口的差距与后续优先级。

### 当前主要差距

#### 1. canonical memory backend 仍未完成

当前 XTile 已经进入 allocator-first v1，但离 Iris 风格的 canonical `allocator + export/import-map/access` 统一底座还有实质差距。

当前缺口主要是：

- 还没有把 export / import / map / access 收成单一状态机
- external mapping 仍未实现，`fd passing + DMA-BUF` 仍缺位
- segmented import-map 还没有形成正式主路径

所以当前准确口径应是：**allocator-first 第一阶段已落地，但 canonical backend 仍未完成。**

#### 2. 高层公共语义还要继续收口

当前 public host-side op 家族已经建立，但还没有完全收成“普通用户只接触稳定 contract，expert 才接触内部细节”的最终形态。

当前仍需继续处理的点是：

- `gemm_allscatter.shard/full` 仍应保持 unsupported；这不是少一个 wrapper，而是当前并不存在稳定 local-shard basis
- `pattern.execute(...)` 虽然已不再是推荐主入口，但在仓库中仍偏显眼，后续还要继续退到 expert/internal surface
- public contract 命名、benchmark contract spelling 与文档术语还需要继续统一，避免消费层分叉重新出现

#### 3. multiprocess 正式支持面仍然偏窄

当前真正可以保守写成正式支持面的，是：

- `world_size=2`
- `transport_strategy=ctypes_ipc`
- 代表性 dtype / shape 下的 correctness 与结构化 benchmark

当前还不能写成“已支持”的，是：

- `pytorch_ipc`
- `peer_access_pointer_exchange`
- `world_size > 2`
- 更大 shape 与更长时间 stress
- 更明确的 performance contract

所以 multiprocess 相关 public 口径仍必须保持 conservative；当前问题不是“完全不能用”，而是“正式支持面还太窄”。

#### 4. collective 的生产级验证与性能闭环仍不足

当前 collective 层已经有稳定高层 contract，但离“成熟通信库”的标准还差两层闭环：

- **支持面闭环**：当前 public surface 主要还是 `world_size=2 + ctypes_ipc`
- **性能闭环**：P2P 仍约 `82.8% - 83.0%` 峰值，`8192^3` GEMM 仍约 `83.0% - 83.5%` of cuBLAS，comm-only collective 与 NCCL 也仍有明显差距

所以当前准确说法只能是：**功能基础已建立，但性能、stress、长稳和更大支持面还没有闭环。**

### 对 Iris 的差距归纳

与 Iris 的实质差异前文已经在第 4 层展开；这里仅保留对当前工程差距的收口判断：

- Iris 更接近 canonical substrate，优势在 allocator-first、fd passing、DMA-BUF、统一 import/map 语义
- XTile 当前优势在显式 contract、显式 support matrix、显式 feature gate 与诊断透明度
- XTile 不应简单照抄 Iris；正确方向是保留现在这套可观测和可诊断的工程外壳，同时把底层逐步收口到 canonical allocator/import-map/access

### 下一步优先级

1. **P0-next：完成 canonical allocator/import-map/access 底座**
   - 把当前 allocator-first v1 继续推进成统一 export / import / map / access 运行时
   - 为后续 `fd passing + DMA-BUF` 和更稳定的 multiprocess substrate 打底

2. **P1：继续收紧公共语义面**
   - 让默认用户路径继续收敛到 `xtile.ops.*`
   - 进一步弱化直接 `pattern.execute(...)` 的 public 可见度
   - 持续统一 contract 命名与文档术语

3. **P2：扩大 multiprocess 的真实证据面**
   - 在不放松标准的前提下扩到 `world_size > 2`
   - 补更大 shape、更多 dtype、更多 stress
   - 只有通过真实矩阵的 transport 才能进入正式支持面

4. **P3：建立更严格的性能与回归门禁**
   - 固定 canonical headline、环境口径、出图脚本和阈值
   - 在当前 public surface 之上继续推进 P2P、GEMM 与 collective 的 perf regression

5. **P4：最后再做跨节点与跨平台扩展**
   - UCX / GDR
   - AMD 真机验证

当前更稳妥的结论是：XTile 已经具备继续工程化推进的基础，但距离理想通信库仍差 canonical substrate、broader support surface，以及完整的性能与生产级验证闭环。

## 与 NCCL 的差距归因及未来方向

这一章单独回答一个更尖锐、也更实际的问题：**如果把 XTile 当成“独立通信库”来衡量，它和 NCCL 现在到底差在哪里，下一步应当怎么补。**

### 当前对比口径

当前结论基于最新 communication-only 结构化 benchmark，而不是主观印象：

- 数据文件：`figures/data/collective_comm_only_latest.json`
- 图文件：`figures/fig6_collective_comm_only.png`
- 生成时间：`2026-03-23T11:52:00+00:00`
- 机器环境：`GPU SKU = 2 x NVIDIA H100 PCIe`，`GPU-GPU interconnect = NVLink (NV12)`
- runtime 口径：`world_size=2`、`dtype=float32`、`transport=ctypes_ipc`
- 消息尺寸：`4 KiB`、`64 KiB`
- benchmark 口径：对比当前**稳定 public primitive surface**，而不是只测内部某个实验性 raw kernel

这里必须特别说明：当前这组数据代表的是“**今天真实可部署、真实可回归的 XTile collective 层**”，不是“理论上如果把内部 fast path 全部写满后可能达到的上限”。

### 当前结果

在当前 compact canonical run 中，5 个 communication-only collective 的最佳点如下：

| collective | 最佳消息尺寸 | XTile | NCCL | XTile / NCCL |
|------|------|------|------|------|
| `allreduce` | `64 KiB` | `0.16 GB/s` | `1.41 GB/s` | `0.11x` |
| `allgather` | `64 KiB` | `0.26 GB/s` | `0.52 GB/s` | `0.49x` |
| `scatter` | `64 KiB` | `0.67 GB/s` | `0.85 GB/s` | `0.78x` |
| `reduce_scatter` | `64 KiB` | `0.60 GB/s` | `1.30 GB/s` | `0.47x` |
| `broadcast` | `64 KiB` | `0.70 GB/s` | `1.37 GB/s` | `0.51x` |

这组结果说明：

- **XTile 现在最弱的是 `allreduce`。**
- `scatter` 相对最好，说明简单 direct-push / root-push 形态已经有一定基础。
- `allgather`、`reduce_scatter`、`broadcast` 都明显落后，说明“collective 架构层”还没有真正收口成高性能实现。

### 补充内部口径：XTile vs `bulk_sync`

如果当前目标不是和 NCCL 正面对标，而是向项目申请材料说明“XTile 的 collective-specific kernel 相比 naive internal baseline 到底带来了什么”，那么更合适的补充口径是：

- 数据文件：`figures/data/collective_bulk_sync_latest.json`
- 图文件：`figures/fig7_collective_bulk_sync.png`
- 生成时间：`2026-03-23T13:30:30+00:00`
- 机器环境：`GPU SKU = 2 x NVIDIA H100 PCIe`，`GPU-GPU interconnect = NVLink (NV12)`
- runtime 口径：`world_size=2`、`dtype=float32`、`transport=peer_access`
- baseline 定义：`bulk_sync = single-process peer_access 下，由更低级 point-to-point copy + 显式阶段同步 拼成的 host-orchestrated internal baseline`

这张图的用途要说清楚：

- 它**不是**“XTile 已经超过外部成熟通信库”的证据。
- 它是“在 XTile 自己的通信栈内部，collective-specific kernel 相对 naive bulk_sync baseline 的结构性收益”。
- 这更适合说明：XTile 已经不只是“能通信”，而是开始具备“专用 collective kernel 的体系化收益”。

当前 best speedup 如下：

| collective | best speedup vs `bulk_sync` | best size |
|------|------|------|
| `allreduce` | `2.05x` | `4 KiB` |
| `allgather` | `1.78x` | `16 KiB` |
| `scatter` | `1.00x` | `64 KiB` |
| `reduce_scatter` | `2.22x` | `64 KiB` |
| `broadcast` | `0.65x` | `16 KiB` |

### XTile / bulk_sync / NCCL 端到端执行流程对比

| 阶段 | `XTile` | `bulk_sync` | `NCCL` |
|------|------|------|------|
| 公开抽象 | tile-level collective | tile-level bulk-synchronous composition | tensor-level collective |
| 输入单元 | 每个 rank 提供一个或多个 tile / shard | 每个 rank 提供一个或多个 tile / shard | 每个 rank 提供一个 tensor / tensor shard |
| 数据布局前提 | symmetric heap 上的对称分配，跨 rank 偏移一致 | symmetric heap 上的对称分配，跨 rank 偏移一致 | 常规 device tensor，由 NCCL 直接接管 |
| 地址语义 | 先建立 `translate_ptr` 语义，再在设备侧访问 peer 对应地址 | 使用 mirror pointer 确定目标槽位，再做 peer copy | 不暴露远端地址翻译，地址与传输由 NCCL 内部管理 |
| 调度入口 | XTile collective primitive / launcher | host 侧 phase-by-phase 编排 | NCCL collective API |
| 执行粒度 | tile granularity | tile granularity | tensor granularity |
| 核心执行方式 | 设备侧 collective kernel 直接完成 gather / scatter / reduce / broadcast 语义 | 将 collective 拆成低级 point-to-point copy、局部归约、阶段同步后逐步拼装 | 由 NCCL 内部 collective engine 完成 tensor 级传输、归约、分发 |
| 数据移动方式 | tile 直接读写本地与 peer 对应 tile 槽位 | tile 在各 phase 中显式 copy 到 stage / dst 槽位 | tensor / chunk 在 NCCL 内部通道中流动 |
| 归约发生位置 | 设备侧 collective 路径内部完成 | host 编排下由各 rank 本地显式 `add`/reduce 完成 | NCCL 内部完成 |
| 同步方式 | collective 路径内部同步，必要时由 host barrier 收口 | 每个 phase 结束后由 host 显式同步 | 由 NCCL collective 语义与内部调度收口 |
| `allreduce` 形态 | 当前实现等价于 `reduce_scatter -> allgather` 收口 | 先朴素 reduce-scatter，再朴素 allgather | 单个 tensor collective，由 NCCL 内部实现 |
| 输出形态 | 结果写回目标 tile / shard / gathered buffer | 结果逐 phase 落到目标 tile / shard / gathered buffer | 结果写回目标 tensor / tensor shard |
| 工程控制面 | kernel、heap、pointer translation、runtime glue 都在 XTile 内部 | phase、copy、stage buffer、同步都在 XTile host 侧显式控制 | 算法、协议、通道、调度都在 NCCL 内部 |

这组内部对比更准确地说明了当前状态：

- `allgather` 和 `reduce_scatter` 已经能稳定体现出 **collective-specific kernel** 相对 bulk-synchronous 朴素拼装的收益。
- `allreduce` 也能在较小消息上体现优势，但大消息仍不稳定，说明它的 public path 仍然过厚。
- `scatter` 基本接近持平，说明 root-push 场景里当前 kernel 还没有明显赢过简单 bulk copy 组合。
- `broadcast` 仍落后于 bulk baseline，说明 flat root-push 版本还没有形成真正值得公开宣称的高性能形态。

因此，对外口径应当拆开：

- 如果是说明“XTile 相比成熟通信库还有多远”，看 `fig6_collective_comm_only.png`。
- 如果是说明“XTile 当前专用 collective kernel 已经比 naive internal baseline 好在哪里”，看 `fig7_collective_bulk_sync.png`。

### 差距归因

#### 1. `allreduce` 已经完成单一路径收口，但仍不是成熟高性能 collective

当前 `xtile.primitives.collectives.allreduce(...)` 的默认 public path 已经不再是 `reduce_scatter + allgather + copy` 的 host-side 组合，而是显式 resolve execution spec 后直接进入 staged device path：

```python
def allreduce(
    tensor: torch.Tensor,
    heap: "xtile.memory.SymmetricHeap",
    op: str = "sum",
) -> None:
    spec = resolve_allreduce_execution(tensor, heap=heap, op=op)
    workspace = _allreduce_workspace(tensor, heap=heap, spec=spec)
    _allreduce_staged_kernel[(spec.grid_size,)](
        tensor,
        workspace.staging,
        workspace.published_epoch,
        workspace.consumed_count,
        heap.get_heap_bases(),
        ...,
    )
```

这条新路径已经解决了之前最厚的三层问题：

- 默认 public `allreduce(...)` 不再依赖 `reduce_scatter + barrier + allgather + barrier + copy`
- host-side 热路径不再显式插两次 `heap.barrier()`
- 不再分配完整 `gathered` buffer，也不再在结尾额外 `tensor.copy_(...)`
- `AllReducePlan` 现在已经显式携带 `implementation / protocol / chunk_elems / num_chunks / workspace_bytes`

但这还不能等价于“已经达到 NCCL 级 allreduce”，因为当前实现虽然已经是**slot-based 的 staged multi-CTA path**，但仍然偏保守：

- 设备侧协议目前是 `slot_epoch_pipeline`
- 已经具备 `pipeline_slots / grid_size / num_warps` 这类 launch metadata
- 当前主要验证面仍然是 `world_size=2 + ctypes_ipc`
- 还没有 pipelined forwarding、多 channel 并发、message-regime protocol 分层

所以更准确的说法应当是：**方向 1 的“单路径 public allreduce 收口”在当前 public surface 上已经完成，但 `allreduce` 与 NCCL 的性能差距仍然存在，而且原因已经从“host-side 厚组合路径”转移到“协议与并发形态还不够强”。**

#### 2. `reduce_scatter` 当前明确是 correctness-first，而不是 performance-first

`tile_reduce_scatter(...)` 当前真实实现采用的是“每个 rank 直接读取所有 peer 的对应 chunk，然后本地归约写回自身”的保守路径。源码注释已经把设计意图写得很清楚：这是一个**correctness-first** 的 multiprocess/device path。

这意味着它当前缺少：

- pipelined ring
- 分块流式 forwarding
- channel 化并发
- double buffer
- 更成熟的 cross-rank staging / synchronization

所以当前它虽然已经能作为 stable public surface 的基础，但还不能拿来和 NCCL 的成熟 reduce-scatter 内核正面对打。

#### 3. `allgather` / `scatter` / `broadcast` 仍然是朴素 flat collective

当前几类 device primitive 的风格都比较直接：

- `tile_allgather(...)`：每个 rank 直接把自己的 tile 写到所有 peer 的目标位置
- `tile_scatter(...)`：只有 root 干活，逐个 target 推送
- `tile_broadcast(...)`：当前仍是 flat root push

这几类实现的优点是：

- 逻辑直白
- 容易验证
- 对 `translate_ptr + tl.store` 路径暴露充分

但它们缺少 NCCL 那种成熟 collective 的关键要素：

- 拓扑感知
- 多 CTA / 多 channel 并发
- 分块流水化
- 大消息 protocol 分层
- 更细粒度的 launch / sync 设计

因此当前只能说“功能成立”，不能说“性能形态已经成立”。

#### 4. 底层内存模型仍然偏 fallback-first，而不是 canonical collective substrate

当前 multiprocess device path 的真实 auto transport 只有 `ctypes_ipc`；`pytorch_ipc` 与 `peer_access_pointer_exchange` 仍然只适合作为 forced diagnostics。

这带来的影响不是只有“transport 少两个”这么简单，而是：

- collective 层目前还建立在 transport-aware bring-up 之上
- 还没有真正落到 allocator / export / import / map / access 统一底座
- 还没有形成一个与具体 bring-up 细节解耦的 canonical communication substrate

这也是 XTile 与 NCCL、以及与 Iris 当前更偏 allocator-first / import-map 的路线之间的核心差距之一。

#### 5. 当前 benchmark 已经放大了小消息延迟问题，但这不是主要借口

当前 communication-only canonical run 是 compact sweep，只覆盖 `4 KiB` 和 `64 KiB` 两个点。这会放大小消息上的 launch / barrier / host orchestration 成本。

但这不能被用来掩盖核心问题，因为：

- NCCL 是在**同一台机器、同一组 GPU、同一轮 benchmark 口径**下对比出来的
- 即使只看 `64 KiB` 这个当前 best point，XTile 的 `allreduce` 仍然只有 `0.11x NCCL`
- 说明主问题不是“图选得不巧”，而是 collective 路径本身还比较原型化

更准确的判断应当是：**小消息 sweep 放大了问题，但没有制造问题。**

### 当前判断

如果只从“XTile 是不是已经是一套接近 NCCL 的通信库”来问，当前答案必须是：

- **还不是。**

更细一点地说：

- XTile 的 **P2P substrate** 与 **GEMM 基础路径** 并不弱，至少已经形成了可继续工程化推进的底子。
- 真正明显落后的，是 **collective 层的架构收口程度**。
- 当前 communication-only layer 更接近“**正确性优先、可验证、可演进的原型级实现**”，还不是“成熟、高吞吐、可与 NCCL 同级竞争的工业级 collective runtime”。

所以文档口径也必须继续保持克制：

- 不能把当前 XTile 写成“已经达到 NCCL 级别”
- 也不能因为 collective 还弱，就把已经完成的 contract / benchmark / support matrix 基础说成“什么都没有”

### 未来方向

如果目标是逐步缩小和 NCCL 的差距，而不是只把功能继续堆上去，那么优先级应当非常明确。

#### 方向 1：先把 `allreduce` 重写成真正的单一路径 public collective（当前 public surface 已完成）

这是最高优先级。

这一步在当前 validated public surface 上已经完成：默认 public `allreduce(...)` 已收口成 staged multi-CTA device path，不再走 `reduce_scatter + allgather + copy` 组合主链。

已经落地的部分包括：

- 去掉 host-side `heap.barrier()` 热路径
- 去掉完整 `gathered` materialization / 结尾 `copy`
- 显式引入 `resolve_allreduce_execution(...)` 与 `AllReducePlan` execution metadata
- 把 public fast path 从 single-CTA 扩成 slot-based multi-CTA pipeline
- 把 benchmark JSON 里的 allreduce execution metadata 补到 `implementation / protocol / chunk_elems / num_chunks / pipeline_slots / grid_size / num_warps / workspace_bytes`
- 让 `xtile.ops.allreduce(...)` 和 `xtile.primitives.collectives.allreduce(...)` 统一到同一条 launcher 主链

当前这一方向剩余的工作，不再是“把 composed path 改掉”，而是把这条新主路径继续从保守版推进到高性能版：

- 继续扩大 `world_size > 2` 的真实验证面
- 补更细的 protocol 分层和更强的 launch geometry
- 在 structured benchmark 与 regression system 中长期跟踪这条 public path

因此，这一方向最难、也最关键的一步现在已经完成；后面不再是“补完方向 1”，而是“围绕新的单路径主干做性能化和扩大支持面”。下面保留这一路线拆解，作为后续继续推进这条主路径的细化目标。

1. **先固定 public 目标形态**
   - `xtile.ops.allreduce(...) -> build_allreduce_plan(...) -> plan.execute(...)` 这一层稳定入口可以保持不变。
   - 这一步现在已经完成：`xtile.primitives.collectives.allreduce(...)` 的默认 public path 已经切到 staged single-path launcher。
   - 后续要做的不是再换 API 入口，而是继续约束 fallback 角色，避免旧 composed path 重新回到默认 benchmark 口径。

2. **把执行 contract 从“只有 `block_size`”扩成真正的 allreduce plan**
   - 这一步也已经完成：`AllReducePlan` 现在已经显式保存 `implementation`、`protocol`、`chunk_elems`、`num_chunks`、`pipeline_slots`、`grid_size`、`num_warps`、`workspace_bytes` 等关键 execution metadata。
   - 下一步要继续补的是更细粒度的信息，例如 staging slot / epoch policy、grid / CTA family、message regime 选择。
   - 这一步的目的不是把 plan 做复杂，而是让后续性能回归能准确定位“到底是协议、chunking，还是 launch policy 在退化”。
   - 对应地，旧的 `_allreduce_workspaces(...)` 二元组已经退出默认主路径；现在主路径只保留协议真正需要的 staging / sync 空间。

3. **先落一个最小但真实的 device fast path**
   - 这一步也已经完成当前 public surface 版本：`op="sum"`、contiguous tensor、symmetric heap、`world_size=2` 的默认 public path 已经落成。
   - 但即使是这个最小版本，也应当做到三件关键事情：host 热路径不再显式插两次 `heap.barrier()`；不再分配完整 `gathered` buffer 并在结尾再 `tensor.copy_(...)`；workspace 不再重复 materialize 一份 full tensor。
   - 当前实现已经不是 single-CTA 原型，而是 slot-based 的 staged multi-CTA kernel；关键不在于“绝对只能一发 kernel”，而在于 public API 对外只呈现一个统一 fast path，而不是继续由 host 手工拼 collective。
   - 当前执行链已经接近下面这种形态：

    ```python
    def allreduce(tensor, heap, op="sum"):
        spec = resolve_allreduce_execution(tensor, heap=heap, op=op)
        _allreduce_fast_path[spec.grid](
            tensor,
            spec.workspace,
            spec.sync_state,
            heap.get_heap_bases(),
            ...
        )
    ```

4. **把完成标准写成明确门禁**
   - correctness 侧应当至少覆盖现有 `tests/test_e2e/_run_allreduce_multiprocess.py`、`tests/test_ops.py`、`tests/benchmarks/bench_collective_comm_only.py` 这三类入口，而且新路径要成为默认 public 路径，而不是只在一个实验 benchmark 里单独跑。
   - benchmark 侧不应只写“有提升”；更准确的阶段目标应当是：新路径先稳定超过当前 composed path，再稳定超过当前 internal `bulk_sync` baseline，然后才谈继续缩小和 NCCL 的差距。
   - 可观测性也要补上：structured benchmark JSON 最好能记录 `allreduce_impl`、`protocol`、`chunk regime`，否则后面即使性能有波动，也很难区分是 kernel 退化、同步退化，还是 workspace / launch policy 变了。

换句话说，方向 1 不是“再写一个 allreduce kernel”这么简单，而是要把 `allreduce` 从“public API 名字已经存在、但内部仍是组合实现”升级成真正的一等 collective。只有这一层先收口，后面方向 2 对 `reduce_scatter` / `allgather` 的优化，才会稳定沉淀到用户真正调用的主路径上。

#### 方向 2：把 `reduce_scatter` / `allgather` 做成 pipelined、chunked、多 CTA collective

这是第二优先级。

当前这两类操作实际上决定了后续 `allreduce`、`gemm_reducescatter`、`gemm_allgather` 的上限。下一阶段应当把它们从 correctness-first 原型推进到性能导向版本：

- pipelined ring / tree
- chunk / tile 级双缓冲
- 多 CTA 并发
- 更清晰的 rank-local / peer-local staging
- 消息尺寸分层，而不是一个 kernel 跑所有区间

#### 方向 3：把 `scatter` / `broadcast` 从 flat root-push 推到 topology-aware 版本

这两项相对当前最好，但不应被当前 `0.78x` 的 `scatter` 掩盖。

下一步应当补：

- tree / hierarchical broadcast
- 更明确的大消息 chunking
- root hot-spot 规避
- launch geometry 与 message regime 的匹配

目标不是只让它“更快一点”，而是让它们从“简单 direct push”进化成可维护的 collective family。

#### 方向 4：把底层 substrate 向 canonical memory model 收口

这一项和性能不是分开的，而是性能能否持续推进的前提。

正确方向不是简单照抄 NCCL 或 Iris，而是：

- 保留 XTile 当前 `support matrix + feature gate + diagnostics` 的工程透明度
- 同时把底层逐步收口到 allocator / export / import / map / access 的统一状态机
- 让 collective 层不再直接依赖“当前 transport 恰好怎么 bring-up 成功”

只有这样，后面做 `world_size > 2`、更多 transport、更多平台，才不会每往前走一步就重新撕开一次运行时语义。

#### 方向 5：把性能验证环境和制度补齐

当前 communication-only benchmark 已经说明问题，但它还不是完整 perf system。

下一步要补的是：

- 更安静、更独占的 benchmark 环境
- 更大的消息尺寸 sweep
- `world_size > 2`
- 固定 headline
- 固定 regression threshold
- 失败 case 自动归因

否则后面即使把 collective kernel 写快了，也很难稳定判断“到底是真的提升，还是环境噪声”。

### 对后续工作的直接结论

如果只看和 NCCL 的差距，当前最该避免的事情有两类：

1. **不要把精力继续分散在次要 transport bring-up 上**
   - `pytorch_ipc`、`peer_access_pointer_exchange` 可以继续保留为 diagnostics
   - 但当前不应把主要工程时间继续投在“让更多 transport 勉强可跑”
   - 真正决定对 NCCL 差距的，不是 transport 数量，而是 collective 架构

2. **不要用局部参数微调替代架构收口**
   - 现在不是缺少一个 `num_warps` 或 `num_stages` 调参
   - 而是缺少单路径 collective、pipelining、multi-CTA、canonical substrate 这些基础结构

因此，就“与 NCCL 的差距”这件事本身，当前最准确的路线图是：

1. 先重构 `allreduce`
2. 再重构 `reduce_scatter / allgather`
3. 然后推进 `scatter / broadcast` 的 topology-aware 版本
4. 同时把底层 substrate 向 canonical allocator/import-map/access 收口
5. 最后在更干净的环境里建立完整 perf regression system

这条路线比“继续堆功能”更难，但如果目标真的是长期可维护、工业级、专业严谨的通信库，这一条才是正确主线。
