# XTile 现状流程：GEMM + AllScatter 全栈代码流

> 文档定位
> - 本文基于当前仓库源码、测试与结构化实验结果整理。
> - 目标是说明当前实现、已验证范围、已知边界与后续工作。
> - 原始文档保留不动；本文件作为当前可维护版本持续更新。

## 任务定义

与设想文档相同：4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
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

### 与设想的差距

| 设想 | 现状 | 差距 |
|------|------|------|
| `xtile.fused_gemm_scatter(A, B, C, strategy="auto")` | `xtile.ops.gemm_allscatter(...)` 已可用 | 高层 API 已开始建立，但命名与最终 op 集合仍可继续收敛 |
| `ctx.randn(...)` 直接在堆上分配 | `XTileContext` 已支持 `empty/zeros/randn()` | 这一项已打通 |
| 自动检测后端 `backend="auto"` | `xtile.init()` 默认就是 `backend="auto"` | 这一项已集成 |
| `xtile.init()` 返回的 ctx 可直接执行 pattern | `xtile.init(..., heap=...)` / `heap_size=...` / `init_local(...)` 都可直接返回可运行 pattern 的真实 ctx | runtime ctx 主路径已统一 |
| benchmark / tests / CLI 与 runtime 入口一致 | 已统一到 `XTileContext` | 这一项已打通 |
| 高层 op API | `xtile.ops.gemm_allscatter(...)` / `xtile.ops.gemm_allgather(...)` / `xtile.ops.allgather(...)` / `xtile.ops.allreduce(...)` / `xtile.ops.reduce_scatter(...)` / `xtile.ops.gemm_reducescatter(...)` 已接入，并统一走显式 `plan` 主链 | 在当前 public multiprocess surface（`world_size=2 + ctypes_ipc`）上，`allgather / allreduce / reduce_scatter / gemm_allscatter / gemm_allgather / gemm_reducescatter` 已完成同级别 host contract、回归和结构化 benchmark 验收；当前差距已转为更大 world size、更多 transport、stress 与性能门禁 |
| full-shape correctness path 与 benchmark shard path | 两者都存在，但都已开始通过显式 contract + plan builder 收敛 | 仍需把更多调用点统一迁到高层 op API，并继续弱化直接 `pattern.execute(...)` 的 public 角色 |

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

### 与设想的对比

**一致**：
- Persistent kernel + round-robin 调度 ✅
- 每算完一个 tile 立刻 scatter ✅
- `scatter_tile_to_peer` 使用 `translate_ptr` + `tl.store` ✅
- `heap_bases` 作为 kernel 参数传入 ✅

**差异**：核心通信原理接近 Iris。当前 XTile 已经把 full/shard 语义提升成显式 host-side contract，而不是继续在 pattern 中隐式猜 `B.shape[1]`。现在真正还没完全收敛的，是“默认 public API 是否全部走 `xtile.ops.*`”以及“是否继续保留多种输出 layout 模型”。

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

### 与设想的对比

**基本一致**：`translate_ptr + tl.store` 这条核心路径成立，编译器也确实全可见；现在 scatter helper 已经不再把 full-buffer / shard-buffer 语义写死在 device helper 里，而是显式消费 host-side contract 传下来的 offset / leading-dim 元数据。
**额外优化**：默认使用 `.wt` (write-through) cache modifier 减少 L2 污染。

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

### 与设想的对比

| 设想 | 现状 | 状态 |
|------|------|------|
| cudaIpcGetMemHandle + Open | ctypes Structure by-value 修复 | ✅ 调用约定已修正 |
| IPC 在 ptrace_scope=1 下工作 | 当前真实环境下 `ctypes_ipc` 已复测通过 | ✅ 已解决 |
| 多进程模式（torchrun） | auto path 已收窄为 `ctypes_ipc`；`pytorch_ipc` / `peer_access_pointer_exchange` 仅保留 forced diagnostics | ✅ 已收口 |
| 跨节点 IPC | 不支持 | ⚠️ 需 UCX/GDR |

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

总纲不变，但详细动作已经并入下文“当前实现状态、差距与下一步计划”。核心原则只有三条：

1. 保留 XTile 当前“多 transport 显式实现 + 受控强制诊断”的工程优势，但 auto path 继续只走已验证的 device-safe transport。
2. 吸收 Iris 的 allocator-first / export-import-map canonical path，把 allocator、segment metadata、FD/DMA-BUF 映射能力做成底层后端抽象。
3. 对上统一为稳定 public contract：逻辑 shape、layout metadata、heap ownership、external import 语义不能再跟具体 fallback 路径绑死。

---

## 第 5 层：硬件执行

```
GPU0 上的 warp 执行 tl.store(translated_ptr, tile_data, cache_modifier=".wt")
    │
    ▼
translated_ptr 指向 GPU2 的 HBM 地址
    │
    ▼
H100 NVLink 硬件自动路由：
  GPU0 SM → tl.store → L2 miss → NVLink (NV12, 300 GB/s) → GPU2 HBM

P2P benchmark 最新 canonical serial rerun 显示 best read ≈ 248.74 GB/s、best write ≈ 248.43 GB/s；scatter 更相关的是写带宽，量级约 82.8%–82.9% 峰值。
```

### 与设想的差距

| 设想 | 现状 | 差距 |
|------|------|------|
| NVIDIA + AMD 均可运行 | 仅在 NVIDIA H100 上实测 | AMD 待硬件验证（代码已就绪） |
| ≥ 95% P2P 带宽 | 当前约 82.7%–82.9% 峰值 | NVLink 协议开销 + Triton PTX 限制 |

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

### 与设想的差距

**基本一致**，但需要补两点：
- `auto_select(...)` 当前识别的 op 名包括 `gemm_allscatter`、`gemm_allgather`、`gemm_reducescatter`，不是 `gemm_scatter`；其中 `gemm_reducescatter(...)` 现已具备稳定 host-side public contract，但当前并不走 Triton fused pattern auto-select，而是“local GEMM materialize + packed reduce_scatter”主链。
- 对本例 `N=4608, world_size=4`，`n_per_rank = 1152`，不会命中 `fused_sequential` 的 `< 1024` 分支，而会更接近 `producer_consumer` 分支。
- 当前实现还带 `compute_intensity` 与 `N > 4096` 的补充分支，不再只是最初那三条简单规则。

---

## 当前实现状态、差距与下一步计划

本节按 2026-03-22 当前仓库源码、测试和结构化实验产物重新整理，不再把已经落地的事项继续写成“待做计划”，而是直接回答四个问题：

1. 哪些已经实现了。
2. 哪些还没有实现。
3. 现在和 Iris 的差距到底在哪里。
4. 距离理想通信库还差什么，以及下一步该先做什么。

### 状态总览

当前文档和代码主链已经**基本对齐**。需要处理的重点不是“代码还没做”，而是把“已经落地的第一版基础能力”和“尚未闭环的工业级验证面”明确区分开。

当前更准确的结论应当是：

- **XTile 的基础框架已经搭起来了。**
  - `XTileContext`、`SymmetricHeap`、4 种 overlap pattern、`translate_ptr` 5 指令、runtime support matrix、高层 `ops` 主链、benchmark JSON、plot、文档导出，这些都已经形成第一版闭环。
- **高层 public contract 的第一版已经成立。**
  - `xtile.ops.gemm_allscatter(...)`
  - `xtile.ops.gemm_allgather(...)`
  - `xtile.ops.allgather(...)`
  - `xtile.ops.reduce_scatter(...)`
  - `xtile.ops.gemm_reducescatter(...)`
  - 现在都已经进入“build plan -> execute plan”主链，而不是继续停留在“直接裸调 pattern.execute(...)”。
- **allocator-first memory substrate 已经落地第一阶段。**
  - `xtile.memory.allocators` 已引入 allocator backend 边界。
  - 当前默认 backend 是 `torch_bump`。
  - `SymmetricHeap.import_external_tensor(...)` / `as_symmetric(...)` 与 `XTileContext.as_symmetric(...)` 已接入。
  - 这还不是 Iris 风格的 canonical import/map，但已经不再是“所有内存语义都硬编码在 SymmetricHeap 里”。
- **和 Iris 的最大差距，不再是有没有 kernel，而是有没有 canonical allocator / export-import-map / runtime substrate。**
  - 这一层 Iris 更整、更统一。
  - XTile 当前更像“显式 fallback + 显式 gate + 显式 contract”的 bring-up 体系。
- **距离理想通信库，当前主要还差四类东西：**
  - 统一内存语义到底层；
  - 收敛公共契约到更少、更稳的 API；
  - 把 multiprocess 支持面从“已验证的局部路径”扩成“可信的完整支持面”；
  - 补齐性能、压力、长稳和生产化验证。

### 已实现

#### 1. shape / layout / output contract 主链已经统一

当前 shape / layout / output contract 主链已经统一，边界 contract 仍有限制。

已经落地的事实包括：

- `xtile/patterns/contracts.py` 已新增 `PatternExecutionSpec`，pattern 不再从裸 tensor shape 猜逻辑语义。
- 4 个主 pattern 已统一先走 `resolve_execution(...)` / `resolve_pattern_execution(...)`。
- benchmark 路径已经显式传入 `full_N + b_layout + c_layout`。
- `scatter_tile_to_peer(...)` 已显式消费 `src_col_offset / valid_cols / dst_leading_dim / dst_col_offset`。
- `full/full`、`full/shard`、`shard/shard` 已有明确 contract；`shard/full` 现在是**有意识地拒绝**，而不是“还没想清楚先放着”。

这意味着当前的问题不是“contract 主链没搭起来”，而是“contract 主链已经搭起来了，但还没有覆盖所有想象中的 layout 组合”。

#### 2. 高层 `ops` 第一版基础已经齐备

当前真实主链已经是“高层 contract -> build plan -> plan.execute(...)”。

源码摘录如下，来自 `xtile/ops.py`：

```python
def build_gemm_reducescatter_plan(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    implementation: str = "auto",
    storage_kind: str = "symmetric",
) -> GemmReduceScatterPlan:
    """Resolve a reusable host-side plan for GEMM + reduce-scatter.

    Public contract:
    - ``A``: local rank contribution of shape ``(M, K)``
    - ``B``: full RHS matrix of shape ``(K, N)``
    - ``C``: rank-local output shard of shape ``(M, N / world_size)``
    """
    ...
    reduce_scatter_plan = ReduceScatterPlan(...)
    return GemmReduceScatterPlan(...)


def gemm_reducescatter(
    A: Any,
    B: Any,
    C: Any,
    *,
    ctx: xtile.XTileContext | None = None,
    implementation: str = "auto",
    storage_kind: str = "symmetric",
) -> Any:
    plan = build_gemm_reducescatter_plan(...)
    return plan.execute(A, B, C, validate=False)
```

这段代码本身已经说明两点：

- `gemm_reducescatter(...)` 已经不再是占位符，也不再是 `NotImplementedError`。
- 高层 public API 现在确实在复用 plan 对象，而不是靠文档假设。

当前已经落地的高层能力包括：

- `GemmAllScatterPlan` + `build_gemm_allscatter_plan(...)`
- `gemm_allscatter(...)`
- `gemm_allscatter_sharded(...)`
- `GemmAllGatherContract` + `GemmAllGatherPlan`
- `build_gemm_allgather_plan(...)`
- `gemm_allgather(...)`
- `AllGatherPlan` + `build_allgather_plan(...)`
- `allgather(...)`
- `ReduceScatterPlan` + `build_reduce_scatter_plan(...)`
- `reduce_scatter(...)`
- `GemmReduceScatterPlan` + `build_gemm_reducescatter_plan(...)`
- `gemm_reducescatter(...)`

当前更准确的表述是：

- `gemm_allscatter(...)`：单进程主路径已成立；`world_size=2 + ctypes_ipc` 的 multiprocess public baseline correctness 与结构化 benchmark 已闭环。
- `gemm_allgather(...)`：单进程主路径已成立；`world_size=2 + ctypes_ipc` 已完成 `2 shapes × 3 dtypes × 2 transport selections = 12/12` 真机矩阵。
- `allgather(...)`：单进程主路径已成立；`world_size=2 + ctypes_ipc` 的 primitive / kernel / high-level API 结构化矩阵已闭环。
- `allreduce(...)`：高层 `AllReducePlan` / `build_allreduce_plan(...)` / `xtile.ops.allreduce(...)` 已落地；single-process 顺序调用与 multiprocess `ctypes_ipc` 两阶段 composed path 均已补齐验证。
- `reduce_scatter(...)`：单进程 reference 与 multiprocess device 路径都已在当前 public surface 上完成真实验收；更大 world size / 更多 transport 仍未闭环。
- `gemm_reducescatter(...)`：稳定 host-side contract 已成立；`world_size=2 + ctypes_ipc` 的 dtype × transport 结构化矩阵与高层 API 验收已闭环。

#### 3. runtime support / capability matrix 第一版已经完成

runtime support / capability matrix 第一版已经完成，当前已经有统一源码出口、CLI 出口和测试出口。

源码摘录如下，来自 `xtile/support.py`：

```python
ops = {
    "gemm_allscatter": gemm_allscatter_status,
    "gemm_allgather": gemm_allgather_status,
    "allgather": allgather_status,
    "reduce_scatter": SupportStatus(
        reduce_scatter_state,
        reduce_scatter_detail,
    ),
    "gemm_reducescatter": gemm_reducescatter_status,
}

contracts = {
    "gemm_allscatter.full/full": SupportStatus("supported", ...),
    "gemm_allscatter.shard/shard": SupportStatus("supported", ...),
    "gemm_allscatter.full/shard": SupportStatus("supported", ...),
    "gemm_allscatter.shard/full": SupportStatus("unsupported", ...),
    "gemm_allgather.shard/full": SupportStatus(...),
    "gemm_reducescatter.full/shard": SupportStatus(
        gemm_reducescatter_status.state,
        ...
    ),
}
```

这意味着：

- `xtile.describe_runtime_support(ctx)` 已经存在。
- `ctx.support_matrix()` 已经存在。
- `xtile support --json` 已经存在。
- benchmark JSON、plot 和文档导出已经开始消费这份 `runtime_support` snapshot；其中 benchmark JSON 现同时携带 `runtime_metadata`，pattern benchmark 还会按 size 记录对应 heap/runtime metadata。
- 当前状态判断已经是 `heap_mode + transport_strategy + operation contract` 感知，而不是只看“有没有函数名”。

#### 4. allocator-first memory substrate 第一阶段已经落地

当前已经完成的部分是：

- `xtile.memory.allocators` 已建立 allocator backend 抽象。
- 默认 allocator backend 已显式命名为 `torch_bump`，而不是继续把分配细节完全藏在 `SymmetricHeap` 私有实现里。
- multiprocess heap bring-up 已开始通过 allocator-owned export/import surface 收口：
  - `export_peer_memory(...)`
  - `import_peer_memory(...)`
  - `PeerMemoryExportDescriptor`
- `SymmetricHeap.peer_export_descriptors()` / `peer_export_metadata()` / `peer_memory_map()` / `peer_memory_map_metadata()` 已接入，peer export/import/map 元数据不再只能靠调试器看内部状态。
- `MemorySegmentDescriptor`、`SymmetricHeap.segment_descriptors()` / `segment_metadata()` 已接入，allocator-owned local segment catalog 现已显式可见；`peer_exports` / `peer_imports` / `peer_memory_map` 也已带 `segment_id` / `segment_kind`。
- `ImportedPeerMemory`、`SymmetricHeap.peer_imports()` / `peer_import_metadata()` 已接入，peer import state 不再只是 `mapped_ptr + cleanup resource` 的内部临时结构，而是正式结构化 surface。
- `PeerMemoryExportDescriptor` / `ImportedPeerMemory` 现都显式带 `peer_rank`；peer export/import records 不再只靠列表位置隐式表达 rank。
- `peer_imports` 现在已经是 `SymmetricHeap` import-map 的单一真实状态源；当前 `translate()` 通过 primary `segment_id` 走 `peer_import_segment(...)`，`heap_bases` 也已改为从 primary-segment import catalog 派生，`peer_memory_map()` 则继续从结构化 peer import records 构建，不再额外维护 `_remote_ptrs` / `_peer_map` 这类并行派生缓存。
- `heap_bases` 的刷新链路也已经收口到 `_refresh_heap_bases()`；`create_all(...)`、single-rank init 与 multiprocess transport setup 不再各自手工覆写 `_heap_bases`。
- `SymmetricHeap._validate_peer_mapping_state(...)` 已接入；`peer_exports` / `peer_imports` 现在在发布前会校验 world-size、segment metadata、export/import 对齐关系，以及 local-rank import 是否仍精确指向 `local_base`。
- `SymmetricHeap._apply_peer_mapping_state(...)` 现在会先按 `peer_rank` 归一化 incoming peer records，再进行校验与发布；内部 contract 不再要求调用方手工先按 rank 排序。
- `_apply_peer_mapping_state(...)` 现在是 fail-closed 的：非法 peer state 不会污染 `_peer_exports`、`_peer_imports` 或 `heap_bases`。
- `context` / benchmark artifact 侧的回归也已经补上：`peer_exports`、`peer_imports`、`peer_memory_map` 的 `peer_rank` 可见性现在有消费层测试锁定。
- `SymmetricHeap.peer_export_descriptor(rank)` / `peer_import(rank)` 已接入；host-side peer lookup 现已有显式 rank-addressed accessor，不必再直接依赖内部列表下标。
- allocator metadata 现已不只暴露 capability flags，还显式带 `external_tensor_import_mode`；当前 `torch_bump` 的真实语义已明确写成 `copy`。
- `ImportedPeerMemory` / `peer_memory_map` 现已显式带 `access_kind`；当前 runtime 已能区分 `transport`（建立路径）与 `access_kind`（local / peer_direct / mapped_remote / remote_pointer 的访问语义）。
- allocator metadata 现还显式带 `peer_transport_modes` 与 `peer_import_access_kinds`；这表示 allocator surface 自身能表达的 peer 语义目录，不等同于 public support matrix 的验证结论。
- allocator metadata 现已新增结构化 `memory_model`，把 `local_segment_layout`、`peer_import_model`、`peer_mapping_model`、`external_tensor_import_mode`、`external_mapping_mode` 收口到统一 schema。
- `SymmetricHeap.allocator_memory_model_descriptor()` / `allocator_memory_model()` 已接入，allocator `memory_model` 不再只能通过嵌套 metadata 间接读取。
- allocator metadata / heap surface 现已新增结构化 `segment_layout`；当前单 exportable segment 的现状已能通过 `layout_kind`、`primary_segment_id`、`exportable_segment_ids` 正式表达。
- allocator metadata / heap surface 现还显式区分 `segments` 与 `exportable_segments`；当前两者仍相同，但边界已正式建立，便于后续 multi-segment / segmented import-map 扩展。
- `SymmetricHeap.segment_descriptor(segment_id)` / `exportable_segment_descriptor(segment_id)` 已接入；single-segment runtime 现在也有显式的 `segment_id` lookup surface，而不再只能默认“主 segment 就是唯一 segment”。
- allocator metadata / heap surface 现还显式带结构化 `external_memory_interface`；当前 external interop 语义已能正式表达 `import_mode=copy`、`mapping_mode=none`、`zero_copy_mapping_supported=false`。
- allocator metadata 现已显式带 `capabilities`，包括 `external_import_copy`、`external_mapping`、`fd_passing`、`dmabuf_mapping` 等布尔能力位；这让“copy-based import 已有、zero-copy external mapping 未有”可以直接从 runtime metadata 读取。
- `SymmetricHeap._validate_peer_mapping_state(...)` 现在除了校验 world-size、rank 对齐和 local-base 一致性，还会显式校验 peer export 的 `segment_id` 必须存在于 allocator `exportable_segments`，且 `segment_kind` 必须与 exportable segment catalog 一致。
- `SymmetricHeap` 现还维护 segment-scoped peer export/import catalog：`peer_export_segments(rank)` / `peer_export_segment(rank, segment_id)` / `peer_import_segments(rank)` / `peer_import_segment(rank, segment_id)` 已接入，host-side peer state 不再只能按 flat list 消费。
- 现有 `peer_export_descriptor(rank)` / `peer_import(rank)` 也已经改为通过 primary `segment_id` 走 segment-scoped catalog；当前仍是单 exportable segment runtime，但 host-side access shape 已不再把“一 rank 一条记录”写死成唯一形式。
- heap metadata 现已新增 `peer_export_catalog` / `peer_import_catalog` 两个 grouped surface；context、benchmark artifact 与 support matrix 现都能显式消费 segment-scoped peer catalog。
- `heap_bases` 的刷新链路也已进一步收口：当前地址表不再直接扫 flat `peer_imports`，而是显式从每个 rank 的 primary-segment peer import catalog 派生。
- `SymmetricHeap.allocate_tensor(...)`、ownership 检查、`import_external_tensor(...)`、`as_symmetric(...)` 已统一走 allocator。
- `XTileContext.as_symmetric(...)` / `is_symmetric(...)` 已接入，普通 device tensor 现可显式 materialize 到 heap。
- `XTileContext.heap_metadata()` / `runtime_metadata()` 已接入，runtime / heap / peer-map 现在有统一结构化出口。
- `tests/benchmarks/bench_gemm.py`、`tests/benchmarks/bench_p2p_translate.py`、`tests/benchmarks/bench_patterns.py` 生成的结构化 JSON 已统一带出 `runtime_metadata`；其中 pattern benchmark 因 heap size 随 problem size 变化，额外按 size 记录对应 runtime metadata。
- support matrix 已把 `symmetric_heap_allocator_first_import_map` 从完全未开始提升为 `partial`，并新增 `symmetric_heap.external_import`、`symmetric_heap.external_mapping`、`symmetric_heap.segment_metadata`、`symmetric_heap.peer_import_metadata`、`symmetric_heap.peer_segment_catalog` 与 `symmetric_heap.peer_mapping_metadata` 状态。

当前准确口径是：

- **allocator-first 的第一阶段已经实现。**
- **canonical import/map 还没有实现。**

也就是说，XTile 现在已经具备 allocator boundary、allocator-owned peer export/import surface 与 external import surface，但还没有做到 Iris 那种 allocator/export/import/map/access 一体化底座。

#### 5. `gemm_allgather(...)` 的第一版基础工作已经完成

`xtile/ops.py` 里的 `gemm_allgather(...)` 已经作为独立 public host contract 落地，不再需要继续把 `shard/full` 塞进 `gemm_allscatter(...)`。

当前已经完成的部分是：

- public contract 已固定：
  - `A(M, K)` 是完整 LHS；
  - `B(K, N / world_size)` 是本 rank RHS shard；
  - `C(M, N)` 是 full output。
- 当前实现是保守但稳定的 host-side 组合：
  - `local GEMM materialize`
  - `allgather` 到 `rank-major shard` workspace
  - 再 materialize 成 full output
- `C` 必须位于 attached symmetric heap；
  - `A/B_shard` 可以是普通 device tensor。
- single-process correctness 已有真实回归。
- multiprocess `ctypes_ipc` 已有 2-GPU baseline correctness 矩阵。

当前直接证据包括：

- `docs/generated/gemm_allgather_multiprocess_matrix.json`
- `tests/test_gemm_allgather_multiprocess.py`
- `tests/test_e2e/_run_gemm_allgather_multiprocess.py`

当前矩阵结论是：

- 共 `12` 个 case。
- `6` 个通过，`6` 个失败。
- 通过面是 `auto/ctypes_ipc` 与 forced `ctypes_ipc`，`dtype = fp16 / bf16 / fp32`。
- `pytorch_ipc` 和 `peer_access_pointer_exchange` 当前仍失败，因此 multiprocess 仍需保持 `partial`。

同时还要把“支持面内扩验”和“全 transport 矩阵”区分开写：

- 全 transport 矩阵仍然是 `6/12`。
- 但在当前正式支持面 `auto/ctypes_ipc + forced ctypes_ipc` 内，已经新增两组 shape 的真机扩验：
  - `128x256x128`
  - `256x512x256`
- 这组 `2 shapes × 3 dtypes × 2 transport selections` 的 shape-grid 结果为 `12/12` 全通过。
- 因此当前更准确的表述不是“只在一个最小 baseline case 上成立”，而是“**在当前收窄后的正式支持面内，已经开始形成更可信的多 shape baseline**”。

#### 6. `gemm_reducescatter(...)` 的第一版基础工作已经完成

`xtile/ops.py` 里的 `gemm_reducescatter(...)` 已经不是“待做项”，而是已经落地的第一版 public host contract。

当前已经完成的部分是：

- public contract 已固定：
  - `A(M, K)` 是本 rank 本地贡献；
  - `B(K, N)` 是完整 RHS；
  - `C(M, N / world_size)` 是本 rank 输出 shard。
- 当前实现是保守但稳定的 host-side 组合：
  - `local GEMM materialize`
  - `按 rank-major 列分片 pack`
  - `复用 ReduceScatterPlan`
- `C` 必须位于 attached symmetric heap；
  - 这是 reduce-scatter 输出与 workspace 的要求。
  - `A/B` 可以是普通 device tensor，不需要强行一起放进 heap。
- single-process correctness 已有真实回归。
- opt-in multiprocess `ctypes_ipc` 已有 2-GPU baseline correctness 证据。
- support matrix、测试和文档口径已经同步。

当前直接证据包括：

- `docs/generated/gemm_reducescatter_multiprocess_matrix.json`
- `tests/test_gemm_reducescatter_multiprocess.py`
- `tests/test_e2e/_run_gemm_reducescatter_multiprocess.py`

当前矩阵结论也要写准确：

- 共 `12` 个 case。
- `6` 个通过，`6` 个失败。
- 通过面是 `auto/ctypes_ipc` 与 forced `ctypes_ipc`，`dtype = fp16 / bf16 / fp32`。
- `pytorch_ipc` 和 `peer_access_pointer_exchange` 当前仍失败，因此还不能把 multiprocess 写成 fully supported。

#### 7. benchmark / JSON / plot / 文档导出已经开始同源

benchmark / JSON / plot / 文档导出在当前范围内已经建立起第一版闭环：

- `tests/benchmarks/bench_patterns.py` 输出 `figures/data/pattern_overlap_latest.json`
- `tests/benchmarks/bench_gemm.py` 输出 `figures/data/gemm_latest.json`
- `tests/benchmarks/bench_p2p_translate.py` 输出 `figures/data/p2p_latest.json`
- benchmark JSON 已携带 `runtime_support` 快照
- `scripts/plot_figures.py` 已优先从 JSON 出图
- `scripts/export_benchmark_summary.py` 已导出 `docs/generated/benchmark_runtime_summary.md`
- 写入 `figures/data/` 的 canonical benchmark 已有 repo-global lock，避免并发实验污染正式结果

因此这部分当前准确的结论是：

- **正式图已经有统一结构化数据源。**
- **剩下的主要是展示层增强和更多实验覆盖，不是主链没搭起来。**

### 未实现

#### 1. allocator-first canonical backend 还没有收口完成

这是当前和 Iris 最大的实质差距，也是 XTile 还没有真正“工程化收口”的地方。

现在的 XTile 已经具备：

- allocator abstraction
- `torch_bump` allocator backend
- allocator capability metadata
- copy-based `import_external_tensor(...)` / `as_symmetric(...)`
- local segment metadata
- peer import metadata
- transport-aware fallback + gate + support matrix

但它还没有做到：

- export/import canonical path
- external mapping / segmented import-map
- 统一 import-map-access 运行时

这一整层 canonical substrate。

所以这一项当前必须明确记为：**第一阶段已实现，但 canonical backend 仍未完成。**

#### 2. 高层 GEMM op 家族还没有完全收口

这里需要把“已经完成的”和“还没完成的”分开写清楚。

已经完成的部分：

- `gemm_allscatter.full/full`
- `gemm_allscatter.full/shard`
- `gemm_allscatter.shard/shard`
- `gemm_allgather.shard/full`
- `gemm_reducescatter.full/shard`

这些 contract 现在都已经有明确的 public host-side 入口或 expert 入口。

当前仍未完成的部分是：

- `gemm_allscatter.shard/full` 仍然应保持 unsupported；
  - 这不是“少一个 wrapper”，而是因为当前 `gemm_allscatter_sharded(...)` 暴露的是 peer-scatter ownership contract，不是 stable local-shard basis。
- `pattern.execute(...)` 虽然已经不再是推荐 public 主入口，但在仓库里仍然偏显眼，后续还要继续退到 expert/internal surface。
- 当前 multiprocess public support 明确只覆盖 `world_size=2 + ctypes_ipc`；
  - 这不是文案保守，而是当前真实验证边界。
- `gemm_allgather(...)`、`gemm_reducescatter(...)`、`allreduce(...)` 虽然已经完成当前 public surface 的 contract / benchmark / 回归闭环，但更大 world size、更多 shape、更多 transport 与长时间 stress 仍未完成。

#### 3. multiprocess 支持面还没有闭环

当前有直接证据的 multiprocess 支持面是：

- `ctypes_ipc`
- 2-GPU
- 代表性 dtype
- baseline correctness

当前还没有闭环的是：

- `pytorch_ipc`
- `peer_access_pointer_exchange`
- `world_size > 2`
- 更大 shape
- 长时间 stress
- performance contract

所以当前所有 multiprocess 相关 public 口径都必须继续保守，不应写成 fully supported。

#### 4. tile collective 的生产级验证还没有完成

当前不能把 collective 写得过满，但也不能再把已经闭环的部分写成“未完成”。更准确的状态是：

- `allgather`、`allreduce`、`reduce_scatter` 的高层 contract 都已经有真实可跑的 public 主链。
- `world_size=2 + ctypes_ipc` 这一当前 public surface 上，collective correctness 与结构化 benchmark 已经闭环：
  - `pytest -q tests/test_feature_gates.py tests/test_support.py tests/test_benchmark_results.py tests/test_cli_support.py tests/test_collectives_host.py tests/test_ops.py` → `67 passed`
  - `pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py tests/test_gemm_allscatter_multiprocess.py tests/test_gemm_allscatter_auto_patterns_multiprocess.py tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py tests/test_allreduce_multiprocess.py` → `15 passed`
  - `docs/generated/allgather_multiprocess_matrix.json` → `6/6` cases passed
  - `docs/generated/reduce_scatter_multiprocess_matrix.json` → `6/6` cases passed
  - `docs/generated/allreduce_multiprocess_matrix.json` → `6/6` cases passed
  - `docs/generated/gemm_allgather_multiprocess_ctypes_shapes.json` → `12/12` cases passed
  - `docs/generated/gemm_reducescatter_multiprocess_matrix.json` → `6/6` cases passed
  - `docs/generated/gemm_allscatter_multiprocess_matrix.json` → `12/12` cases passed
  - `docs/generated/gemm_allscatter_multiprocess_auto_patterns.json` → `8/8` cases passed
- 还没有完成的是 broader public surface，而不是当前 public surface 本身：
  - `pytorch_ipc`
  - `peer_access_pointer_exchange`
  - `world_size > 2`
  - 更大 shape
  - 长时间 stress
  - performance contract

#### 5. 性能闭环还没有完成

当前文档如果要保持严谨，必须把“基础能力已成立”和“性能目标已达成”分开写。

截至当前：

- P2P 仍只有约 `82.8% - 83.0%` 峰值，离 `>=95%` 目标有明显差距。
- `8192^3` GEMM 最新 canonical rerun 仍只有约 `83.0% - 83.5%` of cuBLAS，未达到 `>=90%` 目标。

所以当前不能把 XTile 写成“已经完成性能闭环”，只能写成“功能基础已建立，但性能目标尚未达标”。

### 与 Iris 的主要差异

当前 Iris 和 XTile 的差异，不宜再简单写成“谁 transport 多、谁 transport 少”。真正要看的，是体系结构的收口位置。

| 维度 | Iris 当前形态 | XTile 当前形态 | 当前判断 |
|------|---------------|----------------|----------|
| 底层内存语义 | allocator + fd passing + DMA-BUF 映射，偏 canonical import/map | fallback-first，多 transport bring-up + gate | **Iris 更整** |
| public contract 表达 | 更偏底层内存/映射体系之上的 op 组合 | contract 显式化、`full/shard` 语义更直白 | **XTile 的上层语义更容易讲清** |
| multiprocess bring-up | 底层路径更统一 | 显式区分 `ctypes_ipc` / `pytorch_ipc` / `peer_access_pointer_exchange` | **XTile 诊断更透明，但收口不够** |
| 运行时状态表达 | 更偏底层实现能力本身 | 已有 support matrix，可表达 mode/transport/contract | **XTile 现阶段的可观测性更强** |
| 长期工程形态 | 更像 canonical substrate | 更像务实演进中的实验型通信库 | **Iris 更接近长期目标形态** |

因此，当前更准确的判断是：

- **Iris 在底层 substrate 完整性上更强。**
- **XTile 在“显式 contract + 显式 support matrix + 显式 gate”这层的工程透明度上有明显优点。**
- **XTile 不应该简单照抄 Iris；正确方向是保留现在这套显式 contract / support / diagnostics，再把 allocator-first canonical backend 补到底层。**

### 距离目标形态的主要差距

如果目标不是“论文 demo 可跑”，而是“长期可维护、工业级、专业严谨的通信库”，那当前还差下面几层能力：

1. **单一 canonical memory model**
   - 不再让 public 语义依赖具体 transport 细节。
   - allocator / export / import / map / access 要统一成同一状态机。

2. **更收口的 API 分层**
   - 普通用户只接触稳定 contract。
   - `pattern.execute(...)` 继续保留，但应明确为 expert/internal surface。

3. **支持面要由真实证据驱动**
   - `supported / partial / unsupported` 必须继续只由代码 + 测试 + 结构化实验决定。
   - 不能靠文档口径“先写成支持”。

4. **性能与回归要制度化**
   - P2P、GEMM、collective 都需要固定 headline、固定环境、固定出图脚本、固定回归阈值。
   - 现在第一版数据链已经有了，但离完整 perf regression system 还有距离。

5. **生产级验证与可观测性**
   - 长时间 stress
   - 更大 world size
   - 错误分类
   - debug dump / trace / runtime metadata
   - 更明确的故障边界

总体上，当前 XTile 已经不是“基础还没搭起来”，而是“基础已经搭起来，但离理想通信库还差工程化收口和生产级验证”。

### 下一步优先级

基于当前代码和实验状态，下一阶段应避免回到“补 plan、补 API 名字”这类已完成事项，而应按下面顺序推进：

1. **P0-next：把 allocator-first v1 继续推进成 canonical backend**
   - 当前 allocator abstraction 与 external import surface 已存在。
   - 下一步是补 export / import / map / access 统一语义。
   - 把当前 multiprocess fallback 能力收口到底层 canonical layer。

2. **P2：把 multiprocess 支持面从“局部成立”扩成“可信成立”**
   - 继续修 `pytorch_ipc` / `peer_access_pointer_exchange`。
   - 扩到 `world_size > 2`。
   - 补更大 shape、stress 和 performance 验收。

3. **P3：继续收紧 public surface，并弱化 direct pattern surface**
   - 继续把更多 public 调用点收敛到 `build plan -> execute plan` 主链。
   - 让 `pattern.execute(...)` 更明确地退到 expert/internal surface。
   - 统一 public contract 命名与 benchmark contract spelling，避免 `full/full` / `full_full` 这类消费层分叉再次出现。

4. **P4：继续收紧 collective 与性能闭环**
   - 把 P2P 与大尺寸 GEMM 拉向既定目标。
   - 在已闭环的当前 public surface 之上，增加 performance regression threshold。

5. **P5：最后再做跨节点和跨平台扩展**
   - UCX / GDR
   - AMD 真机验证

当前结论可以稳定写成：

- **已经实现的，不应再写成“待做计划”。**
- **没有实现的，也不能被“第一版跑通”掩盖。**
- **XTile 现在已经有了可继续工程化推进的坚实基础，但还没有达到 Iris 那种 canonical substrate 完整度，更没有达到理想通信库的最终形态。**
