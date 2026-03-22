# XTile 现状流程：GEMM + AllScatter 全栈代码流（修订版）

> 修订说明
> - 原始文档保留不动；本文件是在原文基础上的核对修订版。
> - `【修订】` 表示原文该处已按当前仓库现实改写。
> - `【新增核对】` 表示为避免误解而补充的仓库现状说明。
> - `【新增状态更新 2026-03-21】` 表示基于最新代码、benchmark 与绘图脚本的状态更新。

## 任务定义

与设想文档相同：4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
先算 C_local = A × B（本地 GEMM），然后把 C_local scatter 到所有其他 GPU。

M=8192, N=4608, K=36864, 4 GPUs.

【新增核对】当前仓库里实际长期共存着两套调用约定：

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

【修订】这部分不再放手写调用示例，直接贴当前 `xtile/ops.py` 里的真实入口实现。现在推荐用户入口仍是 `xtile.ops.gemm_allscatter(...)`，但其内部已经升级成“build plan → execute plan”的主链：

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

【新增核对】`tests/benchmarks/bench_patterns.py` 当前 benchmark 仍然走 shard-buffer 约定，但不再手工拼 contract + 手工调 pattern，而是也改成走统一的 plan builder 主链：

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

【新增状态更新 2026-03-21】需要特别注意：benchmark 路径今天已经**不再依赖 `B.shape[1]` 隐式猜 full-N**；并且 benchmark / CLI helper / 高层 op 现在已经开始共享同一条 `build_gemm_allscatter_plan(...)` 主链。

### 与设想的差距

| 设想 | 现状 | 差距 |
|------|------|------|
| `xtile.fused_gemm_scatter(A, B, C, strategy="auto")` | `xtile.ops.gemm_allscatter(...)` 已可用 | 高层 API 已开始建立，但命名与最终 op 集合仍可继续收敛 |
| `ctx.randn(...)` 直接在堆上分配 | `XTileContext` 已支持 `empty/zeros/randn()` | 这一项已打通 |
| 自动检测后端 `backend="auto"` | `xtile.init()` 默认就是 `backend="auto"` | 这一项已集成 |
| `xtile.init()` 返回的 ctx 可直接执行 pattern | `xtile.init(..., heap=...)` / `heap_size=...` / `init_local(...)` 都可直接返回可运行 pattern 的真实 ctx | runtime ctx 主路径已统一 |
| benchmark / tests / CLI 与 runtime 入口一致 | 已统一到 `XTileContext` | 这一项已打通 |
| 高层 op API | `xtile.ops.gemm_allscatter(...)` / `xtile.ops.allgather(...)` / `xtile.ops.reduce_scatter(...)` / `xtile.ops.gemm_reducescatter(...)` 已接入，并统一走显式 `plan` 主链 | `gemm_reducescatter(...)` 现已补齐稳定 host contract，但 multiprocess 仍继承 `reduce_scatter(device)` 的 experimental gate；`full/shard` 已由 host wrapper 打通，而 `shard/full` 已确认不应继续塞进 `gemm_allscatter(...)`，应转入 future `gemm_allgather` 风格 contract |
| full-shape correctness path 与 benchmark shard path | 两者都存在，但都已开始通过显式 contract + plan builder 收敛 | 仍需把更多调用点统一迁到高层 op API，并继续弱化直接 `pattern.execute(...)` 的 public 角色 |

---

## 第 2 层：Pattern 层（以 FusedSequential 为例）

### 实际源码：显式 execution contract

【修订】当前 pattern 层的关键变化不是 kernel 花样，而是 `xtile/patterns/contracts.py` 先把逻辑 shape/layout 解释清楚，再交给 pattern 执行。下面是当前真实源码摘录：

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

**差异（修订）**：核心通信原理接近 Iris。当前 XTile 已经把 full/shard 语义提升成显式 host-side contract，而不是继续在 pattern 中隐式猜 `B.shape[1]`。现在真正还没完全收敛的，是“默认 public API 是否全部走 `xtile.ops.*`”以及“是否继续保留多种输出 layout 模型”。

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

**基本一致（修订）**：`translate_ptr + tl.store` 这条核心路径成立，编译器也确实全可见；现在 scatter helper 已经不再把 full-buffer / shard-buffer 语义写死在 device helper 里，而是显式消费 host-side contract 传下来的 offset / leading-dim 元数据。
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

【新增状态更新 2026-03-21 第四轮核对】跨进程测试（`torch.multiprocessing.spawn`，2× H100，`ptrace_scope=1`）现在的真实状态需要进一步写严：

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

【修订】这说明当前问题已经不再只是“heap / IPC bring-up 会不会崩”，而是更精确的：
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

**与 Iris 的区别（修订）**：不能再简单写成“Iris 仅使用 HIP IPC 单一路径”。当前 Iris 的主实现已经演进到 allocator + fd passing + DMA-BUF 映射；XTile 的特点也不应再表述成“三条 transport 同等可用”。更准确的表述是：XTile 显式实现了 `ctypes_ipc` / `pytorch_ipc` / `peer_access_pointer_exchange` 三条 bring-up 策略，并且已经通过真实矩阵确认只有 `ctypes_ipc` 可作为当前 multiprocess device path 的 auto transport。

### 【新增状态更新 2026-03-21】Iris 与 XTile 的实质差异、优劣与对齐方向

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
| NVIDIA 当前可用性 | 本仓库语境下不是主战场 | 当前更强，尤其在 H100 bring-up 上更顺手 |

**差异本质**：

1. **Iris 更像“一个 canonical 的内存映射体系”**
   - allocator 是一等对象
   - FD / DMA-BUF / map/access 是主路径的一部分
   - 更适合把“跨进程、跨设备、连续 VA 语义”长期做深

2. **XTile 更像“一个现实可用的生存路径体系”**
   - 同一接口下先把 transport 分层实现清楚
   - auto path 只保留当前真实通过 device-side 矩阵的 `ctypes_ipc`
   - 其他 transport 继续保留为 forced diagnostics / bring-up 对照
   - 对当前 NVIDIA bring-up、实验收口和风险控制更友好

**哪个更好？**

- 如果问“**长期架构形态**”，Iris 这一套 allocator + import/map 更好，原因是语义更统一、可扩展性更强、未来更容易把多进程 / 多节点 / 多 allocator 做成同一个 canonical path。
- 如果问“**当前这台 H100 机器上谁更实用**”，XTile 现在更好，因为它把多种现实限制都兜住了，尤其是 `ptrace_scope=1` 这类环境问题下仍能跑通。

**XTile 应该如何向 Iris 对齐，而不是简单照抄？**

总纲不变，但详细动作已经并入下文 `【新增状态更新 2026-03-21】未统一点推进状态与下一步计划`。核心原则只有三条：

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

【新增核对】P2P benchmark 最新 canonical serial rerun 显示 best read ≈ 248.74 GB/s、best write ≈ 248.43 GB/s；scatter 更相关的是写带宽，量级约 82.8%–82.9% 峰值。
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

**基本一致（修订）**，但需要补两点：
- `auto_select(...)` 当前识别的 op 名包括 `gemm_allscatter`、`gemm_allgather`、`gemm_reducescatter`，不是 `gemm_scatter`；【修订 2026-03-22】其中 `gemm_reducescatter(...)` 现已具备稳定 host-side public contract，但当前并不走 Triton fused pattern auto-select，而是“local GEMM materialize + packed reduce_scatter”主链。
- 对本例 `N=4608, world_size=4`，`n_per_rank = 1152`，不会命中 `fused_sequential` 的 `< 1024` 分支，而会更接近 `producer_consumer` 分支。
- 当前实现还带 `compute_intensity` 与 `N > 4096` 的补充分支，不再只是最初那三条简单规则。

---

## 总体差距汇总

## 【新增状态更新 2026-03-21】未统一点推进状态与下一步计划

下面不再把已经落地的事项继续写成“待做计划”，而是按当前代码状态拆成“已完成 / 部分完成 / 未完成 / 当前弱项 / 下一步计划”。

### 优先级调整（最高优先级）

【新增状态更新 2026-03-21】从工程角度，当前最高优先级已经明确调整为：

1. **P0：外部单一契约，内部计划执行**
   - 目标：默认用户入口只表达稳定逻辑语义；内部通过显式 `plan` 统一 contract / pattern / execution。
   - 原因：如果这条主链不先收口，mixed layout wrapper、更多 fused op、allocator-first heap backend 都会继续叠在松散 API 上。
   - 当前阶段验收：
     - `xtile.ops.gemm_allscatter(...)` 默认解释为 public `full/full` 合同
     - 新增显式 `GemmAllScatterPlan`
     - 新增 `build_gemm_allscatter_plan(...)`
     - 新增 `gemm_allscatter_sharded(...)` 作为 shard/shard expert 入口
     - benchmark / helper 开始复用 plan builder，而不是继续各自拼 contract

2. **P1：文档、benchmark、plot 同源**
   - 目标：所有 headline 都能回溯到结构化结果和环境元数据。

3. **P2：补齐剩余高层 op，并把错误归类的 mixed layout 需求重命名**
   - 目标：减少直接 `pattern.execute(...)` 的 public 暴露面；把真正属于 `gemm_allgather` 的需求从 `gemm_allscatter` 路线上拆出来。

4. **P3：heap canonical backend 对齐 Iris**
   - 目标：把当前现实可用 fallback 收敛进 allocator-first canonical layer。

### U1 + U2 + U3：shape/layout/output contract 主链【已完成】

【新增状态更新 2026-03-21】这三项当前阶段已经落地，核心证据有四条：

1. `xtile/patterns/contracts.py` 已新增 `PatternExecutionSpec`，不再让 pattern 从裸 tensor shape 猜逻辑语义。
2. 4 个主 pattern 都先走 `resolve_execution(...)` / `resolve_pattern_execution(...)`，benchmark 路径已经显式传 `full_N + b_layout + c_layout`。
3. `scatter_tile_to_peer(...)` 已改成显式消费 `src_col_offset / valid_cols / dst_leading_dim / dst_col_offset`。
4. contract 校验不再“猜着跑”：`full/shard` 已升级为显式 host wrapper；剩余未完成的 mixed multi-rank layout 仍会被明确拒绝。

【修订】因此，原文里“U1/U2/U3 未统一”这句现在已经不准确。更准确的状态应该是：
- **主链已统一**
- **边界场景仍有限制**

当前还没做完的边界，不属于主链未统一，而属于下一阶段扩展：
- mixed `full/shard` multi-rank contract 已由高层 host wrapper 打通
- mixed `shard/full` multi-rank contract 仍直接拒绝
- correctness path 与 benchmark path 仍同时存在，只是已经共享同一套 contract
- public API 还没有把“逻辑 full 语义”和“内部 shard 执行计划”完全包装成一层

### U4：高层 op API 第一阶段【已完成，第二阶段部分推进】

【新增状态更新 2026-03-21】`xtile.ops.gemm_allscatter(...)` 已经接入，并且这轮已经把它继续收口成显式 plan 主链。与此同时，`xtile.ops.allgather(...)` 与 `xtile.ops.reduce_scatter(...)` 也已经落地到同样的“build plan → execute plan”模式。当前真实实现已经是：
- 解析上下文
- GEMM 路径解析 public layout contract，collective 路径校验 heap/shape contract
- build `GemmAllScatterPlan` / `AllGatherPlan`
- GEMM 路径自动选择 pattern，collective 路径直接绑定底层 collective launcher
- `plan.execute(...)` 默认支持执行前再校验，benchmark 热路径可显式关闭

【新增状态更新 2026-03-21】本轮已落地的第一阶段代码包括：
- `xtile.ops.GemmAllScatterPlan`
- `xtile.ops.build_gemm_allscatter_plan(...)`
- `xtile.ops.gemm_allscatter_sharded(...)`
- `xtile.ops.AllGatherPlan`
- `xtile.ops.build_allgather_plan(...)`
- `xtile.ops.allgather(...)`
- `xtile.ops.ReduceScatterPlan`
- `xtile.ops.build_reduce_scatter_plan(...)`
- `xtile.ops.reduce_scatter(...)`
- `tests/benchmarks/bench_patterns.py` 改走 plan builder
- `xtile.patterns.auto_select.benchmark_all_patterns(...)` 改走 plan builder

所以，原来“高层 op API 缺失”这个说法也需要改成更准确的版本：
- **`gemm_allscatter(...)` 已完成单进程主路径；multiprocess 现已完成 `ctypes_ipc` 下 public `full/full + full/shard` 的 2-GPU baseline correctness，并补齐 representative `auto` pattern coverage，但 broader stress/performance/world-size 闭环仍未完成**
- **`allgather(...)` 已完成单进程主路径；multiprocess `ctypes_ipc` 已通过 2-GPU 真机验证，但当前仍保守记为 `partial`**
- **`reduce_scatter(...)` 已完成单进程 reference 主路径并可用**
- **【修订 2026-03-22】`gemm_reducescatter(...)` 已完成第一版稳定 host contract**

当前阶段的结论不是“没有高层 API”，而是“高层 API 第一阶段已完成，而且已经开始从‘直接 execute pattern’收口到‘build plan → execute plan’；【修订 2026-03-22】`gemm_reducescatter(...)` 也已补齐到这条主链里。下一阶段不再是‘先把它从占位符补出来’，而是继续收紧 multiprocess/public/performance gate，并把错误归类到 `gemm_allscatter(...)` 名下的 `shard/full` 需求转成独立 contract”。
【新增状态更新 2026-03-21 第三轮核对】这里需要进一步收紧：`mixed-layout host wrapper` 已经不是“完全没做”，而是：
- `full/shard` 已完成，并由高层 API 自动 materialize heap-backed full output 后再返回本 rank shard
- `shard/full` 仍未完成，而且今天的真实诊断已经证明问题不只是“少一层 allgather wrapper”，而是它根本不应继续作为 `gemm_allscatter(...)` 的逆向补丁

### U5：benchmark 数据管线与图表口径【已完成（当前范围）】

【新增状态更新 2026-03-21】这项现在可以按当前范围写成“已完成”，因为 benchmark → JSON → plot 主链已经从 pattern 扩展到了 GEMM / P2P / roofline 相关图：

- **已完成**
  - `bench_patterns.py` 会输出 `figures/data/pattern_overlap_latest.json`
  - `bench_gemm.py` 会输出 `figures/data/gemm_latest.json`
  - `bench_p2p_translate.py` 会输出 `figures/data/p2p_latest.json`
  - 3 类 benchmark JSON 现都开始携带 `runtime_support` 快照
  - `fig1_gemm_performance` / `fig2_p2p_bandwidth` / `fig3_pattern_overlap` 已优先从 JSON 自动加载
  - `fig5_roofline` 已改为消费同一份 GEMM benchmark 数据，而不是继续吃独立人工常量
  - `scripts/plot_figures.py` 现会把 `runtime_support` 摘要直接写入 figure footer
  - `scripts/export_benchmark_summary.py` 现会把 canonical benchmark JSON 导出为 `docs/generated/benchmark_runtime_summary.md`
  - quick run 不再污染正式图
  - 写入 `figures/data/` 的 canonical benchmark 现已由 repo-global lock 串行保护，避免并发实验污染正式 benchmark 产物
  - JSON 元数据已记录 `heap_mode`、`transport_strategy`、`layout_mode`、`repeats`、`aggregation`
  - `experiment_log.md` 与 `docs/四项目全栈差异分析_20260320.md` 的旧 overlap headline 已按最新结果修正

【修订】因此，U5 当前真正完成的是“**正式图已经有统一结构化数据源**”。剩下的增强项不再属于主链未完成，而属于展示层增强，例如图注补充更多命令/日期信息。

### U6：symmetric heap canonical allocator/import-map 路线【未完成】

【新增状态更新 2026-03-21】这项目前还是未完成，不能粉饰：

- XTile 当前强项是把 multiprocess transport 显式分层，并且已经用真实矩阵把“可 host bring-up”和“可 Triton device remote access”区分开了
- Iris 当前强项是 allocator + fd passing + DMA-BUF 映射这一类 canonical import/map 路线
- XTile 还没有把 allocator abstraction、external import、segment map/access 做成统一底层语义

所以这项当前仍应明确写成：**未完成**。

### 【新增状态更新 2026-03-21 第二轮核对】P4：runtime support / capability matrix【第一阶段已完成】

【新增状态更新 2026-03-21】这项此前只写在“下一阶段计划”里，但代码里没有统一出口，导致文档容易继续漂。现在已经先落了一版真实 runtime support matrix：

- 新增 `xtile.describe_runtime_support(ctx)`
- 新增 `ctx.support_matrix()`
- 新增 `xtile/support.py`
- 新增 `tests/test_support.py`
- 新增 `xtile support --json`

当前真实源码摘录：

```python
gemm_allscatter_status = _describe_gemm_allscatter_support(
    has_heap=has_heap,
    heap_mode=heap_mode,
    transport_strategy=transport_strategy,
)
allgather_status = _describe_allgather_support(
    has_heap=has_heap,
    heap_mode=heap_mode,
    transport_strategy=transport_strategy,
)

ops = {
    "gemm_allscatter": gemm_allscatter_status,
    "allgather": allgather_status,
    "reduce_scatter": SupportStatus(
        reduce_scatter_state,
        reduce_scatter_detail,
    ),
    "gemm_reducescatter": gemm_reducescatter_status,
}
```

【修订】这意味着“support / capability matrix 完全未开始”这个说法现在也不准确了。更准确的状态应该是：

- **runtime support matrix 第一阶段已完成，并已有 CLI / 测试出口**
- **benchmark 结构化 JSON 已开始内嵌这份矩阵**
- **plot / 文档导出主链现在也已统一消费 `runtime_support` 字段**
- **而且状态已经开始反映 heap / mode / transport 条件，而不是只看有没有 API 入口**

### 当前弱项

【新增状态更新 2026-03-21】基于当前代码和文档复核，真正值得继续推进的弱项是下面这些，而不是重复写已经完成的 U1-U4 主链：

【新增状态更新 2026-03-22】到当前为止，`gemm_reducescatter(...)` 的**第一版基础工作**可以正式记为**已完成**：
- public contract 已固定
- `plan` / 高层 `ops` 已落地
- single-process 与 opt-in multiprocess `ctypes_ipc` 都已有直接真实验收
- support matrix / 测试 / 文档 已同步

因此，下面的“弱项”不再是“基础功能缺失”，而是“运行时支持面与工业级验证面仍需继续收紧”。

1. **高层 op 集合已基本齐，但 `gemm_reducescatter(...)` 仍有边界条件需要继续收紧**
   - 【修订 2026-03-22】`xtile.ops.gemm_reducescatter(...)` 已不再是 `NotImplementedError`；当前 contract 已固定为 `A(M,K)` 本地贡献、`B(K,N)` 完整 RHS、`C(M,N/world_size)` 本 rank 输出分片。
   - 当前实现是稳定 host-side 组合：`local GEMM materialize -> 按列 pack 成 rank-major contiguous input -> 复用 ReduceScatterPlan`。
   - 【新增状态更新 2026-03-22】当前 contract 还额外收紧了一点：只要求 `C` 位于 attached symmetric heap，`A/B` 可以是普通 device tensor；因为真正需要 symmetric heap 的是 reduce-scatter 输出与内部 workspace，而不是输入 GEMM 源张量本身。
   - 单进程 peer-access heap 下，这条 public 路径现已通过 1-GPU / 2-GPU 真实回归；【新增状态更新 2026-03-22】opt-in multiprocess `ctypes_ipc` 下，`build plan` 和高层 `ops` 也都已通过 2-GPU 真机校验；但 multiprocess 仍继承 `reduce_scatter(device)` 的 experimental + transport-aware gate，因此还不能把它写成“所有 mode 下 fully supported”的完成项。
2. **直接 `pattern.execute(...)` 仍然是显式可见的 expert surface**
   - 这对底层调优有价值，但对公共语义收口仍然偏宽。
   - 这轮虽然已经补了 multiprocess unsupported transport 的 host-side 守卫，不再直接掉进 Triton illegal memory access；但它仍属于 expert/internal surface，不应替代高层 op 的稳定 contract。
3. **mixed layout host wrapper 只完成了一半**
   - `full/shard` 已实现并经过 2-GPU 真实回归与重复调用 workspace 复用验证。
   - `shard/full` 仍保持显式拒绝。今天新增的真实 2-GPU 诊断表明，当前 `gemm_allscatter_sharded(...)` 在 multi-rank 下暴露的是 peer-scatter ownership contract，而不是“每个 rank 先稳定产出自己的 local shard”。在 `bulk_sync` 诊断中，`rank0` 相对本地 `A @ B_shard0` 参考的 `max_abs_diff = 35.5`，而 `rank1` 仅约 `4.88e-4`，说明这条链不能直接拿来做 full-output assembly 的基石。
4. **SymmetricHeap 还没有 allocator-first canonical layer**
   - 目前依然更像“务实 fallback 系统”，还不是 Iris 风格的统一 allocator/import-map 框架。
5. **性能闭环仍未完成**
   - P2P 仍只有约 82.8%–83.0% 峰值，离 `>=95%` 目标明显有差距。
   - 8192³ GEMM 最新 canonical serial rerun 仍只有 83.0%–83.5% of cuBLAS，未达 `>=90%` 目标。

### 下一阶段计划

【新增状态更新 2026-03-21 第二轮核对】下一阶段计划需要比上一版更严格，因为这次核对后能确认一件事：`gemm_reducescatter(...)` 不是“下一步直接补一个 plan object”这么简单，它前面有明确前置依赖。

因此，下一阶段计划收敛为下面 4 项，并按依赖顺序推进：

1. **P0-A：把 support matrix 接入正式状态出口、结构化结果、plot 与文档导出【已完成】**
   - 已完成：
     - `xtile.describe_runtime_support(ctx)`
     - `ctx.support_matrix()`
     - `tests/test_support.py`
     - `xtile support --json`
     - `tests/test_benchmark_results.py`
     - GEMM / P2P / pattern benchmark JSON 现已内嵌 `runtime_support`
     - 写入 `figures/data/` 的 canonical benchmark 现已由全局锁串行保护
     - `scripts/plot_figures.py` 现已消费 `runtime_support` 并写入 figure footer
     - `scripts/export_benchmark_summary.py` 现已导出 `docs/generated/benchmark_runtime_summary.md`
     - canonical benchmark 已重跑：
       - `PYTHONPATH=. python tests/benchmarks/bench_gemm.py --repeats 3 --output-json figures/data/gemm_latest.json`
       - `PYTHONPATH=. python tests/benchmarks/bench_p2p_translate.py --output-json figures/data/p2p_latest.json`
       - `PYTHONPATH=. python tests/benchmarks/bench_patterns.py --warmup 3 --iters 10 --output-json figures/data/pattern_overlap_latest.json`
     - `python scripts/plot_figures.py`
     - `python scripts/export_benchmark_summary.py`
     - 最新聚焦回归（含 benchmark lock / reporting）：
       - `pytest -q tests/test_benchmark_results.py tests/test_benchmark_reporting.py tests/test_export_benchmark_summary.py tests/test_collectives_host.py tests/test_cli_support.py tests/test_support.py tests/test_ops.py tests/test_patterns/test_contracts.py tests/test_context.py`
       - `32 passed`
   - 验收标准：
     - 文档状态不再手工罗列与代码不一致的支持面
     - 正式 CLI 可直接输出 support matrix
     - benchmark 结构化结果可回放当次运行的支持面快照
     - 图和 Markdown 导出都能直接引用同一份 support snapshot

2. **P0-B：把 reduce_scatter host contract 从 baseline launcher 推进到稳定 public contract，再谈 `gemm_reducescatter(...)`【第二阶段进行中】**
   - 原因：
     - 当前已有 `tile_reduce_scatter` kernel + baseline `reduce_scatter(...)` host launcher
     - 现在已补 `ReduceScatterPlan` + `xtile.ops.reduce_scatter(...)`
     - 当前 2-GPU multiprocess/device correctness 已打通，但这条路径仍未完成稳定 public/performance contract 所需的更大验证面，因此仍不足以直接公开 fused `gemm_reducescatter(...)`
   - 本轮实验核对：
     - `python -m xtile.cli support --backend cuda --world-size 2 --heap-size-mb 64 --json` 实测输出：
       - `ops.reduce_scatter = supported`
       - `collectives.reduce_scatter_launcher = supported`
     - 真实 2 GPU 高层 API 校验：
       - `xtile.ops.reduce_scatter(...)` 实测 `rank0=12.0, rank1=14.0`
       - 期望 `[12.0, 14.0]`
     - 定向回归：
       - `pytest -q tests/test_benchmark_results.py tests/test_collectives_host.py tests/test_cli_support.py tests/test_support.py tests/test_ops.py tests/test_patterns/test_contracts.py tests/test_context.py`
       - `24 passed`
     - 风险收口：
       - `reduce_scatter(..., implementation="device")` 在 `single_process` 2-GPU 下实测会给出错误值
       - 现在已把这一分支改成显式 `ValueError`，只保留已验证的 `reference/auto` 主路径
     - 第四轮真实诊断 + 第五轮修复验收：
       - `tests/test_backend_ipc.py` 已确认 `get_ipc_handle()` 现返回完整 64-byte payload
       - `python -m tests.test_e2e._run_ipc_test` 现已真实通过，结果含 `ALL MULTI-GPU IPC TESTS PASSED`
       - `XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 python -m tests.test_e2e._run_reduce_scatter_multiprocess` 现已真实通过
       - multiprocess/device `reduce_scatter` 的 primitive 与高层 API 结果均为：
         - `rank0 = 4.0`
         - `rank1 = 6.0`
       - `tile_reduce_scatter` 当前实现已改成 **只远端读 peer chunk、只本地写 output** 的 correctness-first device 路径，避免 peer 覆盖未归约本地 chunk 的 data race
       - 因此 multiprocess/device path 现已新增默认 gate：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES`
       - 默认 public 行为仍保持**显式拒绝** multiprocess `auto/device`
       - 只有显式设置该环境变量时，support matrix 才会把 multiprocess device path 表示为 experimental `partial`
       - opt-in 回归：
         - `XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_backend_ipc.py tests/test_ops.py tests/test_support.py tests/test_collectives_host.py tests/test_reduce_scatter_multiprocess.py tests/test_cli_support.py tests/test_context.py tests/test_patterns/test_contracts.py`
         - `33 passed`
     - 第六轮真实矩阵 + transport 收口：
       - `PYTHONPATH=. XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 python -u tests/benchmarks/bench_reduce_scatter_multiprocess.py --warmup 2 --iters 5 --timeout-sec 60 --output-json docs/generated/reduce_scatter_multiprocess_matrix.json`
       - 结果汇总：`12` 个 case 中 `6` 个通过、`6` 个失败
       - 通过面：
         - `auto` / `ctypes_ipc`
         - `dtype = fp16 / bf16 / fp32`
         - 实际 transport 均为 `ctypes_ipc`
       - 未通过面：
         - `pytorch_ipc`
         - `peer_access_pointer_exchange`
       - 因此 gate 已进一步改成 **transport-aware**
         - 当前即便显式打开 experimental gate，也只允许 `transport_strategy='ctypes_ipc'`
         - 其他 transport 会在 host 侧提前抛 `ValueError`
       - 相关回归：
         - `pytest -q tests/test_feature_gates.py tests/test_support.py tests/test_ops.py tests/test_backend_ipc.py`
         - `29 passed`
         - `XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py`
         - `3 passed`
     - 因此当前可以把**单进程 reference 路径**写成 `supported`，并把 **multiprocess/device correctness** 写成“已通过 2-GPU `ctypes_ipc` 矩阵验证但仍为 experimental”
     - 【新增状态更新 2026-03-22】`gemm_reducescatter(...)` 现也已补上独立的 2-GPU multiprocess 验收：
       - 新增 `tests/test_e2e/_run_gemm_reducescatter_multiprocess.py`
       - 新增 `tests/test_gemm_reducescatter_multiprocess.py`
       - 实测命令：
         - `XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 python -m tests.test_e2e._run_gemm_reducescatter_multiprocess --dtype float32 --launcher all --M 128 --N 256 --K 128 --warmup 0 --iters 1`
       - 实测结果：
         - `rank0: transport_strategy='ctypes_ipc', plan_ok=True, high_level_ok=True, max_abs_diff=0.0`
         - `rank1: transport_strategy='ctypes_ipc', plan_ok=True, high_level_ok=True, max_abs_diff=0.0`
       - 【新增状态更新 2026-03-22】随后又补上了结构化矩阵：
         - `PYTHONPATH=. python -u tests/benchmarks/bench_gemm_reducescatter_multiprocess.py --M 128 --N 256 --K 128 --warmup 0 --iters 1 --timeout-sec 120 --output-json docs/generated/gemm_reducescatter_multiprocess_matrix.json`
         - 结果汇总：`12` 个 case 中 `6` 个通过、`6` 个失败
         - 通过面：
           - `auto` / `ctypes_ipc`
           - `dtype = fp16 / bf16 / fp32`
           - 实际 transport 均为 `ctypes_ipc`
         - 未通过面：
           - `pytorch_ipc`
           - `peer_access_pointer_exchange`
       - 因此当前文档口径需要更新为：
         - `gemm_reducescatter(...)` 的 **single_process stable host contract 已闭环**
         - **multiprocess `ctypes_ipc` 2-GPU baseline correctness 已有直接证据**
         - 但 broader dtype/world-size/stress/performance contract 仍未闭环，因此 support 继续保持 `partial`
     - 但这依然不等于 fused `gemm_reducescatter(...)` 前置条件已全部完成
   - 下一步：
     - 当前环境下 `dtype(fp16/bf16/f32)` 与 transport 第一轮矩阵已经完成；下一步优先级改成：
       - 继续修 `pytorch_ipc` / `peer_access_pointer_exchange` 的 device-side remote access 正确性；在修好前保持 `ctypes_ipc only`
       - 在有更多 GPU 的机器上补 `world_size>2` 真实验收
       - 继续补 benchmark / stress 证据，再决定是否把 multiprocess/device 从 experimental gate 提升为正式 public contract
     - 在 public/performance gate 完成前，`gemm_reducescatter(...)` 仍不直接公开
   - 验收标准：
      - single-process reference 路径保持真实可验并纳入 support matrix
      - multiprocess/device 路径状态继续由真实 gate + 实验日志决定，而不是靠文档口径人工判断
      - 至少完成 `ctypes_ipc` 之外 transport 的真实修复证据，以及更大验证矩阵与 benchmark/stress 证据后，再决定是否公开
      - 再决定是否公开 `xtile.ops.gemm_reducescatter(...)`

3. **P0-C：把 multiprocess `allgather` / `gemm_allscatter` public surface 与真实 transport 证据对齐【本轮已推进，下一阶段继续】**
   - 本轮已完成：
     - `allgather` 新增 2-GPU multiprocess 真机诊断与矩阵
     - `auto/ctypes_ipc` 在 `fp16/bf16/fp32` 的 primitive / kernel / high-level API 全通过
     - forced `pytorch_ipc` / `peer_access_pointer_exchange` 现在会在 host 侧明确拒绝
     - support matrix 已改成 mode/transport 感知，`gemm_allscatter` 无 heap 时不再写成 `supported`
     - `gemm_allscatter` 现已补上 2-GPU multiprocess 真机诊断与矩阵
     - `auto/ctypes_ipc` 在 `full/full + full/shard × fp16/bf16/fp32` 的 plan / high-level API 已全通过
     - forced `pytorch_ipc` / `peer_access_pointer_exchange` 也已确认是 host-side 明确拒绝，而不是 kernel crash
   - 当前仍未闭环：
     - `gemm_allscatter` 虽然已经补齐 representative `auto` pattern 面，但更大 shape、更多 world-size 和更长时间 performance/stress 还没有完成
     - 因此它当前仍应保守写成 `partial`，而不是直接升级成 `supported`
   - 下一步：
     - 扩到更大 shape / 更长时间 stress / 条件变化后的稳定性验证
     - 在当前 2-GPU 之外补更大 world-size 真机证据
     - 在 representative correctness 之外，再补 public performance contract 所需证据
   - 验收标准：
     - `allgather` 与 `gemm_allscatter` 的 support 状态都必须能回溯到真实脚本和结构化 artifact
     - unsupported transport 必须保持 host-side 明确拒绝
     - `gemm_allscatter` 只有在拿到更大 shape / world-size / stress / performance 真机证据后才允许升级状态

4. **P1：把 `shard/full` 从“补 wrapper”改成“定义独立 public contract”**
   - `full/shard` 已完成，但 `shard/full` 经过真实 2-GPU 诊断后，已经确认不应继续作为 `gemm_allscatter(...)` 的逆向补丁。
   - 下一步应单独定义 `gemm_allgather` 风格 contract：先明确 local-output ownership，再决定是否复用现有 pattern、还是走“local GEMM + allgather” 的高层组合路径。
   - 验收标准不再是“能跑起来”，而是“local shard ownership、heap contract、full-output assembly 语义都能被测试稳定证明”。

5. **P2：向 Iris 的 canonical heap 路线对齐**
   - 引入 allocator abstraction
   - 补 `import_external_tensor(...)` / `as_symmetric(...)` 等价能力
   - 把 `create_all()` 与 multiprocess path 收敛到统一状态机

### 核心路径：完全实现

| 组件 | 设想 | 现状 | 状态 |
|------|------|------|------|
| translate_ptr (5 指令) | 匹配 Iris | ✅ 100% 匹配 | 完成 |
| SymmetricHeap | IPC/peer access | ✅ 单进程 `peer_access` + 多进程 auto=`ctypes_ipc`；其余 transport 保留 forced diagnostics | 完成 |
| 4 种 overlap pattern | kernel 实现 | ✅ 全部实现 | 完成 |
| GEMM kernel | ≥ 90% cuBLAS | ✅ 4096³：fp16 94.9%，bf16 91.1%（latest canonical rerun） | 完成 |
| Auto-select | 硬件感知 | ✅ 启发式已实现，且已与统一 `XTileContext` 主路径打通 | 基本完成 |
| tile 级 collective | ring allreduce 等 | ⚠️ 代码已实现，但当前结果里至少 allreduce collective benchmark 仍报 `invalid resource handle` | 部分完成 |
| 跨平台 HAL | CUDA + HIP | ✅ 代码就绪 | 待 AMD 硬件 |

### 未完成

| 组件 | 设想 | 现状 | 原因 |
|------|------|------|------|
| P2P ≥ 95% | 285 GB/s | ❌ 当前约 82.9% 峰值 | Triton/PTX 路径与协议开销 |
| 一键 API | `xtile.fused_gemm_scatter(...)` | 【修订 2026-03-22】⚠️ `xtile.ops.gemm_allscatter(...)` / `xtile.ops.allgather(...)` / `xtile.ops.gemm_reducescatter(...)` 已接入；其中 `gemm_reducescatter(...)` 现为稳定 host-side contract，但 multiprocess/public-performance gate 仍未闭环 | 高层 API 主集合已补齐，当前弱项转向更严格的运行时 gate 与性能闭环 |
| 8192³ GEMM ≥ 90% | kernel 优化 | ❌ 当前约 83.0%–83.5% | 仍需更深 kernel-level 优化 |
| Pattern ≥ 1.3× overlap | 多 GPU overlap | ✅ 当前 full 6-size rerun best = 1.667× | contract 修正后已达标 |
| 跨节点 IPC | UCX/GDR | ❌ 待实现 | 需跨机通信基础设施 |
| AMD 实测 | MI300X 验证 | ❌ 待硬件 | 无 AMD GPU |
