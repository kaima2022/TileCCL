# XTile WG 工作组模式：从上到下完整代码流程报告

## 全局架构总览

```
User Code
    |
    v
+---------------------------+
| Layer 0: xtile.ops        |  <-- gemm_allscatter() 公共入口
+---------------------------+
    |  build plan + resolve contract
    v
+---------------------------+
| Layer 1: Pattern Engine   |  <-- WGSpecializedPattern.execute()
+---------------------------+
    |  single kernel launch
    v
+====================================================+
| Layer 2: Triton Kernel (_wg_specialized_kernel)     |
|                                                     |
|  +------------------+   +------------------------+  |
|  | Compute Workers  |   | Comm Workers           |  |
|  | pid < COMPUTE_SMS|   | pid >= COMPUTE_SMS     |  |
|  |                  |   |                        |  |
|  | GEMM tile loop   |   | wait + scatter loop    |  |
|  |   |              |   |   |                    |  |
|  |   v              |   |   v                    |  |
|  | tile_signal()  ------>  tile_wait()           |  |
|  | (release)        |   |  (acquire)             |  |
|  |                  |   |   |                    |  |
|  +------------------+   |   v                    |  |
|                         | scatter_tile_to_peer() |  |
|                         |   |                    |  |
|                         |   v                    |  |
|                         | translate_ptr()        |  |
|                         |   |                    |  |
|                         |   v                    |  |
|                         | tl.store(remote_C)     |  |
|                         +------------------------+  |
+====================================================+
    |                              |
    v                              v
+---------------------------+  +---------------------------+
| Layer 3: Sync Primitives  |  | Layer 4: Memory           |
| tile_signal / tile_wait   |  | translate_ptr             |
| (atomic_xchg / atomic_cas)|  | SymmetricHeap.heap_bases  |
+---------------------------+  +---------------------------+
```



上文分层图里的 `Layer 3: Sync Primitives` 与 `Layer 4: Memory`
更适合理解为：**Kernel 向下依赖的两个底层能力**，而不是“时间顺序上的下一步初始化流程”。



## 时间顺序上的全局流程

这一节是对上文分层图的**补充说明**，不替代原图，只是把容易混淆的两件事拆开：

1. `heap_bases` / peer mapping 是 **初始化阶段**准备好的运行时元数据
2. `translate_ptr()` / `tile_signal()` / `tile_wait()` 是 **kernel 执行阶段**实际使用的底层原语

### 1. 初始化链路（Host-side，只做一次）

```text
User Code
  |
  v
xtile.init_local(world_size, heap_size)
  |
  v
SymmetricHeap.create_all()
  |
  +--> 为每个 GPU 分配本地 symmetric heap buffer
  |
  +--> 建立 peer access / IPC mapping
  |
  +--> 为每个 rank 构造本 rank 视角下的 heap_bases
  |    heap_bases[i] = "在当前 GPU 地址空间中可解引用的 rank i heap 基址"
  |
  v
XTileContext.attach_heap()
  |
  v
ctx.heap_bases 传入 Triton kernel
```

这条链路的本质职责是：**把跨 GPU 可访问的对称堆环境准备好**，并把
`translate_ptr()` 运行时所需的 `heap_bases` 张量准备出来。

### 2. 运行时链路（Device-side，每次 kernel 执行）

```text
gemm_allscatter()
  |
  v
WGSpecializedPattern.execute()
  |
  v
_wg_specialized_kernel()
  |
  +--> Compute Workers
  |      |
  |      +--> tl.dot(...) 计算 tile
  |      +--> tl.store(local C tile)
  |      +--> tile_signal(locks, tile_id)
  |
  +--> Comm Workers
         |
         +--> tile_wait(locks, tile_id)
         +--> tl.load(local C tile)
         +--> scatter_tile_to_peer(...)
                 |
                 +--> translate_ptr(C_ptr, rank, peer, heap_bases)
                 +--> tl.store(remote_C, ...)
```

这条链路的本质职责是：**在 kernel 内完成 tile 级生产者-消费者同步，并把本地指针翻译成远端可访问指针**。



---

## 第一部分：主线流程

### 1. 用户入口 (xtile/ops.py)

用户通过 `xtile.ops.gemm_allscatter()` 发起一次 GEMM + AllScatter 融合操作：

```python
# 用户代码
import xtile

ctxs = xtile.init_local(world_size=2, heap_size=512 * 1024 * 1024)

for ctx in ctxs:
    A = ctx.randn(4096, 4096, dtype=torch.float16)
    B = ctx.randn(4096, 8192, dtype=torch.float16)
    C = ctx.zeros(4096, 8192, dtype=torch.float16)

    xtile.ops.gemm_allscatter(A, B, C, ctx=ctx, pattern="wg_spec")
```

`gemm_allscatter()` 内部构建一个执行计划并立即执行：

```python
# xtile/ops.py:826-857
def gemm_allscatter(A, B, C, *, ctx=None, full_N=None,
                    b_layout=None, c_layout=None,
                    pattern="auto", hw_info=None,
                    storage_kind="symmetric"):
    plan = build_gemm_allscatter_plan(
        A, B, C, ctx=ctx, full_N=full_N,
        b_layout=b_layout, c_layout=c_layout,
        pattern=pattern, hw_info=hw_info,
        storage_kind=storage_kind,
    )
    return plan.execute(A, B, C, validate=False)
```

### 2. Plan 构建 (xtile/ops.py)

`build_gemm_allscatter_plan()` 完成三件事：

**(a) 解析 layout contract**

默认无 layout hint 时，按 `full/full` 公共契约处理：

```python
# xtile/ops.py:1224-1243
def _resolve_public_layout_contract(*, b_layout, c_layout):
    if b_layout is None and c_layout is None:
        return "full", "full"  # 默认 full/full 公共契约
    ...
```

**(b) 解析 execution spec**

`resolve_pattern_execution()` 把张量 shape + layout 映射为标准化的 `PatternExecutionSpec`：

```python
# xtile/patterns/contracts.py:92-229
def resolve_pattern_execution(A, B, C, *, rank, world_size,
                               full_N=None, b_layout=None,
                               c_layout=None, storage_kind="symmetric"):
    M, K = int(A.shape[0]), int(A.shape[1])
    ...
    # full/full contract: scatter 时源列 = rank * shard_N, scatter_cols = shard_N
    scatter_cols = shard_N
    scatter_src_col_offset = rank * shard_N
    scatter_dst_col_offset = rank * shard_N
    scatter_dst_leading_dim = full_N

    return PatternExecutionSpec(
        M=M, K=K, full_N=full_N, local_N=b_cols,
        rank=rank, world_size=world_size,
        rhs=rhs_spec, output=output_spec,
        scatter_src_col_offset=scatter_src_col_offset,
        scatter_cols=scatter_cols,
        scatter_dst_leading_dim=scatter_dst_leading_dim,
        scatter_dst_col_offset=scatter_dst_col_offset,
    )
```

`PatternExecutionSpec` 核心字段语义：

```
full/full contract, world_size=2, rank=0:
    full_N = 8192, shard_N = 4096

    B(K=4096, N=8192)  -- full buffer
    C(M=4096, N=8192)  -- full buffer

    scatter_src_col_offset = 0 * 4096 = 0    # rank0 产出的列从第 0 列开始
    scatter_cols           = 4096             # 要 scatter 的列数
    scatter_dst_col_offset = 0               # 写入 peer 时的列偏移
    scatter_dst_leading_dim = 8192           # peer 上 C 的行步长
```

**(c) 解析 pattern 实现**

当用户指定 `pattern="wg_spec"` 或 auto_select 选中 WGSpecialized：

```python
# xtile/ops.py:1179-1221
def _resolve_pattern_impl(*, pattern, ctx, execution, hw_info):
    if isinstance(pattern, str):
        pattern_cls = _PATTERN_ALIASES.get(pattern)  # "wg_spec" -> WGSpecializedPattern
        return pattern_cls(ctx)
    ...
```

最终返回 `GemmAllScatterPlan`，其 `execute()` 调用 `pattern_impl.execute(A, B, C, spec=execution)`。

### 3. Pattern 执行 (xtile/patterns/wg_specialized.py)

`WGSpecializedPattern.execute()` 是 WG 模式的核心调度器：

```python
# xtile/patterns/wg_specialized.py:127-202
def execute(self, A, B, C, **kwargs):
    spec = self.resolve_execution(A, B, C, spec=kwargs.get("spec"), ...)
    M, K = A.shape
    N = spec.local_N                       # 本地列数

    # (1) 决定 SM 分配
    compute_sms, comm_sms = self._resolve_sm_split()
    total_sms = compute_sms + comm_sms     # 例如 H100: 106 compute + 26 comm = 132

    # (2) 计算 tile 数量
    num_tiles_m = triton.cdiv(M, self.BLOCK_M)   # 4096/128 = 32
    num_tiles_n = triton.cdiv(N, self.BLOCK_N)   # 8192/128 = 64
    total_tiles = num_tiles_m * num_tiles_n       # 32 * 64 = 2048

    # (3) 分配 lock 缓冲区 (per-tile 信号量)
    locks = self._get_locks(total_tiles, A.device)

    # (4) 单次 kernel launch -- grid = total_sms
    grid = (total_sms,)
    self._wg_specialized_kernel[grid](
        A, B, C, locks, heap_bases,
        M, N, K,
        spec.scatter_src_col_offset, spec.scatter_cols,
        spec.scatter_dst_leading_dim, spec.scatter_dst_col_offset,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        self.ctx.rank, world_size, compute_sms,
        num_tiles_m=num_tiles_m, num_tiles_n=num_tiles_n,
        total_tiles=total_tiles,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        COMPUTE_SMS=compute_sms, COMM_SMS=comm_sms,
        EVEN_K=(K % 64 == 0),
        num_warps=4, num_stages=4,
    )
```

SM 分配策略：

```python
# xtile/patterns/wg_specialized.py:83-99
def _resolve_sm_split(self):
    total_sms = self.ctx.backend.get_device_properties().compute_units
    comm = max(1, int(total_sms * 0.2))   # 默认 20% SM 给通信
    compute = total_sms - comm             # 80% SM 给计算
    return compute, comm                   # H100: (106, 26)
```

### 4. Triton 内核 (xtile/patterns/wg_specialized.py)

这是整个流程的核心——**一个 kernel 内两类 worker 并行运行**。

```
GPU SMs (H100 = 132 SMs):

  SM  0  ─┐
  SM  1   │  Compute Workers (pid 0..105)
  ...     │  持续 GEMM 计算 -> 写入本地 C -> signal
  SM 105 ─┘

  SM 106 ─┐
  SM 107  │  Comm Workers (pid 106..131)
  ...     │  wait tile 完成 -> 读取 C -> scatter 到 peer
  SM 131 ─┘
```

#### 4a. Compute Worker 路径

```python
# _wg_specialized_kernel 内部 (pid < COMPUTE_SMS)
if pid < COMPUTE_SMS:
    for tile_id in range(pid, total_tiles, COMPUTE_SMS):
        # --- tile 坐标计算 ---
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # --- Iris 风格 mask-free K-loop ---
        rm = offs_m % M    # 取模避免边界 mask
        rn = offs_n % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        rk = tl.arange(0, BLOCK_K)
        A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # --- GEMM 主循环 ---
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(A_BASE)
            b = tl.load(B_BASE)
            acc = tl.dot(a, b, acc, allow_tf32=True)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        # --- 写入本地 C ---
        result = acc.to(C_ptr.dtype.element_ty)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, result, mask=c_mask)

        # --- 发信号：此 tile 计算完毕 ---
        tile_signal(locks_ptr, tile_id)  # release 语义
```

#### 4b. Comm Worker 路径

```python
# _wg_specialized_kernel 内部 (pid >= COMPUTE_SMS)
else:
    comm_pid = pid - COMPUTE_SMS

    for tile_id in range(comm_pid, total_tiles, COMM_SMS):
        # --- 等待 compute worker 完成此 tile ---
        tile_wait(locks_ptr, tile_id)  # acquire 语义

        # --- 读取已完成的 tile 数据 ---
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tile_data = tl.load(c_ptrs, mask=c_mask, other=0.0)

        # --- scatter 到所有 peer ---
        for peer in range(world_size):
            if peer != rank:
                scatter_tile_to_peer(
                    C_ptr, tile_data, offs_m, offs_n,
                    rank, peer, heap_bases,
                    scatter_src_col_offset, scatter_cols,
                    scatter_dst_leading_dim, scatter_dst_col_offset,
                    c_mask,
                )
```

### 5. Scatter Helper (xtile/patterns/_helpers.py)

`scatter_tile_to_peer()` 是所有 pattern 共享的核心通信原语：

```python
# xtile/patterns/_helpers.py:18-91
@triton.jit
def scatter_tile_to_peer(
    C_ptr, tile_data, offs_m, offs_n,
    rank, peer, heap_bases,
    src_col_offset, valid_cols,
    dst_leading_dim, dst_col_offset,
    mask, CACHE_MODIFIER: tl.constexpr = ".wt",
):
    # (1) 指针翻译: 本地 C_ptr -> peer 地址空间
    remote_C = translate_ptr(C_ptr, rank, peer, heap_bases)

    # (2) 列映射: 本地列 -> peer 目标列
    col_mask = (offs_n >= src_col_offset) & (offs_n < src_col_offset + valid_cols)
    safe_local_cols = tl.where(col_mask, offs_n - src_col_offset, 0)
    offsets = offs_m[:, None] * dst_leading_dim + (dst_col_offset + safe_local_cols[None, :])
    final_mask = mask & col_mask[None, :]

    # (3) 远端写入 (write-through 绕过 L2 污染)
    tl.store(remote_C + offsets, tile_data, mask=final_mask, cache_modifier=".wt")
```

列映射图示 (full/full contract, world_size=2):

```
Rank 0 本地 C: [M x 8192] (full buffer)
               col 0       col 4095 col 4096     col 8191
               |-- rank0 产出 --|-- rank1 产出 --|

Rank 0 scatter 到 Rank 1:
  src_col_offset = 0
  valid_cols     = 4096
  dst_col_offset = 0
  dst_leading_dim = 8192

  从 C[0:M, 0:4096] 读取 tile_data
  写入 peer1 的 C[0:M, 0:4096] (translate_ptr 翻译后的远端地址)
```

### 6. 指针翻译 (xtile/memory/translation.py)

`translate_ptr` 是 XTile 最关键的 5 条指令：

```python
# xtile/memory/translation.py:38-99
@triton.jit
def translate_ptr(ptr, from_rank, to_rank, heap_bases, HINT: tl.constexpr = 0):
    """
    offset     = ptr - heap_bases[from_rank]
    remote_ptr = heap_bases[to_rank] + offset
    """
    # 1. 加载源 rank 堆基址
    from_base = tl.load(heap_bases + from_rank)

    # 2. 加载目标 rank 堆基址
    to_base = tl.load(heap_bases + to_rank)

    # 3. 计算字节偏移
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base

    # 4. 计算远端指针 (通过 byte-level 指针算术)
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_byte = to_base_byte + offset

    # 5. 类型还原
    translated_ptr = tl.cast(translated_byte, ptr.dtype)
    return translated_ptr
```

内存模型图示：

```
              GPU 0 地址空间                    GPU 1 地址空间
          ┌───────────────────┐           ┌───────────────────┐
          │                   │           │                   │
  0xA000  │ ┌─── Heap 0 ───┐ │           │ ┌─── Heap 1 ───┐ │  0xD000
          │ │               │ │           │ │               │ │
          │ │  C tensor     │ │           │ │  C tensor     │ │
          │ │   ptr = 0xA100│ │           │ │               │ │
          │ │               │ │           │ │               │ │
          │ └───────────────┘ │           │ └───────────────┘ │
          │                   │           │                   │
          │ ┌─── Heap 1 ───┐ │  IPC map  │                   │
  0xB000  │ │  (peer mapped)│ │ ========> │                   │
          │ │               │ │           │                   │
          │ └───────────────┘ │           │                   │
          └───────────────────┘           └───────────────────┘

  heap_bases (on GPU 0): [0xA000, 0xB000]
  heap_bases (on GPU 1): [0xC000, 0xD000]   (各自视角不同)

  translate_ptr(ptr=0xA100, from_rank=0, to_rank=1, heap_bases):
    offset = 0xA100 - 0xA000 = 0x100
    remote = 0xB000 + 0x100  = 0xB100  (GPU0 地址空间中对 GPU1 堆的映射)
```

### 7. 同步原语 (xtile/sync/primitives.py)

Compute worker 和 Comm worker 通过 `tile_signal` / `tile_wait` 形成 release-acquire 对：

```python
# tile_signal: compute worker 完成 tile 后调用
@triton.jit
def tile_signal(locks, tile_id, sem="release", scope="gpu"):
    tl.atomic_xchg(locks + tile_id, 1, sem=sem, scope=scope)

# tile_wait: comm worker 等待 tile 完成
@triton.jit
def tile_wait(locks, tile_id, sem="acquire", scope="gpu"):
    while tl.atomic_cas(locks + tile_id, 1, 0, sem=sem, scope=scope) != 1:
        pass  # CAS 自旋等待
```

时序图：

```
Compute Worker (SM 0)           locks[tile_id]        Comm Worker (SM 106)
       |                              0                       |
       | tl.store(C, tile_data)                               |
       |       (写入 tile 数据)                                |
       |                                                      |
       | tile_signal(locks, tile_id)                          |
       |-----> atomic_xchg(1, release) ----+                  |
       |                              1    |                  |
       |                                   |                  |
       |                                   +--> tile_wait()   |
       |                                   | atomic_cas(1->0) |
       |                              0    |   (acquire)      |
       |                                   |                  |
       |                                   |  tl.load(C, tile)|
       |                                   |  scatter_to_peer |
       v                                   v                  v
```

**release-acquire 保证**：compute worker 在 `tile_signal` 之前对 C 的所有写入，
在 comm worker 的 `tile_wait` 返回后可见。这是 C++ 内存模型的标准保证。

---

## 第二部分：初始化链路

用户代码到 kernel launch 之前的完整初始化路径：

### 1. Runtime 初始化

```python
# xtile/__init__.py:319-354
def init_local(world_size, heap_size, *, backend="auto"):
    backend = _detect_backend()  # "cuda" 或 "hip"

    # 创建所有 GPU 的对称堆
    heaps = SymmetricHeap.create_all(size=heap_size, world_size=world_size,
                                      backend=backend)
    contexts = []
    for rank, heap in enumerate(heaps):
        ctx = _build_context(backend_name=backend, rank=rank,
                             world_size=world_size)
        ctx.attach_heap(heap)
        contexts.append(ctx)
    return contexts
```

### 2. SymmetricHeap 创建

`SymmetricHeap.create_all()` 为每个 GPU 分配堆，启用 peer access，收集 heap_bases：

```python
# xtile/memory/symmetric_heap.py (简化)
@classmethod
def create_all(cls, size, world_size, backend="auto"):
    heaps = []
    for rank in range(world_size):
        torch.cuda.set_device(rank)
        heap = cls(size=size, rank=rank, world_size=world_size, backend=backend)
        heaps.append(heap)

    # 启用 peer access + 收集 bases
    for i in range(world_size):
        peer_bases = []
        for j in range(world_size):
            if i == j:
                peer_bases.append(heaps[j].local_base)
            else:
                # peer access 直接拿对方 device pointer
                enable_peer_access(i, j)
                peer_bases.append(heaps[j].local_base)  # NVLink 可直接解引用
        heaps[i]._heap_bases = torch.tensor(peer_bases, dtype=torch.int64,
                                             device=f"cuda:{i}")
    return heaps
```

### 3. XTileContext 数据流

```
XTileContext
  .rank = 0
  .world_size = 2
  .device = "cuda:0"
  .backend = CUDABackend(...)
  .heap = SymmetricHeap(...)
      .get_heap_bases() -> tensor([0xA000, 0xB000], dtype=int64, device='cuda:0')
      .mode = "single_process"
      .transport_strategy = "peer_access"
```

---

## 第三部分：数据流图

以 `M=4096, K=4096, N=8192, world_size=2, BLOCK=128` 为例：

```
                    GPU 0                                        GPU 1
          ┌─────────────────────┐                    ┌─────────────────────┐
          │  A (4096 x 4096)    │                    │  A (4096 x 4096)    │
          │  B (4096 x 8192)    │                    │  B (4096 x 8192)    │
          │  C (4096 x 8192)    │                    │  C (4096 x 8192)    │
          │                     │                    │                     │
          │  GEMM tiles:        │                    │  GEMM tiles:        │
          │  32 x 64 = 2048     │                    │  32 x 64 = 2048     │
          │                     │                    │                     │
          │  106 compute SMs    │                    │  106 compute SMs    │
          │  + 26 comm SMs      │                    │  + 26 comm SMs      │
          │                     │                    │                     │
          │  Compute: C = A @ B │                    │  Compute: C = A @ B │
          │     (all 2048 tiles)│                    │     (all 2048 tiles)│
          │                     │                    │                     │
          │  Scatter: rank0 把  │    NVLink (NV12)   │  Scatter: rank1 把  │
          │  C[:,0:4096] 写入 --│----- 双向 -------->│  C[:,4096:8192]写入 │
          │  GPU1 的 C[:,0:4096]│                    │  GPU0 的C[:,4096:]  │
          └─────────────────────┘                    └─────────────────────┘

最终效果: 两个 GPU 上的 C 都包含完整的 A @ B 结果
```

---

## 第四部分：细节分析

### 4.1 SM 分配策略

默认 80/20 分配（可配置）：

| 硬件 | Total SMs/CUs | Compute | Comm | 依据 |
|-------|--------------|---------|------|------|
| H100 PCIe | 132 | 106 | 26 | `_COMM_SM_FRACTION = 0.2` |
| MI300X | 304 | 243 | 61 | 同比例 |

分配比例的影响：
- Comm SM 过少 -> scatter 成为瓶颈，compute worker 空等
- Comm SM 过多 -> 算力浪费，GEMM 延迟增大
- 最优比例取决于 compute intensity (2MNK / MN_per_rank) 和互联带宽

### 4.2 Tile 调度：Persistent Kernel

Compute workers 和 comm workers 都使用 **persistent kernel** 模式——每个 SM 通过 for 循环跨步迭代所有 tile：

```
Compute SM 0:  tile 0, 106, 212, 318, ...  (stride = COMPUTE_SMS)
Compute SM 1:  tile 1, 107, 213, 319, ...
...
Compute SM 105: tile 105, 211, 317, ...

Comm SM 0:     tile 0, 26, 52, 78, ...     (stride = COMM_SMS)
Comm SM 1:     tile 1, 27, 53, 79, ...
...
Comm SM 25:    tile 25, 51, 77, ...
```

好处：
- 消除 kernel launch overhead（只需一次 launch）
- SM 利用率 100%（每个 SM 都有 work）
- 自然负载均衡（stride 保证均匀分配）

### 4.3 K-loop 优化

采用 Iris 风格的 mask-free K-loop：

```python
# 取模实现 mask-free 加载
rm = offs_m % M
rn = offs_n % N
rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
```

- `% M` / `% N`：越界时回绕到有效区域，避免 mask 分支
- `max_contiguous` / `multiple_of`：向编译器提示连续性，启用向量化加载
- 尾部 K 块（`EVEN_K=False`）仍需 mask 处理，但只有最后一次迭代

### 4.4 Cache 策略

| 操作 | Cache Modifier | 原因 |
|------|---------------|------|
| A/B 加载 | 默认 (.ca) | GEMM 数据有 K-loop 复用 |
| C 本地写入 | 默认 | 计算结果，comm worker 随后读取 |
| C 本地读取 (comm) | 默认 | 从 L2 读取刚写入的数据 |
| 远端写入 (scatter) | `.wt` (write-through) | 绕过本地 L2 污染，数据不会被本地再次使用 |

### 4.5 PatternExecutionSpec 中的 scatter 参数

`PatternExecutionSpec` 解耦了 pattern 内核与 layout 语义：

| 字段 | full/full (rank=0, ws=2) | shard/shard (rank=0, ws=2) |
|------|--------------------------|---------------------------|
| `scatter_src_col_offset` | 0 | 0 |
| `scatter_cols` | 4096 (= N/ws) | 4096 (= N_per_rank) |
| `scatter_dst_leading_dim` | 8192 (= full_N) | 4096 (= N_per_rank) |
| `scatter_dst_col_offset` | 0 | 0 |

kernel 不关心 full/shard 语义，只按 spec 参数执行写入。Layout 复杂度完全由 host-side `resolve_pattern_execution()` 吸收。

### 4.6 auto_select 何时选中 WG Specialization

```python
# xtile/patterns/auto_select.py (简化决策树)
n_per_rank = N // world_size
total_tiles = cdiv(M, 128) * cdiv(n_per_rank, 128)

if M < 256:
    -> BulkSync              # 行太少，overlap 无意义
elif n_per_rank < 1024 and K > 12288:
    -> FusedSequential       # 小 shard，大 K，简单 fusion 即可
elif n_per_rank < 2048 and K > 6144:
    -> ProducerConsumer      # 中等 shard，需要双流并行
elif total_tiles >= sm_count and K > 4096:
    -> WGSpecialized         # 大问题：tiles 多到足以饱和两个 SM pool
elif N > 4096 and K > 8192:
    -> WGSpecialized         # 大 N 后备路径
else:
    -> BulkSync              # 安全默认
```

核心判据：**tile 数量 >= SM 数量 + K 足够大** = WG Specialization 最优。

---

## 第五部分：层级总结

| 层级 | 模块 | 职责 | 核心类/函数 |
|------|------|------|------------|
| L0 - 公共 API | `xtile/ops.py` | 用户入口，contract 解析，plan 构建 | `gemm_allscatter()`, `GemmAllScatterPlan` |
| L1 - Pattern | `xtile/patterns/wg_specialized.py` | SM 分配，kernel launch，tile 调度 | `WGSpecializedPattern.execute()` |
| L2 - Kernel | 同上 (static method) | Compute/Comm worker 并行，GEMM + scatter | `_wg_specialized_kernel` |
| L3 - Scatter | `xtile/patterns/_helpers.py` | 列映射，远端写入 | `scatter_tile_to_peer()` |
| L4 - 翻译 | `xtile/memory/translation.py` | 对称堆指针翻译 (5 条指令) | `translate_ptr()` |
| L5 - 同步 | `xtile/sync/primitives.py` | Tile 级 release-acquire 信号 | `tile_signal()`, `tile_wait()` |
| L6 - Contract | `xtile/patterns/contracts.py` | Shape/layout 标准化 | `PatternExecutionSpec` |
| L7 - Context | `xtile/__init__.py` | Runtime 初始化，堆管理 | `XTileContext`, `init_local()` |
| L8 - Memory | `xtile/memory/symmetric_heap.py` | 对称堆分配，peer mapping | `SymmetricHeap.create_all()` |

完整调用链：

```
gemm_allscatter()
  -> build_gemm_allscatter_plan()
    -> resolve_pattern_execution()    # L6: 解析 contract
    -> _resolve_pattern_impl()        # 选择 WGSpecialized
  -> GemmAllScatterPlan.execute()
    -> WGSpecializedPattern.execute()  # L1: SM 分配 + launch
      -> _wg_specialized_kernel()      # L2: Triton kernel
        -> [Compute] tl.dot() loop     # GEMM 累积
        -> [Compute] tile_signal()     # L5: release 信号
        -> [Comm]    tile_wait()       # L5: acquire 等待
        -> [Comm]    scatter_tile_to_peer()  # L3
          -> translate_ptr()           # L4: 指针翻译
          -> tl.store(remote, .wt)     # 远端写入
```

---

# XTile WG 工作组模式第 4-6 节源码解读补充

本文是对文档 [XTile WG工作组模式完整代码流程报告.md](/home/makai/XTile/docs/XTile%20WG%E5%B7%A5%E4%BD%9C%E7%BB%84%E6%A8%A1%E5%BC%8F%E5%AE%8C%E6%95%B4%E4%BB%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E6%8A%A5%E5%91%8A.md) 中以下三节的补充解读：

- `4. Triton 内核 (xtile/patterns/wg_specialized.py)`
- `5. Scatter Helper (xtile/patterns/_helpers.py)`
- `6. 指针翻译 (xtile/memory/translation.py)`

目标不是改写原文，而是结合一个固定实例，按原文展示的源码顺序，把这些代码在矩阵中的含义解释清楚。

---

## 一、固定实例

后文统一使用这个例子：

- `world_size = 2`
- `full/full` contract
- 全局输出 `C` 的列数为 `8`
- 为了便于看清 tile 位置，假设：
  - `BLOCK_M = 4`
  - `BLOCK_N = 2`

这样，输出矩阵 `C` 会被切成 `2 x 4` 个 tile：

```text
C 的 tile 网格

            tile_n=0   tile_n=1   tile_n=2   tile_n=3
            cols 0-1   cols 2-3   cols 4-5   cols 6-7
          ---------------------------------------------
tile_m=0      T00        T01        T02        T03
tile_m=1      T10        T11        T12        T13
```

在 `full/full, world_size=2` 下，两边都会计算完整的 `C`，但 scatter 的列所有权是按列区间划分的：

```text
rank0 负责列 [0, 4)   -> T00 T01 T10 T11
rank1 负责列 [4, 8)   -> T02 T03 T12 T13
```

这个“负责”说的是 scatter 阶段谁对哪段列形成有效远端写入，不是说谁只计算哪一半。

---

## 二、对第 4 节的解读：Triton 内核

原文第 4 节给出的核心判断是：

```python
pid = tl.program_id(0)

if pid < COMPUTE_SMS:
    ...
else:
    ...
```

这段代码的直接含义是：

- 同一次 kernel launch 内，前一部分 program instance 是 compute worker
- 后一部分 program instance 是 comm worker
- 两类 worker 在同一个 kernel 内并行执行

这和原文图示是一一对应的：前面的 SM 负责 GEMM，后面的 SM 负责 wait + scatter。

### 2.1 Compute Worker 路径

原文第 4a 节代码：

```python
if pid < COMPUTE_SMS:
    for tile_id in range(pid, total_tiles, COMPUTE_SMS):
        # --- tile 坐标计算 ---
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # --- Iris 风格 mask-free K-loop ---
        rm = offs_m % M
        rn = offs_n % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        rk = tl.arange(0, BLOCK_K)
        A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # --- GEMM 主循环 ---
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(A_BASE)
            b = tl.load(B_BASE)
            acc = tl.dot(a, b, acc, allow_tf32=True)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        # --- 写入本地 C ---
        result = acc.to(C_ptr.dtype.element_ty)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, result, mask=c_mask)

        # --- 发信号：此 tile 计算完毕 ---
        tile_signal(locks_ptr, tile_id)  # release 语义
```

下面按原代码顺序解读。

#### 2.1.1 `for tile_id in range(pid, total_tiles, COMPUTE_SMS)`

这句的含义必须严格按代码理解：

- 当前 compute worker 从自己的 `pid` 开始领取 tile
- 每次跨 `COMPUTE_SMS` 个 tile 继续处理下一个
- 所有 compute worker 共同覆盖 `0 .. total_tiles-1`

它并不是“先处理 rank0 负责的左半边，再处理 rank1 负责的右半边”。

如果示意地假设 `COMPUTE_SMS = 4`，那么：

```text
pid 0 -> tile_id 0, 4
pid 1 -> tile_id 1, 5
pid 2 -> tile_id 2, 6
pid 3 -> tile_id 3, 7
```

对应到 tile 网格：

```text
pid 0 -> T00, T10
pid 1 -> T01, T11
pid 2 -> T02, T12
pid 3 -> T03, T13
```

从这句代码本身就可以看出，tile 的计算调度是全局 round-robin。

#### 2.1.2 `tile_m / tile_n / offs_m / offs_n`

原文代码：

```python
tile_m = tile_id // num_tiles_n
tile_n = tile_id % num_tiles_n
offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

这四句是 tile 定位的核心。

拿 `tile_id = 6` 举例：

- `num_tiles_n = 4`
- `tile_m = 6 // 4 = 1`
- `tile_n = 6 % 4 = 2`

所以这个 tile 是 `T12`。

再代入本例中 `BLOCK_M = 4, BLOCK_N = 2`：

- `offs_m = 1 * 4 + [0, 1, 2, 3] = [4, 5, 6, 7]`
- `offs_n = 2 * 2 + [0, 1] = [4, 5]`

也就是说：

```text
tile_id = 6
=> tile_m = 1, tile_n = 2
=> 这个 tile 对应 C[4:8, 4:6]
=> 它就是图中的 T12
```

所以这四句代码不是抽象意义上的“算 tile 坐标”，而是精确地把一维 `tile_id` 恢复成矩阵中的二维 tile 位置。

#### 2.1.3 `rm / rn`

原文代码：

```python
rm = offs_m % M
rn = offs_n % N
rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
```

这几句的作用是：

- `rm` / `rn` 是真正用于 A/B 访存的行列索引
- `% M`、`% N` 让边界 tile 也能复用同一条主路径
- `tl.multiple_of` / `tl.max_contiguous` 是给编译器的约束与向量化提示

在本例里，如果 `tile_id = 6` 对应 `T12`，那么主路径中：

- `rm` 代表输出 tile 的行块 `[4, 5, 6, 7]`
- `rn` 代表输出 tile 的列块 `[4, 5]`

#### 2.1.4 `A_BASE / B_BASE / acc`

原文代码：

```python
rk = tl.arange(0, BLOCK_K)
A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

这三句定义了当前输出 tile 的一次 K 分块计算所需的两块输入：

- `A_BASE` 指向 `A` 的一个 `BLOCK_M x BLOCK_K` 子块
- `B_BASE` 指向 `B` 的一个 `BLOCK_K x BLOCK_N` 子块
- `acc` 是当前输出 tile 的寄存器累加器

继续用 `T12 = C[4:8, 4:6]` 举例：

- `A` 需要的是固定行块 `4:8`
- `B` 需要的是固定列块 `4:6`
- 然后沿着 `K` 维不断前进

所以这里不是“一次取完整的 A/B”，而是：

```text
第一轮：取 A[4:8, 0:BLOCK_K]，B[0:BLOCK_K, 4:6]
第二轮：取 A[4:8, BLOCK_K:2*BLOCK_K]，B[BLOCK_K:2*BLOCK_K, 4:6]
...
```

#### 2.1.5 GEMM 主循环

原文代码：

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(A_BASE)
    b = tl.load(B_BASE)
    acc = tl.dot(a, b, acc, allow_tf32=True)
    A_BASE += BLOCK_K * stride_ak
    B_BASE += BLOCK_K * stride_bk
```

这段代码的含义是：

- 当前 worker 正在计算某个固定输出 tile
- 它每轮加载一对 `A/B` 的 K 分块
- 做一次 `tl.dot`
- 然后把 A、B 的指针都沿 K 维推进一块

对 `T12` 来说，这段循环实际上是在做：

```text
T12 =
  A[4:8, 0:BK]    @ B[0:BK, 4:6]
+ A[4:8, BK:2BK]  @ B[BK:2BK, 4:6]
+ A[4:8, 2BK:3BK] @ B[2BK:3BK, 4:6]
+ ...
```

也就是说，`K` 是 GEMM 的归约维：

- `A` 的列维
- `B` 的行维
- 每个输出元素做内积时累加的那一维

#### 2.1.6 写回本地 C 并 signal

原文代码：

```python
result = acc.to(C_ptr.dtype.element_ty)
c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, result, mask=c_mask)

tile_signal(locks_ptr, tile_id)
```

这几句必须连起来看：

1. `c_ptrs` 定位本地 `C` 中当前 tile 的位置
2. `tl.store` 把计算好的 tile 写回本地 `C`
3. `tile_signal(locks_ptr, tile_id)` 通知 comm worker：这个 tile 可以消费了

还是用 `tile_id = 6 / T12` 举例：

```text
Compute worker 在 rank0 上算出 T12
-> 写到 rank0 本地 C[4:8, 4:6]
-> 把 locks[6] 标成 ready
```

到这里为止，compute worker 的责任已经结束。它只保证：

- tile 已经被算出来
- tile 已经放进本地 `C`
- tile 对 comm worker 可见

它并不负责 scatter。

### 2.2 Comm Worker 路径

原文第 4b 节代码：

```python
else:
    comm_pid = pid - COMPUTE_SMS

    for tile_id in range(comm_pid, total_tiles, COMM_SMS):
        # --- 等待 compute worker 完成此 tile ---
        tile_wait(locks_ptr, tile_id)  # acquire 语义

        # --- 读取已完成的 tile 数据 ---
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tile_data = tl.load(c_ptrs, mask=c_mask, other=0.0)

        # --- scatter 到所有 peer ---
        for peer in range(world_size):
            if peer != rank:
                scatter_tile_to_peer(
                    C_ptr, tile_data, offs_m, offs_n,
                    rank, peer, heap_bases,
                    scatter_src_col_offset, scatter_cols,
                    scatter_dst_leading_dim, scatter_dst_col_offset,
                    c_mask,
                )
```

下面按原顺序解释。

#### 2.2.1 `comm_pid = pid - COMPUTE_SMS`

这句的作用只是把 comm worker 的编号从 `0` 开始重新编号。它没有改变 tile 的逻辑，只是把：

- 全局 `pid`

映射成：

- comm 池内部编号 `comm_pid`

#### 2.2.2 `for tile_id in range(comm_pid, total_tiles, COMM_SMS)`

这句和 compute worker 的那句是对称的：

- 每个 comm worker 也按 round-robin 领一串 `tile_id`
- 它不是只领取“本 rank 负责的 tile”
- 它先按全局 `tile_id` 领任务，再在 scatter 阶段决定哪些列有效

这个结论必须直接从源码读出来，不能替换成“owner-first”这类更直观但与当前代码不完全一致的表述。

#### 2.2.3 `tile_wait(locks_ptr, tile_id)`

这句的语义是：

- comm worker 等待对应的 compute worker 完成当前 `tile_id`
- 只有在 compute worker 已经对该 tile 执行完 `tl.store(...)` 之后，它才会返回

所以如果当前 comm worker 轮到 `tile_id = 6`，它等待的就是 `T12` 这块 tile 的本地写回已经完成。

#### 2.2.4 再次恢复 `tile_m / tile_n / offs_m / offs_n`

原文代码：

```python
tile_m = tile_id // num_tiles_n
tile_n = tile_id % num_tiles_n
offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

这和 compute worker 使用的是同一套坐标恢复逻辑。原因很直接：

- compute worker 按这套坐标把 tile 写进了本地 `C`
- comm worker 必须用完全相同的坐标把同一个 tile 从本地 `C` 读出来

所以对 `tile_id = 6` 而言：

- compute worker 写的是 `C[4:8, 4:6]`
- comm worker 读的也必须是 `C[4:8, 4:6]`

#### 2.2.5 `tile_data = tl.load(c_ptrs, ...)`

这句的作用非常明确：

- 从本地 `C` 中读出刚刚算好的 tile
- 结果保存在寄存器里的 `tile_data`

所以 comm worker 并不重新做 GEMM，它只是消费 compute worker 产出的 tile。

#### 2.2.6 `for peer in range(world_size)`

原文这里没有写任何“只对左半边 tile 才 scatter”的条件，而是：

- 对所有 `peer`
- 只跳过 `peer == rank`
- 然后统一调用 `scatter_tile_to_peer(...)`

这说明 comm worker 阶段并不是靠 `if tile_n ...` 这种条件提前裁掉 tile，而是把：

- 当前 tile 的坐标 `offs_m / offs_n`
- 当前 rank 的 scatter contract

一起交给 `scatter_tile_to_peer()` 去做最后判断。

---

## 三、对第 5 节的解读：Scatter Helper

原文第 5 节代码：

```python
@triton.jit
def scatter_tile_to_peer(
    C_ptr, tile_data, offs_m, offs_n,
    rank, peer, heap_bases,
    src_col_offset, valid_cols,
    dst_leading_dim, dst_col_offset,
    mask, CACHE_MODIFIER: tl.constexpr = ".wt",
):
    # (1) 指针翻译: 本地 C_ptr -> peer 地址空间
    remote_C = translate_ptr(C_ptr, rank, peer, heap_bases)

    # (2) 列映射: 本地列 -> peer 目标列
    col_mask = (offs_n >= src_col_offset) & (offs_n < src_col_offset + valid_cols)
    safe_local_cols = tl.where(col_mask, offs_n - src_col_offset, 0)
    offsets = offs_m[:, None] * dst_leading_dim + (dst_col_offset + safe_local_cols[None, :])
    final_mask = mask & col_mask[None, :]

    # (3) 远端写入 (write-through 绕过 L2 污染)
    tl.store(remote_C + offsets, tile_data, mask=final_mask, cache_modifier=".wt")
```

### 3.1 `remote_C = translate_ptr(...)`

这句只是把本地 `C_ptr` 翻译成当前 rank 地址空间中对 peer 堆的映射地址。

它并不决定 scatter 哪些列有效，只决定：

- 写入的目标地址从“本地 C 基址”
- 变成“peer 的 C 基址映射”

### 3.2 `col_mask`

真正决定“这个 tile 对当前 rank 是否属于有效 scatter 区间”的是这句：

```python
col_mask = (offs_n >= src_col_offset) & (offs_n < src_col_offset + valid_cols)
```

在 `full/full, world_size=2` 下：

- `rank0` 的 contract 是：
  - `src_col_offset = 0`
  - `valid_cols = 4`
- `rank1` 的 contract 是：
  - `src_col_offset = 4`
  - `valid_cols = 4`

因此对 `rank0` 来说：

- `T00` 的 `offs_n = [0, 1]`，有效
- `T01` 的 `offs_n = [2, 3]`，有效
- `T02` 的 `offs_n = [4, 5]`，无效
- `T03` 的 `offs_n = [6, 7]`，无效

所以不是 comm worker 前面只等待左半边 tile，而是：

- comm worker 会按自己的 `tile_id` 序列消费 tile
- 到了 `scatter_tile_to_peer()` 这一步，才由 `col_mask` 决定这块 tile 是否落在当前 rank 负责的列区间里

### 3.3 `safe_local_cols / offsets / final_mask`

原文代码：

```python
safe_local_cols = tl.where(col_mask, offs_n - src_col_offset, 0)
offsets = offs_m[:, None] * dst_leading_dim + (dst_col_offset + safe_local_cols[None, :])
final_mask = mask & col_mask[None, :]
```

这几句是在做“局部列坐标到远端目标列坐标”的映射。

如果当前是 `rank0`，负责列 `[0, 4)`，那么：

- 对 `T01`，`offs_n = [2, 3]`
- `safe_local_cols = [2, 3]`
- `final_mask = true`

因此 `T01` 会真正落到远端对应位置。

但对 `T02`：

- `offs_n = [4, 5]`
- `col_mask = false`
- `final_mask = false`

于是虽然 `tile_data` 已经从本地 `C` 中被读出来，这个 tile 最终并不会形成 rank0 负责的有效远端写入。

### 3.4 `tl.store(remote_C + offsets, tile_data, mask=final_mask, ...)`

这句是 scatter helper 的最后一步：

- 把当前 tile 中属于本 rank scatter 区间的部分
- 写到 peer 的远端 `C`

如果 `final_mask` 全为 `false`，那么这次 store 对当前 rank 的 scatter 语义来说就是空操作。
