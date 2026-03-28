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
