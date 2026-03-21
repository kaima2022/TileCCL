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
   - 这条路径今天确实在跑，但它和 pattern `execute()` 对 `N` 的解释还没有完全统一

因此，下面文档会明确区分“**当前 correctness 主路径**”和“**当前 benchmark 主路径**”，不再混写成一个完全统一的调用模型。

---

## 第 1 层：用户代码

### 实际代码（方式 A：当前仓库可执行路径，非一键 API）

【修订】这部分原先写成“先 `xtile.init()`，再手工拼 `_PatternCtx`，再把 `heap_bases` 塞回去”。这在 2026-03-20 之后已经不是当前主路径。现在 pattern / CLI / benchmarks 都应复用真实 `XTileContext`。

```python
import torch
import xtile

contexts = xtile.init_local(
    world_size=4,
    heap_size=1 << 30,
)
ctx = contexts[0]  # rank 0

A = ctx.randn(M, K, dtype=torch.float16)
B = ctx.randn(K, N, dtype=torch.float16)
C = ctx.zeros(M, N, dtype=torch.float16)

pattern = ctx.auto_select_pattern(
    "gemm_allscatter",
    M=M,
    N=N,
    K=K,
)
pattern.execute(A, B, C)
```

### 实际代码（方式 B：当前 benchmark 主路径）

【新增核对】`tests/benchmarks/bench_patterns.py` 当前使用的是“每 rank 只给本地列分片”的形状约定：

```python
import torch
import xtile

N_per_rank = N // world_size
contexts = xtile.init_local(world_size=world_size, heap_size=heap_size)
ctx = contexts[0]

A = ctx.randn(M, K, dtype=torch.float16)
B = ctx.randn(K, N_per_rank, dtype=torch.float16)
C = ctx.zeros(M, N_per_rank, dtype=torch.float16)

pattern = ctx.auto_select_pattern(
    "gemm_allscatter",
    M=M,
    N=N,
    K=K,
)
pattern.execute(A, B, C)
```

【新增状态更新 2026-03-21】需要特别注意：这条 benchmark 路径今天**能跑通并产出相对性能数据**，但 pattern `execute()` 对 `B.shape[1]` 的解释还没有和“full-N 语义”完全统一。换句话说，它是当前真实 benchmark 路径，但还不是已经完全收敛的最终 API 语义。

### 与设想的差距

| 设想 | 现状 | 差距 |
|------|------|------|
| `xtile.fused_gemm_scatter(A, B, C, strategy="auto")` | `pattern = auto_select(...); pattern.execute(...)` | 缺少一键 API，用户仍需手动组织调用 |
| `ctx.randn(...)` 直接在堆上分配 | `XTileContext` 已支持 `empty/zeros/randn()` | 这一项已打通 |
| 自动检测后端 `backend="auto"` | `xtile.init()` 默认就是 `backend="auto"` | 这一项已集成 |
| `xtile.init()` 返回的 ctx 可直接执行 pattern | `xtile.init(..., heap=...)` / `heap_size=...` / `init_local(...)` 都可直接返回可运行 pattern 的真实 ctx | runtime ctx 主路径已统一 |
| benchmark / tests / CLI 与 runtime 入口一致 | 已统一到 `XTileContext` | 这一项已打通 |
| 高层 op API | 仍缺 `xtile.ops.gemm_allscatter(...)` | 这是现在真正剩下的用户层缺口 |
| full-shape correctness path 与 benchmark shard path | 两者都存在，但 shape 语义尚未完全统一 | 这是当前 API 层最需要继续收敛的地方 |

---

## 第 2 层：Pattern 层（以 FusedSequential 为例）

### 核心 kernel 逻辑（从实际代码摘取，非逐参数原样转录）

【新增核对】下面这段保留了当前 `xtile/patterns/fused_sequential.py` 的核心控制流和地址计算，但省略了 `heap_bases` 的真实参数位置、`num_tiles_m/num_tiles_n` 等样板参数。因此它应理解为“贴近源码的整理版”，不是逐行拷贝。

```python
# xtile/patterns/fused_sequential.py — _fused_kernel

@triton.jit
def _fused_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K, N_per_rank,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    rank, world_size,
    heap_bases,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # ====== 与 Iris 完全相同的 persistent kernel 循环 ======
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # --- Iris 风格 GEMM ---
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)

        A_BASE = A_ptr + rm[:, None] * stride_am
        B_BASE = B_ptr + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # EVEN_K 分离式 K-loop：主循环零 mask
        if EVEN_K:
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                rk = tl.arange(0, BLOCK_K)
                a = tl.load(A_BASE + rk[None, :] * stride_ak)
                b = tl.load(B_BASE + rk[:, None] * stride_bk)
                acc = tl.dot(a, b, acc)
                A_BASE += BLOCK_K * stride_ak
                B_BASE += BLOCK_K * stride_bk
        else:
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                rk = tl.arange(0, BLOCK_K)
                k_mask = (k * BLOCK_K + rk) < K
                a = tl.load(A_BASE + rk[None, :] * stride_ak, mask=k_mask[None, :])
                b = tl.load(B_BASE + rk[:, None] * stride_bk, mask=k_mask[:, None])
                acc = tl.dot(a, b, acc)
                A_BASE += BLOCK_K * stride_ak
                B_BASE += BLOCK_K * stride_bk

        c = acc.to(C_ptr.type.element_ty)

        # --- 本地 store ---
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        C_tile_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(C_tile_ptr, c, mask=c_mask)

        # --- 立刻 scatter 到所有 peer（Iris fused 风格）---
        for peer in range(world_size):
            if peer != rank:
                dst_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_per_rank)
                scatter_tile_to_peer(
                    C_ptr, c, offs_m, offs_n,
                    rank, peer, N, N_per_rank,
                    heap_bases, dst_mask,
                )
```

### 与设想的对比

**一致**：
- Persistent kernel + round-robin 调度 ✅
- 每算完一个 tile 立刻 scatter ✅
- `scatter_tile_to_peer` 使用 `translate_ptr` + `tl.store` ✅
- `heap_bases` 作为 kernel 参数传入 ✅

**差异（修订）**：核心通信原理接近 Iris，但当前 XTile 的 shape 约定还没有完全收敛。correctness tests 使用完整 `B(K,N)`、完整 `C(M,N)`；benchmarks / CLI 则使用 `B(K,N_per_rank)`、`C(M,N_per_rank)`；而 pattern `execute()` 今天仍然普遍从 `B.shape[1]` 推导 `N`。因此这里不宜再写成“无实质差异”，更准确的说法是“**底层 primitive 路线成立，但 API 形状约定仍需统一**”。

---

## 第 3 层：通信底层

### 实际 scatter_tile_to_peer 实现

```python
# xtile/patterns/_helpers.py

@triton.jit
def scatter_tile_to_peer(
    C_ptr, tile_data, offs_m, offs_n,
    rank, peer, N, N_per_rank,
    heap_bases, mask,
    CACHE_MODIFIER: tl.constexpr = ".wt",  # 默认 write-through
):
    # 步骤 1：翻译指针到 peer 的地址空间
    remote_C = translate_ptr(C_ptr, rank, peer, heap_bases)

    # 步骤 2：计算目标偏移（考虑 column-shard 布局）
    dst_col_offset = rank * N_per_rank
    offsets = offs_m[:, None] * N + (dst_col_offset + offs_n[None, :])

    # 步骤 3：写入远端内存
    if CACHE_MODIFIER == ".wt":
        tl.store(remote_C + offsets, tile_data, mask=mask, cache_modifier=".wt")
    else:
        tl.store(remote_C + offsets, tile_data, mask=mask)
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

**基本一致（修订）**：`translate_ptr + tl.store` 这条核心路径成立，编译器也确实全可见；但输出 buffer 是完整 `(M,N)` 还是本地 shard `(M,N_per_rank)`，当前仓库内部仍未完全统一。
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

### 模式 B：多进程 — _setup_multiprocess（三级 fallback）

```python
# xtile/memory/symmetric_heap.py

def _setup_multiprocess(self):
    """多进程模式，三级 fallback："""
    self._backend.init_ipc()

    # --- 策略 1: ctypes IPC handle 交换 ---
    # cudaIpcGetMemHandle + cudaIpcOpenMemHandle
    # 需要 ptrace_scope=0，否则 error 201
    try:
        local_handle = self._backend.get_ipc_handle(self._local_ptr)
        # ... dist.all_gather 交换 handle，open_ipc_handle 打开 ...
        return
    except RuntimeError:
        pass  # 继续尝试策略 2

    # --- 策略 2: PyTorch IPC（文件描述符共享）---
    # storage._share_cuda_() + _new_shared_cuda()
    # 使用 Unix domain socket 传递 fd，绕过 ptrace_scope ✅
    try:
        storage = self._buffer.untyped_storage()
        share_info = storage._share_cuda_()

        all_infos = [None] * self._world_size
        dist.all_gather_object(all_infos, share_info)

        remote_ptrs = []
        for r in range(self._world_size):
            if r == self._rank:
                remote_ptrs.append(self._local_ptr)
            else:
                remote_storage = torch.UntypedStorage._new_shared_cuda(*all_infos[r])
                remote_ptrs.append(remote_storage.data_ptr())

        self._heap_bases = torch.tensor(remote_ptrs, dtype=torch.int64, device=self._device)
        return
    except Exception:
        pass  # 继续尝试策略 3

    # --- 策略 3: peer access 指针交换（同节点 fallback）---
    # dist.all_gather 交换 raw data_ptr()
    local_base = torch.tensor([self._local_ptr], dtype=torch.int64, device=self._device)
    all_bases = [torch.zeros(1, dtype=torch.int64, device=self._device)
                 for _ in range(self._world_size)]
    dist.all_gather(all_bases, local_base)
    self._heap_bases = torch.cat(all_bases)
```

### IPC 诊断实测结果

跨进程测试（`torch.multiprocessing.spawn`，2× H100，ptrace_scope=1）：

| 方法 | 结果 | 原理 |
|------|------|------|
| peer access (非 IPC) | ✅ PASS | cudaDeviceEnablePeerAccess，单进程直接寻址 |
| ctypes IPC (Structure) | ❌ FAIL | cudaIpcOpenMemHandle 被 ptrace_scope=1 阻止 |
| **PyTorch IPC** (_share_cuda_) | ✅ **PASS** | Unix domain socket fd 共享，不受 ptrace_scope 限制 |

### 与设想的对比

| 设想 | 现状 | 状态 |
|------|------|------|
| cudaIpcGetMemHandle + Open | ctypes Structure by-value 修复 | ✅ 调用约定已修正 |
| IPC 在 ptrace_scope=1 下工作 | PyTorch IPC fallback 绕行 | ✅ 已解决 |
| 多进程模式（torchrun） | 三级 fallback 完整实现 | ✅ 可用 |
| 跨节点 IPC | 不支持 | ⚠️ 需 UCX/GDR |

**与 Iris 的区别（修订）**：不能再简单写成“Iris 仅使用 HIP IPC 单一路径”。当前 Iris 的主实现已经演进到 allocator + fd passing + DMA-BUF 映射；XTile 的特点是显式实现了 ctypes IPC → PyTorch IPC → peer access 三层 fallback。

### 【新增状态更新 2026-03-21】Iris 与 XTile 的实质差异、优劣与对齐方向

这里需要把“都能建立 symmetric memory”拆开看，二者今天其实代表两种不同风格：

| 维度 | Iris | XTile |
|------|------|------|
| 主抽象 | `SymmetricHeap` + allocator backend | `SymmetricHeap` + mode/fallback |
| 内存来源 | `TorchAllocator` / `VMemAllocator` 可切换 | 当前主要是 `torch.empty(uint8)` bump allocator |
| 远端映射主路径 | FD passing + DMA-BUF export/import + VA map/access | ctypes IPC → PyTorch IPC → peer access fallback |
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
   - 同一接口下优先尝试最直接的 ctypes IPC
   - 不行就退到 PyTorch IPC
   - 再不行就退到单节点 peer access
   - 对当前 NVIDIA bring-up 和 benchmark 迭代更友好

**哪个更好？**

- 如果问“**长期架构形态**”，Iris 这一套 allocator + import/map 更好，原因是语义更统一、可扩展性更强、未来更容易把多进程 / 多节点 / 多 allocator 做成同一个 canonical path。
- 如果问“**当前这台 H100 机器上谁更实用**”，XTile 现在更好，因为它把多种现实限制都兜住了，尤其是 `ptrace_scope=1` 这类环境问题下仍能跑通。

**XTile 应该如何向 Iris 对齐，而不是简单照抄？**

总纲不变，但详细动作已经并入下文 `【新增计划 2026-03-21】` 的每一个“未统一点”中。核心原则只有三条：

1. 保留 XTile 当前 `ctypes IPC → PyTorch IPC → peer access` 的现实 fallback，不牺牲 NVIDIA bring-up 成功率。
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

【新增核对】P2P benchmark 显示 best read ≈ 248.70 GB/s、best write ≈ 248.14 GB/s；scatter 更相关的是写带宽，量级约 82.7%–82.9% 峰值。
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
- 当前支持的 op 名是 `gemm_allscatter`、`gemm_allgather`、`gemm_reducescatter`，不是 `gemm_scatter`。
- 对本例 `N=4608, world_size=4`，`n_per_rank = 1152`，不会命中 `fused_sequential` 的 `< 1024` 分支，而会更接近 `producer_consumer` 分支。
- 当前实现还带 `compute_intensity` 与 `N > 4096` 的补充分支，不再只是最初那三条简单规则。

---

## 总体差距汇总

## 【新增计划 2026-03-21】文档中所有“未统一点”的改进计划

下面只列当前文档里已经明确暴露出来、且确实还没统一的点，不额外扩张范围。

### U1. full-shape correctness path 与 benchmark shard path 未统一

**现状**：
- tests 常用 `B(K, N)` / `C(M, N)`
- benchmark 常用 `B(K, N_per_rank)` / `C(M, N_per_rank)`
- 同一个 `pattern.execute(A, B, C)` 在不同路径下背负了不同语义

**风险**：
- 用户无法仅靠函数签名理解输入输出语义
- benchmark 结果与 correctness 路径不完全同构
- 后续一键 API 很难定义稳定契约

**改进计划**：
1. 明确定义两类公开语义，只保留其中一类作为主语义。
2. 推荐主语义选 `full-N API`：
   - `A(M, K)`
   - `B(K, N)`
   - `C(M, N)` 或显式 `C_local(M, N)` / `C_shard(M, N_per_rank)`
3. benchmark 若需要 shard-friendly fast path，应下沉为内部 helper，不再直接复用公开语义名字。
4. 为 shard path 增加显式入口，例如：
   - `execute_sharded(A, B_shard, C_shard, full_N=...)`
   - 或 host-side wrapper 先补齐 metadata，再调统一内核入口。
5. 与 Iris 对齐时，用户层只暴露“逻辑全量 shape”的稳定契约，不把 benchmark shard path 当第一类 public API；就像 Iris 的 `allocate(...)` / `as_symmetric(...)` 不会把底层 allocator 或导入路径直接暴露给用户。
6. `xtile.ops.gemm_allscatter(...)` 这类高层入口应负责把 full-shape 语义规范化成内部 shard 计划，pattern 层只接收已经解歧义后的 plan object。

**验收标准**：
- `pattern.execute(...)` 只有一种明确公开语义
- tests / benchmarks / CLI 都不再混用同名接口承载两种 shape 语义
- 对外文档与示例统一站在“逻辑全量张量”视角，内部 shard 只是执行实现细节

### U2. pattern `execute()` 对 `B.shape[1]` 的解释未统一

**现状**：
- host 侧很多 pattern 直接用 `_, N = B.shape`
- 这会让 `N` 在不同调用路径下代表“full N”或“local shard N”

**风险**：
- `N_per_rank = N // world_size` 之类逻辑会在 shard path 下继续缩一轮
- auto-select 与真实 kernel work decomposition 可能错位

**改进计划**：
1. 所有 pattern host API 明确接收 `full_N` 元数据，禁止再隐式把 `B.shape[1]` 当 full `N`。
2. 内部统一为：
   - `full_N`
   - `local_N`
   - `n_per_rank`
3. 在所有 pattern `execute()` 开头加入 shape contract 校验。
4. 对不符合契约的调用直接 `raise ValueError`，不再“能跑就跑”。
5. 与 Iris 对齐时，不再从裸 tensor shape 反推逻辑语义，而是像 Iris 对 allocator / segment / import 状态那样，把关键语义提升成显式 metadata。
6. 新增统一的 `LayoutSpec` / `ShardSpec` 描述对象，至少包含：
   - `full_shape`
   - `local_shape`
   - `rank`
   - `world_size`
   - `layout_kind`
   - `storage_kind`
7. auto-select、pattern host code、benchmark 统计口径都只消费这份显式 spec，不再各自重复猜 `B.shape[1]` 的含义。

**验收标准**：
- `B.shape[1]` 不再被当作唯一的 `N` 来源
- 每个 pattern 都能清楚区分 full-N 与 shard-N
- 调度决策与执行分解不再依赖“从 shape 猜语义”

### U3. 输出缓冲区语义未统一：完整 `(M, N)` 还是本地 shard `(M, N_per_rank)`

**现状**：
- correctness path 里 `C` 常是完整输出
- benchmark path 里 `C` 常是本地 shard
- scatter helper 又按“peer 保存完整输出缓冲区中某一列分片”来写

**风险**：
- 存储布局与 scatter 布局容易出现隐式假设
- 输出 buffer 的越界 / mask / offset 语义不稳定

**改进计划**：
1. 明确定义输出布局模型：
   - 方案 A：所有 rank 都持完整 `C(M, N)`，scatter 填自己的 shard 区间
   - 方案 B：所有 rank 只持本地 `C_shard(M, N_per_rank)`，all-scatter 另有目的缓冲区
2. 推荐把 correctness 与 benchmark 都迁到同一个模型。
3. `scatter_tile_to_peer(...)` 改成显式携带 `layout` 或更明确的 host-side wrapper。
4. tests 新增：
   - full-buffer layout correctness
   - shard-buffer layout correctness
   - 二者若都保留，则必须名字不同、契约不同
5. 与 Iris 对齐时，输出 buffer 的“逻辑布局”与“底层存储来源”必须拆开表达；不能因为张量在 symmetric heap、外部导入张量、或普通本地张量上，就改变 `C` 的逻辑语义定义。
6. 为输出张量引入统一 descriptor，至少记录：
   - `logical_layout`（full / shard）
   - `storage_kind`（symmetric / imported / local）
   - `owner_rank`
   - `full_N`
   - `local_N`
7. 如果 full-buffer 与 shard-buffer 两种模型都保留，则 API 名、文档名、测试名都必须显式区分，不能再共享同一个 `C` 叙事。

**验收标准**：
- 文档、tests、benchmark、pattern 内核对 `C` 的语义一致
- 输出布局切换不再依赖隐式假设，而依赖显式 descriptor / wrapper

### U4. 高层 op API 缺失，用户层入口未统一

**现状**：
- 当前用户仍需 `ctx.auto_select_pattern(...); pattern.execute(...)`
- 缺少统一高层入口

**风险**：
- 用户直接暴露在 pattern / shape / layout 细节之上
- API 难以稳定演进

**改进计划**：
1. 新增 `xtile.ops` 模块。
2. 第一批只做：
   - `gemm_allscatter(...)`
   - `gemm_reducescatter(...)`
   - `allgather(...)`
3. 高层 API 负责：
   - 选择 pattern
   - 校验 shape
   - 规范 full_N / shard_N 元数据
   - 在必要时分配中间缓冲区
4. pattern 类退回“专家接口”，不再作为默认用户入口。
5. 与 Iris 对齐时，默认用户入口应首先暴露“稳定操作语义 + heap/context 能力”，而不是让用户直接接触 `pattern`、`heap_bases`、`translate_ptr` 和 fallback 分支。
6. 高层 API 应与 `XTileContext` / `heap` 配合，统一承担：
   - 对称堆分配
   - `on_symmetric_heap(...)` 检查
   - 必要时的 `as_symmetric(...)` 或等价导入动作
   - layout / shard metadata 规范化
7. pattern 继续保留，但角色要更接近 backend implementation detail / expert tuning surface，而不是默认 public API。

**验收标准**：
- README / CLI / benchmark 示例统一改用 `xtile.ops.*`
- pattern 仍可保留，但不再承担默认 public API 角色
- 默认示例不再要求用户手工处理 pattern 选择与 heap 建立细节

### U5. benchmark 数据管线与图表口径未统一

**现状**：
- 之前 `fig3_pattern_overlap` 曾直接硬编码旧值
- benchmark 结果与绘图脚本并未自动联动

**风险**：
- 图表可能长期滞后于真实结果
- 文档、图、benchmark 输出三套口径分叉

**改进计划**：
1. benchmark 输出统一落到结构化 JSON。
2. 绘图脚本只读最新结果文件，不再手写硬编码数值。
3. 图上显式标注：
   - 测试日期
   - 运行命令
   - 聚合方式（min / mean / median）
4. `figures/` 重生成流程纳入同一 benchmark pipeline。
5. 与 Iris 对齐时，结果元数据必须能回答“这次实验到底走了哪条内存建立路径”，至少记录：
   - allocator/backend
   - heap mode（single-process / multiprocess）
   - transport strategy（ctypes IPC / PyTorch IPC / peer access / future DMA-BUF）
   - layout mode（full / shard）
6. 对外比较 Iris / XTile 时，只允许比较 metadata 可追溯的结果，禁止再出现“图上 headline 无法还原到具体运行配置”的情况。
7. 图表脚本、实验日志、文档引用统一读取同一份 JSON 元数据，确保后续扩展到 Iris-style allocator backend 时无需再补写手工说明。

**验收标准**：
- 修改 benchmark 后，重画图不需要手改数值
- 图表与 `experiment_log.md` 同源
- 任一图表都能回溯到对应的 heap backend / transport strategy / layout mode

### U6. symmetric heap 建立路径语义未统一

**现状**：
- XTile 现在是“ctypes IPC → PyTorch IPC → peer access”三层 fallback
- 但单进程 `create_all()` 与多进程 `_setup_multiprocess()` 仍是两套不同风格
- Iris 则更接近 allocator + import/map 的 canonical path

**风险**：
- 单进程 / 多进程语义收敛慢
- 后续要引入更强 allocator / DMA-BUF 路径时改动面大

**改进计划**：
1. 为 XTile 增加 allocator abstraction，并把当前 `torch.empty(uint8)` bump allocator 上提为默认 `TorchAllocator` 风格实现，而不是把“怎么分配 heap”硬编码在 `SymmetricHeap` 里。
2. 对齐 Iris 现有抽象，把 heap backend 能力拆成稳定接口：
   - `allocate`
   - `get_base_address`
   - `get_allocation_segments`
   - `export_handle`
   - `import_handle`
   - `map_segments`
   - `set_access`
   - `owns_tensor`
   - `import_external_tensor`
3. 将 `create_all()` 与 `_setup_multiprocess()` 收敛到同一个内部状态机：
   - allocate local heap
   - publish local base / metadata
   - choose transport strategy
   - import or map peer segments
   - establish access
   - materialize `heap_bases`
4. 保留当前 `ctypes IPC → PyTorch IPC → peer access` fallback，但把它们下沉为 backend strategy / capability matrix，而不是散落在 `SymmetricHeap` 主逻辑里。
5. 在支持的后端上补 Iris 风格 segmented export/import/map 路径；`peer access` 保留为 fast path，而不是唯一的单进程语义。
6. 补齐 `as_symmetric(...)` / `import_external_tensor(...)` 等价能力，让“外部张量转入 symmetric heap”成为正式语义，而不是用户自己猜测 tensor 是否可直接参与 collective。
7. 为单进程 / 多进程 / 未来 DMA-BUF backend 建立统一 capability 文档与测试矩阵，明确每条路径支持什么、缺什么、退化到哪一层。
8. 整体原则上，XTile 向 Iris 对齐不是“删除 fallback 改成单一路径”，而是“保留 fallback 的现实可用性，同时增加 allocator-first canonical layer”。

**验收标准**：
- `create_all()` 与 multiprocess 共享更大比例的底层契约
- heap 对上层暴露的是统一语义，而不是模式分叉细节
- 单进程 / 多进程 / 外部导入 tensor 在 API 层看起来属于同一套 heap contract

### 推荐推进顺序

1. **先做 U1 + U2 + U3**
   - 因为这三项直接决定 pattern API 是否可被正确理解
2. **再做 U4**
   - 先把 shape / layout 契约定死，再包一键 API
3. **然后做 U5**
   - 让 benchmark / doc / figures 从同一数据源出图
4. **最后做 U6**
   - 这是更底层的长期架构收敛，价值很高，但不该阻塞前面的 API 统一

## 【新增状态更新 2026-03-21】我到底改了什么？是不是负优化？

先说结论：

- **不能把当前 headline 变差简单解释成“负优化”**。
- 更准确的说法是：**runtime / benchmark /图表口径被收紧以后，之前那个最好看的 `1.067×` 不再能当稳定结论**。
- 目前证据更支持“以前的 pattern headline 带有单尺寸单轮次偏乐观成分”，而不是“我把 pattern kernel 真正优化坏了”。

### 这轮我实际做过的修改与优化

1. **runtime context 统一**
   - `XTileContext` 现在持有 `backend`、可选 `heap`、`heap_bases`
   - `xtile.init(..., heap=...)`、`xtile.init(..., heap_size=...)`、`xtile.init_local(...)` 打通
   - `ctx.empty/zeros/randn/barrier/auto_select_pattern` 已补齐

2. **tests / CLI / benchmark 切到真实 runtime 入口**
   - pattern tests 不再手工塞 `_Ctx`
   - `bench_patterns.py` 改走 `xtile.init_local(...)`
   - CLI `xtile bench pattern` 支持 `--warmup/--iters/--heap-size-mb`

3. **pattern benchmark 从“能跑”改到“更可复现”**
   - 不再写死 `512 MiB` heap
   - 改成根据 `(M, N, K)` 自动估算每 rank 对称堆需求
   - 这修掉了大尺寸 benchmark 直接失败的问题（原 `P5-003`）

4. **pattern 辅助开销削减**
   - `ProducerConsumerPattern` 复用 lock buffer 与 streams
   - `WGSpecializedPattern` 复用 lock buffer
   - 这是主机侧辅助路径优化，不是 fused kernel 算法本身的大改

5. **GEMM benchmark 口径修正**
   - `xtile.kernels.gemm.gemm(...)` 现在显式支持 `num_warps` / `num_stages`
   - 避免以前 benchmark 看起来在调参，实际 wrapper 并没真的吃到这些参数

6. **绘图脚本修正**
   - 原来 `scripts/plot_figures.py` 里 `fig3_pattern_overlap` 直接把 `1.067×` 写死
   - 现在已经改成基于最新 full 6-size rerun 的稳定结果出图，不再保留这个过期 headline

### 为什么现在结果看起来没有以前好？

主要有 4 个原因：

1. **旧的 `1.067×` 不是当前最稳结论**
   - 它来自旧阶段的单尺寸单轮次结果
   - 当前统一 runtime + 动态 heap + 6 尺寸完整复测后，best stable 只有 `1.004×`

2. **旧图表曾经直接画了过期硬编码数据**
   - 之前的 `fig3_pattern_overlap` 不是“重跑实验自动出图”，而是“把旧数值重新画一遍”
   - 所以你会看到图还写着 `1.067×`

3. **这轮改动主要优化的是“工程正确性与复现性”，不是 pattern 内核性能本身**
   - 真正明确改善的是 runtime 统一、benchmark 稳定性、heap 配置、实验口径
   - 不是 fused pattern 的大规模算法升级

4. **pattern 路径本来就还没有被证明稳定优于 baseline**
   - 这轮复测只是把这个事实更清楚地暴露出来
   - 从工程视角，这是“去掉了过强叙事”，不是“做出了可证实的负优化”

### 那有没有真实的正向改进？

有，但主要集中在工程主路径和 GEMM，而不是 overlap headline：

- `XTileContext` 主路径统一了，这是真的正向工程改进
- pattern benchmark 大尺寸能稳定跑完了，这是真的正向改进
- GEMM 8192³ 的 official helper 3 次复测中位数现在约：
  - fp16: `84.2%`
  - bf16: `86.1%`
- 相比此前文档里长期引用的约 `79%`，这是更好的当前口径

### 当前最准确的现状判断

- **不是“XTile 被我做坏了”**
- **而是“runtime 与 benchmark 更严谨了，旧的 pattern 最优结论站不住了”**
- 现在 XTile 真正已经站稳的是：
  - transparent primitive 路线
  - unified runtime context
  - dynamic symmetric heap benchmark path
  - 4096³ GEMM 达标
- 现在仍然没站稳的是：
  - overlap pattern 的稳定性能优势
  - 8192³ GEMM ≥ 90%
  - 一键式高层 API
  - shape 约定完全统一

### 核心路径：完全实现

| 组件 | 设想 | 现状 | 状态 |
|------|------|------|------|
| translate_ptr (5 指令) | 匹配 Iris | ✅ 100% 匹配 | 完成 |
| SymmetricHeap | IPC/peer access | ✅ 三级 fallback (ctypes IPC → PyTorch IPC → peer access) | 完成 |
| 4 种 overlap pattern | kernel 实现 | ✅ 全部实现 | 完成 |
| GEMM kernel | ≥ 90% cuBLAS | ✅ 4096³：fp16 100.2%，bf16 91.1%（official helper 3 次复测中位数） | 完成 |
| Auto-select | 硬件感知 | ✅ 启发式已实现，且已与统一 `XTileContext` 主路径打通 | 基本完成 |
| tile 级 collective | ring allreduce 等 | ⚠️ 代码已实现，但当前结果里至少 allreduce collective benchmark 仍报 `invalid resource handle` | 部分完成 |
| 跨平台 HAL | CUDA + HIP | ✅ 代码就绪 | 待 AMD 硬件 |

### 未完成

| 组件 | 设想 | 现状 | 原因 |
|------|------|------|------|
| P2P ≥ 95% | 285 GB/s | ❌ 当前约 82.7%–82.9% 峰值 | Triton/PTX 路径与协议开销 |
| 一键 API | `xtile.fused_gemm_scatter(...)` | ❌ 需手动组合 | 便利封装未做 |
| 8192³ GEMM ≥ 90% | kernel 优化 | ❌ 当前约 84.2%–86.1% | 仍需更深 kernel-level 优化 |
| Pattern ≥ 1.3× overlap | 多 GPU overlap | ❌ 当前 best stable 仅 1.004× | 旧 1.067× 单点结果未复现 |
| 跨节点 IPC | UCX/GDR | ❌ 待实现 | 需跨机通信基础设施 |
| AMD 实测 | MI300X 验证 | ❌ 待硬件 | 无 AMD GPU |
