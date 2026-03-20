# XTile 现状流程：GEMM + AllScatter 全栈代码流（修订版）

> 修订说明
> - 原始文档保留不动；本文件是在原文基础上的核对修订版。
> - `【修订】` 表示原文该处已按当前仓库现实改写。
> - `【新增核对】` 表示为避免误解而补充的仓库现状说明。

## 任务定义

与设想文档相同：4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
先算 C_local = A × B（本地 GEMM），然后把 C_local scatter 到所有其他 GPU。

M=8192, N=4608, K=36864, 4 GPUs.

---

## 第 1 层：用户代码

### 实际代码（方式 A：当前仓库可执行路径，非一键 API）

【修订】原文示例把 `XTileContext` 直接当作 pattern 运行时上下文，并把 `gemm_scatter` 当作合法 op；这两点在当前仓库都不成立。下面改成与现有实现一致的写法。

```python
import torch
import xtile
from xtile.backends import get_backend
from xtile.memory.symmetric_heap import SymmetricHeap
from xtile.patterns.auto_select import auto_select

runtime = xtile.init(backend="auto", rank=0, world_size=4)

heaps = SymmetricHeap.create_all(
    size=1 << 30,
    world_size=4,
    backend=runtime.backend,
)
heap = heaps[runtime.rank]

class _PatternCtx:
    pass

ctx = _PatternCtx()
ctx.rank = runtime.rank
ctx.world_size = runtime.world_size
ctx.heap_bases = heap.get_heap_bases()
ctx.backend = get_backend(runtime.backend)

# 当前实现里通常只要求参与远端写入/翻译的输出张量在 heap 上
A = torch.randn((M, K), device=runtime.device, dtype=torch.float16)
B = torch.randn((K, N), device=runtime.device, dtype=torch.float16)
C = heap.allocate_tensor((M, N), dtype=torch.float16)

pattern = auto_select("gemm_allscatter", M, N, K, world_size=4, ctx=ctx)
# → N/world_size = 1152，不满足 fused_sequential 的 < 1024 条件
# → 按当前 heuristics，本例更接近 ProducerConsumerPattern
pattern.execute(A, B, C)
```

### 与设想的差距

| 设想 | 现状 | 差距 |
|------|------|------|
| `xtile.fused_gemm_scatter(A, B, C, strategy="auto")` | `pattern = auto_select(...); pattern.execute(...)` | 缺少一键 API，用户仍需手动组织调用 |
| `ctx.randn(...)` 直接在堆上分配 | 仍需 `heap.allocate_tensor(...)` 或普通 `torch.randn(...)` | 缺少 context 级便利方法 |
| 自动检测后端 `backend="auto"` | `xtile.init()` 默认就是 `backend="auto"` | 这一点已集成，不应再写成“未完全集成” |
| `xtile.init()` 返回的 ctx 可直接执行 pattern | 仍需手工补齐 `heap_bases` 和 backend 对象 | runtime ctx 与 pattern ctx 尚未统一 |

---

## 第 2 层：Pattern 层（以 FusedSequential 为例）

### 实际 kernel 代码

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
    num_pid_n = tl.cdiv(N_per_rank, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # ====== 与 Iris 完全相同的 persistent kernel 循环 ======
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # --- Iris 风格 GEMM ---
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N_per_rank
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
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_per_rank)
        C_tile_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(C_tile_ptr, c, mask=mask)

        # --- 立刻 scatter 到所有 peer（Iris fused 风格）---
        for peer in range(world_size):
            if peer != rank:
                scatter_tile_to_peer(
                    C_ptr, c, offs_m, offs_n,
                    rank, peer, N, N_per_rank,
                    heap_bases, mask,
                )
```

### 与设想的对比

**一致**：
- Persistent kernel + round-robin 调度 ✅
- 每算完一个 tile 立刻 scatter ✅
- `scatter_tile_to_peer` 使用 `translate_ptr` + `tl.store` ✅
- `heap_bases` 作为 kernel 参数传入 ✅

**差异（修订）**：核心通信原理接近 Iris，但当前 XTile 的 shape / ctx 约定还没有完全收敛。tests 使用完整 `B(K,N)`、完整 `C(M,N)`；benchmarks / CLI 则使用 `B(K,N_per_rank)`、`C(M,N_per_rank)`；而 `fused_sequential.py` 又会从 `B.shape[1]` 推导 `N` 再计算 `N_per_rank`。因此这里不宜再写成“无实质差异”。

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
    total_tiles = tl.cdiv(M, 128) * tl.cdiv(n_per_rank, 128)

    if M < 256:
        return BulkSyncPattern       # 太小，不值得 overlap
    elif n_per_rank < 1024 and K > int(12288 * bw_scale):
        return FusedSequentialPattern # 通信小、计算大
    elif n_per_rank < 2048 and K > int(6144 * bw_scale):
        return ProducerConsumerPattern
    elif total_tiles >= sm_count and K > int(4096 * bw_scale):
        return WGSpecializedPattern
    else:
        return BulkSyncPattern
```

### 与设想的差距

**基本一致（修订）**，但需要补两点：
- 当前支持的 op 名是 `gemm_allscatter`、`gemm_allgather`、`gemm_reducescatter`，不是 `gemm_scatter`。
- 对本例 `N=4608, world_size=4`，`n_per_rank = 1152`，不会命中 `fused_sequential` 的 `< 1024` 分支，而会更接近 `producer_consumer` 分支。

---

## 总体差距汇总

### 核心路径：完全实现

| 组件 | 设想 | 现状 | 状态 |
|------|------|------|------|
| translate_ptr (5 指令) | 匹配 Iris | ✅ 100% 匹配 | 完成 |
| SymmetricHeap | IPC/peer access | ✅ 三级 fallback (ctypes IPC → PyTorch IPC → peer access) | 完成 |
| 4 种 overlap pattern | kernel 实现 | ✅ 全部实现 | 完成 |
| GEMM kernel | ≥ 90% cuBLAS | ✅ 4096³: 100.7% | 完成 |
| Auto-select | 硬件感知 | ✅ 启发式已实现，但未与一键 API / 统一 ctx 完全打通 | 基本完成 |
| tile 级 collective | ring allreduce 等 | ⚠️ 代码已实现，但当前 collective benchmark 仍有 `invalid resource handle` | 部分完成 |
| 跨平台 HAL | CUDA + HIP | ✅ 代码就绪 | 待 AMD 硬件 |

### 未完成

| 组件 | 设想 | 现状 | 原因 |
|------|------|------|------|
| P2P ≥ 95% | 285 GB/s | ❌ 当前约 82.7%–82.9% 峰值 | Triton/PTX 路径与协议开销 |
| 一键 API | `xtile.fused_gemm_scatter(...)` | ❌ 需手动组合 | 便利封装未做 |
| 8192³ GEMM ≥ 90% | kernel 优化 | ❌ 79% | PTX-level 瓶颈 |
| Pattern ≥ 1.3× overlap | 多 GPU overlap | ❌ 1.067× | 2 GPU 限制 |
| 跨节点 IPC | UCX/GDR | ❌ 待实现 | 需跨机通信基础设施 |
| AMD 实测 | MI300X 验证 | ❌ 待硬件 | 无 AMD GPU |
