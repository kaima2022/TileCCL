# XTile：集各家之长的下一代 Tile 通信库

## 详细分析与开发计划

## 引言

**XTile 应以 Iris 为技术基石，但解决它的三大局限**——仅 AMD、仅 intra-node、无 pattern 自动选择。具体来说：

从 **Iris** 继承纯 Triton 实现策略、symmetric memory 抽象、value/pointer 双 API、以及完整的 overlap 模式 taxonomy（这是 Iris 最大的学术贡献）。Iris 论文中的五个 Listing 代码是最直接的实现参考。

从 **TileScale** 借鉴多尺度统一抽象（thread→warp→block→GPU→node），以及 communication 作为与 compute/memory 同等地位的一等原语的设计哲学。

从 **TileLink** 借鉴细粒度的 tile 级信号原语（`tile_signal`/`tile_wait`），这比 Iris 现有的 `atomic_cas` 自旋锁更优雅。

从 **ThunderKittens** 借鉴硬件感知的 tile 尺寸设计（16×16 匹配 tensor core）和 Load-Store-Compute-Finish 流水线模板。

计划分四个 Phase、12 个月，优先确保在 AMD 和 NVIDIA 上都能达到 95%+ 的 P2P 带宽利用率，然后构建 Pattern Library 和 Auto-Select 引擎。

---

## 第一部分：现有库深度分析

### 1.1 竞争格局全景

当前 tile 通信库生态可以按两个维度分类：**语言层次**（高级 Python/DSL vs 低级 C++/CUDA）和**通信能力**（纯单 GPU vs 多 GPU 分布式）。

| 库 | 语言层 | 多GPU通信 | 计算-通信重叠 | 编译器可见性 | 硬件覆盖 |
|---|---|---|---|---|---|
| TileScale | Python DSL | ✅ collective + P2P | ✅ 编译器自动调度 | ✅ 全可见 | NVIDIA (实验) |
| TileLink/Triton-Dist | Triton + 编译器 | ✅ tile级信号原语 | ✅ 1.17x-20x 加速 | ⚠️ 部分（包装xSHMEM） | NVIDIA + AMD |
| Iris | Python + Triton | ✅ symmetric memory | ✅ 多模式 taxonomy | ✅ 全可见（纯 Triton） | AMD（MI300X） |
| ThunderKittens | C++20 CUDA | ✅ NVLink/NVSwitch | ✅ Load-Store-Compute-Finish | ❌ 不透明 | NVIDIA (H100/B200) |
| HipKittens | C++ HIP | ⚠️ 有目录无原语 | ❌ 依赖外部 ROCm | ❌ 不透明 | AMD (MI300X/MI350X) |
| cuTile | Python DSL (MLIR) | ❌ 纯单GPU | ❌ | ✅ Tile IR | NVIDIA |
| TileFusion | C++ 模板 | ❌ 纯单GPU | ❌ | ❌ | NVIDIA |

### 1.2 各库核心优势提取

**TileScale 的精华：统一多尺度抽象**
- 三位一体原语模型：compute / memory / communication 同等地位
- 跨尺度一致性：thread → warp → block → GPU → 多节点使用相同 tile 接口
- "mega-device" 理念：把整个集群虚拟化为一个巨大的计算设备
- **XTile 应借鉴**：统一的多尺度抽象架构，communication 作为一等公民

**TileLink 的精华：tile 级信号/数据原语**
- 精细的生产者-消费者原语：`producer_tile_notify`、`consumer_tile_wait`、`tile_push_data`、`tile_pull_data`
- 动态/静态 mapping 支持不同 tile size 的独立优化
- 编译器级别的 compute-comm fusion
- **XTile 应借鉴**：tile 级信号原语设计，动态 tile mapping

**Iris 的精华：纯 Triton 原生实现 + 完整的 overlap 模式 taxonomy**
- 全 Python/Triton 实现，编译器完全可见
- symmetric memory 抽象 + 指针翻译机制
- 完整的 overlap 模式分类：bulk-synchronous → producer-consumer → workgroup specialization → wavefront specialization
- value-based 和 pointer-based 双 API 模式
- acquire/release 内存语义，而非 SHMEM 的 quiet/wait_until
- **XTile 应借鉴**：纯原生实现策略、overlap taxonomy、双 API 模式、C++/HIP 内存模型

**ThunderKittens 的精华：极致的硬件利用**
- 16×16 基础 tile 完美匹配 tensor core
- Load-Store-Compute-Finish 四阶段流水线模板
- NVLink + NVSwitch 直接支持
- MXFP8 / NVFP4 新精度支持
- **XTile 应借鉴**：硬件感知的 tile 尺寸设计、流水线模板

**cuTile 的精华：MLIR 编译栈**
- 基于 CUDA Tile IR / MLIR 的编译优化
- 不可变 tensor-like tile 结构，类型安全
- 自动管理 tensor core 和线程映射
- **XTile 应借鉴**：tile 类型系统设计，MLIR 编译栈思路

**TileFusion 的精华：三级内存层次 tile 管理**

- global ↔ shared ↔ registers 三级 tile 传输
- CTA 线程协作数据搬运模式
- Macro kernel 组合模板
- **XTile 应借鉴**：三级内存层次的显式管理模型

### 1.3 现有库的核心缺陷分析

**缺陷一：硬件孤岛**
- ThunderKittens 只支持 NVIDIA（H100/B200），放弃了 Ampere
- HipKittens 只支持 AMD，通信能力不完整
- Iris 当前仅验证在 AMD MI300X 上
- 没有一个库能同时高效覆盖 NVIDIA 和 AMD

**缺陷二：通信与计算的抽象割裂**
- cuTile 和 TileFusion 完全没有通信能力
- ThunderKittens 的通信是硬编码在 C++ 模板中，无法灵活组合
- NCCL/RCCL 在 kernel 外部运行，无法 tile 级 overlap

**缺陷三：缺乏 inter-node 支持**
- Iris 明确声明只支持 intra-node（IPC），未来才做 RDMA
- TileScale 实验阶段，多节点不稳定
- ThunderKittens 仅 NVLink/NVSwitch，无 RDMA

**缺陷四：编译器可见性不一致**

- Triton-Distributed 包装 xSHMEM 为不透明字节码
- ThunderKittens/HipKittens 是 header-only C++，编译器无法跨通信边界优化

---

## 第二部分：XTile 总体设计

### 2.1 设计目标

XTile 的核心定位：**第一个同时实现"跨硬件可移植 + 编译器全可见 + 多尺度统一 + 多节点通信"的 tile 通信库**。

五大设计目标：

1. **Communication as First-Class Primitive**（借鉴 TileScale + Iris）：通信与计算、内存同等地位，不是附加功能
2. **Full Compiler Visibility**（借鉴 Iris）：全部用目标语言原生实现，编译器可跨计算-通信边界优化
3. **Hardware Portability**（弥补所有库的缺陷）：同一套 API 覆盖 NVIDIA (Hopper/Blackwell) 和 AMD (CDNA3/CDNA4)
4. **Multi-Scale Unified Abstraction**（借鉴 TileScale）：thread → warp → block → GPU → node 统一 tile 接口
5. **Production-Grade Overlap Patterns**（借鉴 Iris taxonomy）：内建完整的 compute-communication overlap 模式库

### 2.2 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    XTile 用户层 (Python API)              │
│  xtile.load / xtile.store / xtile.allreduce / ...       │
│  xtile.Tile / xtile.SymmetricHeap / xtile.Pattern       │
├─────────────────────────────────────────────────────────┤
│                   Pattern Library 层                      │
│  BulkSync / Sequential / ProducerConsumer / WGSpecial   │
│  WavefrontSpecial / WorkQueue / PipelinedOverlap        │
├─────────────────────────────────────────────────────────┤
│                 Core Primitive 层                         │
│  ┌───────────┬────────────┬────────────────────────┐    │
│  │ Compute   │  Memory    │  Communication          │    │
│  │ tile.dot  │  tile.load │  tile.remote_load       │    │
│  │ tile.acc  │  tile.store│  tile.remote_store      │    │
│  │ tile.reduce│ tile.copy │  tile.signal / tile.wait│    │
│  │           │            │  tile.allreduce         │    │
│  │           │            │  tile.allgather         │    │
│  │           │            │  tile.scatter           │    │
│  └───────────┴────────────┴────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                Synchronization 层                         │
│  acquire / release / acq_rel 语义                        │
│  scope: wavefront / workgroup / device / system          │
│  tile_signal / tile_wait / atomic_cas / barrier          │
├─────────────────────────────────────────────────────────┤
│              Memory Management 层                         │
│  SymmetricHeap + IPC + RDMA (future)                     │
│  Pointer Translation Engine                              │
│  Cache-Aware Tiling: L1 → L2 → LLC → Remote             │
├─────────────────────────────────────────────────────────┤
│              Hardware Abstraction 层 (HAL)                │
│  ┌──────────────────┬──────────────────────┐            │
│  │  NVIDIA Backend   │   AMD Backend         │            │
│  │  Triton (NVIDIA)  │   Triton (ROCm)       │            │
│  │  NVLink/NVSwitch  │   Infinity Fabric      │            │
│  │  CUDA IPC         │   HIP IPC              │            │
│  │  NVSHMEM (可选)   │   rocSHMEM (可选)      │            │
│  └──────────────────┴──────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 2.3 核心设计决策

**决策 1：以 Triton 为主要实现语言（借鉴 Iris）**

理由：
- Triton 同时支持 NVIDIA 和 AMD，是目前唯一跨硬件的 tile-based 编译框架
- 纯 Triton 实现保证编译器全可见性
- Python 前端对 AI 开发者最友好
- Iris 已证明纯 Triton 实现的可行性和性能（1.79× vs RCCL）

**决策 2：symmetric memory 作为基础通信模型（借鉴 Iris + OpenSHMEM）**

理由：
- symmetric heap 提供可预测的内存布局，指针翻译开销极低
- one-sided 通信模式天然适合 GPU 的 massively parallel 架构
- Iris 已验证 symmetric memory + IPC 在 AMD 上的效率

**决策 3：C++/HIP/CUDA 内存模型语义（借鉴 Iris，摒弃 SHMEM）**

理由：
- acquire/release 语义 GPU 开发者已经熟悉
- 比 SHMEM 的 quiet/wait_until 更直观
- 直接映射到硬件的 memory scope（wavefront/workgroup/agent/system）

**决策 4：value-based + pointer-based 双 API（借鉴 Iris）**

- value-based：寄存器直接到远端内存，适合细粒度 tile 通信
- pointer-based：内存到内存的批量传输，适合大块数据移动

**决策 5：内置 Overlap Pattern Library（借鉴 Iris taxonomy + TileLink 原语）**

- 提供预定义的 fused pattern 模板
- 用户可组合原语构建自定义 pattern
- 支持自动 pattern 选择（根据问题尺寸和硬件拓扑）

---

## 第三部分：详细 API 设计

### 3.1 Host-Side API

```python
import xtile

# 初始化
ctx = xtile.init(backend="auto")  # auto-detect NVIDIA/AMD
# ctx.rank, ctx.world_size, ctx.device, ctx.topology

# Symmetric Heap 管理
heap = ctx.create_heap(size=1024 * 1024 * 1024)  # 1GB per GPU

# Tensor 创建（在 symmetric heap 中）
A = ctx.zeros((M, K), dtype=xtile.float16)
B = ctx.randn((K, N), dtype=xtile.float16)
C = ctx.empty((M, N), dtype=xtile.float32)

# Host-side 通信
ctx.barrier()
ctx.broadcast(tensor, src_rank=0)

# Pattern 选择与调度
pattern = xtile.patterns.auto_select(
    op="gemm_allscatter",
    shape=(M, N, K),
    world_size=ctx.world_size,
    topology=ctx.topology
)
pattern.execute(A, B, C)
```

### 3.2 Device-Side API（Triton 内核中使用）

```python
@triton.jit
def fused_gemm_allscatter(A, B, C, heap_bases, ...):
    pid = tl.program_id(0)
    rank = xtile.get_rank()
    
    # ====== 计算原语 ======
    acc = xtile.tile_zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = xtile.tile_load(A, [rm, rk])  # 本地 tile load
        b = xtile.tile_load(B, [rk, rn])
        acc = xtile.tile_dot(a, b, acc)   # tile GEMM
    
    # ====== 通信原语 (value-based) ======
    # 直接将寄存器中的结果 scatter 到远端
    for dst in range(world_size):
        xtile.tile_remote_store(
            C, acc, 
            src_rank=rank, dst_rank=dst,
            heap_bases=heap_bases,
            offset=offset, mask=mask
        )
    
    # ====== 通信原语 (pointer-based) ======
    xtile.tile_put(C + local_offset, C + remote_offset,
                   src_rank=rank, dst_rank=dst,
                   heap_bases=heap_bases, mask=mask)
    
    # ====== 信号原语 (借鉴 TileLink) ======
    xtile.tile_signal(locks, tile_id, sem="release", scope="device")
    xtile.tile_wait(locks, tile_id, sem="acquire", scope="device")
    
    # ====== 集体通信原语 ======
    result = xtile.tile_allreduce(acc, op="sum")
    gathered = xtile.tile_allgather(acc)
```

### 3.3 Overlap Pattern API

```python
# 模式 1：Bulk-Synchronous（最简单）
@xtile.pattern("bulk_sync")
def gemm_then_scatter(A, B, C, ctx):
    gemm_kernel[grid](A, B, C, ...)
    ctx.barrier()
    scatter_kernel[grid](C, ...)

# 模式 2：Fused Sequential（Iris Listing 4 风格）
@xtile.pattern("fused_sequential")
@triton.jit
def fused_sequential(A, B, C, heap_bases, ...):
    for tile_id in range(pid, total_tiles, NUM_SMS):
        acc = xtile.gemm_loop(A, B, tile_id, ...)
        xtile.tile_scatter(C, acc, tile_id, heap_bases, ...)

# 模式 3：Workgroup Specialization（Iris Listing 5 风格）
@xtile.pattern("wg_specialized")
@triton.jit
def wg_specialized(A, B, C, locks, heap_bases,
                   COMPUTE_SMS: tl.constexpr,
                   COMM_SMS: tl.constexpr, ...):
    pid = tl.program_id(0)
    if pid < COMPUTE_SMS:
        xtile.compute_worker(A, B, C, locks, pid, ...)
    else:
        xtile.comm_worker(C, locks, pid - COMPUTE_SMS, heap_bases, ...)

# 模式 4：自动选择
result = xtile.auto_fused_gemm_scatter(
    A, B, C,
    strategy="auto",  # auto / bulk_sync / sequential / wg_specialized
    ctx=ctx
)
```

### 3.4 Cache-Aware Tiling API（借鉴 Iris + TileFusion）

```python
# 多级缓存感知
@triton.jit
def cache_aware_kernel(...):
    # L1 级 tile（compute unit 内）
    tile_l1 = xtile.tile_load(ptr, cache_level="L1")
    
    # L2 级 tile（跨 compute unit，同一 XCD）
    tile_l2 = xtile.tile_load(ptr, cache_level="L2",
                               swizzle=GROUP_SIZE_M)
    
    # LLC 级 tile（跨 XCD，同一 GPU）
    tile_llc = xtile.tile_load(ptr, cache_level="LLC",
                                chiplet_swizzle=True)
    
    # 远端 store with cache modifier
    xtile.tile_remote_store(ptr, tile, cache_modifier=".wt")
```

---

## 第四部分：技术实现路线

### Phase 0：基础设施搭建（Month 1-2）

**目标**：建立项目框架、CI/CD、测试基础

| 任务 | 详情 | 预计时间 |
|------|------|----------|
| 项目结构初始化 | Python package + Triton 集成 + 构建系统 | 1 周 |
| HAL 层骨架 | 硬件检测、backend 选择（NVIDIA/AMD）| 1 周 |
| 测试框架 | 单元测试 + 多 GPU benchmark harness | 1 周 |
| CI/CD | GitHub Actions + 多 GPU 测试环境（借 AMD/NVIDIA 机器）| 1 周 |
| Iris 代码深度审计 | Fork Iris，逐行理解指针翻译、IPC 建立、内存模型 | 2 周 |
| Triton 上游跟踪 | 追踪 Triton 的 Gluon、atomic 语义等新特性 | 持续 |

**关键交付物**：
- `xtile` Python package 可安装
- 在单个 AMD MI300X 上能 import 并检测硬件
- Iris 的核心代码已 fork 并可独立运行

### Phase 1：核心原语层（Month 3-5）

**目标**：实现 XTile 的三大基础原语（compute / memory / communication）

**1.1 Symmetric Memory 子系统**

以 Iris 的实现为起点，但做以下增强：

```python
# Iris 原始实现（仅 HIP IPC）
hipMalloc → hipIpcGetMemHandle → hipIpcOpenMemHandle

# XTile 增强：统一的 IPC 抽象
class SymmetricHeap:
    def __init__(self, size, backend):
        if backend == "hip":
            self._impl = HIPSymmetricHeap(size)  # 基于 Iris
        elif backend == "cuda":
            self._impl = CUDASymmetricHeap(size)  # 新增
        # 未来: RDMA backend
    
    def translate(self, local_ptr, to_rank):
        """统一的指针翻译（跨 backend）"""
        return self._impl.translate(local_ptr, to_rank)
```

**1.2 Device-Side 原语实现**

优先级排序：

| 优先级 | 原语 | 类型 | 基于 |
|--------|------|------|------|
| P0 | `remote_load` / `remote_store` | value-based | Iris load/store |
| P0 | `get` / `put` / `copy` | pointer-based | Iris get/put/copy |
| P0 | `atomic_*` (add, cas, xchg, ...) | 同步 | Iris atomics |
| P0 | `tile_signal` / `tile_wait` | 信号 | TileLink 启发 |
| P1 | `allreduce` / `allgather` / `scatter` | collective | 新实现 |
| P1 | `barrier` | 同步 | Iris barrier |
| P2 | `all2all` | collective | TileScale 启发 |
| P2 | `broadcast` | collective | 新实现 |

**1.3 NVIDIA Backend 适配**

Iris 当前仅 AMD，NVIDIA 适配的关键差异：

| 方面 | AMD (Iris 已有) | NVIDIA (XTile 新增) |
|------|----------------|-------------------|
| IPC | hipIpcGetMemHandle | cudaIpcGetMemHandle |
| 内存模型 | SC-HRF, agent scope | similar, device scope |
| 互联 | Infinity Fabric | NVLink / NVSwitch |
| 线程粒度 | wavefront (64) | warp (32) |
| 计算单元 | CU (304 on MI300X) | SM (132 on H100) |
| atomic scope | block/gpu/sys | block/gpu/sys |

**关键适配任务**：
- CUDA IPC 建立流程
- warp 大小差异处理（影响 wavefront specialization）
- NVLink 拓扑检测与利用
- CUDA memory scope 映射

### Phase 2：Pattern Library（Month 6-8）

**目标**：实现完整的 compute-communication overlap 模式库

**2.1 模式实现优先级**

| 阶段 | 模式 | 复杂度 | 性能预期 | 代码基础 |
|------|------|--------|----------|---------|
| P2-A | Bulk-Synchronous | 低 | baseline | Iris Listing 3 |
| P2-A | Fused Sequential | 中 | 1.3-1.5× | Iris Listing 4 |
| P2-B | Unfused Producer-Consumer | 中 | 1.2-2.5× | Iris Section 4.1.2 |
| P2-B | Fused WG Specialization | 高 | 1.2-1.8× | Iris Listing 5 |
| P2-C | Fused Wavefront Specialization | 高 | TBD | 需 Gluon 支持 |
| P2-C | Work Queue | 高 | TBD | 研究阶段 |

**2.2 Auto-Select 引擎**

根据 Iris 论文的实验观察，不同 pattern 适合不同的问题形状：

```python
def auto_select_pattern(M, N, K, world_size, hw_info):
    """
    决策逻辑（基于 Iris 论文 Section 5 的实验结果）：
    
    1. 小 N + 大 K → Fused Sequential 最优
       (通信开销小，GEMM 需要更多资源)
       例：8192 × 4608 × 36864
    
    2. 小 N + 大 K + 多 GPU → Producer-Consumer 最优
       (可完全隐藏通信于 GEMM 之后)
       例：8192 × 3584 × 14336, 8 GPUs
    
    3. 大 N + 大 K → WG Specialization 最优
       (需要专用资源处理通信)
       例：8192 × 8192 × 30720
    
    4. 通信量 >> 计算量 → Bulk-Synchronous 最简单可靠
    """
    comm_compute_ratio = estimate_ratio(M, N, K, world_size)
    
    if comm_compute_ratio < 0.1:
        return "fused_sequential"
    elif N // world_size < threshold_small_n:
        return "producer_consumer"
    elif N * K > threshold_large:
        return "wg_specialized"
    else:
        return "bulk_synchronous"
```

### Phase 3：高级特性（Month 9-12）

**3.1 Cache-Aware Tiling（借鉴 Iris Section 3.4 + TileFusion）**

- Chiplet-aware swizzling（AMD XCD, NVIDIA GPC）
- Multi-level cache tile building：L1 → L2 → LLC → Remote
- Cache modifier 支持：`.wt`（write-through）、`.cg`（cache-global）等

**3.2 Profiling & Instrumentation（借鉴 Iris 的优势）**

由于全 Triton 实现，XTile 可提供：
- Triton Proton 集成的细粒度 profiling
- 通信操作级别的计时和带宽测量
- Overlap 效率可视化工具
- Tile 级 communication heatmap

**3.3 Gluon Backend（追踪 Triton 上游）**

- 利用 `@gluon.jit` 和 `@aggregate` 封装 backend state
- 消除手动传递 `heap_bases` 的需要
- 支持 wavefront-level specialization

**3.4 Multi-Node 支持（长期目标）**

- RDMA backend 扩展（InfiniBand / RoCE）
- 两级通信层次：intra-node (IPC/NVLink) + inter-node (RDMA)
- 自动 topology-aware 的通信策略选择

---

## 第五部分：与 Iris 的关系定位

### 5.1 Fork vs. From-Scratch 决策

**推荐策略：以 Iris 为参考实现，但从头构建 XTile**

理由：
1. Iris 是 AMD-only，XTile 需要从架构层面支持双硬件
2. Iris 的 HAL 层不存在（直接调用 HIP），需要重新设计
3. Iris 没有 Pattern Library 的抽象层
4. 但 Iris 的核心算法（指针翻译、IPC 建立、atomic 同步模式）极有参考价值

**具体策略**：
- 核心算法移植：指针翻译、symmetric heap 管理
- API 设计继承：value-based + pointer-based 双模式、atomic 语义
- 模式代码参考：Listing 3/4/5 的 pattern 实现
- 架构重新设计：HAL 层、Pattern Library、Cache-Aware 层

### 5.2 与上游 Triton 的关系

- 追踪 Triton 主线的 atomic 语义、memory scope 支持
- 利用 Triton 的 NVIDIA + AMD dual-backend 能力
- 关注 Gluon 的演进（aggregate type 对通信上下文很关键）
- 可能向 Triton 上游贡献通信原语 PR

---

## 第六部分：Benchmark 计划

### 6.1 Microbenchmark 矩阵

| 测试类型 | 指标 | 对比基线 |
|---------|------|---------|
| P2P load/store | 归一化带宽 (%) | Iris, NCCL/RCCL |
| Atomic operations | 归一化带宽 (%) | Iris |
| All-load / All-store | 不同 buffer size 的带宽 | Iris |
| AllReduce | 延迟 + 带宽 | NCCL, RCCL, TileScale |
| AllGather | 延迟 + 带宽 | NCCL, RCCL |

### 6.2 应用级 Benchmark

| 场景 | 模式 | 问题尺寸 | 对比目标 |
|------|------|---------|---------|
| GEMM + AllScatter | 全 taxonomy | Iris 论文的 6 个尺寸 | PyTorch + RCCL, Iris |
| GEMM + AllGather | WG Spec | LLM inference 典型尺寸 | Triton-Distributed |
| Self-Attention | Fused | seq_len=2048/4096/8192 | FlashAttention + NCCL |
| MoE Dispatch | All2All | Top-2, 8/16 experts | DeepSeek 风格 |

### 6.3 性能目标

| 指标 | 目标 |
|------|------|
| P2P 带宽利用率 | ≥ 95%（达到 Iris 水平） |
| Collective 带宽利用率 | ≥ 90% |
| GEMM+AllScatter 加速比 | ≥ 1.3× vs bulk-sync baseline |
| 跨硬件性能差异 | NVIDIA vs AMD 性能差 ≤ 15% |
| API 行数开销 | 在 Triton kernel 中增加 ≤ 5 行实现通信 |

---

## 第七部分：团队与时间线

### 7.1 推荐团队结构

| 角色 | 人数 | 职责 |
|------|------|------|
| 架构师/Tech Lead | 1 | 整体设计、API 审查、upstream 协调 |
| NVIDIA Backend 工程师 | 1-2 | CUDA IPC、NVLink 适配、H100/B200 优化 |
| AMD Backend 工程师 | 1-2 | 基于 Iris 的 HIP 实现、MI300X/MI350X 优化 |
| Pattern Library 工程师 | 1 | Overlap 模式实现、auto-select 引擎 |
| Benchmark / QA | 1 | 性能测试、回归测试、文档 |

### 7.2 12 个月 Milestone

```
Month 1-2   [Phase 0] 基础设施 + Iris 审计
            → 交付：可安装的 xtile package，Iris 代码理解文档

Month 3-5   [Phase 1] 核心原语
            → 交付：AMD 上 P2P 原语达到 Iris 性能
            → 交付：NVIDIA 上 P2P 原语初步可用
            → 交付：Microbenchmark 达到 95% 带宽利用率

Month 6-8   [Phase 2] Pattern Library
            → 交付：4 种 overlap 模式可用
            → 交付：Auto-select 引擎 v1
            → 交付：GEMM+AllScatter 在两个平台上超过 baseline

Month 9-10  [Phase 3a] 高级特性
            → 交付：Cache-aware tiling
            → 交付：Profiling 工具
            → 交付：Gluon backend（如 Triton 上游就绪）

Month 11-12 [Phase 3b] 稳定化 + 社区
            → 交付：完整文档 + tutorial
            → 交付：Multi-node RDMA 原型
            → 交付：开源发布 + 论文提交
```

---

## 第八部分：风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| Triton 上游 API 变动 | 原语需要重写 | 中 | 抽象层隔离，紧跟 Triton nightly |
| NVIDIA IPC 行为与 AMD 差异大 | 跨平台统一困难 | 中 | HAL 层充分抽象，允许 backend-specific 优化路径 |
| Gluon 不稳定/延迟发布 | wavefront specialization 受阻 | 高 | 先用标准 Triton API，Gluon 作为可选 backend |
| 多节点 RDMA 复杂度 | 延期或降级 | 高 | 先聚焦 intra-node，RDMA 作为 Phase 4 |
| 性能不及 ThunderKittens (C++) | 社区接受度低 | 低 | Iris 已证明 Triton 可达到 95%+ 带宽，focus on overlap 优势 |

---

## 第九部分：XTile 的差异化价值总结

| 维度 | 现有最佳 | XTile 的目标 |
|------|---------|------------|
| 硬件覆盖 | 单平台（Iris=AMD, TK=NVIDIA） | **双平台统一** |
| 编译器可见性 | Iris（纯 Triton） | **继承 + 扩展到 NVIDIA** |
| Overlap 模式 | Iris（4种模式） | **6+ 模式 + Auto-Select** |
| 多尺度抽象 | TileScale（实验） | **生产就绪的多尺度** |
| 多节点 | 无成熟方案 | **Intra-node 优先 + RDMA 路线图** |
| 缓存感知 | Iris（swizzle + modifier） | **统一的多级缓存 tile 管理** |
| 易用性 | Iris（几行代码） | **Pattern Library + Auto-Select** |
