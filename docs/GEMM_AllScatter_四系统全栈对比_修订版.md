# 固定任务：GEMM + AllScatter 全栈代码流对比

## 任务定义

4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
每个 GPU 先算 C_local = A × B（本地 GEMM），然后把 C_local scatter 到所有其他 GPU，
最终每个 GPU 都拥有完整的 C_global[M, N×4]。

M=8192, N=4608, K=36864, 4 GPUs.

---

## 系统一：TileScale

### 第 1 层：用户代码（TileLang DSL）

TileScale 当前公开的用户层接口是 `T.putmem_nbi_block(...)`、`T.put_block(...)`、`T.get_block(...)`、`T.wait_eq(...)`、`T.barrier_all()` / `T.sync_all()` 这类 distributed primitives。普通 `T.copy(src, dst)` 仍然存在，但当前 repo 里没有公开 `T.scatter()` 或 `T.copy(..., dst=...)` 这种分布式用户接口。

```python
import tilelang
import tilelang.language as T

# examples/distributed/example_allgather.py
T.copy(A[local_base : local_base + block_M, :], A_shared)
T.putmem_nbi_block(
    T.address_of(B[global_base, 0]),
    T.address_of(A_shared[0, 0]),
    block_M * N * dtype_map[dtype].itemsize,
    peer,
)

# examples/distributed/example_overlapping_allgather.py
rank = T.alloc_local([1], "uint64")
rank[0] = T.get_rank()
T.put_block(
    src=T.address_of(src[bx * block_M]),
    dst=T.address_of(dst[bx * block_M + rank[0] * M]),
    size=block_M,
    dst_pe=k,
)
```

**关键点**：TileScale 当前公开的是较低层的 distributed/NVSHMEM primitives。`T.scatter()` 不是公开接口；普通 `T.copy(src, dst)` 也不是公开的 remote-copy 用户接口。

### 第 2 层：TVM 编译器（基于 Apache TVM 项目）

**TVM 编译器是开源深度学习编译器框架**。TileLang/TileScale 是建在 TVM 之上的。

TVM标准过程：

```
用户写的 TileLang 代码（Python DSL）
    │ TileLang 前端把它翻译成 TVM 的 PrimFunc IR
    ▼
TVM 的优化管线
    │ 一系列 optimization passes（循环展开、向量化、内存规划等）
    ▼
TVM 的 CodeGen
    │ 把优化后的 IR 翻译成 CUDA C 代码
    ▼
nvcc（NVIDIA 的 CUDA 编译器）
    │ 编译 CUDA C → PTX → cubin
    ▼
可执行的 GPU kernel
```

TileLang/TileScale的TVM 编译器：

```
用户 Python DSL
    │
    ▼
TileLang Frontend（解析 Python → TVM PrimFunc IR）
    │
    ▼
Phase 1: LowerAndLegalize
  ├─ T.gemm → 展开为 warp-level WMMA/WGMMA 指令
  ├─ T.copy(src, dst) → 本地 copy / TMA / LSU / async copy
  ├─ T.Pipelined → 插入 async copy + barrier 实现多级流水线
  ├─ T.putmem_* / T.getmem_* / T.barrier_all / T.sync_all
  │    → 直接 codegen 到 `nvshmemx_*` / `nvshmem_*` device API
  ├─ T.put_block / T.get_block / T.put_warp / T.get_warp
  │    → 先做 remote base remap，再 lower 到 `tl::cp_block` / `tl::cp_warp` extern helper
  └─ T.wait_eq / T.wait_ne / T.wait_ge / ...
       → 分布式 wait primitive
    │
    ▼
Phase 2: OptimizeForTarget
  ├─ Layout inference（自动内存布局优化）
  ├─ Swizzle（L2 cache 局部性优化）
  └─ Cost model 选择最优实现
    │
    ▼
CodeGenTileLangCUDA / CodeGenHIP
  ├─ 生成 CUDA/HIP kernel 代码
  └─ nvcc / hipcc 编译为 .so
```

TileScale 的 distributed primitive 不会全部 lower 到同一种 NVSHMEM 调用。当前 repo 里至少能明确看到两条路径：

```c
// 路径 A：putmem/getmem/barrier/sync family
nvshmemx_putmem_nbi_block(dst, src, bytes, pe);
nvshmemx_getmem_nbi_block(dst, src, bytes, pe);
nvshmem_barrier_all();

// 路径 B：tileop family
remote_dst = get_remote_base_ptr(dst_pe) + offset_to_base;
call_extern("tl::cp_block<...>", remote_dst, src_addr);
```

共同点是：

```
✓ TVM 可以决定这些调用插在 kernel 的哪个位置
✗ 但 extern helper / NVSHMEM 调用内部对上层优化仍然是不透明的
✗ 因而编译器不敢跨这些调用自由重排有依赖的内存操作
```



### 第 3 层：通信底层（NVSHMEM）

```c
// 当前 repo 里能明确对应的 lowering 至少有两类：

// 1. putmem/getmem/barrier/sync family
nvshmemx_putmem_nbi_block(dst, src, bytes, pe);
nvshmemx_getmem_nbi_block(dst, src, bytes, pe);
nvshmem_barrier_all();

// 2. put_block/get_block family
remote_dst = get_remote_base_ptr(dst_pe) + offset_to_base;
call_extern("tl::cp_block<...>", remote_dst, src_addr);

// ⚠️ barrier_all / sync_all 是单独的 primitive，不是 putmem 自动附带的后处理。
// ⚠️ 这些 extern / NVSHMEM 调用对上层编译优化仍然是不透明的。
```

### 第 4 层：内存建立

```python
# 当前实现：
# 1. torch.distributed.init_process_group(..., backend="nccl")
# 2. TP_GROUP = torch.distributed.new_group(...)
# 3. pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
# 4. 通过 tilelang/distributed/launch.sh 启动；其内部调用 torch.distributed.run
$ GPUS=4 ./tilelang/distributed/launch.sh examples/distributed/example_allgather.py

# 内部流程：
# 1. torch.distributed 初始化
# 2. 创建 TP_GROUP
# 3. pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
# 4. 后续分布式原语通过 NVSHMEM symmetric memory 工作
```

### 第 5 层：硬件执行

```
GPU0 ──NVLink──► GPU1
  │                │
  ▼                ▼
GPU3 ◄──NVLink── GPU2

具体走哪条互连、是否发生 fallback，由 NVSHMEM runtime 和机器拓扑决定；
这部分不是 TileScale 的 Python / TVM 层能直接看到的内容
```

---

## 系统二：TileLink / Triton-Distributed

### 第 1 层：用户代码（Triton + TileLink 原语）

Triton-distributed 当前公开的是 low-level primitives：`wait`、`consume_token`、`rank`、`num_ranks`、`symm_at`、`notify`，以及 `libshmem_device.*`。README 和 `docs/primitives.md` 把高层 tile-centric primitives 标为尚未发布。

```python
import triton_dist
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device

@triton_dist.jit
def producer_consumer_kernel(rank: tl.constexpr, num_ranks: tl.constexpr, queue_ptr, signal_ptr, BLOCK_SIZE: tl.constexpr):
    peer_rank = (rank + 1) % num_ranks
    offs = tl.arange(0, BLOCK_SIZE)

    token = dl.wait(
        dl.symm_at(signal_ptr, peer_rank),
        1,
        "gpu",
        "acquire",
        waitValue=0,
    )
    queue_ptr = dl.consume_token(queue_ptr, token)
    data = tl.load(queue_ptr + offs)
    tl.store(dl.symm_at(queue_ptr, peer_rank) + offs, data)
    libshmem_device.fence()
    dl.notify(signal_ptr, peer_rank, signal=1, sig_op="set", comm_scope="intra_node")
```

**关键点**：当前公开接口是“低层同步原语 + 对称地址转换 + NVSHMEM device functions”，不是 `BlockChannel` 这套高层 tile API。

### 第 2 层：TileLink 编译器 （Triton JIT 编译器（基于 OpenAI Triton 项目））

```
用户 Python（`wait` / `consume_token` / `rank` / `num_ranks` / `symm_at` / `notify` / `libshmem_device.*`）
    │
    ▼
`wait` / `consume_token` / `rank` / `num_ranks` / `symm_at` / `notify`
    │
    ▼
Triton Distributed Dialect
    │
    ├─ NVIDIA 路径：
    │    - `wait` → PTX spin-wait
    │    - `notify` → intra-node PTX store/atomics，或 inter-node `nvshmemx_signal_op`
    │    - `symm_at` → `nvshmem_ptr`
    ├─ AMD 路径：
    │    - `wait` → LLVM polling loop + `gpu.barrier`
    │    - `consume_token` → identity
    │    - `symm_at` → `rocshmem_ptr_wrapper` / `mori_shmem_ptr`
    └─ `libshmem_device.*` → 直接 SHMEM device primitive extern call
    ▼
LLVM IR → PTX / AMDGPU ISA → cubin / HSACO
```

### 第 3 层：通信底层（NVIDIA 为 NVSHMEM，AMD 为 ROCSHMEM / MORI）

```c
// NVIDIA 路径：
// wait(...)
asm volatile("spin: ld.global.acquire.gpu.b32 %0, [%1]; ...");

// notify(..., intra-node path)
// -> membar + PTX store / atomics lowering

// notify(..., comm_scope = inter_node)
nvshmemx_signal_op(sig_addr, signal, sig_op, pe);

// symm_at(...)
remote_ptr = nvshmem_ptr(symm_addr, pe);

// AMD 路径：
// wait(...) -> LLVM load/poll loop + gpu.barrier
// symm_at(...) -> rocshmem_ptr_wrapper(...) / mori_shmem_ptr(...)

// Python-exposed SHMEM path：
libshmem_device.putmem_signal_nbi_block(...);
```

### 第 4 层：内存建立

```python
# 当前仓库启动方式是 torchrun / scripts/launch.sh，
# 初始化逻辑基于 torch.distributed ProcessGroup，再初始化 SHMEM 后端。
$ bash ./scripts/launch.sh tutorials/01-distributed-notify-wait.py

# 内部流程：
# 1. torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", ...)
# 2. pg = torch.distributed.new_group(..., backend="nccl")
# 3. initialize_distributed(...) 内部按后端调用
#    - CUDA: init_nvshmem_by_torch_process_group(pg)
#    - HIP:  init_rocshmem_by_torch_process_group(pg) / init_mori_by_torch_process_group(pg)
# 4. 对称 tensor 由 SHMEM backend 创建
```

### 第 5 层：硬件执行

```
与 TileScale 类似：底层链路由 SHMEM runtime 和机器拓扑共同决定；
这部分不是 Triton-distributed Python API 直接暴露的内容。
区别在于，当前 Triton-distributed 公开给用户的是更低层的 wait/notify/symm_at/NVSHMEM primitive 组合。
```

---

## TileLink 的 mapping 到底做什么（论文中的高层 tile-centric 设计）

下面这节描述论文中的高层 tile-centric mapping 模型。released Triton-distributed 仓库当前公开的是 `distributed.wait`、`distributed.consume_token`、`distributed.symm_at`、`distributed.notify` 等低层 op，而不是 `BlockChannel` 用户接口。

### 核心问题

一个 fused kernel 里有很多 tile（比如 GEMM 产生了 256 个 tile），通信也要按 tile 粒度进行。问题是：**tile #42 的数据应该发给谁？发到远端的哪个位置？通过哪个 barrier 同步？**

mapping 就是回答这三个问题的。

### 三种 mapping 的具体含义

假设我们有 4 个 GPU，GEMM 产生的输出 C 被分成 16 个 tile（4×4 网格）：

```
        N 维度 (分成 4 列)
        col0  col1  col2  col3
row0  [ t0    t1    t2    t3  ]
row1  [ t4    t5    t6    t7  ]     M 维度 (分成 4 行)
row2  [ t8    t9    t10   t11 ]
row3  [ t12   t13   t14   t15 ]
```

AllScatter 的语义：GPU0 计算所有 16 个 tile，然后：

- col0 的 tile (t0,t4,t8,t12) 留在 GPU0
- col1 的 tile (t1,t5,t9,t13) 发到 GPU1
- col2 的 tile (t2,t6,t10,t14) 发到 GPU2
- col3 的 tile (t3,t7,t11,t15) 发到 GPU3

#### Shape Mapping：tile_id → 张量切片

```
shape_mapping(tile_id=5) = C[row1, col1]
                         = C[BLOCK_M*1 : BLOCK_M*2, BLOCK_N*1 : BLOCK_N*2]
```

就是说：tile #5 对应 C 矩阵中哪块区域？编译器需要知道这个才能生成正确的地址计算代码。

#### Rank Mapping：tile_id → 目标 GPU

```
rank_mapping(tile_id=5) = 1    (col1 → GPU1)
rank_mapping(tile_id=10) = 2   (col2 → GPU2)
rank_mapping(tile_id=0) = 0    (col0 → GPU0, 留本地)
```

就是说：tile #5 应该发给哪个 GPU？对于 AllScatter 来说，列号决定目标 rank。

#### Channel Mapping：tile_id → barrier 编号

```
channel_mapping(tile_id=5) = 5   (用 barrier #5 来同步这个 tile)
channel_mapping(tile_id=10) = 10 (用 barrier #10 来同步这个 tile)
```

每个 tile 有自己独立的 barrier——生产者写完 tile #5 后 signal barrier #5，消费者只等 barrier #5，不用等所有 tile 都算完。这就是"tile 粒度同步"的关键。

### 静态 vs 动态 mapping

```
静态 mapping（编译期确定）：
  rank_mapping(tile_id) = tile_id % world_size
  → 编译器直接生成：dst_rank = tid % 4
  → 适用于：AllScatter, AllGather 等固定 pattern

动态 mapping（运行期确定）：
  rank_mapping(tile_id) = routing_table[tile_id]  ← 运行时填入
  → 编译器生成：dst_rank = tl.load(routing_table + tid)
  → 适用于：MoE 的动态路由（每个 token 去不同的 expert）
```

### 为什么 Iris 不需要 mapping？

因为 Iris 不做编译器层面的映射——用户在 kernel 里自己写 for 循环和 if 判断：

```python
# Iris：用户显式写出 mapping 逻辑
for remote_rank in range(world_size):          # ← rank mapping
    offset = rm * stride + (rn + cur_rank * N) * stride  # ← shape mapping
    iris.store(C + offset, c, cur_rank, remote_rank, ...)
# 如果要做 tile 级同步，用户还会显式写 tl.load/tl.store 或 iris.atomic_* 逻辑
```

mapping 的三个问题在 Iris 里是用户用 Python/Triton 代码直接回答的，不是编译器自动推导的。

---



## 系统三：Iris

### 第 1 层：用户代码（纯 Triton，选择 Fused Sequential 模式）

```python
import triton
import triton.language as tl
import iris

@triton.jit
def fused_gemm_all_scatter(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm_global, stride_cn_global,
    cur_rank, world_size,
    heap_bases,          # <--- Iris 特有：所有 rank 的堆基址
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # ====== persistent kernel: 每个 SM 处理多个 tile ======
    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # ====== GEMM 计算（标准 Triton）======
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        c = acc.to(C.type.element_ty)

        # ====== 就在这里——GEMM tile 算完，立刻 scatter ======
        # 计算全局偏移（考虑当前 rank 的 N 维度偏移）
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        offset = rm[:, None] * stride_cm_global + \
                 (rn[None, :] + cur_rank * N) * stride_cn_global

        # iris.store: 寄存器中的 tile → 远端内存（value-based）
        for remote_rank in range(world_size):
            iris.store(                          # <--- Iris 核心
                C + offset, c,                   # 目标地址 + 数据
                cur_rank, remote_rank,           # 源和目标 rank
                heap_bases,                      # 堆基址表
                mask=mask
            )
        # ↑ 这就是 Fused Sequential 的全部！
        #   没有额外的 barrier，没有单独的通信 kernel
        #   每算完一个 tile 立刻发出去

# ====== Host-side 启动 ======
def main(rank, world_size):
    iris_ctx = iris.iris(heap_size=2**30)  # 1GB symmetric heap
    A = iris_ctx.randn((M, K), dtype=torch.float16)
    B = iris_ctx.randn((K, N), dtype=torch.float16)
    C = iris_ctx.zeros((M, N * world_size), dtype=torch.float16)

    heap_bases = iris_ctx.get_heap_bases()
    fused_gemm_all_scatter[(NUM_SMS,)](
        A, B, C, M, N, K, ...,
        cur_rank=rank, world_size=world_size,
        heap_bases=heap_bases, ...
    )
    iris_ctx.barrier()
```

**关键点**：Iris 的核心增量是“远端 store / put 的那一段循环”，没有额外的 distributed compiler layer，也没有外部 SHMEM 库参与 device-side 远端访问。repo 里的 examples 还包含 `global_offset`、mask、本地 fast path、可选 timestamp 等脚手架。

### 第 2 层：编译（标准 Triton，无特殊步骤）

```
用户 Python + @triton.jit
    │
    ▼
标准 Triton 编译流水线（与单 GPU kernel 走同一套 pipeline）
  ├─ Triton IR
  ├─ Triton GPU IR
  ├─ LLVM IR
  └─ AMDGPU ISA → HSACO（当前仓库主路径）

# iris.store 内部展开为：
#   __translate() → tl.store()
# 这些都是标准 Triton 操作，至少在 IR 形式上对编译器可见
# ⚠️ 这里没有额外的 distributed compiler layer；是否发生跨通信优化，取决于标准 Triton / LLVM pass
```

### 第 3 层：iris.store 内部实现（纯 Triton）

```python
@triton.jit
def store(pointer, value, from_rank, to_rank, heap_bases, mask=None):
    # 步骤 1：指针翻译
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    # 步骤 2：标准 Triton store（写到远端内存）
    tl.store(translated_ptr, value, mask=mask)
    #         ↑ 就是一个普通的 tl.store ！
    #           GPU 硬件沿底层设备互连把写入路由到远端地址

@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases):
    # 加载两个 rank 的堆基地址
    from_base = tl.load(heap_bases + from_rank)  # 一次 L1 cache hit
    to_base   = tl.load(heap_bases + to_rank)    # 一次 L1 cache hit
    # 纯算术：remote_ptr = to_base + (ptr - from_base)
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_ptr_byte = to_base_byte + offset
    translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)
    return translated_ptr

# 这就是 Iris 的全部通信实现！
# 没有 NVSHMEM，没有 inline ASM，没有不透明调用
# 编译器看到的就是：load + 减法 + 加法 + store
```

### 第 4 层：内存建立

```python
# 当前 SymmetricHeap 主实现：
# 1. 选择 TorchAllocator / VMemAllocator
# 2. setup_fd_infrastructure(...) 建立 fd 传递通道
# 3. distributed_allgather 收集各 rank base
# 4. export_dmabuf_handle / mem_import_from_shareable_handle / mem_map
#    把 peer 段映射进本进程地址空间
# 5. 构建 heap_bases tensor 给 device-side __translate 使用
```

### 第 5 层：硬件执行

```
GPU0 上的一个 wavefront 执行 tl.store(translated_ptr, tile_data)
    │
    ▼
translated_ptr 指向 GPU2 的内存地址
    │
    ▼
底层设备互连负责把这次远端写路由到目标 GPU：
  GPU0 的 store 指令 → 互连链路 → GPU2 的内存系统

# 关键：这不是"调用通信库"，而是"写一个远端地址"
# 具体走哪条链路取决于机器拓扑和运行环境，不由 Iris Python API 决定
```

---

## 系统四：XTile

> 维护说明
>
> - 为降低维护成本和口径漂移风险，本文件不再维护 XTile 的分层流程展开。
> - XTile 当前流程、公开入口、边界、support surface 与性能口径，统一以 [XTile现状流程_修订版.md](./XTile现状流程_修订版.md) 为准。

---

## 关键差异总结表

| 方面 | TileScale | Triton-distributed | Iris | XTile |
| --- | --- | --- | --- | --- |
| 用户接口层级 | TileLang distributed primitives | Triton distributed low-level ops | Triton kernel 内远端访存原语 | Host ops + patterns + primitives |
| 编译表示 | TVM IR + extern / NVSHMEM 调用 | Distributed dialect + backend lowering | 标准 Triton load/store | 标准 Triton + pointer translation / load / store / atomic |
| overlap 控制 | 用户编排，编译器与硬件配合 | 用户显式同步与通信 | 用户手写 kernel pattern | 显式 pattern 或 auto-select |
| 通信机制 | NVSHMEM 与远端拷贝 helper | SHMEM runtime + distributed op lowering | Pointer translation + remote load/store | Pointer translation + load/store/atomic + tile collectives |
| 运行时依赖 | `torch.distributed` + NVSHMEM | `torch.distributed` + SHMEM runtime | PyTorch + `torch.distributed` + ROCm runtime | PyTorch + Triton + CUDA/HIP runtime（多进程 heap 建立与 IPC handle 交换还依赖 `torch.distributed`） |
| 目标硬件 | NVIDIA | NVIDIA / AMD | AMD / ROCm | NVIDIA / AMD |
| collective 抽象 | distributed primitives 组合 | low-level primitives 组合 | 远端访存原语组合 | host collective ops + tile collectives |

---

## TVM 编译器和 Triton JIT 编译器的区别；AST 是什么

### TVM 编译器 vs Triton JIT 编译器

它们是**两个完全不同的编译器**，服务于不同的系统：

```
TileScale 用 TVM 编译器（基于 Apache TVM 项目）
Iris、TileLink、XTile 用 Triton JIT 编译器（基于 OpenAI Triton 项目）
```

**TVM 编译器**是什么：

TVM 是 Apache 的一个开源深度学习编译器框架。TileLang/TileScale 是建在 TVM 之上的——它们用 TVM 的 IR（中间表示）和优化 pass 作为基础设施。

```
用户写的 TileLang 代码（Python DSL）
    │ TileLang 前端把它翻译成 TVM 的 PrimFunc IR
    ▼
TVM 的优化管线
    │ 一系列 optimization passes（循环展开、向量化、内存规划等）
    ▼
TVM 的 CodeGen
    │ 把优化后的 IR 翻译成 CUDA C 代码
    ▼
nvcc（NVIDIA 的 CUDA 编译器）
    │ 编译 CUDA C → PTX → cubin
    ▼
可执行的 GPU kernel
```

**Triton JIT 编译器**是什么：

Triton 是 OpenAI 创建的一个 GPU 编程语言。它自带一个 JIT（即时）编译器。

```
用户写的 @triton.jit 函数（Python）
    │ Triton 在第一次调用时触发 JIT 编译
    ▼
Triton IR（Triton 自己的中间表示）
    ▼
Triton GPU IR（针对 GPU 特性的中间表示）
    ▼
LLVM IR（通用的低级中间表示）
    ▼
PTX（NVIDIA）或 AMDGPU ISA（AMD）
    ▼
cubin（NVIDIA）或 HSACO（AMD）
```

**关键区别**：

| 方面     | TVM（TileScale 用）            | Triton JIT（Iris/TileLink/XTile 用） |
| -------- | ------------------------------ | ------------------------------------ |
| 编译时机 | 可以提前编译（AOT）            | 第一次调用时编译（JIT）              |
| IR 基础  | TVM PrimFunc                   | Triton 自研 IR → LLVM IR             |
| 代码生成 | 生成 CUDA C 源码 → nvcc 编译   | 直接生成 LLVM IR → 直出机器码        |
| 硬件支持 | 很广（CPU/GPU/TPU/各种加速器） | GPU 为主（NVIDIA + AMD）             |
| 社区     | 学术界为主                     | 工业界广泛使用（PyTorch 生态）       |

### AST 是什么

AST = Abstract Syntax Tree（抽象语法树）。它是**编译器理解你的代码结构的第一步**。

举个例子，你写了这段 Python：

```python
for k in range(0, K, BLOCK_K):
    a = tl.load(A_ptr + k)
    acc += tl.dot(a, b)
```

Python 解释器（或编译器）首先把它解析成一棵树：

```
          ForLoop
         /   |    \
      var   range   body
       k   (0,K,BK)  │
                   [Assign, Assign]
                   /            \
              a = Call         acc += Call
                  |                  |
              tl.load            tl.dot
              /                  / \
          BinOp(+)              a   b
          / \
       A_ptr  k
```

这就是 AST——把文本代码变成一棵结构化的树，方便编译器分析和变换。

**TileLink 为什么要做 AST 解析？**

released Triton-distributed 仓库没有直接公开 `block_channel` 这类对象，也没有可直接指认的 `shape_mapping` / `rank_mapping` / `channel_mapping` 用户级 API。能直接对应到仓库里的，是 `distributed.wait`、`distributed.consume_token`、`distributed.symm_at`、`distributed.notify` 这些低层 op；mapping 这一节描述的是论文里的概念模型。

---

## 问题 4：TileScale 和 TileLink 如何接入 NVSHMEM / SHMEM backend

### 先理解 NVSHMEM 是什么

NVSHMEM 是 NVIDIA 提供的一个库，有两套 API：

```
Host-side API（在 CPU 上调用）：
  nvshmem_init()         → 初始化分布式环境
  nvshmem_malloc(size)   → 分配 symmetric heap
  nvshmem_barrier_all()  → 全局同步

Device-side API（在 GPU kernel 内部调用）：
  nvshmem_float_put_nbi(dst, src, count, peer)  → 非阻塞远端写
  nvshmem_float_get(dst, src, count, peer)       → 远端读
  nvshmemx_signal(addr, val, peer)               → 远端 signal
  nvshmem_quiet()                                → 等待所有操作完成
```

### TileScale 如何使用 NVSHMEM

**通过 C 函数调用**只对 `putmem_*` / `getmem_*` / `barrier_all` / `sync_all` 这一族直接成立；当前 repo 还有一条 `put_block/get_block/put_warp/get_warp` 通过 remapped address + `tl::cp_block` / `tl::cp_warp` extern helper 的路径。

```c
// 路径 A：putmem family
nvshmemx_putmem_nbi_block(remote_ptr, local_ptr, tile_size_in_bytes, dst_rank);
nvshmem_barrier_all();

// 路径 B：tileop family
remote_dst = get_remote_base_ptr(dst_pe) + offset_to_base;
call_extern("tl::cp_block<...>", remote_dst, local_src);
```

### TileLink 如何使用 NVSHMEM

Triton-distributed 的 NVIDIA 路径同时包含：
- `WaitOp` / `NotifyOp` 这类自定义 distributed op 的 LLVM/PTX lowering；
- `libshmem_device.*` 这类直接暴露给 Python 的 NVSHMEM device primitives。

下面是 PTX/底层信号示意：

```c
// 在 Triton lowering 中，wait/notify 之类 distributed op 可能对应 PTX 级实现：
asm volatile(
    "membar.sys; st.relaxed.sys.global.b32 [%0], %1;"
    :
    : "l"(remote_ptr), "r"(data)
    : "memory"
);
```

### 两种方式的区别

```
TileScale：
  - `putmem_*` / `barrier_all` 这一路，codegen 直接生成 `nvshmemx_*` / `nvshmem_*` 调用
  - `put_block/get_block` 这一路，先 remap remote address，再走 `tl::cp_block` / `tl::cp_warp` extern helper

Triton-distributed：
  - `wait` / `notify` / `symm_at` 先进入 Distributed Dialect，再做 backend-specific lowering
  - NVIDIA 路径里，`wait` 和部分 `notify` 会变成 PTX inline asm
  - 但 `symm_at`、inter-node `notify`、`libshmem_device.*` 仍然会调用 SHMEM device functions
  - AMD 路径里，`wait` 是 LLVM polling loop，`symm_at` / extern calls 走 rocshmem / mori wrapper

共同点：
  - 只要最终落到外部 SHMEM 调用，这部分对上层优化器仍然是不透明的
  - Triton-distributed 的 distributed op 本身，比纯 extern call 多一层可分析的 IR
```

用人话说：

```
TileScale：请了一个翻译（NVSHMEM 库）来帮你跟外国人（远端 GPU）说话。
           你不懂外语，翻译说了什么你也不知道。

TileLink：你自己背了几句外语（PTX 内联汇编），直接跟外国人说。
          比 TileScale 少了一个中间人，但你说的那几句话，
          你的助手（Triton 优化器）也听不懂。

Iris：    你发现其实不需要说外语——你可以直接把东西放到对方桌上（tl.store）。
          你的助手（Triton 编译器）在 IR 层会把它当作普通 store 来看，
          因为它和本地 store 属于同一类 Triton 操作。
```

---

## 什么是编译器"透明/半透明/不透明"

### "编译器"指的是谁

在这个上下文中，"编译器"指的是**把你的 kernel 代码翻译成 GPU 机器码的那个工具**，具体就是 **Triton JIT 编译器**（对于 Iris、TileLink、XTile）或 **TVM 编译器**（对于 TileScale）。

编译器的核心工作除了翻译，还包括**优化**。优化的前提是：**编译器必须理解代码在做什么**。

### 不透明（TileScale 的 extern / NVSHMEM 调用）

```python
# TVM lower / CUDA codegen 视角（概念性展示）：
acc = ...
call_extern("nvshmemx_putmem_nbi_block", dst, src, bytes, pe)   # ← 编译器：外部调用
call_extern("tl::cp_block<128>", remote_dst, local_src)         # ← 编译器：外部调用
```

编译器看到这类 extern call 时，能直接利用的高层语义很少：

- 它知道这里有一个**外部调用边界**
- 参数列表是可见的
- 但调用内部的内存语义、延迟模型、实现细节并没有在上层 IR 中展开

这类调用会形成比较保守的优化边界。当前仓库代码没有提供足够高层语义，让上层 pass 把它像普通 `tt.load` / `tt.store` 那样分析。

**这就是"不透明"——编译器完全不知道这个操作内部在做什么，只能保守处理。**

### 半透明（Triton-distributed 的 NVIDIA inline PTX 路径）

```python
# 编译器看到的 LLVM IR（概念性展示）：
%acc = call <1024 x float> @triton_dot(...)     # ← 编译器理解
call void asm sideeffect                         # ← 编译器部分理解
    "st.global.relaxed.gpu.u32 [$0], $1;         #    知道它做了一个 store
     fence.proxy.async;",                         #    知道有一个 memory fence
    "l,r"(i64 %remote_ptr, i32 %data)            #    知道输入是 ptr 和 data
```

编译器看到 inline ASM 时，比纯 extern call 多暴露了一些信息：

- PTX 文本本身可见
- 操作数约束可见
- `sideeffect` / fence 这类附加信息可见

但它仍然不是 Triton/LLVM 能完全建模的普通 IR op。当前仓库里能直接证实的是：NVIDIA 路径的 `wait` / 部分 `notify` 会 lower 成 PTX inline asm；文档不再把某一条具体优化“能不能做”写成已验证事实。

**这就是"半透明"——编译器能看到一些外部特征，但看不到内部逻辑。**

### 全透明（Iris 的 `tl.store`；XTile 的 `translate_ptr + load/store/atomic`）

```python
# 编译器看到的 Triton IR（概念性展示）：
%acc = tt.dot %a, %b                              # 矩阵乘法
%from_base = tt.load %heap_bases_ptr               # 加载堆基址
%to_base = tt.load %heap_bases_ptr + %to_rank      # 加载远端基址
%offset = arith.subi %ptr_int, %from_base          # 计算偏移
%remote_ptr = arith.addi %to_base, %offset         # 计算远端地址
tt.store %remote_ptr, %acc, %mask                  # 存储到远端
```

从 IR 形态上看，Iris 这条路径里出现的仍是标准 Triton 操作：

- `tt.load`
- `arith.subi` / `arith.addi`
- `tt.store`

XTile 当前源码里的 device path 也属于同一类“IR 可见”的 Triton 组合，只是具体形态更准确地说是：

- `translate_ptr(...)`
- `tt.load` / `tt.store`
- `tt.atomic_*`

而不是把它简化成“只有一个裸 `tl.store`”。这意味着它同样不会额外引入 extern / inline-asm 边界。至于 LICM、向量化、重排等具体优化是否真的发生，要看当下 Triton / LLVM pass；当前仓库里我没有看到专门针对跨 GPU `tt.store` 或 translated remote access 的新 pass。

**这就是"全透明"——在 IR 形态上，通信仍表现为普通的 load/store，没有新增 extern / asm 黑盒边界。**

### 直觉对比

```
不透明（TileScale）：
  你跟编译器说"请快递公司把包裹送到 GPU2"。
  编译器不知道快递公司怎么送，不敢假设任何事情。
  → 上层优化器通常会更保守地跨这个边界处理。

半透明（TileLink）：
  你跟编译器说"我自己开车把包裹送到 GPU2，走高速公路"。
  编译器知道你在开车（store 指令），知道走高速（memory fence），
  但不知道你具体走哪条路线，开多快。
  → 可见性比纯 extern 更高，但仍然有明显黑盒边界。

全透明（Iris/XTile）：
  你跟编译器说"把包裹放到架子上"。
  在 IR 层，它跟放到本地架子上属于同一种 load/store/atomic 操作家族。
  只不过这个架子恰好在另一个房间（GPU2），搬运工（硬件）自己会处理。
  → 至少在 IR 层，没有额外的 extern / asm 黑盒边界。
```

### 现实中的影响有多大？

仓库代码直接体现的是“IR 更透明”，并没有给出“已经因为这件事拿到了多少现成性能收益”的证据。

1. 我没有在当前仓库里看到专门针对跨 GPU `tt.store` 的额外 Triton pass。

2. examples 里的 overlap 仍主要来自用户显式写出的 pattern、同步和 kernel 结构。

3. Iris / XTile 保留了更多编译器分析空间，但当前仓库没有实现自动通信调度优化。

---

## 各种 IR 是什么？PTX vs cubin

### IR 是什么

IR = Intermediate Representation（中间表示）。就是代码在编译过程中的**中间形态**。

一段代码从"人写的 Python"到"GPU 执行的二进制"要经过多次变形。每次变形后的形态就是一种 IR。就像把中文翻译成法语，可能先翻译成英语作为中间语言。

### Triton 的编译链路中的所有 IR

让我用一个具体的 Triton kernel 追踪每一层：

#### 用户代码（Python）

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

#### Triton IR（Triton 自己的高级 IR）

```mlir
// Triton IR 使用 MLIR 方言（一种标准 IR 框架的"语言"）
// 这一层还保留了 tile 的概念

module {
  tt.func @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>,
                       %out_ptr: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id {axis = 0 : i32} : i32
    %block_start = arith.muli %pid, %c1024 : i32      // pid * BLOCK
    %offsets = tt.make_range {start = 0, end = 1024}   // arange(0, BLOCK)
    %addr = tt.addptr %x_ptr, %offsets                 // x_ptr + offsets
    %mask = arith.cmpi "slt", %offsets, %n             // offsets < n
    %x = tt.load %addr, %mask                          // tl.load
    // ... 类似处理 y ...
    %sum = arith.addf %x, %y                           // x + y
    tt.store %out_addr, %sum, %mask                    // tl.store
    tt.return
  }
}

// 特点：
// - 操作的是 tile（一整块 1024 个元素同时处理）
// - tt.load / tt.store 是 tile 级操作
// - 没有 thread 的概念，没有 shared memory 的概念
```

#### Triton GPU IR（针对 GPU 硬件特化的 IR）

```mlir
// Triton GPU IR 引入了 GPU 特有的概念：
// - 线程布局（哪些线程处理 tile 的哪些元素）
// - shared memory（数据先到 shared memory 再到寄存器）
// - 内存合并访问（coalesced access）

module {
  tt.func @add_kernel(%x_ptr: !tt.ptr<f32>, ...) {
    // 这里出现了"布局"注解——告诉硬件如何把 tile 分给 32 个线程
    %x = tt.load %addr, %mask
        {layout = #blocked<{sizePerThread=[32], threadsPerWarp=[32],
                            warpsPerCTA=[4], order=[0]}>}

    // 如果需要，会插入 shared memory 操作
    %x_shared = ttg.local_alloc %x : tensor<1024xf32>
        -> !ttg.memdesc<1024xf32, #shared>
    %x_reg = ttg.local_load %x_shared
        : !ttg.memdesc<1024xf32, #shared> -> tensor<1024xf32, #blocked>

    // 特点：
    // - 出现了 warp、CTA、thread 的概念
    // - 出现了 shared memory
    // - 但还不是最终的机器码——还有一层 LLVM IR
  }
}
```

#### TileLink 的 Distributed IR（当前 repo 里真实存在的 op 名）

```mlir
// 在 Triton GPU IR 之外，TileLink 新增了一套平行的 IR
// 专门处理分布式通信指令

module {
  // 当前 repo 里能直接对应到的 distributed op 更接近这些
  %token = distributed.wait %barrier_ptr, %num_barriers, %wait_value,
      scope = "gpu", semantic = "acquire"
  %remote = distributed.symm_at %symm_ptr, %rank
  %value2 = distributed.consume_token %value, %token
  distributed.notify %sig_addr, %signal_val, %rank,
      sig_op = "set", comm_scope = "intra_node"

  // `libshmem_device.*` 则是另一条 extern SHMEM primitive 路径，
  // 不一定表现成这组 distributed op。
}
```

#### LLVM IR（通用低级 IR）

```llvm
; LLVM IR 是一种"接近机器码但不依赖具体硬件"的表示
; 它是 Triton（和很多其他编译器）的最后一个"跨平台"阶段

define void @add_kernel(float* %x_ptr, float* %y_ptr, float* %out_ptr, i32 %n) {
entry:
  ; 获取 thread ID
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

  ; 计算地址
  %idx = add i32 %tid, %offset
  %addr_x = getelementptr float, float* %x_ptr, i32 %idx

  ; 加载（可能是向量化的）
  %x = load float, float* %addr_x, align 4

  ; ... 类似处理 y ...

  ; 加法
  %sum = fadd float %x, %y

  ; 存储
  store float %sum, float* %out_addr, align 4

  ; 如果有 TileLink 的通信操作，会出现 inline ASM：
  call void asm sideeffect
    "st.global.relaxed.gpu.u32 [$0], $1; fence.proxy.async;",
    "l,r"(i64 %remote_ptr, i32 %data)

  ; 特点：
  ; - 已经没有 tile 的概念，变成了单个元素/向量的操作
  ; - 已经有了 thread ID、地址计算等具体细节
  ; - 但还没有变成特定 GPU 的指令集
  ret void
}
```

#### PTX（NVIDIA 虚拟汇编）

```ptx
// PTX = Parallel Thread Execution
// 是 NVIDIA GPU 的"虚拟汇编语言"
// "虚拟"的意思是：它不是 GPU 直接执行的——还需要一步转换

.entry add_kernel(.param .u64 x_ptr, .param .u64 y_ptr, ...) {
    .reg .f32 %f<4>;        // 声明浮点寄存器
    .reg .u32 %r<8>;        // 声明整数寄存器
    .reg .u64 %rd<4>;       // 声明 64 位寄存器
    .reg .pred %p<2>;       // 声明谓词寄存器（用于条件判断）

    // 获取 thread ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mad.lo.s32 %r2, %r1, 1024, %r0;    // idx = blockIdx * 1024 + threadIdx

    // 检查边界
    setp.lt.s32 %p0, %r2, %r7;          // mask = idx < n

    // 加载 x[idx]
    @%p0 ld.global.f32 %f0, [%rd0];     // if (mask) x = load(x_ptr + idx)

    // 加载 y[idx]
    @%p0 ld.global.f32 %f1, [%rd1];

    // 加法
    add.f32 %f2, %f0, %f1;              // sum = x + y

    // 存储
    @%p0 st.global.f32 [%rd2], %f2;     // if (mask) store(out + idx, sum)

    ret;
}

// 特点：
// - 看起来像汇编，但是"虚拟"的（不是真正的机器码）
// - GPU 上不同代际的硬件有不同的真实指令集
// - PTX 是所有 NVIDIA GPU 通用的，由 ptxas 翻译成真实指令
```

#### cubin（真正的 GPU 机器码）

```
cubin 是二进制文件，人类不可读。
它包含的是 NVIDIA GPU 真正执行的机器码（SASS 指令集）。

PTX → cubin 的转换由 ptxas（NVIDIA 的 PTX 汇编器）完成。
不同的 GPU 架构（sm_80 = A100, sm_90 = H100）生成不同的 cubin。

用 cuobjdump 可以反汇编 cubin，看到真实的 SASS 指令：

/*0000*/ MOV R1, c[0x0][0x28] ;         // 硬件寄存器操作
/*0010*/ S2R R0, SR_TID.X ;             // 读 thread ID（特殊寄存器）
/*0020*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x160] ;
/*0030*/ ISETP.GE.AND P0, PT, R0, R2, PT ;
/*0040*/ @P0 EXIT ;
/*0050*/ LDG.E R3, [R4] ;              // 全局内存加载
/*0060*/ LDG.E R5, [R6] ;
/*0070*/ FADD R7, R3, R5 ;             // 浮点加法
/*0080*/ STG.E [R8], R7 ;              // 全局内存存储

// 这才是 GPU 真正执行的东西！
```

#### AMD 的对应物

```
AMD 侧：
  LLVM IR → AMDGPU ISA（AMD 的汇编）→ HSACO（AMD 的可执行格式）

  AMDGPU ISA 示例：
  v_add_f32 v0, v1, v2          // 浮点加法
  global_load_dword v3, v[4:5]  // 全局内存加载
  global_store_dword v[6:7], v8 // 全局内存存储

  HSACO 相当于 NVIDIA 的 cubin——是真正的机器码。
```

AMD和NVIDIA

```
AMD:
LLVM IR（通用低级 IR） → AMDGPU ISA（AMD 的汇编）→ HSACO（AMD 的可执行格式）

NVIDIA:
LLVM IR（通用低级 IR） → PTX（人类可以阅读和检查的汇编） → cubin（GPU 实际执行的二进制）
```



### 为什么有时候说 PTX 有时候说 cubin？

```
PTX 和 cubin 是同一条流水线上的两个阶段：
  LLVM IR → PTX → cubin

说"编译到 PTX"：强调的是中间产物（人类可以阅读和检查的汇编）
说"编译到 cubin"：强调的是最终产物（GPU 实际执行的二进制）

在实际使用中：
  - Triton 默认直接编译到 cubin（跳过人类不需要看的 PTX 文本阶段）
  - 调试时可以让 Triton 输出 PTX 来检查生成的代码质量
  - TileLink 论文说"编译到 PTX"是因为它们的 Distributed IR
    先翻译成 PTX 级别的 inline ASM，然后再由 ptxas 编译成 cubin
```

完整的等式：

```
PTX ≈ 虚拟汇编（跨 GPU 代际通用，人类可读）
cubin ≈ 真实机器码（特定 GPU 架构，人类不可读）

类比 CPU 世界：
  PTX ≈ x86 汇编（.s 文件）
  cubin ≈ 可执行文件（.o / .exe）
```

---
