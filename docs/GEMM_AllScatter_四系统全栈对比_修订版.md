# 固定任务：GEMM + AllScatter 全栈代码流对比（修订版）

> 修订说明
> - 原始文档保留不动；本文件是在原文基础上的核对修订版。
> - `【修订】` 表示原文该处已按当前仓库现实改写。
> - `【新增核对】` 表示为避免误解而补充的仓库现状说明。

## 任务定义

4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
每个 GPU 先算 C_local = A × B（本地 GEMM），然后把 C_local scatter 到所有其他 GPU，
最终每个 GPU 都拥有完整的 C_global[M, N×4]。

M=8192, N=4608, K=36864, 4 GPUs.

---

## 系统一：TileScale

### 第 1 层：用户代码（TileLang DSL）

【修订】原文把 TileScale 当前用户接口写成了 `T.scatter()`。当前仓库公开接口更接近 `T.copy(..., dst=...)`、`T.putmem_nbi_block(...)`、`T.barrier()`；下面改成与当前代码路径一致的示意。

```python
import tilelang
import tilelang.language as T

@tilelang.jit(pass_configs={"tl.disable_rdc": True})
def fused_gemm_scatter(rank, num_ranks, M, N, K_per_rank, block_M, block_N, block_K):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K_per_rank), "float16"),
        B: T.Tensor((N, K_per_rank), "float16"),
        C: T.Tensor((M // block_M, N // block_N, block_M, block_N), "float16"),
    ):
        with T.Kernel(N // block_N, M // block_M, threads=128) as (bx, by):
            # ... GEMM 计算 ...
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by, bx, :, :])
            peer = by // M_blocks_per_rank
            T.putmem_nbi_block(
                T.address_of(C[by, bx, 0, 0]),
                T.address_of(C[by, bx, 0, 0]),
                block_M * block_N * 2,
                peer,
            )
            T.barrier()
```

**关键点（修订）**：当前 repo 暴露的是较低层的 distributed/NVSHMEM primitives，不是原文里的公开 `T.scatter()` 高层接口。

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
  ├─ T.copy → 展开为 TMA / LSU / async copy（编译器自动选最优）
  ├─ T.Pipelined → 插入 async copy + barrier 实现多级流水线
  ├─ distributed `T.copy(..., dst=...)` / `T.putmem_*`
  │    → 此处编译到 NVSHMEM 调用（见下层）
  └─ T.barrier / barrier_all → 编译到 NVSHMEM barrier
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

"编译到 NVSHMEM 调用"具体是什么意思？

不是"编译成 .so 文件然后调用 NVSHMEM"。实际流程是这样的：

```
TileScale 编译器输出的是一个 CUDA kernel（.cu 文件）。
在这个 kernel 的 __device__ 函数中，包含了对 NVSHMEM 的设备端 API 调用。
然后用 nvcc 编译这个 .cu 文件时，nvcc 会链接 NVSHMEM 的静态库（.a 文件）。
最终生成一个包含 NVSHMEM 代码的 .so / .cubin。
```

用具体代码说明：

```
// TileScale 编译器生成的 CUDA kernel（中间产物）
__global__ void gemm_scatter_kernel(float* A, float* B, float* C, ...) {
    // --- 编译器生成的 GEMM 部分（编译器完全理解）---
    float acc[BLOCK_M][BLOCK_N] = {0};
    for (int k = 0; k < K; k += BLOCK_K) {
        // ... WMMA 指令，编译器可以自由重排 ...
    }

    // --- 编译器生成的通信部分（编译器不理解内部）---
    nvshmem_float_put_nbi(dst_ptr, acc, size, dst_rank);
    //     ↑ 这是一个函数调用
    //       nvcc 编译时会链接 NVSHMEM 库
    //       NVSHMEM 库的实现是预编译的二进制代码
    //       TileScale 的编译器（TVM）看不到这个函数内部做了什么
}
```

关键区别：

```
✓ 编译器可以决定"在哪里插入 nvshmem_put"（GEMM 循环之后？每个 tile 之后？）
✗ 编译器不能决定"nvshmem_put 内部怎么执行"（走哪条 NVLink？分几个包？）
✗ 编译器不能把 nvshmem_put 和 GEMM 的 load 指令交错
  （因为不知道 nvshmem_put 会读写哪些寄存器/内存）
```



### 第 3 层：通信底层（NVSHMEM）

```c
// 【修订】当前仓库里更贴近现实的 lowering 是：
// distributed `T.copy(..., dst=...)` / `T.putmem_nbi_block(...)`
// 生成 NVSHMEM device-side put / barrier 调用，而不是公开的 `T.scatter()` API。
__device__ void put_tile(void* dst, void* src, int bytes, int pe) {
    nvshmemx_putmem_nbi_block(dst, src, bytes, pe);
    nvshmem_barrier_all();
}
// ⚠️ 这里的 NVSHMEM 调用对上层编译优化仍然是不透明的。
```

### 第 4 层：内存建立

```python
# 【修订】当前仓库主路径不是“必须 mpirun + MPI API”。
# 当前实现更接近：
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

NVSHMEM 内部选择：NVLink 直传 or PCIe fallback
对用户和编译器都不透明
```

---

## 系统二：TileLink / Triton-Distributed

### 第 1 层：用户代码（Triton + TileLink 原语）

【修订】原文把论文/未来高层接口写成了当前公开 API。当前 Triton-distributed 已发布的是 low-level primitives：`wait`、`consume_token`、`rank`、`num_ranks`、`symm_at`、`notify`，以及 `libshmem_device.*`；README 和 `docs/primitives.md` 都明确说高层 tile-centric primitives 还没发布。

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

**关键点（修订）**：当前公开接口是“低层同步原语 + 对称地址转换 + NVSHMEM device functions”，不是 `BlockChannel` 这套高层 tile API。

### 第 2 层：TileLink 编译器 （Triton JIT 编译器（基于 OpenAI Triton 项目））

```
【修订】当前 released repo 更接近下面这条编译路径：

用户 Python（`wait` / `consume_token` / `symm_at` / `notify` / `libshmem_device.*`）
    │
    ▼
Triton Distributed Dialect / extern NVSHMEM calls
    │
    ├─ WaitOp → PTX spin-wait lowering
    ├─ NotifyOp → PTX/NVSHMEM signal lowering
    ├─ GetRank/GetNumRanks/SymmAt → NVSHMEM device call lowering
    └─ `libshmem_device.*` → 外部 NVSHMEM device-side primitive
    ▼
LLVM IR → PTX → cubin（NVIDIA GPU 可执行文件）
```

### 第 3 层：通信底层（仍是 NVSHMEM）

```c
// 【修订】当前仓库更能直接对应下面这些 lowerings：
// 1. wait(...) -> PTX spin-wait
// 2. notify(...) -> PTX/NVSHMEM signal
// 3. libshmem_device.putmem_* -> NVSHMEM device primitive

// wait(...)
asm volatile("spin: ld.global.acquire.gpu.b32 %0, [%1]; ...");

// notify(...)
asm volatile("membar.sys; st.relaxed.sys.global.b32 [%0], %1;");

// libshmem_device.putmem_signal_nbi_block(...)
nvshmemx_putmem_signal_nbi_block(dst, src, bytes, sig_addr, signal, sig_op, pe);
```

### 第 4 层：内存建立

```python
# 【修订】当前仓库启动方式是 torchrun / scripts/launch.sh，
# 初始化逻辑基于 torch.distributed ProcessGroup，再初始化 SHMEM 后端。
$ bash ./scripts/launch.sh tutorials/01-distributed-notify-wait.py

# 内部流程：
# 1. torch.distributed.init_process_group(...)
# 2. pg = torch.distributed.new_group(...)
# 3. initialize_distributed(...) 内部按后端调用
#    - CUDA: init_nvshmem_by_torch_process_group(pg)
#    - HIP:  init_rocshmem_by_torch_process_group(pg) / mori_shmem
# 4. 对称 tensor 由 SHMEM backend 创建
```

### 第 5 层：硬件执行

```
与 TileScale 类似：底层仍由 SHMEM runtime 选择 NVLink / PCIe / ROCm fabric 等路径。
区别在于，当前 Triton-distributed 公开给用户的是更低层的 wait/notify/symm_at/NVSHMEM primitive 组合。
```

---

## TileLink 的 mapping 到底做什么（概念/未来高层接口，不是当前公开 API）

【新增核对】这一节保留原文的概念解释价值，但它描述的是论文里的高层 tile-centric 设计方向，不是当前 Triton-distributed 仓库已经公开的 `BlockChannel` 用户接口。

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
# Iris：用户手动实现 mapping 逻辑
for remote_rank in range(world_size):          # ← rank mapping
    offset = rm * stride + (rn + cur_rank * N) * stride  # ← shape mapping
    iris.store(C + offset, c, cur_rank, remote_rank, ...)
# barrier 用 atomic_cas 手动实现                 # ← channel mapping
```

mapping 的三个问题在 Iris 里是用户用 Python/Triton 代码直接回答的，不是编译器自动推导的。

---



## 系统三：Iris

### 第 1 层：用户代码（纯 Triton，选择 Fused Sequential 模式）

```python
import triton
import triton.language as tl
import iris  # 【修订】Iris 现仓库已明显超出早期“小型原型”规模，这里不再用固定代码行数描述

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

**关键点**：只比标准 Triton GEMM 多了 5 行代码（`for remote_rank ... iris.store(...)`）。没有编译器介入，没有外部库。

### 第 2 层：编译（标准 Triton，无特殊步骤）

```
用户 Python + @triton.jit
    │
    ▼
标准 Triton 编译流水线（与单 GPU kernel 完全相同）
  ├─ Triton IR
  ├─ Triton GPU IR
  ├─ LLVM IR
  └─ AMDGPU ISA（或 PTX → cubin）

# iris.store 内部展开为：
#   __translate() → tl.store()
# 这些都是标准 Triton 操作，编译器完全可见
# ⚠️ 但编译器不会主动优化通信调度——只是不阻碍
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
    #           GPU 硬件负责通过 NVLink/InfinityFabric 路由

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

### 第 4 层：内存建立（当前主路径已不是旧式 HIP IPC 单一路径）

```python
# 【修订】当前 SymmetricHeap 主实现更接近：
#
# 1. 选择 TorchAllocator / VMemAllocator
# 2. setup_fd_infrastructure(...) 建立 fd 传递通道
# 3. distributed_allgather 收集各 rank base
# 4. export_dmabuf_handle / mem_import_from_shareable_handle / mem_map
#    把 peer 段映射进本进程地址空间
# 5. 构建 heap_bases tensor 给 device-side __translate 使用
#
# 旧式 hipMalloc + hipIpcGetMemHandle + hipIpcOpenMemHandle 的思路
# 仍然有历史背景价值，但已不是当前仓库的主路径。
```

### 第 5 层：硬件执行

```
GPU0 上的一个 wavefront 执行 tl.store(translated_ptr, tile_data)
    │
    ▼
translated_ptr 指向 GPU2 的内存地址
    │
    ▼
AMD Infinity Fabric 硬件自动路由：
  GPU0 的 store 指令 → L2 cache miss
  → Infinity Fabric link → GPU2 的 L2 cache → GPU2 的 HBM

# 关键：这不是"调用通信库"，而是"写一个远端地址"
# GPU 硬件的 memory controller 自动处理跨 GPU 的数据路由
# 与本地 tl.store 的唯一区别是延迟更高（走了 Infinity Fabric）
```

---

## 系统四：XTile（设想）

【修订】本节保留“目标接口 / 设想 API”的表达方式，但不应再被理解为当前 `XTile/` 仓库已全部公开这些接口。当前真实实现与限制，请以《XTile现状流程_修订版.md》为准。

【新增核对】下面这节里的部分代码块仍然是“目标态伪接口”，例如 `ctx.randn(...)`、`ctx.empty(...)`、`xtile.fused_gemm_scatter(...)`。这些名字当前并不是仓库里可直接调用的一键 API。

### 第 1 层：用户代码（Triton + XTile auto-select）

```python
import triton
import triton.language as tl
import xtile

# ====== 方式 A：一键自动（最简单，目标态伪代码）======
def run_auto(rank, world_size):
    ctx = xtile.init(backend="auto")  # 自动检测 NVIDIA / AMD
    A = ctx.randn((M, K), dtype=torch.float16)
    B = ctx.randn((K, N), dtype=torch.float16)
    C = ctx.empty((M, N * world_size), dtype=torch.float16)

    # auto-select 根据 (M,N,K,world_size,hw_info) 选择最优 pattern
    result = xtile.fused_gemm_scatter(
        A, B, C,
        strategy="auto",  # 自动选择 pattern
        ctx=ctx
    )
    # 【修订】如果代入当前仓库里的 heuristics：
    #   N=4608, K=36864, 4 GPUs → N/world_size=1152
    #   不满足 fused_sequential 的 < 1024 分支
    #   → 更接近 ProducerConsumerPattern，而不是 fused_sequential

# ====== 方式 B：手动控制（高级用户）======
@triton.jit
def manual_fused_gemm_allscatter(
    A, B, C,
    M, N, K,
    cur_rank, world_size,
    heap_bases,       # <--- 与 Iris 相同的 symmetric heap
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # --- GEMM 计算（与 Iris 完全相同）---
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
            acc += tl.dot(a, b)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        c = acc.to(tl.float16)

        # --- 通信：与 Iris 原语级别相同 ---
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        offset = rm[:, None] * stride_cm + (rn[None, :] + cur_rank * N) * stride_cn

        for remote_rank in range(world_size):
            xtile.tile_remote_store(         # <--- XTile 原语
                C + offset, c,
                cur_rank, remote_rank,
                heap_bases, mask=mask
            )

# ====== 方式 C：Workgroup Specialization（用同一个库切换 pattern）======
@triton.jit
def wg_specialized_gemm_allscatter(
    A, B, C, locks,
    heap_bases,
    COMPUTE_SMS: tl.constexpr, COMM_SMS: tl.constexpr,
    ...
):
    pid = tl.program_id(0)

    if pid < COMPUTE_SMS:
        # 计算 worker：做 GEMM + signal
        for tile_id in range(pid, total_tiles, COMPUTE_SMS):
            acc = gemm_tile(A, B, tile_id, ...)
            tl.store(C + offset, acc, mask=mask, cache_modifier=".wt")
            xtile.tile_signal(locks, tile_id, sem="release", scope="gpu")
    else:
        # 通信 worker：wait + scatter
        comm_pid = pid - COMPUTE_SMS
        for tile_id in range(comm_pid, total_tiles, COMM_SMS):
            xtile.tile_wait(locks, tile_id, sem="acquire", scope="gpu")
            for remote_rank in range(world_size):
                if remote_rank != cur_rank:
                    xtile.tile_put(
                        C + offset, C + offset,
                        cur_rank, remote_rank,
                        heap_bases, mask=mask
                    )

# ⚠️ 三种方式使用【同一套原语】，只是组合方式不同
# ⚠️ 如果映射回当前仓库，auto-select 选择的是具体 pattern，而不是只有方式 B / C 两种
```

### 第 2 层：编译（标准 Triton，无特殊步骤）

```
与 Iris 完全相同：标准 Triton 编译流水线。
xtile.tile_remote_store 内部就是 `translate_ptr + tl.store`。
【修订】xtile.tile_signal 内部是 `tl.atomic_xchg`；`tile_wait` 才是 `tl.atomic_cas` 自旋等待。
编译器对所有操作完全可见。

唯一的差异在 host-side：
  xtile.init(backend="auto")
    → 检测硬件 → 选择 CUDA IPC 或 HIP IPC
    → 建立 symmetric heap
    → 返回 heap_bases tensor
```

### 第 3 层：通信底层（统一的纯 Triton 实现）

```python
@triton.jit
def tile_remote_store(pointer, value, from_rank, to_rank, heap_bases, mask=None):
    # 与 Iris 的 iris.store 实现完全相同
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    tl.store(translated_ptr, value, mask=mask)

@triton.jit
def tile_signal(locks, tile_id, sem="release", scope="gpu"):
    # 信号原语（借鉴 TileLink 的概念，用 Iris 的 atomic 实现）
    tl.atomic_xchg(locks + tile_id, 1, sem=sem, scope=scope)

@triton.jit
def tile_wait(locks, tile_id, sem="acquire", scope="gpu"):
    while tl.atomic_cas(locks + tile_id, 1, 0, sem=sem, scope=scope) == 0:
        pass

# 新增：tile 级 collective（Iris 没有的）
@triton.jit
def tile_allreduce(local_tile, heap_bases, rank, world_size, op="sum"):
    # Ring AllReduce 的 tile 粒度实现
    for step in range(world_size - 1):
        peer = (rank + step + 1) % world_size
        # 从 peer 读取对应 tile
        peer_tile = tile_remote_load(buffer, peer, rank, heap_bases, mask)
        # 本地 reduce
        if op == "sum":
            local_tile += peer_tile
        # 同步
        tile_signal(barriers, step, sem="release", scope="sys")
        tile_wait(barriers, step + world_size, sem="acquire", scope="sys")
    return local_tile

# ⚠️ 所有通信都是 tl.load / tl.store / tl.atomic_*
# ⚠️ 编译器完全可见
# ⚠️ 在 NVIDIA 和 AMD 上用完全相同的 Triton 代码
```

### 第 4 层：内存建立（跨平台 HAL）

```python
class XTileContext:
    def __init__(self, backend="auto"):
        # 自动检测硬件
        if backend == "auto":
            if torch.cuda.get_device_properties(0).name.startswith("NVIDIA"):
                self.backend = CUDABackend()
            else:
                self.backend = HIPBackend()

        self.heap = self.backend.create_symmetric_heap(size)

    # ====== CUDA Backend ======
    class CUDABackend:
        def create_symmetric_heap(self, size):
            ptr = cuda.cudaMalloc(size)
            handle = cuda.cudaIpcGetMemHandle(ptr)
            # ... 交换 handles（与 Iris 的 HIP 流程对称）
            peer_ptr = cuda.cudaIpcOpenMemHandle(peer_handle)
            return heap_bases

    # ====== HIP Backend ======
    class HIPBackend:
        def create_symmetric_heap(self, size):
            ptr = hip.hipMalloc(size)
            handle = hip.hipIpcGetMemHandle(ptr)
            # ... 交换 handles（与 Iris 完全相同）
            peer_ptr = hip.hipIpcOpenMemHandle(peer_handle)
            return heap_bases

    # ⚠️ Host-side 是唯一有平台差异的地方
    # ⚠️ Device-side 的 __translate / tl.store 在两个平台完全相同
```

### 第 5 层：硬件执行

```
NVIDIA H100:
  tl.store(translated_ptr) → SM → L2 miss → NVLink → 远端 GPU HBM

AMD MI300X:
  tl.store(translated_ptr) → CU → L2 miss → Infinity Fabric → 远端 GPU HBM

# 硬件行为完全相同：一个 store 指令写到远端地址
# GPU 的 memory controller 自动路由
# XTile 不关心底层是 NVLink 还是 Infinity Fabric
```

### 新增：Auto-Select 引擎

【修订】如果把这一节从“设想”落回当前仓库，真实签名更接近
`auto_select(op, M, N, K, world_size, hw_info=None, ctx=None)`；
而对本例 `M=8192, N=4608, K=36864, world_size=4`，当前 heuristics 会更接近 `ProducerConsumerPattern`。

```python
def auto_select(op, M, N, K, world_size, hw_info=None, ctx=None):
    """
    更接近当前仓库实现的 heuristic（省略了 bw_scale / total_tiles 等细节）：

    输入: M=8192, N=4608, K=36864, world_size=4
    """
    n_per_rank = N // world_size  # 1152

    if M < 256:
        return BulkSyncPattern(ctx)

    elif n_per_rank < 1024 and K > 12288:
        return FusedSequentialPattern(ctx)

    elif n_per_rank < 2048 and K > 6144:
        return ProducerConsumerPattern(ctx)

    elif N > 4096 and K > 8192:
        return WGSpecializedPattern(ctx)

    else:
        return BulkSyncPattern(ctx)  # 安全默认

    # 本例: n_per_rank=1152 < 2048, K=36864 > 6144
    # → 更接近 ProducerConsumerPattern ✓
```

---

## 关键差异总结表

【修订】下面的表把“当前仓库公开接口”和“概念/设想接口”区分开。

| 方面 | TileScale | Triton-distributed | Iris | XTile |
| --- | --- | --- | --- | --- |
| 用户写通信的方式 | `T.copy(..., dst=...)` / `putmem_*` | `wait/notify/symm_at` + `libshmem_device.*` | `iris.store` + for-loop | 当前以 pattern + primitives 组合为主 |
| 编译器看到什么 | NVSHMEM 调用，不透明 | distributed ops + NVSHMEM/PTX lowering | `tl.load` + `tl.store`，透明 | `tl.load` + `tl.store`，透明 |
| overlap 谁决定 | 用户写原语 + 编译器/硬件配合 | 用户写 low-level sync/comm | 用户手写 pattern | heuristics + pattern |
| 通信底层实现 | NVSHMEM API | wait/notify lowering + NVSHMEM device primitives | `__translate + tl.store` | `__translate + tl.store` |
| 外部依赖 | `torch.distributed` + NVSHMEM | `torch.distributed` + SHMEM backend | PyTorch + ROCm runtime | CUDA 或 HIP；现状仍需手工补齐 pattern ctx |
| 硬件平台 | NVIDIA | NVIDIA / HIP 路径也在扩展 | AMD | NVIDIA + AMD（代码目标） |
| tile collective | `T.allreduce`（经 NVSHMEM） | 尚无已发布高层 tile collective API | 无 | 代码已写出，但验证未完全稳定 |
| 【修订】当前文档最易误导处 | 把低层 put/copy 写成 `T.scatter()` | 把未来 `BlockChannel` 写成现状 | 把 heap 建立仍写成旧式 HIP IPC 主路径 | 把设想接口和当前仓库现状混写 |

---

## TVM 编译器和 Triton JIT 编译器的区别；AST 是什么

### TVM 编译器 vs Triton JIT 编译器

它们是**两个完全不同的编译器**，服务于不同的系统：

```
TileScale 用 TVM 编译器（基于 Apache TVM 项目）
Iris 和 TileLink 用 Triton JIT 编译器（基于 OpenAI Triton 项目）
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

| 方面     | TVM（TileScale 用）            | Triton JIT（Iris/TileLink 用） |
| -------- | ------------------------------ | ------------------------------ |
| 编译时机 | 可以提前编译（AOT）            | 第一次调用时编译（JIT）        |
| IR 基础  | TVM PrimFunc                   | Triton 自研 IR → LLVM IR       |
| 代码生成 | 生成 CUDA C 源码 → nvcc 编译   | 直接生成 LLVM IR → 直出机器码  |
| 硬件支持 | 很广（CPU/GPU/TPU/各种加速器） | GPU 为主（NVIDIA + AMD）       |
| 社区     | 学术界为主                     | 工业界广泛使用（PyTorch 生态） |

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

【修订】如果讨论的是论文里的高层 tile-centric 设计，这个解释成立；但当前 released Triton-distributed 仓库并没有把 `BlockChannel` 作为公开 API 暴露出来，因此下面这段更应理解为“概念化说明 / 未来高层接口设想”。

```python
# 用户写的代码
block_channel.tile_push_data(c, tile_id=tid, dst_rank=dst)

# TileLink 的 AST 变换把它改写为：
# （伪代码，展示 AST 层面的变换）
__barrier_ptr = block_channel._barriers[channel_mapping(tid)]
__dst_addr = shape_mapping(tid) + rank_mapping(tid) * stride
asm("nvshmem_putmem_nbi", __dst_addr, c_ptr, size, dst)
asm("st.release.gpu", __barrier_ptr, 1)
```

---

## 问题 4：TileScale 和 TileLink 如何调用 NVSHMEM

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

**通过 C 函数调用**——这一点是对的，但当前更贴近仓库现实的入口是 `putmem_*` / distributed `T.copy(..., dst=...)`，而不是已公开的 `T.scatter()`：

```c
// TileScale 编译器生成的 CUDA kernel
__global__ void kernel(...) {
    // ... GEMM 计算 ...

    // TileScale 编译器把 putmem/copy 分布式原语翻译成：
    nvshmemx_putmem_nbi_block(
        remote_ptr,
        local_ptr,
        tile_size,
        dst_rank
    );
}
```

### TileLink 如何使用 NVSHMEM

【修订】当前 released repo 不能简单概括成“高层 `tile_push_data` 被 AST 改写成 inline ASM”。更准确地说，它同时包含：
- `WaitOp` / `NotifyOp` 这类自定义 distributed op 的 LLVM/PTX lowering；
- `libshmem_device.*` 这类直接暴露给 Python 的 NVSHMEM device primitives。

下面保留一个“PTX/底层信号”示意，但不再把它说成当前公开的 `tile_push_data` 用户接口：

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
TileScale 调用 NVSHMEM：
  用户代码 → TVM 编译器 → 生成 CUDA C 代码 → 代码中有 nvshmem_put() 函数调用
  → nvcc 编译 → 链接 NVSHMEM 库 → 最终二进制文件

TileLink 调用 NVSHMEM：
  用户代码 → TileLink 编译器 → 在 Triton IR 中插入 inline ASM
  → ASM 的内容是 PTX 指令（这些 PTX 指令实现了 NVSHMEM 的功能）
  → LLVM 把 inline ASM 原样传递到最终 PTX → nvcc 编译为 cubin

关键区别：
  TileScale：编译器生成"调用别人的代码"的指令
  TileLink：编译器生成"自己就是那些代码"的指令

但对于上层优化器来说，两者都是"看不懂的东西"：
  TileScale 的优化器看到 nvshmem_put → "我不知道这个函数做什么，不敢动"
  TileLink 的优化器看到 inline ASM   → "我不知道这段汇编做什么，不敢动"
```

用人话说：

```
TileScale：请了一个翻译（NVSHMEM 库）来帮你跟外国人（远端 GPU）说话。
           你不懂外语，翻译说了什么你也不知道。

TileLink：你自己背了几句外语（PTX 内联汇编），直接跟外国人说。
          比 TileScale 少了一个中间人，但你说的那几句话，
          你的助手（Triton 优化器）也听不懂。

Iris：    你发现其实不需要说外语——你可以直接把东西放到对方桌上（tl.store）。
          你的助手（Triton 编译器）完全理解"放东西到桌上"这个动作，
          因为这跟"放东西到自己桌上"是同一个操作。
```

---

## 什么是编译器"透明/半透明/不透明"

### "编译器"指的是谁

在这个上下文中，"编译器"指的是**把你的 kernel 代码翻译成 GPU 机器码的那个工具**，具体就是 **Triton JIT 编译器**（对于 Iris、TileLink、XTile）或 **TVM 编译器**（对于 TileScale）。

编译器的核心工作除了翻译，还包括**优化**。优化的前提是：**编译器必须理解代码在做什么**。

### 不透明（TileScale 的 NVSHMEM 调用）

```python
# 编译器看到的 Triton IR（概念性展示）：
%acc = tt.dot %a, %b                    # ← 编译器理解：矩阵乘法
%result = arith.addf %acc, %bias        # ← 编译器理解：加法
call @nvshmem_float_put_nbi(%dst, %src, %n, %rank)  # ← 编译器：???
```

编译器看到 `nvshmem_float_put_nbi` 时，它知道的只有：

- 这是一个**外部函数调用**
- 它接受 4 个参数
- 它**可能**读写**任何**内存（保守假设）
- 它**可能**有**任何**副作用

编译器**不知道**的：

- 这个函数会不会修改 `%acc` 用到的内存？
- 这个函数需要多少时间？
- 这个函数和前面的 `tt.dot` 有没有数据依赖？

因此编译器**不敢做的优化**：

```
❌ 不敢把 nvshmem_put 移到 dot 前面（可能有依赖）
❌ 不敢删除 nvshmem_put 前面的 store（put 可能在读那块内存）
❌ 不敢把两个 nvshmem_put 合并成一个（不知道语义是否允许）
❌ 不敢在 nvshmem_put 和下一个 dot 之间插入预取指令（不知道 put 修改了什么）
```

**这就是"不透明"——编译器完全不知道这个操作内部在做什么，只能保守处理。**

### 半透明（TileLink 的 inline ASM）

```python
# 编译器看到的 LLVM IR（概念性展示）：
%acc = call <1024 x float> @triton_dot(...)     # ← 编译器理解
call void asm sideeffect                         # ← 编译器部分理解
    "st.global.relaxed.gpu.u32 [$0], $1;         #    知道它做了一个 store
     fence.proxy.async;",                         #    知道有一个 memory fence
    "l,r"(i64 %remote_ptr, i32 %data)            #    知道输入是 ptr 和 data
```

编译器看到 inline ASM 时，它比 `nvshmem_put` 多知道一些：

- **知道**：这是一个 store 操作（从 PTX 指令可以推断）
- **知道**：它写入的地址是 `%remote_ptr`（从操作数约束可以推断）
- **知道**：有一个 memory fence（`fence.proxy.async`）

编译器**仍然不知道**的：

- `%remote_ptr` 指向哪个 GPU 的内存？
- 这个 store 和本地的 load 有没有别名（alias）关系？
- 能不能把这个 ASM 和另一个 ASM 重排序？

因此：

```
✓ 编译器敢把不相关的本地计算移到 ASM 前面（因为知道 ASM 只写 remote_ptr）
❌ 仍然不敢跨 ASM 做寄存器分配优化（ASM 可能使用任何寄存器）
❌ 仍然不敢分析 ASM 内部指令的延迟来做调度优化
```

**这就是"半透明"——编译器能看到一些外部特征，但看不到内部逻辑。**

### 全透明（Iris / XTile 的 tl.store）

```python
# 编译器看到的 Triton IR（概念性展示）：
%acc = tt.dot %a, %b                              # 矩阵乘法
%from_base = tt.load %heap_bases_ptr               # 加载堆基址
%to_base = tt.load %heap_bases_ptr + %to_rank      # 加载远端基址
%offset = arith.subi %ptr_int, %from_base          # 计算偏移
%remote_ptr = arith.addi %to_base, %offset         # 计算远端地址
tt.store %remote_ptr, %acc, %mask                  # 存储到远端
```

编译器看到的**全部是标准 Triton 操作**：

- `tt.load`：加载（编译器完全理解语义和内存模型）
- `arith.subi` / `arith.addi`：整数算术（编译器完全理解）
- `tt.store`：存储（编译器完全理解语义和内存模型）

编译器**能做的优化**：

```
✓ 把 %from_base 的加载提升到循环外面（因为 heap_bases 不会变）
✓ 把偏移计算和地址计算做常量折叠
✓ 把 store 和后续计算重排序（如果分析出没有依赖）
✓ 在 store 前面插入预取指令
✓ 跨 store 边界做寄存器分配
✓ 把多个小 store 合并成一个大 store（向量化）
```

**这就是"全透明"——编译器把通信操作视为普通的 load/store，用同样的优化 pass 处理。**

### 直觉对比

```
不透明（TileScale）：
  你跟编译器说"请快递公司把包裹送到 GPU2"。
  编译器不知道快递公司怎么送，不敢假设任何事情。
  → 编译器在快递之前/之后插入"等待"，确保安全。

半透明（TileLink）：
  你跟编译器说"我自己开车把包裹送到 GPU2，走高速公路"。
  编译器知道你在开车（store 指令），知道走高速（memory fence），
  但不知道你具体走哪条路线，开多快。
  → 编译器可以安排你出发前的准备工作，但不能优化你的开车过程。

全透明（Iris/XTile）：
  你跟编译器说"把包裹放到架子上"。
  编译器完全理解"放东西到架子上"——它跟放到本地架子上是同一个动作。
  只不过这个架子恰好在另一个房间（GPU2），搬运工（硬件）自己会处理。
  → 编译器可以优化你放包裹的顺序、你在放包裹前后做的所有事情。
```

### 现实中的影响有多大？

**诚实地说：目前影响不大。** 这是因为：

1. 当前的 Triton 编译器虽然"能看到" Iris 的通信操作，但**还没有专门为跨 GPU store 做优化 pass**。它只是不会因为看不懂而插入不必要的 barrier。

2. 真正的性能差异更多来自于**overlap pattern 的选择**（是 Iris 的贡献），而非编译器优化。

3. 但从**架构正确性**来看，全透明是正确的方向——未来当 Triton 编译器变得更智能时，它可以自动发现 overlap 机会，前提是它能看到通信操作。Iris/XTile 的纯 Triton 路线为这个未来做好了准备。

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

#### TileLink 的 Distributed IR（TileLink 新增的）

```mlir
// 在 Triton GPU IR 之外，TileLink 新增了一套平行的 IR
// 专门处理分布式通信指令

module {
  // 这些是 TileLink 特有的操作
  dist.signal_barrier %barrier_ptr, %value
      {semantics = "release", scope = "gpu"}
  dist.wait_barrier %barrier_ptr, %expected
      {semantics = "acquire", scope = "gpu"}
  dist.remote_store %dst_ptr, %data, %dst_rank
      {nvshmem_op = "putmem_nbi"}

  // 这一层 IR 最终被翻译成 LLVM IR 中的 inline ASM 指令
  // 而标准的 Triton GPU IR 走正常的 LLVM 翻译路径
  // 两条路径最后在 LLVM IR 层合流
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
