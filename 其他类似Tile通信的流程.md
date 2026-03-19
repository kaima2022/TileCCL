# 固定任务：GEMM + AllScatter 全栈代码流对比

## 任务定义

4 个 GPU，每个 GPU 持有 A[M,K] 和 B[K,N]。
每个 GPU 先算 C_local = A × B（本地 GEMM），然后把 C_local scatter 到所有其他 GPU，
最终每个 GPU 都拥有完整的 C_global[M, N×4]。

M=8192, N=4608, K=36864, 4 GPUs.

---

## 系统一：TileScale

### 第 1 层：用户代码（TileLang DSL）

```python
import tilelang as T

@T.jit(target="cuda")
def gemm_allscatter(
    A, B,
    block_M: int = 128, block_N: int = 128, block_K: int = 64,
):
    M, N, K = T.const('M, N, K')
    A: T.Tensor[[M, K], T.float16]
    B: T.Tensor[[K, N], T.float16]
    C = T.empty([M, N * T.world_size()], T.float16)  # 全局输出

    # --- scale=device 级别的 kernel ---
    with T.Kernel(
        T.ceildiv(N, block_N), T.ceildiv(M, block_M),
        threads=128,
        scale="device"      # <--- TileScale 特有：指定运行尺度
    ) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local  = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        # --- 通信：一行搞定 scatter ---
        T.scatter(C_local, C, dst="all")  # <--- TileScale 的通信原语
        #   等价于：把 C_local tile 写到 C_global 的对应位置，
        #   C_global 在所有 rank 的 symmetric heap 上

        T.barrier(scope="device")  # <--- 跨 GPU 同步
```

**关键点**：用户只写 tile 逻辑 + `T.scatter()` + `T.barrier()`，overlap 由编译器自动处理。

### 第 2 层：TVM 编译器

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
  ├─ T.scatter →  此处编译到 NVSHMEM 调用（见下层）
  └─ T.barrier →  此处编译到 NVSHMEM barrier
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

### 第 3 层：通信底层（NVSHMEM）

```c
// 编译器将 T.scatter() 展开为类似这样的 NVSHMEM 设备端调用：
__device__ void scatter_tile(float* C_local, float* C_global, int rank, int world_size) {
    // NVSHMEM 设备端 put
    for (int dst = 0; dst < world_size; dst++) {
        if (dst != rank) {
            nvshmem_float_put_nbi(             // <--- NVSHMEM API
                C_global + dst_offset,          // 远端地址
                C_local,                        // 本地数据
                tile_size,                      // 大小
                dst                             // 目标 rank
            );
        }
    }
    nvshmem_barrier_all();  // <--- NVSHMEM barrier
}
// ⚠️ 这段代码对 Triton/TVM 编译器是【不透明的二进制字节码】
// 编译器无法对 nvshmem_float_put_nbi 做指令重排序或 fusion
```

### 第 4 层：内存建立

```python
# TileScale 的 host-side 启动脚本
# 依赖 NVSHMEM 的分布式初始化
$ mpirun -np 4 python run_tilescale.py

# 内部流程：
# 1. NVSHMEM 初始化（nvshmem_init）
# 2. NVSHMEM 分配 symmetric heap（nvshmem_malloc）
# 3. 所有 rank 自动获得对等访问
# ⚠️ 必须有 MPI + NVSHMEM 运行时环境
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

```python
import triton
import triton.language as tl
from triton_dist import BlockChannel  # <--- TileLink 特有

@triton.jit
def gemm_allscatter_tilelink(
    A, B, C_global,
    M, N, K,
    block_channel: BlockChannel,  # <--- 封装分布式元数据
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # ... tile 坐标计算（与 Iris 类似，省略）

    # ====== GEMM 计算部分（标准 Triton）======
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    # ====== TileLink 通信原语 ======
    tile_id = pid
    # 通知所有消费者：这个 tile 已经计算完成
    block_channel.producer_tile_notify(tile_id)  # <--- TileLink 原语

    # 将 tile 数据推送到所有远端
    for dst in range(block_channel.world_size):
        if dst != block_channel.rank:
            block_channel.tile_push_data(        # <--- TileLink 原语
                c,                                # 本地 tile 数据
                tile_id=tile_id,
                dst_rank=dst,
            )

    # 等待所有远端的 tile 都到达
    block_channel.consumer_tile_wait(tile_id)    # <--- TileLink 原语
```

**关键点**：用户写的是 `producer_tile_notify` / `tile_push_data` / `consumer_tile_wait` 这些 tile 级原语，比 NCCL 细粒度得多，但比 Iris 的 `tl.store` 抽象级更高。

### 第 2 层：TileLink 编译器

```
用户 Python (Triton + TileLink primitives)
    │
    ▼
Python AST 解析
    │ BlockChannel 参数被分解为：
    │   - shape mapping:  tile_id → tensor[row_start:row_end, col_start:col_end]
    │   - rank mapping:   tile_id → dst_rank
    │   - channel mapping: tile_id → barrier_channel_id
    │
    ▼
Triton IR 生成
    │ TileLink 原语 → Triton ElementwiseInlineAsmOp
    │   producer_tile_notify → 内联 PTX 指令（signal barrier）
    │   tile_push_data       → 内联 PTX 指令（NVSHMEM put）
    │   consumer_tile_wait   → 内联 PTX 指令（wait barrier）
    │
    ▼
分叉为两条 IR：
  ├─ Triton GPU IR（标准 Triton 计算部分）
  └─ Distributed IR（TileLink 新增，处理通信指令）
    │
    ▼
Distributed IR → LLVM IR
    │ 将 ElementwiseInlineAsmOp 翻译为 NVSHMEM 的底层调用
    │ 插入 memory fence 指令保证 consistency
    │
    ▼
LLVM IR → PTX → cubin（NVIDIA GPU 可执行文件）
```

### 第 3 层：通信底层（仍是 NVSHMEM）

```c
// TileLink 编译器将 tile_push_data 展开为：
// （这些是编译器生成的 PTX 内联汇编，不是用户写的）

// producer_tile_notify 展开为：
asm volatile("fence.proxy.async;");     // memory fence
asm volatile("st.relaxed.gpu [%0], 1;"  // 写 signal flag
             : : "l"(barrier_ptr));

// tile_push_data 展开为：
asm volatile(
    "nvshmem_putmem_nbi %0, %1, %2, %3;"  // NVSHMEM 非阻塞 put
    : : "r"(dst_ptr), "r"(src_ptr),
        "r"(size), "r"(dst_rank)
);

// consumer_tile_wait 展开为：
asm volatile(                              // 自旋等待 signal
    "spin: ld.relaxed.gpu %0, [%1];"
    "      setp.ne.u32 p, %0, 1;"
    "      @p bra spin;"
    : "=r"(val) : "l"(barrier_ptr)
);

// ⚠️ 比 TileScale 好一点：编译器【生成】了这些指令（不是链接不透明库）
//    但对 Triton 的后续优化 pass 来说，
//    ElementwiseInlineAsmOp 仍然是不可分析的黑盒
```

### 第 4 层：内存建立

```python
# TileLink 的 host-side 启动
$ torchrun --nproc_per_node=4 run_tilelink.py

# 内部流程：
# 1. NVSHMEM 初始化（nvshmem_init_thread）
# 2. NVSHMEM 分配 symmetric heap
# 3. TileLink runtime 在 NVSHMEM 堆上创建 BlockChannel 对象
#    - 分配 barrier 数组（每个 tile_id 一个 barrier slot）
#    - 分配 data buffer（每个 rank 的输出 tile 空间）
# ⚠️ 同样必须有 NVSHMEM + MPI 运行时
```

### 第 5 层：硬件执行

```
与 TileScale 相同——底层都是 NVSHMEM 选择 NVLink 直传路径。
但 TileLink 的 tile 粒度 barrier 使得 GPU 可以在第一个 tile
完成后立即开始通信，不必等整个 GEMM 结束。
```

---

## 系统三：Iris

### 第 1 层：用户代码（纯 Triton，选择 Fused Sequential 模式）

```python
import triton
import triton.language as tl
import iris  # <--- 约 370 行 Python + Triton 代码

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

### 第 4 层：内存建立（HIP IPC）

```python
class iris:
    def __init__(self, heap_size):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 步骤 1：本地分配
        self.local_ptr = hip.hipMalloc(heap_size)

        # 步骤 2：导出 IPC handle
        local_handle = hip.hipIpcGetMemHandle(self.local_ptr)

        # 步骤 3：通过 PyTorch distributed 交换 handles
        all_handles = [None] * world_size
        dist.all_gather_object(all_handles, local_handle)

        # 步骤 4：打开远端 handles
        self.peer_ptrs = []
        for i, handle in enumerate(all_handles):
            if i == rank:
                self.peer_ptrs.append(self.local_ptr)
            else:
                remote_ptr = hip.hipIpcOpenMemHandle(handle)
                self.peer_ptrs.append(remote_ptr)

        # 步骤 5：构建 heap_bases tensor（64 字节，L1 常驻）
        self.heap_bases = torch.tensor(
            [ptr for ptr in self.peer_ptrs],
            dtype=torch.uint64, device="cuda"
        )
        # heap_bases = [GPU0_base, GPU1_base, GPU2_base, GPU3_base]
        # 之后 device-side __translate 只需读这个 64 字节的 tensor

    # ⚠️ 依赖：只有 PyTorch + HIP runtime
    # ⚠️ 没有 NVSHMEM，没有 MPI（PyTorch distributed 处理进程通信）
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

### 第 1 层：用户代码（Triton + XTile auto-select）

```python
import triton
import triton.language as tl
import xtile

# ====== 方式 A：一键自动（最简单）======
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
    # 内部 auto-select 判断：
    #   N=4608, K=36864, 4 GPUs → N/world_size=1152 < 2048, K > 8192
    #   → 选择 "fused_sequential" 模式

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
# ⚠️ auto-select 在方式 A 中自动选择方式 B 或 C
```

### 第 2 层：编译（标准 Triton，无特殊步骤）

```
与 Iris 完全相同：标准 Triton 编译流水线。
xtile.tile_remote_store 内部就是 __translate + tl.store。
xtile.tile_signal 内部就是 tl.atomic_cas。
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
    tl.atomic_cas(locks + tile_id, 0, 1, sem=sem, scope=scope)

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

```python
def auto_select(M, N, K, world_size, hw_info):
    """
    基于 Iris 论文 Section 5 实验数据的 heuristic：

    输入: M=8192, N=4608, K=36864, world_size=4
    """
    n_per_gpu = N // world_size  # 1152
    comm_tiles = M * N * world_size  # 通信量
    compute_flops = 2 * M * N * K   # 计算量
    ratio = comm_tiles / compute_flops  # ~0.00003 → 通信远小于计算

    if n_per_gpu < 2048 and K > 16384:
        # 通信小、计算大 → Fused Sequential 最优
        # Iris 论文验证：8192×4608×36864, 4 GPU → 1.8× speedup
        return FusedSequentialPattern(ctx)

    elif n_per_gpu < 4096 and K > 8192:
        # 可隐藏通信 → Producer-Consumer
        # 需要分配 CU/SM：256 compute + 48 comm (MI300X)
        return ProducerConsumerPattern(ctx,
            compute_sms=hw_info.total_sms - 48,
            comm_sms=48)

    elif N > 4096:
        # 通信量大 → Workgroup Specialization
        return WGSpecializedPattern(ctx,
            compute_sms=hw_info.total_sms * 0.8,
            comm_sms=hw_info.total_sms * 0.2)

    else:
        return BulkSyncPattern(ctx)  # 安全默认

    # 本例: n_per_gpu=1152 < 2048, K=36864 > 16384
    # → 选择 FusedSequentialPattern ✓
```

---

## 关键差异总结表

```
┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│                  │  TileScale   │  TileLink    │    Iris      │   XTile      │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 用户写通信的方式  │ T.scatter()  │ tile_push    │ iris.store   │ xtile.store  │
│                  │ 一行 DSL     │ _data()      │ + for loop   │ 或一键 auto  │
│                  │              │ 信号原语      │ 手动模式     │              │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 编译器看到什么    │ NVSHMEM      │ inline ASM   │ tl.load +    │ tl.load +    │
│                  │ 不透明调用    │ 半透明       │ tl.store     │ tl.store     │
│                  │              │              │ 全透明       │ 全透明       │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ overlap 谁决定   │ 编译器自动    │ 编译器映射   │ 用户手写     │ auto-select  │
│                  │              │              │ pattern      │ + 用户覆盖   │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 通信底层实现     │ NVSHMEM API  │ NVSHMEM PTX  │ __translate  │ __translate  │
│                  │              │              │ + tl.store   │ + tl.store   │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 外部依赖         │ NVSHMEM+MPI  │ NVSHMEM+MPI  │ HIP only     │ CUDA or HIP  │
│                  │              │              │              │ (auto)       │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 硬件平台         │ NVIDIA       │ NVIDIA       │ AMD          │ NVIDIA+AMD   │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ tile collective  │ T.allreduce  │ 无内建       │ 无           │ tile_allreduce│
│ (纯 Triton)     │ (→NVSHMEM)   │              │              │ (纯 Triton)  │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 通信代码行数     │ 1-2 行       │ 3-5 行       │ 5-8 行       │ 1 行(auto)   │
│ (在 kernel 内)  │              │              │              │ 或 5-8(手动) │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```