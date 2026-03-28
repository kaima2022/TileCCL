# TNCC (Tile Native Collective Communication) 远端访存底层机制报告

日期：`2026-03-27`

## 目的

回答一个经常被混淆的问题：

- TNCC 既然不调用 `NVSHMEM`，那 `tl.load` / `tl.store` 访问远端 GPU 内存时，底层到底走什么路径。
- 它和 `NCCL`、`NVSHMEM`、`GPUDirect RDMA (GDR)` 的关系分别是什么。
- 为什么这种路径会对 `SM` 占用敏感。

## 结论摘要

先给结论，再展开。

- TNCC 当前远端访存路径不是 `NVSHMEM`，也不是 `NCCL`。
- TNCC 的核心机制是：先把远端 GPU heap 映射成“当前 rank 可解引用的 GPU 虚拟地址”，再由 Triton 的 `tl.load` / `tl.store` 对该地址发普通 global memory 读写。
- 单进程多卡时，走的是 `CUDA peer access + UVA` 的直接 peer pointer。
- 多进程时，默认走的是 `CUDA IPC handle` 映射后的远端指针；TNCC 当前默认验证的 transport 是 `ctypes_ipc`。
- 物理链路取决于机器拓扑：
  - 有 `NVLink`，就走 `NVLink P2P`
  - 没有 `NVLink` 但支持 peer access，就走 `PCIe P2P`
- 它不是通常所说的 `GPUDirect RDMA`。
- 如果把 `GPUDirect` 当成大类，这条路径更接近 `GPUDirect P2P / peer access`，不是 NIC 参与的 `RDMA` 路径。
- `tl.load` / `tl.store` 这条路径是由执行 kernel 的线程在 `SM` 上发起的普通访存指令，不是独立 DMA engine 替你推进；因此它天然会受 `SM` 调度和协议推进影响。

## `tl.load` / `tl.store` 到底会 lower 成什么

这是本报告最关键的一步。为了避免只停留在概念层，本机直接编译了两个最小 Triton kernel：

1. 一个普通 `tl.load` / `tl.store` kernel
2. 一个把 TNCC 的 `translate_ptr` 内联进去的 kernel

环境：

- `Triton 3.2.0`
- `PyTorch 2.6.0+cu124`
- `target sm_90a`

### 证据 1：普通 `tl.load` / `tl.store`

在 Triton TTIR / TTGIR 中，能直接看到：

```text
tt.load ... cacheModifier = cg
tt.store ... cacheModifier = wt
```

继续 lower 到 PTX 后，直接变成：

```text
ld.global.cg.v2.b32
st.global.wt.v2.b32
```

这说明 `tl.load` / `tl.store` 在 NVIDIA 后端上，本质就是 global memory load/store 指令，不是什么隐藏的 SHMEM runtime 调用。

### 证据 2：把 `translate_ptr` 一起编进去

把 TNCC 的 `translate_ptr` 内联进最小 kernel 后，TTIR 中可以直接看到如下链路：

```text
tt.load heap_bases[from_rank]
tt.load heap_bases[to_rank]
tt.ptr_to_int local_ptr
subi
tt.int_to_ptr to_base
tt.addptr
tt.bitcast
tt.load translated_ptr
tt.store out_ptr
```

继续 lower 到 PTX 后，关键指令序列是：

```text
ld.global.b64    ; load heap_bases[from_rank]
ld.global.b64    ; load heap_bases[to_rank]
sub.s64          ; ptr - from_base
add.s64          ; to_base + offset
ld.global.cg.b32 ; load from translated remote ptr
st.global.wt.b32 ; store result
```

这条证据非常关键，因为它证明了：

- `translate_ptr` 没有神秘 runtime
- 它只是基址加载 + 指针算术
- 真正的数据访存依然是普通 `ld.global` / `st.global`

换句话说，TNCC 的远端访存是“对远端映射地址发普通 global 访存”，不是“调用 NVSHMEM 的 put/get”。

## 一句话答案

TNCC 这里走的不是 `NVSHMEM`，也不是 `GDR`；更准确地说，它走的是：

`peer access / CUDA IPC + UVA/mapped remote GPU virtual address + Triton global load/store`

## TNCC 源码中的真实路径

### 1. collectives 不依赖 NCCL / RCCL

TNCC 的 collective 原语在模块说明里已经写明：

- 位置：`tncc/primitives/collectives.py`
- 结论：纯 `@triton.jit`，无 `NCCL / RCCL` 依赖

关键代码：

- [`tncc/primitives/collectives.py`](../tncc/primitives/collectives.py)

对应说明：

- `Tile-level collective operations implemented entirely in Triton`
- `no NCCL / RCCL dependency`
- `All remote memory access goes through translate_ptr`

这意味着 collectives 的数据面不是通过外部通信库发起的，而是靠 Triton kernel 自己访问远端地址。

### 2. `translate_ptr` 只做地址翻译

远端访问的核心在 [`tncc/memory/translation.py`](../tncc/memory/translation.py)。

`translate_ptr(ptr, from_rank, to_rank, heap_bases)` 的逻辑非常直接：

1. 读取 `from_rank` 的 heap base
2. 读取 `to_rank` 的 heap base
3. 计算 `ptr` 在源 heap 内的 byte offset
4. 用目标 heap base 加上这个 offset
5. 生成 `remote_ptr`

也就是：

```text
offset     = ptr - heap_bases[from_rank]
remote_ptr = heap_bases[to_rank] + offset
```

关键点不在这段算术本身，而在 `heap_bases[to_rank]` 的含义：

- 它不是“远端原始地址”
- 它是“该远端 heap 在当前 rank 地址空间中的可访问映射基址”

源码里已经写得很明确：

- `heap_bases` 保存的是 each rank's heap base address `as mapped into this rank's address space`

### 3. `remote_load` / `remote_store` 本质上只是 `tl.load` / `tl.store`

同文件中的 `remote_load` / `remote_store` 只有两步：

1. `translate_ptr(...)`
2. 对翻译后的指针执行 `tl.load` / `tl.store`

也就是说，TNCC 并没有在这里调用一个“远端通信 runtime API”。它做的是：

- 先把地址翻译成远端映射地址
- 然后像访问本地 global memory 一样访问它

这也是为什么 TNCC 的远端访问能保持在 Triton 编译器可见的 IR 内，而不是掉进一个外部黑盒调用。

## 映射地址是怎么来的

这里要分单进程和多进程。

### 单进程多卡：`CUDA peer access`

单进程路径在 [`tncc/memory/symmetric_heap.py`](../tncc/memory/symmetric_heap.py) 的 `create_all()`。

它做了三件事：

1. 遍历 GPU 对，调用 `enable_peer_access`
2. 在每张 GPU 上分配 heap buffer
3. 把各 GPU 的 `data_ptr()` 直接塞进 `heap_bases`

这条路径的源码语义非常明确：

- `Peer access is enabled between all GPU pairs`
- `heap_bases contains direct device pointers (no IPC)`

因此单进程时，TNCC 走的是：

- `cudaDeviceEnablePeerAccess` 风格的 peer memory access
- 同一 64-bit UVA 下的 peer pointer 直接解引用

这和 NVIDIA CUDA Programming Guide 里的 peer access 定义一致：设备之间启用 peer access 后，kernel 可以直接解引用另一张卡的内存地址。

### 多进程：默认 `CUDA IPC`

多进程路径在 [`tncc/memory/symmetric_heap.py`](../tncc/memory/symmetric_heap.py) 的 `_setup_multiprocess()`。

当前默认策略只有一个：

- `ctypes_ipc`

对应注释写得很清楚：

- `Raw ctypes IPC — cudaIpcGetMemHandle / cudaIpcOpenMemHandle`

而 allocator 的 export/import 进一步坐实了这条链路：

- [`tncc/memory/allocators.py`](../tncc/memory/allocators.py)
- `export_peer_memory(..., transport="ctypes_ipc")`
  - 调 `backend.get_ipc_handle(self.base_ptr)`
- `import_peer_memory(...)`
  - 调 `backend.open_ipc_handle(export.payload)`
  - 返回 `mapped_ptr`

CUDA backend 自己也写得很直白：

- `get_ipc_handle`: 返回 CUDA IPC handle
- `open_ipc_handle`: 打开 remote CUDA IPC handle，并返回 local device pointer

因此多进程时，TNCC 的地址建立流程是：

1. 每个 rank 导出本地 heap 的 CUDA IPC handle
2. 各 rank 交换 handle
3. 每个 rank 用 `cudaIpcOpenMemHandle` 把其他 rank 的 heap 映射进自己进程
4. 把映射后的 `mapped_ptr` 写入本 rank 的 `heap_bases`
5. kernel 内再由 `translate_ptr` 做基址平移

这里的关键语义不是“复制数据”，而是“导入一个可被当前进程 GPU-side 直接解引用的远端映射地址”。

### TNCC 自己如何定义这些访问语义

allocator 里已经把这几类 transport 的访问语义分类好了：

- `peer_access` -> `peer_direct`
- `ctypes_ipc` / `pytorch_ipc` -> `mapped_remote`
- `peer_access_pointer_exchange` -> `remote_pointer`

这和上面的结论完全一致：

- 单进程是 direct peer access
- 多进程默认是 imported / mapped remote memory



## 它物理上到底走哪条链路

逻辑地址路径和物理互连是两层事情。

逻辑上，TNCC kernel 只知道自己在访问一个 peer-accessible / IPC-mapped GPU 地址。

物理上，真正经过哪条链路，取决于 GPU 拓扑：

- GPU 之间如果有 `NVLink`，P2P 流量会走 `NVLink`
- 如果没有 `NVLink` 但支持 peer access，则走 `PCIe P2P`

本机实测：

```text
$ nvidia-smi topo -m
GPU0  GPU1
GPU0   X    NV12
GPU1  NV12   X
```

这说明当前这台机器上，`GPU0 <-> GPU1` 之间是 `NV12`，也就是 bonded NVLink 连接；因此 TNCC 的 peer 远端访存物理上会走 NVLink，而不是普通 PCIe 路径。

## 它和 GDR 到底是什么关系

这是最容易混淆的点。

### 精确结论

- 如果你说的是 `GPUDirect RDMA`
  - 那么答案是：不是
- 如果你把 `GPUDirect` 当作大类
  - 那么它更接近 `GPUDirect P2P / peer access`

### 为什么不是 GPUDirect RDMA

NVIDIA 官方对 `GPUDirect RDMA` 的定义重点是：

- GPU 与第三方 peer device 直接交换数据
- 典型对象是 `NIC / RDMA adapter`
- 依赖 `nvidia-peermem` 这类内核侧支持

而 TNCC 当前讨论的路径是：

- 同机 `GPU <-> GPU`
- `peer access` / `CUDA IPC`
- 由执行中的 Triton kernel 直接解引用远端 GPU 地址

这里没有出现：

- RDMA NIC
- verbs / IB / RoCE
- `nvidia-peermem`
- 第三方 PCIe device 直接 DMA 进 GPU 内存

所以把 TNCC 当前这条路径叫 `GDR` 并不准确。

## 它和 NVSHMEM、NCCL 的本质区别

### 相比 NVSHMEM

NVSHMEM 的典型模型是：

- 先建立 SHMEM runtime 与 symmetric heap
- 远端访问通过 `nvshmem_*` / `nvshmemx_*` 设备 API 表达
- 编译器看到的是外部库语义，而不是普通 Triton `load/store`

TNCC 当前不是这条路。

从代码库看，当前 `tncc/` 实现路径里没有 `NVSHMEM` runtime 接入；collective 与 remote access 的关键路径都建立在：

- symmetric heap
- peer/IPC 映射
- `translate_ptr`
- Triton `load/store/atomic`

之上。

因此，TNCC 的优势是：

- 通信路径仍然处于 Triton 编译器可见范围
- 指针翻译、同步、数据搬运可以在同一个 kernel/IR 世界里被联合优化

### 相比 NCCL

NCCL 的接口语义是 collective library：

- 用户提交 `all_reduce / broadcast / all_gather`
- NCCL 运行时/内核负责选算法、分块、调度和推进

TNCC 不是 library call 驱动的数据面；它更接近：

- 用户或上层 primitive 自己组织协议
- kernel 内显式 remote load/store/atomic
- 协议推进与计算/同步逻辑同处于 Triton kernel 中

因此它的灵活性更高，但也更暴露出协议设计和 kernel 调度细节。

## 为什么这条路径会对 SM 占用敏感

这是理解 benchmark 抖动的关键。

### 1. 这不是“独立 DMA engine 在后台搬数据”

TNCC 当前远端访存的动作，本质是：

- warp 在 `SM` 上执行
- 发出 `ld.global` / `st.global`
- 访问目标是远端 peer / IPC 映射地址

因此：

- 没有独立的“通信线程外包器”帮你推进协议
- 也不是 copy engine 自动替你完成整段 collective

远端读写本身以及协议推进，都需要 kernel 的 warp 被正常调度执行。

### 2. 协议推进和链路带宽是两回事

即使底层链路是 `NVLink`，也不代表只要没人“抢 NVLink 带宽”，通信就一定不受影响。

只要有下面任一情况，collective 仍然会变慢：

- 本地 GPU 的 `SM` 长期被别的 kernel 占满
- peer GPU 的协议响应无法及时执行
- 轮询、发布、确认、epoch 递进这些控制面操作得不到及时调度

也就是说：

- 链路空闲，不等于协议会自己推进
- 对 TNCC 这种 kernel-driven remote access 模型来说，`SM progress` 和 `interconnect bandwidth` 同样重要

### 3. 这和之前观测到的噪声现象是对得上的

此前 benchmark 噪声分析里已经看到：

- 纯大块 `P2P read/write` steady-state 带宽受影响较小
- 但含有 staged publish / consume / ack / wait 的 collective 明显更容易在共享 GPU 环境下变慢

这和本报告的底层机制完全一致：

- 大块单边 remote access 更像“持续流式访存”
- collective 则包含更强的协议推进依赖

因此 TNCC 当前路径对 `SM` 干扰敏感，不是偶然现象，而是机制上可以解释的结果。

## `cache_modifier` 改变了什么，没有改变什么

TNCC 代码里常见：

- 远端读用 `.cg`
- 远端写用 `.wt`

这只改变 cache policy，不改变 transport 本身。

也就是说：

- `.cg` 不会把“PCIe 路径”变成“NVLink 路径”
- `.wt` 也不会把“普通 global store”变成“NVSHMEM put”

它们影响的是：

- 是否绕过某级 cache
- 是否尽量减少本地 cache pollution

但不决定“到底走哪条互连”。

## 当前机器上的精确表述

结合源码与本机拓扑，当前这台机器上更准确的完整表述应为：

> TNCC 当前的远端访存机制，是在 host 侧先通过 CUDA peer access 或 CUDA IPC 把对端 symmetric heap 建立为本 rank 可访问的 GPU 虚拟地址，再在 Triton kernel 内通过 `translate_ptr` 做基址平移，并由普通 `tl.load` / `tl.store` lower 成 `ld.global` / `st.global` 去访问该远端地址。当前机器 `GPU0 <-> GPU1 = NV12`，所以实际物理链路是 NVLink P2P，而不是 GPUDirect RDMA。

## 容易说错的几句话

下面这些说法不够准确：

- “TNCC 的 `tl.load/store` 底层是 NVSHMEM”
- “TNCC 这里走的是 GDR”
- “既然只是通信，没有人抢 NVLink 就不该受影响”
- “只要地址能翻译出来，通信就会自动推进”

更准确的说法是：

- “TNCC 走的是 peer access / CUDA IPC 映射后的远端 GPU 地址直接解引用”
- “如果说 GDR，必须区分 GPUDirect P2P 和 GPUDirect RDMA；这里不是 RDMA”
- “TNCC 通信是 kernel-driven 的 global memory remote access，既依赖链路，也依赖 SM progress”

## 对工程判断的意义

这份机制分析直接决定后续优化方向。

如果要继续把 TNCC 做成可维护、可解释、可追 benchmark 的工业级实现，重点不应只盯：

- 算法名义带宽
- interconnect 峰值

更要盯：

- 协议推进是否过度依赖 busy-poll
- control plane 是否造成过多短事务 remote access
- remote load/store 是否具备足够好的 batching / staging / coalescing
- 是否能减少必须由双方 `SM` 持续在线推进的环节

## 参考证据

### TNCC 源码

- [`tncc/primitives/collectives.py`](../tncc/primitives/collectives.py)
- [`tncc/memory/translation.py`](../tncc/memory/translation.py)
- [`tncc/memory/symmetric_heap.py`](../tncc/memory/symmetric_heap.py)
- [`tncc/memory/allocators.py`](../tncc/memory/allocators.py)
- [`tncc/backends/cuda.py`](../tncc/backends/cuda.py)

### 官方资料

- NVIDIA CUDA C++ Programming Guide, Peer-to-Peer Memory Access
  - https://docs.nvidia.com/cuda/archive/12.9.0/pdf/CUDA_C_Programming_Guide.pdf
- NVIDIA GPUDirect RDMA documentation
  - https://docs.nvidia.com/cuda/gpudirect-rdma/

### 本机实测

- `nvidia-smi topo -m`
- Triton `CompiledKernel.asm`
  - `ttir`
  - `ttgir`
  - `llir`
  - `ptx`
