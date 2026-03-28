# TNCC (Tile Native Collective Communication) 矩阵乘计算路径分析报告

## 1. 结论摘要

当前 TNCC 的矩阵乘实现不是“自研一个独立的低层 GEMM runtime”，而是建立在 Triton tile kernel 之上的一套分层方案：

1. 最底层的实际乘法由 Triton `tl.dot` 完成。
2. TNCC 自己控制的是 tile 切分、persistent 调度、K 维循环组织、远端地址翻译，以及计算和通信的融合方式。
3. 在高层 API 中，不同操作的“计算路径成熟度”并不完全一样：
   - `tncc.kernels.gemm.gemm(...)` 是单机/本地的 Triton GEMM。
   - `tncc.ops.gemm_allscatter(...)` 是当前最原生的 TNCC 融合路径，GEMM 与 scatter 都在 pattern/kernel 体系里完成。
   - `tncc.ops.gemm_allgather(...)` 与 `tncc.ops.gemm_reducescatter(...)` 当前实现更保守，先做本地 GEMM，再调用高层 collective 计划完成后续通信。

因此，如果问题是“TNCC 现在如何做矩阵乘”，最准确的回答是：

- 计算内核层面：用 Triton 分块加载 + `tl.dot(a, b, acc)` 做 tile GEMM。
- 系统层面：用 pattern 把 tile 级矩阵乘与远端 scatter/gather/reduce-scatter 组织起来。
- 现阶段最完整体现 TNCC 设计目标的路径是 `gemm_allscatter`，不是 `gemm_allgather` 或 `gemm_reducescatter`。

## 2. 总体架构

README 把体系分成了几层：用户 API、Pattern Library、Core Primitives、Synchronization、Memory Management、HAL，见 [README.md](../README.md)。其中与矩阵乘最相关的是：

- 用户侧入口：`tncc.ops.*`
- pattern 层：`BulkSyncPattern`、`FusedSequentialPattern`、`ProducerConsumerPattern`、`WGSpecializedPattern`
- 计算原语：`tile_dot`
- 远端访存基础：`translate_ptr`
- 同步原语：`tile_signal` / `tile_wait`

这说明 TNCC 的设计目标不是“只做 GEMM”，而是把“计算 tile”和“通信 tile”放进同一套编译器可见的 Triton 代码里。

## 3. 最底层实际怎么做矩阵乘

### 3.1 独立 GEMM 内核

单独的 GEMM 实现在 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L52)。

这个 kernel 的核心结构是：

1. 把输出矩阵 `C(M, N)` 切成 `BLOCK_SIZE_M x BLOCK_SIZE_N` 的 tile。
2. 每个 Triton program 不是只算一个 tile，而是通过 persistent loop 轮转处理多个 tile，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L115)。
3. 每个输出 tile 对应一次 K 维分块累加：
   - 从 `A` 取一个 `BLOCK_M x BLOCK_K` tile
   - 从 `B` 取一个 `BLOCK_K x BLOCK_N` tile
   - 调用 `acc = tl.dot(a, b, acc)`，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L146)
4. K 维被拆成“整块主循环 + 尾块 remainder”，主循环尽量不带 mask，尾块单独处理，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L141)。
5. 最后只在写回 `C` 时做 M/N 边界 mask，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L162)。

### 3.2 关键实现特征

这个 GEMM kernel 有几个非常明确的工程特征：

- `acc` 使用 `fp32` 累加，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L138)。
- 通过 `rm % M`、`rn % N` 这种 modular wrapping 消掉大部分边界判断，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L125)。
- 通过 `tl.max_contiguous` 和 `tl.multiple_of` 给编译器向量化提示，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L129)。
- 采用 `GROUP_SIZE_M` 做 tile swizzle，改善 L2 局部性，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L117)。
- 使用 persistent grid，grid 大小跟 SM 数有关，而不是简单等于 tile 数量，见 [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L174)。

### 3.3 是否手写了 MMA / Tensor Core 指令

从源码层面看，没有。

TNCC 源码里真正执行矩阵乘的是 `tl.dot`，见：

- [tncc/kernels/gemm.py](../tncc/kernels/gemm.py#L149)
- [tncc/primitives/compute.py](../tncc/primitives/compute.py#L11)

也就是说，TNCC 没有自己手写 `mma.sync` / `wmma` / `mfma` 级别指令序列。更底层是否映射到 Tensor Core 或 AMD 对应矩阵指令，取决于 Triton 后端和目标硬件。这一点应理解为：

- TNCC 控制算法结构和数据/通信流；
- Triton 控制更底层的机器指令选择。

## 4. `tile_dot` 在整体中的角色

`tncc.primitives.compute.tile_dot()` 只是 `tl.dot` 的薄包装，见 [tncc/primitives/compute.py](../tncc/primitives/compute.py#L11)。

这意味着：

1. TNCC 把 tile GEMM 抽象成一个 primitive，方便在更复杂 kernel 里复用。
2. 但当前真正高性能的 GEMM pattern 并没有统一调用这个 wrapper，而是把 `tl.dot` 直接内联进各个 pattern kernel 中。

换句话说，`tile_dot` 更像“抽象语义层 primitive”，而 `bulk_sync` / `fused_sequential` / `producer_consumer` / `wg_specialized` 里的 GEMM 循环才是现在实际执行路径。

## 5. `gemm_allscatter`：当前最原生的 TNCC 矩阵乘路径

### 5.1 host 侧如何进入 pattern

`gemm_allscatter` 的高层入口在 [tncc/ops.py](../tncc/ops.py#L826)。

调用链大致是：

1. `gemm_allscatter(...)`
2. `build_gemm_allscatter_plan(...)`
3. 解析 layout / shape contract
4. 选择 pattern
5. 执行对应 pattern 的 `execute(...)`

plan 构建逻辑在 [tncc/ops.py](../tncc/ops.py#L718)，它做了三件事：

- 验证张量契约；
- 解析 full/full、shard/shard、部分 mixed layout 的语义；
- 根据 problem shape 和 context 选择 pattern 实现。

其中 layout contract 的标准化在 [tncc/patterns/contracts.py](../tncc/patterns/contracts.py#L84)。这个模块的意义很大，因为 pattern 不再靠“猜 shape”推断语义，而是消费一个明确的 `PatternExecutionSpec`。

### 5.2 auto-select 选什么 pattern

自动选择逻辑在 [tncc/patterns/auto_select.py](../tncc/patterns/auto_select.py#L34)。

它主要根据：

- `M`
- `N / world_size`
- `K`
- tile 数量
- 设备 SM 数
- 链路带宽

来选择：

- `BulkSyncPattern`
- `FusedSequentialPattern`
- `ProducerConsumerPattern`
- `WGSpecializedPattern`

这说明 TNCC 的“矩阵乘怎么做”不仅取决于 GEMM 内核，还取决于“算出来的 tile 什么时候发、谁来发、发和算是否并行”。

## 6. 四种 pattern 里的矩阵乘到底怎么执行

这四种 pattern 在计算部分几乎是同一套 GEMM 内核骨架，区别主要在“GEMM tile 算完之后怎么和通信拼接”。

### 6.1 BulkSyncPattern

实现见 [tncc/patterns/bulk_sync.py](../tncc/patterns/bulk_sync.py#L67)。

它分三步：

1. 启动 GEMM kernel，计算本地 `C`
2. `backend.synchronize()` 做全设备同步
3. 再启动 scatter kernel，把 `C` 的 tile 推到 peer

对应代码：

- GEMM launch: [tncc/patterns/bulk_sync.py](../tncc/patterns/bulk_sync.py#L106)
- barrier: [tncc/patterns/bulk_sync.py](../tncc/patterns/bulk_sync.py#L126)
- scatter launch: [tncc/patterns/bulk_sync.py](../tncc/patterns/bulk_sync.py#L129)

它的计算部分是最朴素的 persistent tile GEMM，见 [tncc/patterns/bulk_sync.py](../tncc/patterns/bulk_sync.py#L160)。

特点是：

- 计算和通信完全分离；
- 没有 overlap；
- 最适合作为基线和 correctness 参考。

### 6.2 FusedSequentialPattern

实现见 [tncc/patterns/fused_sequential.py](../tncc/patterns/fused_sequential.py#L73)。

这个 pattern 在单个 kernel 内部对每个 tile 做：

1. GEMM 累加
2. 把结果写到本地 `C`
3. 立即 scatter 到所有 peer

对应代码在：

- GEMM 累加: [tncc/patterns/fused_sequential.py](../tncc/patterns/fused_sequential.py#L206)
- 本地 store: [tncc/patterns/fused_sequential.py](../tncc/patterns/fused_sequential.py#L236)
- 对 peer scatter: [tncc/patterns/fused_sequential.py](../tncc/patterns/fused_sequential.py#L240)

它的 overlap 不是显式 producer/consumer 机制，而是依赖硬件让“上一块 tile 的 remote store”和“下一块 tile 的 GEMM”形成一定重叠。

### 6.3 ProducerConsumerPattern

实现见 [tncc/patterns/producer_consumer.py](../tncc/patterns/producer_consumer.py#L140)。

它把工作分成两个 kernel、两个 stream：

- producer kernel: 做 GEMM，算完 tile 后 `tile_signal`
- consumer kernel: `tile_wait`，等 tile ready 后 scatter

关键代码：

- launch producer: [tncc/patterns/producer_consumer.py](../tncc/patterns/producer_consumer.py#L189)
- launch consumer: [tncc/patterns/producer_consumer.py](../tncc/patterns/producer_consumer.py#L211)
- producer 中 GEMM + signal: [tncc/patterns/producer_consumer.py](../tncc/patterns/producer_consumer.py#L242)

这里的矩阵乘仍然是同样的 tile GEMM，只是计算 worker 和通信 worker 被解耦到了不同 stream。

### 6.4 WGSpecializedPattern

实现见 [tncc/patterns/wg_specialized.py](../tncc/patterns/wg_specialized.py#L127)。

这是最接近“原生一体化流水线”的实现。单次 kernel launch 里：

- `pid < COMPUTE_SMS` 的 program 负责 GEMM + signal
- `pid >= COMPUTE_SMS` 的 program 负责 wait + scatter

关键代码：

- 单 kernel launch: [tncc/patterns/wg_specialized.py](../tncc/patterns/wg_specialized.py#L173)
- compute worker: [tncc/patterns/wg_specialized.py](../tncc/patterns/wg_specialized.py#L258)
- comm worker: [tncc/patterns/wg_specialized.py](../tncc/patterns/wg_specialized.py#L310)

所以如果要找“TNCC 现在最像论文式 tile-level compute-communication fusion 的矩阵乘实现”，答案通常是这个 pattern。

## 7. 远端 scatter 如何接在 GEMM 后面

### 7.1 不是调 NCCL / NVSHMEM 黑盒

TNCC 当前模式下，scatter 的核心不是调用外部通信 runtime 完成一整个 collective，而是：

1. 先把本地指针翻译成远端 heap 中对应位置的指针；
2. 再对这个远端地址直接执行 Triton `tl.store`。

最核心的函数是 [tncc/memory/translation.py](../tncc/memory/translation.py#L38) 的 `translate_ptr(...)`：

- 读取 `from_rank` heap base
- 读取 `to_rank` heap base
- 计算原指针相对本地 heap 的偏移
- 把偏移加到目标 rank 的 heap base 上

公式就是：

```text
offset     = ptr - heap_bases[from_rank]
remote_ptr = heap_bases[to_rank] + offset
```

### 7.2 scatter helper 做了什么

`scatter_tile_to_peer(...)` 在 [tncc/patterns/_helpers.py](../tncc/patterns/_helpers.py#L18)。

它做的事情是：

1. 调用 `translate_ptr(C_ptr, rank, peer, heap_bases)` 拿到 peer 视角下的 `C` 指针；
2. 根据 contract 把“本地列偏移”映射成“peer 目标列偏移”；
3. 对 `remote_C + offsets` 直接 `tl.store(...)`。

这一步很关键。因为这说明 TNCC 把“通信”也保留在 Triton 编译器可见的代码里，而不是把数据交给外部黑盒 runtime 处理。

## 8. tile 级同步怎么做

`ProducerConsumerPattern` 和 `WGSpecializedPattern` 依赖本地 lock tensor 做 tile 级同步。

同步原语在 [tncc/sync/primitives.py](../tncc/sync/primitives.py#L357)：

- `tile_signal(...)`：本质是 `atomic_xchg(lock, 1)`，默认 release 语义，见 [tncc/sync/primitives.py](../tncc/sync/primitives.py#L357)
- `tile_wait(...)`：循环 `atomic_cas(lock, 1, 0)`，默认 acquire 语义，见 [tncc/sync/primitives.py](../tncc/sync/primitives.py#L390)

这意味着 pattern 间的区别不是“有没有 GEMM”，而是“GEMM tile ready 后用什么同步机制把它交给通信侧”。

## 9. `gemm_allgather` 和 `gemm_reducescatter` 现在怎么做计算

这是一个容易误判的点。

如果只看 `gemm_allscatter`，会以为 TNCC 所有 GEMM+collective 都已经走 Triton fused kernel。当前并不是这样。

### 9.1 gemm_allgather

`build_gemm_allgather_plan(...)` 的注释已经明确写出：当前实现是保守方案，先做一次本地 GEMM，再复用高层 allgather，见 [tncc/ops.py](../tncc/ops.py#L887)。

执行时：

1. 先分配本地输出 shard workspace
2. 调 `_run_local_gemm(A, B_shard, out=local_shard)`
3. 再执行 allgather 组装完整输出

具体代码在：

- `GemmAllGatherPlan.execute`: [tncc/ops.py](../tncc/ops.py#L274)
- `_run_local_gemm`: [tncc/ops.py](../tncc/ops.py#L1597)

而 `_run_local_gemm` 实际上优先用 `torch.mm(A, B, out=out)`，失败再退到 `torch.matmul`，见 [tncc/ops.py](../tncc/ops.py#L1597)。

所以 `gemm_allgather` 当前不是 Triton fused GEMM kernel 主导，而是“本地 PyTorch GEMM + TNCC allgather”。

### 9.2 gemm_reducescatter

`build_gemm_reducescatter_plan(...)` 也在注释里说明：当前实现是先做本地 GEMM 到 workspace，再重排，再复用高层 `reduce_scatter`，见 [tncc/ops.py](../tncc/ops.py#L956)。

所以它当前的“计算”部分也不是 pattern 里的 Triton fused GEMM，而是 host 侧本地 GEMM 方案。

## 10. 现在 TNCC 的矩阵乘有哪些实际分支

综合代码现状，可以把“矩阵乘怎么做”总结成三条路径。

### 10.1 路径 A：独立本地 Triton GEMM

入口：

- `tncc.kernels.gemm.gemm(...)`

特点：

- 真正的 Triton persistent GEMM
- 用 `tl.dot`
- 不涉及通信融合

### 10.2 路径 B：`gemm_allscatter` 的 fused/pattern GEMM

入口：

- `tncc.ops.gemm_allscatter(...)`
- `tncc.ops.gemm_allscatter_sharded(...)`

特点：

- GEMM 在 Triton pattern kernel 里执行
- 计算 tile 后直接做 scatter / signal / wait / remote store
- 是当前最完整体现 TNCC 设计哲学的路径

### 10.3 路径 C：保守 host-side GEMM + collective

入口：

- `tncc.ops.gemm_allgather(...)`
- `tncc.ops.gemm_reducescatter(...)`

特点：

- 本地 GEMM 由 `torch.mm` / `torch.matmul` 完成
- 后续通信由 TNCC 高层 collective plan 承担
- 更像“稳定 API 先打通”，而不是最终形态的 fused Triton kernel

## 11. 对“TNCC 现在如何做矩阵乘”的准确表述

如果要给出一个严格、不误导的表述，可以写成：

> TNCC 当前的矩阵乘核心是基于 Triton 的 tile GEMM，乘法本身由 `tl.dot` 完成；TNCC 负责 tile 切分、persistent 调度、layout contract、远端地址翻译和计算通信融合。  
> 在高层操作里，`gemm_allscatter` 已经走 pattern 化的 Triton fused 路径，而 `gemm_allgather` / `gemm_reducescatter` 目前仍是更保守的“本地 GEMM + collective”实现。

## 12. 工程判断

从工程状态看，当前仓库的主线很清楚：

1. TNCC 已经把“计算 tile + 远端 tile 通信”这件事在 `gemm_allscatter` 上做到了 Triton 编译器可见。
2. 但“所有 GEMM 类操作都 fully fused 到同一套 Triton pattern”这件事还没有完全收口。
3. 因此如果后续要继续演进，最自然的方向不是再去重写一个新的独立 GEMM kernel，而是把 `allgather` / `reducescatter` 路径也逐步向 `gemm_allscatter` 这种 contract + pattern + translated remote access 的体系收敛。

## 13. 最后结论

一句话总结：

当前 TNCC 的矩阵乘不是“一个统一的单体 GEMM 实现”，而是“一个 Triton tile GEMM 核心，加上一层 pattern 化的计算通信编排系统”。

其中：

- 乘法本身：`tl.dot`
- 调度方式：persistent tile loop
- 融合机制：bulk-sync / fused-sequential / producer-consumer / workgroup-specialization
- 远端通信：`translate_ptr` 后直接 `tl.store` / `tl.load`
- 当前最原生的矩阵乘主线：`gemm_allscatter`
