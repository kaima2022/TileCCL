# TNCC (Tile Native Collective Communication) 相关术语解释

日期：`2026-03-27`

## 目的

这份文档专门解释 TNCC 里最容易混淆的一批术语，尤其是：

- `full`
- `shard`
- `full_N`
- `local_N`
- `rank`
- `world_size`
- `layout`
- `contract`
- `scatter`

这些词在分布式 GEMM、all-scatter、WG pattern 文档里会反复出现。如果不先统一这些概念，后面的流程文档会很难读。

本文只讲 TNCC 当前代码里的专业含义，不讲泛泛而谈的分布式术语。

## 一个先导例子

先看一个最小例子。

假设：

- 总输出矩阵逻辑大小是 `M x N = 4096 x 8192`
- 一共 `2` 张 GPU
- 当前是 `rank=0`

那么：

- `world_size = 2`
- `full_N = 8192`
- 每张卡负责的列数是 `8192 / 2 = 4096`
- 也就是 `N_per_rank = local_N = 4096`

如果某个 tensor 是 `full` 布局，它的列数就是 `8192`。  
如果某个 tensor 是 `shard` 布局，它在每张卡上的列数就是 `4096`。

这是后面所有术语的基础。

## 1. `rank`

`rank` 就是当前参与分布式执行的编号。

在你当前的使用场景里，可以简单理解为：

- `rank 0` -> 第 0 张 GPU
- `rank 1` -> 第 1 张 GPU

更严格一点说：

- `rank` 是“当前执行实体在通信组里的编号”
- 在单机多 GPU 场景中，通常与 GPU 编号一一对应
- 但从抽象上，它首先是“分布式参与者编号”，其次才是“物理 GPU 编号”

TNCC 里的 `ctx.rank` 就是这个意思。

例子：

- 两张卡时，合法 `rank` 只有 `0` 和 `1`
- 如果当前代码运行在第 1 张卡上，那么当前 `rank` 通常就是 `1`

## 2. `world_size`

`world_size` 是参与当前这次分布式执行的总 rank 数。

在单机双卡里：

- `world_size = 2`

在单卡本地调试里：

- `world_size = 1`

它不等于“机器上总共有多少 GPU”，而是：

- 这次 TNCC 上下文里，实际被纳入同一个通信/执行组的 rank 数

例子：

- 机器有 8 张 GPU，但这次只用前 2 张跑 TNCC
- 那么这里的 `world_size` 仍然是 `2`，不是 `8`

## 3. `M / N / K`

这些是 GEMM 的标准维度：

- `A` 的形状通常是 `(M, K)`
- `B` 的形状通常是 `(K, N)`
- `C` 的形状通常是 `(M, N)`

TNCC 文档里最容易让人迷糊的是：

- `N` 有时指逻辑全量列数
- 有时 pattern 内部又会出现本地列数

所以 TNCC 代码里专门拆成了：

- `full_N`
- `local_N`

## 4. `full_N`

`full_N` 是逻辑上的完整输出宽度，也就是“全局 N”。

它表示：

- 如果把所有 rank 的输出拼起来，完整结果一共有多少列

例子：

- 2 张卡
- 每张卡本地存 `4096` 列
- 那么逻辑完整输出就是 `8192` 列
- 所以 `full_N = 8192`

一个最重要的理解是：

- `full_N` 是“全局问题规模”的概念
- 不是“当前这张卡实际拿着多少列”

## 5. `local_N`

`local_N` 是当前 rank 本地实际持有或实际计算的列数。

它表示：

- 从当前 rank 的物理 tensor 视角看，当前张量有多少列

例子：

- `world_size = 2`
- `full_N = 8192`
- 如果当前张量采用 `shard` 布局
- 那么 `local_N = 4096`

如果张量采用 `full` 布局，那么：

- `local_N = full_N`

所以可以把它理解成：

- `full_N` 是逻辑全局宽度
- `local_N` 是当前 rank 看到的物理本地宽度

## 6. `full`

`full` 是一种张量布局语义。

它表示：

- 这个 tensor 在当前 rank 上保存的是“完整列宽”的版本
- 也就是它的列数等于 `full_N`

例子：

- `full_N = 8192`
- 如果 `B` 是 `full` 布局
- 那么当前 rank 上的 `B.shape[1] = 8192`

同理：

- 如果 `C` 是 `full` 布局
- 那么当前 rank 上的 `C.shape[1] = 8192`

直观理解：

- `full` = 当前 rank 拿的是完整版本，不只是自己那一份分片

## 7. `shard`

`shard` 也是一种张量布局语义。

它表示：

- 这个 tensor 在当前 rank 上只保存“属于自己的一段”
- 不是完整列宽，而是全局列宽的一部分

最常见的是列方向切分：

- `shard_N = full_N / world_size`

例子：

- `world_size = 2`
- `full_N = 8192`
- 则每个 rank 的 shard 宽度是 `4096`
- 如果 `C` 是 `shard` 布局
- 那么当前 rank 上的 `C.shape[1] = 4096`

直观理解：

- `shard` = 当前 rank 只拿局部切片

## 8. `layout`

`layout` 在这里不是指内存连续性或 row-major/col-major，而是：

- 这个 tensor 的“逻辑分布方式”

TNCC 当前在 pattern contract 里关心的布局主要是：

- `full`
- `shard`

所以当代码里写：

- `b_layout="full"`
- `c_layout="shard"`

它的意思不是：

- B 用某种底层内存格式
- C 用另一种底层内存格式

而是：

- `B` 在逻辑上是完整矩阵
- `C` 在逻辑上是每个 rank 只持有一个分片

这是“分布式布局语义”，不是普通线性代数库里那种“存储顺序语义”。

## 9. `rhs`

`rhs` 是 `right-hand side` 的缩写，表示 GEMM 里的右操作数。

在 `A @ B = C` 里：

- `A` 是左矩阵
- `B` 是右矩阵
- 所以 `B` 常被叫作 `rhs`

TNCC 的 `PatternExecutionSpec` 里有：

- `rhs`
- `output`

分别就是：

- `rhs` -> 对 `B` 的布局与形状描述
- `output` -> 对 `C` 的布局与形状描述

## 10. `output`

`output` 指输出张量，也就是 GEMM 的结果 `C`。

在 TNCC 的执行合同里，`output` 不只是“这是个输出”，还会记录：

- 它的逻辑完整形状
- 它在当前 rank 的本地形状
- 它是 `full` 还是 `shard`

所以 `output layout` 本质是在回答：

- 当前这张卡上的 `C` 是完整结果，还是只是一块分片

## 11. `contract`

`contract` 在 TNCC 里可以理解为：

- 执行约定
- 语义合同
- 调用约束

它回答的是：

- 这次操作到底要算什么
- 输入输出各自是什么布局
- 当前 rank 的物理张量形状应该是多少
- scatter 的源列和目标列应该怎么解释

为什么需要这个概念？

因为只看 tensor shape 往往不能唯一确定语义。

例如：

- `B.shape = (K, 4096)`

这到底表示：

1. 全局问题本来就只有 4096 列
2. 还是全局其实有 8192 列，只是当前 rank 拿了 4096 列 shard

如果没有 `contract`，kernel 很容易“猜错用户意图”。

所以 TNCC 会先把这些语义解析成结构化对象，例如：

- `GemmAllScatterContract`
- `PatternExecutionSpec`

这样 pattern 和 kernel 不用猜，直接按合同执行。

## 12. `PatternExecutionSpec`

这是 TNCC pattern 层最核心的“执行合同对象”。

它包含：

- `M`
- `K`
- `full_N`
- `local_N`
- `rank`
- `world_size`
- `rhs`
- `output`
- `scatter_src_col_offset`
- `scatter_cols`
- `scatter_dst_leading_dim`
- `scatter_dst_col_offset`

可以把它理解成：

- 这次 GEMM + scatter 的完整执行说明书

pattern 真正消费的是这个对象，而不是自己从原始 tensor 瞎猜。

## 13. `full_shape`

`full_shape` 表示这个 tensor 在逻辑上的完整全局形状。

例子：

- `output.full_shape = (4096, 8192)`

意思是：

- 从全局语义上，输出矩阵 `C` 的完整尺寸是 `4096 x 8192`

即使当前 rank 只拿到其中一部分，这个值也仍然是全局尺寸。

## 14. `local_shape`

`local_shape` 表示当前 rank 上实际物理持有的张量形状。

例子：

- `output.local_shape = (4096, 4096)`

意思是：

- 当前 rank 这张卡上，真实分配出来的 `C` 张量尺寸是 `4096 x 4096`

如果 `output.layout_kind == "shard"`，那么 `local_shape` 往往小于 `full_shape`。  
如果 `layout_kind == "full"`，那么 `local_shape` 和 `full_shape` 通常一致。

## 15. `storage_kind`

`storage_kind` 表示张量从“存储归属/放置方式”看属于哪一类。

在你当前看到的 TNCC 代码里，最常见的是：

- `storage_kind = "symmetric"`

它表示：

- 这个张量放在 TNCC 的 symmetric heap 里
- 因而它可以参与 `translate_ptr` 这种基于 heap offset 的远端地址翻译

为什么这件事重要？

因为：

- 不是任意普通 `torch.cuda.Tensor` 都天然适合做 TNCC 的远端对称翻译
- TNCC 当前很多远端操作默认要求 tensor 在 symmetric heap 里

## 16. `symmetric heap`

`symmetric heap` 可以直译成“对称堆”。

它的核心意思是：

- 每个 rank 都有一块专门用于远端访问的显存区域
- 不同 rank 之间，这些区域在语义上是对齐的
- 同一个 offset 在不同 rank 的 heap 里代表“对应位置”

所以 `translate_ptr` 才能成立：

```text
offset = ptr - local_heap_base
remote_ptr = remote_heap_base + offset
```

如果没有 symmetric heap，这种“同 offset 平移”就没有可靠语义基础。

## 17. `heap_bases`

`heap_bases` 是一个长度为 `world_size` 的 `int64` 张量。

其中：

- `heap_bases[i]` 表示 rank `i` 的 symmetric heap 基址
- 而且这个基址必须是“当前 rank 可直接访问的地址”

例子：

- `heap_bases[0]` -> rank 0 的 heap 基址
- `heap_bases[1]` -> rank 1 的 heap 基址

在单进程时，它通常来自 peer access 直接地址。  
在多进程时，它通常来自 CUDA IPC 导入后的映射地址。

## 18. `scatter`

`scatter` 在 TNCC 这里，不要简单理解成“随便发数据”，它是有明确方向和语义的。

这里的 scatter 一般表示：

- 当前 rank 把自己算出来的一部分输出列
- 写到其他 rank 对应的目标位置

在 GEMM + all-scatter 语境下，它通常意味着：

- 本地先产出一个逻辑上更完整的结果
- 再把其中属于各 rank 的那一段推送给对应 peer

所以 scatter 不是“复制全部输出到所有人”，而是：

- 按约定把某些列段分发到对应 rank

## 19. `scatter_src_col_offset`

这是 scatter 时，从源张量哪一列开始取数据。

例子：

- `rank = 1`
- `world_size = 2`
- `full_N = 8192`
- 每个 rank 对应 `4096` 列

那么 rank 1 对应的列段是：

- `[4096, 8192)`

此时：

- `scatter_src_col_offset = 4096`

意思是：

- 从本地完整输出里，第 4096 列开始，才是应该发给 peer 的那一段

## 20. `scatter_cols`

这是 scatter 时要发送多少列。

例子：

- `full_N = 8192`
- `world_size = 2`

那么每个 rank 的列块宽度是：

- `8192 / 2 = 4096`

所以：

- `scatter_cols = 4096`

它回答的是：

- 本次 scatter 的有效列宽是多少

## 21. `scatter_dst_col_offset`

这是 scatter 到目标 rank 时，应写入目标张量的哪一列起点。

它和 `scatter_src_col_offset` 不一定总是相同概念。

简单说：

- `scatter_src_col_offset` 是“从源里哪开始拿”
- `scatter_dst_col_offset` 是“往目标里哪开始放”

在某些 `full/full` 合同下，这两者可能相同。  
在更复杂的布局映射中，这两者可以不同。

## 22. `scatter_dst_leading_dim`

这表示 scatter 目标布局里，一行的跨度是多少。

你可以把它理解成：

- 目标矩阵每一行有多少列
- 或者更工程一点：目标二维张量在线性地址中的行跨度

如果目标是 `full` 布局：

- 它通常等于 `full_N`

如果目标是 `shard` 布局：

- 它通常等于当前 shard 的列数

它的作用是帮助 kernel 正确计算：

- 第 `m` 行、第 `n` 列在目标张量中的线性偏移

## 23. `full/full`、`shard/shard`、`full/shard`

这些写法表示输入输出布局组合。

### 23.1 `full/full`

意思是：

- `B` 是完整矩阵
- `C` 也是完整矩阵

例子：

- 每张卡都持有完整的 `B(K, full_N)`
- 每张卡也写出完整的 `C(M, full_N)`

之后再由 scatter 合同决定把哪一段推给 peer。

### 23.2 `shard/shard`

意思是：

- `B` 是本地 shard
- `C` 也是本地 shard

例子：

- `B.shape = (K, full_N / world_size)`
- `C.shape = (M, full_N / world_size)`

这是更贴近真正分片执行的模式。

### 23.3 `full/shard`

意思是：

- `B` 是完整矩阵
- `C` 是本地分片

这类情况通常需要宿主层 wrapper 做中间物化，因为直接从 kernel 角度看，它是“输入完整、输出分片”的混合合同。

TNCC 当前代码对 mixed layout 是有限支持的，不是所有组合都开放。

## 24. `peer`

`peer` 指“除了当前 rank 之外的其他 rank”。

在双卡场景里：

- 如果当前 `rank=0`
- 那么 `peer` 通常就是 `1`

在 WG kernel 里常看到：

```python
for peer in range(world_size):
    if peer != rank:
        ...
```

意思就是：

- 把本地需要发出去的 tile 依次写给其他所有 rank

## 25. `peer access`

这是 CUDA/ROCm 提供的同机 GPU 直访能力。

在 TNCC 里，它表示：

- 一张 GPU 可以直接访问另一张 GPU 的显存地址

单进程多卡时，TNCC 通常用这条路径建立 `heap_bases`。

你可以把它理解成：

- 远端 GPU 内存被映射成了当前 GPU 可以解引用的地址

## 26. `CUDA IPC`

这是多进程场景下跨进程共享 GPU 内存的一种机制。

在 TNCC 里，它表示：

- 一个 rank 导出自己 heap 的 IPC handle
- 其他 rank 打开这个 handle
- 得到一个当前进程可用的 `mapped_ptr`

然后这个 `mapped_ptr` 被写入 `heap_bases`，供 `translate_ptr` 使用。

所以：

- 单进程时常见的是 `peer access`
- 多进程时常见的是 `CUDA IPC`

## 27. `translate_ptr`

`translate_ptr` 是 TNCC 远端访问的核心函数。

它的意思是：

- 给我一个“本 rank heap 里的地址”
- 我把它换算成“peer rank heap 里对应 offset 的地址”

公式是：

```text
offset     = ptr - heap_bases[from_rank]
remote_ptr = heap_bases[to_rank] + offset
```

专业上它不是“通信调用”，而是：

- 基于对称堆语义做的地址翻译

## 28. `remote store` / `remote load`

这两个词表示：

- 对翻译后的远端地址做读写

在 TNCC 当前实现里，它们本质上就是：

- `translate_ptr(...)`
- 然后 `tl.load(...)` 或 `tl.store(...)`

所以这里的“remote”不代表 NVSHMEM/NCCL，而是：

- 目标地址属于 peer rank 的 heap

## 29. `logical` 和 `physical`

这是理解 `full_shape/local_shape` 最重要的一组词。

### `logical`

表示：

- 从算法语义上看，这个 tensor 应该有多大
- 不关心当前 rank 实际拿到了多少

例如：

- 全局输出逻辑上是 `(4096, 8192)`

### `physical`

表示：

- 当前 rank 实际分配出来、实际持有的 tensor 形状

例如：

- 当前 rank 只持有 `(4096, 4096)`

所以：

- `logical` 更接近全局视角
- `physical` 更接近当前 GPU 视角

## 30. `plan`

`plan` 是在 `contract` 之上的下一层。

可以这样区分：

- `contract` 更强调“语义说明”
- `plan` 更强调“可执行对象”

例如 TNCC 里的 `GemmAllScatterPlan`：

- 内部带着已经解析好的 execution contract
- 也带着具体绑定好的 pattern 实现
- 可以直接 `plan.execute(A, B, C)`

所以 `plan` 可以理解成：

- “带执行器的合同”

## 一张总表

最容易混淆的词，可以用下面这张表压缩记忆。

| 术语 | 简单理解 | 更准确的专业含义 |
| --- | --- | --- |
| `rank` | 当前这张卡的编号 | 当前分布式参与者编号 |
| `world_size` | 一共几张卡参与 | 当前通信/执行组的总 rank 数 |
| `full_N` | 全局总列数 | 逻辑完整输出宽度 |
| `local_N` | 当前卡实际列数 | 当前 rank 物理持有宽度 |
| `full` | 完整版本 | 当前 rank 上保存完整列宽 tensor |
| `shard` | 分片版本 | 当前 rank 上只保存局部列块 |
| `layout` | 张量是 full 还是 shard | 分布式布局语义 |
| `contract` | 执行约定 | 张量形状、布局、scatter 语义的结构化说明 |
| `plan` | 可执行方案 | 绑定了 contract 和具体 pattern 的执行对象 |
| `scatter` | 把一段列发出去 | 按合同把源列段写入目标 rank 的目标位置 |
| `symmetric heap` | 专门的对称显存区 | 支持 offset-preserving 跨 rank 地址翻译的显存区域 |
| `heap_bases` | 各 rank 的 heap 基址表 | `translate_ptr` 使用的 per-rank 可访问基址张量 |

## 最后总结

如果只记三句话，记这三句就够了：

1. `full_N` 是全局宽度，`local_N` 是当前 rank 实际宽度。
2. `full/shard` 说的是分布式布局语义，不是普通内存排布。
3. `contract` 就是把“这次执行到底是什么意思”先说明白，避免 kernel 自己猜语义。
