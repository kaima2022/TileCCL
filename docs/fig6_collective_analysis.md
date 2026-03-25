# Fig6 现象与根因

基于 [figures/data/collective_comm_only_latest.json](../figures/data/collective_comm_only_latest.json) 的当前 `fig6` 解读。

## 口径

- 环境：`2 x NVIDIA H100 PCIe`，双卡互联 `NVLink (NV12)`，`world_size=2`，`transport=ctypes_ipc`
- 计时：`host wall end-to-end with CUDA completion`
- 上排：统一只看 `4 KiB / 16 KiB / 64 KiB` 的小消息 latency
- 左下：统一只看 `256 KiB` 的跨 collective 带宽对比
- 右下：单独看 `allreduce` 在 `256 KiB / 1 MiB / 2 MiB` 的大消息带宽
- 这是一张 split-view 分析图，不是单一 sweep 的单口径曲线图；但三个视图使用的是同一机器、同一 world size、同一 timing mode

## 图上现象

1. 小消息 latency 全部聚在 `2.39 ~ 2.41 ms`，随消息尺寸变化很小。
2. 小消息段里，XTile 与 NCCL 基本持平。
3. `256 KiB` 时，`allgather` 和 `scatter` 高于 NCCL，`broadcast` 基本持平。
4. `256 KiB` 时，`allreduce` 明显落后，`reduce_scatter` 则出现断崖式塌缩。
5. `allreduce` 从 `256 KiB` 拉到 `1 MiB / 2 MiB` 后，NCCL 带宽继续上升，XTile 没有随消息尺寸正常扩展。

## 关键数值

- `allreduce @ 256 KiB`：XTile `0.0175 GB/s`，NCCL `0.0557 GB/s`，约 `0.31x`
- `allgather @ 256 KiB`：XTile `0.0556 GB/s`，NCCL `0.0378 GB/s`，约 `1.47x`
- `scatter @ 256 KiB`：XTile `0.0554 GB/s`，NCCL `0.0378 GB/s`，约 `1.47x`
- `reduce_scatter @ 256 KiB`：XTile `0.00056 GB/s`，NCCL `0.1090 GB/s`，约 `0.005x`
- `broadcast @ 256 KiB`：XTile `0.1096 GB/s`，NCCL `0.1091 GB/s`，基本持平
- `allreduce @ 1 MiB`：XTile `0.00312 GB/s`，NCCL `0.2223 GB/s`
- `allreduce @ 2 MiB`：XTile `0.00580 GB/s`，NCCL `0.4435 GB/s`

## 根因归因

### 1. 上排小消息主要反映的是观测开销，不是 steady-state 带宽

当前计时口径是 `host wall + result.wait() + cuda synchronize`，再加上 multiprocess benchmark 的 barrier 协调，小消息区间基本被固定开销主导。

因此，上排能说明的是：

- 公开调用路径没有出现数量级额外开销
- 小消息段 XTile 与 NCCL 的调用侧成本接近

上排不能直接推出“大消息协议已经接近 NCCL”。

### 2. `allgather / scatter / broadcast` 在 `world_size=2` 下更容易接近或超过 NCCL

这三类 primitive 当前都属于比较直接的 device-side 远端读写路径：

- `tile_allgather(...)` 是 direct-write 到所有 peer 的目标位置
- `tile_scatter(...)` 是 root 直接把各 chunk 推到各 rank
- `tile_broadcast(...)` 当前是 flat root push

在 `world_size=2 + NVLink` 这个很友好的条件下，这类简单路径没有经历额外的归约协议、分阶段同步和复杂 forwarding，因而更容易接近链路上限。

对应代码：

- [xtile/primitives/collectives.py](../xtile/primitives/collectives.py)

### 3. `reduce_scatter` 的塌缩不是画图问题，是实现本身当前就是 correctness-first

当前 `tile_reduce_scatter(...)` 的真实策略是：

- 每个 rank 只读取“自己负责的那一块”
- 直接从所有 peer 的对称地址把该 chunk 读回来
- 在本地做 reduction
- 最后只写回自己的 `dst`

这条路径的优点是安全、直接、容易验证；缺点也很明确：

- 没有 pipelined ring
- 没有分块 forwarding
- 没有 channel 化并发
- 没有 double buffer
- 没有更成熟的 cross-rank staging/sync

所以它能作为稳定 public surface 的 correctness 基础，但不是高性能 `reduce_scatter` 内核。

对应代码与说明：

- [xtile/primitives/collectives.py](../xtile/primitives/collectives.py)
- [XTile现状流程_修订版.md](/home/makai/XTile/docs/XTile现状流程_修订版.md#L854)

### 4. `allreduce` 已经从 host-side 组合路径收口，但新主路径仍然偏保守

当前 public `allreduce` 已经不是旧的 `reduce_scatter + allgather + copy` 热路径，而是 `device_staged_pipeline`：

- 先把 chunk snapshot 到 staging
- 用 `published_epoch / consumed_count` 做显式 slot 握手
- 等 peer 发布同 epoch 后再读取 peer staging
- 本地归约后写回自己的 tensor

这条路径解决了“host 厚组合路径”的问题，但性能上仍然保守：

- 当前 protocol 还是 `slot_epoch_pipeline`
- 目标 chunk 只有 `16 KiB`
- pipeline slot 上限只有 `8`
- 每个 chunk 都要做 sys-scope 的 publish/consume 握手
- 还没有 pipelined forwarding、多 channel 并发、消息分层 protocol、拓扑感知

因此，大消息段会暴露出很重的协议与同步成本，表现为：NCCL 随消息变大继续扩展，XTile 没有跟上。

对应代码：

- [xtile/primitives/collectives.py](/home/makai/XTile/xtile/primitives/collectives.py#L507)
- [xtile/primitives/collectives.py](/home/makai/XTile/xtile/primitives/collectives.py#L1120)
- [xtile/ops.py](/home/makai/XTile/xtile/ops.py#L1043)

### 5. 为什么小消息看起来接近，大消息又突然拉开

不是结果自相矛盾，而是两个区域在看不同主导项：

- 小消息：固定调用/同步成本主导，所以大家都被压在同一个 `~2.4 ms` 平台上
- 大消息：协议组织能力主导，这时 XTile 当前的保守 allreduce / reduce_scatter 路径就会被放大出来

所以正确解读是：

- `fig6` 上排说明“调用面没有炸”
- `fig6` 下排说明“真正的协议层差距仍然很大”

## 结论

`fig6` 当前最重要的结论不是“XTile collective 全面落后”，而是：

- 小消息调用侧已经接近可用
- 简单 direct-write collective 在 `world_size=2` 下不差
- 真正的短板集中在 `reduce_scatter` 和大消息 `allreduce` 的协议强度

后续优先级应保持明确：

1. 先把 `reduce_scatter` 从 correctness-first 改成 performance-first
2. 再继续增强 `allreduce` 的大消息 protocol 和并发形态
3. 最后再扩大到 `world_size > 2` 和更多 transport 的稳定验证面
