# XTile 

本文件是给 LLM 的仓库入口说明，只保留当前状态、索引和下一步计划。

## 目标

灵感来自于Iris，但不止局限于AMD生态。要做到通用，可维护，易优化，暴露出来更多优化的接口，包括但不限于将runtime运行时的信息拿来优化内核或通信库。

建立Tile粒度的原生通信库，把作为通信一等公民看待。计算-通信以tile为单位，且尽量不调用NCCL和NVSHMEM等，整个内核对编译器可见。

## 项目位置

"/home/makai/XTile/"

### 参考项目

"/home/makai/tilescale/"

"/home/makai/Triton-distributed/"

"/home/makai/iris/"

## 维护规则

- `CLAUDE.md` 与 `docs/experiment_log.md` 都是 LLM 上下文文档，必须保持短、准、可维护。
- 只写当前实现状态、canonical 文档/数据索引、以及简短下一步建议。
- 不要追加日记式日志、阶段流水账、原始命令输出、长篇排障过程、重复 benchmark 表。
- 状态变化时直接原位改写，不保留“第几轮修改”“某日记录”式历史堆叠。
- 历史细节应放到代码、测试、生成物或独立分析文档，不应继续堆在这里。

## 当前实现状态

- 项目主线已经收口到 `XTileContext + SymmetricHeap + xtile.ops + runtime support matrix`。
- 当前硬件基线是 `2 x NVIDIA H100 PCIe`，双卡互联是 `NVLink (NV12)`。
- 当前最稳定主路径是单进程多 GPU：
  - `heap_mode=single_process`
  - `transport_strategy=peer_access`
- 当前多进程公开 baseline 是：
  - `world_size=2`
  - `transport_strategy=ctypes_ipc`
- `pytorch_ipc` 和 `peer_access_pointer_exchange` 仍只算诊断/bring-up 路径，不算正式支持面。

## 当前公开能力

- 高层入口已经存在并应优先使用：
  - `xtile.ops.gemm_allscatter(...)`
  - `xtile.ops.gemm_allgather(...)`
  - `xtile.ops.gemm_reducescatter(...)`
  - `xtile.ops.allgather(...)`
  - `xtile.ops.allreduce(...)`
  - `xtile.ops.reduce_scatter(...)`
- `pattern.execute(...)` 仍应视为 expert/internal surface，而不是默认公共入口。
- 当前 contract 结论：
  - `gemm_allscatter.full/full`: supported
  - `gemm_allscatter.shard/shard`: supported
  - `gemm_allscatter.full/shard`: supported
  - `gemm_allscatter.shard/full`: intentionally unsupported
  - `allreduce`: 稳定的高层 in-place contract

## 内存模型现状

- allocator-first 第一阶段已落地，默认 allocator 是 `torch_bump`。
- 已有能力：
  - allocator-backed `SymmetricHeap`
  - segment metadata
  - peer export/import metadata
  - `import_external_tensor(...)` / `as_symmetric(...)`
  - `runtime_metadata()` / support matrix
- 尚未完成的核心差距：
  - 还没有 Iris 风格的 canonical `allocator + export/import-map/access` 统一底座
  - 还没有 `fd passing + DMA-BUF` 零拷贝 external mapping

## Canonical 索引

- 总状态文档：`docs/XTile现状流程_修订版.md`
- benchmark 摘要：`docs/generated/benchmark_runtime_summary.md`
- 实验状态：`docs/experiment_log.md`
- 关键源码：
  - `xtile/ops.py`
  - `xtile/support.py`
  - `xtile/memory/`

## 下一步计划

1. 先完成内存底座收口：把当前 allocator-first partial 继续推进到 canonical `export/import-map/access` 运行时。
2. 继续收缩公共语义面：所有默认用户路径都应优先通过 `xtile.ops.*`，减少直接暴露 pattern 细节。
3. 扩大真实验证面：从 `world_size=2 + ctypes_ipc` 扩到更完整的 multiprocess/world-size/transport 矩阵。
