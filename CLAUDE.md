# XTile

LLM 仓库入口。只保留当前状态、索引和下一步计划。

## 目标

Tile 粒度原生通信库。通信是一等公民，计算-通信以 tile 为单位，不依赖 NCCL/NVSHMEM，整个内核对编译器完全可见。灵感来自 Iris，但不限于 AMD 生态——通用、可维护、易优化，暴露 runtime 信息供内核/通信优化。

## 参考项目

- `/home/makai/tilescale/`
- `/home/makai/Triton-distributed/`
- `/home/makai/iris/`

## 维护规则

- `CLAUDE.md` 是 LLM 上下文文档，必须保持短、准、可维护。
- 只写当前实现状态、canonical 索引、简短下一步。
- 禁止日记式日志、流水账、原始命令输出、长篇排障。
- 状态变化时原位改写，不保留历史堆叠。

## 硬件基线

- `2 x NVIDIA H100 PCIe`，双卡 NVLink (NV12) 互联。

## 运行时路径

| 路径 | heap_mode | transport_strategy | 状态 |
|------|-----------|-------------------|------|
| 单进程多 GPU（主路径） | `single_process` | `peer_access` | stable |
| 多进程 baseline | `multi_process` | `ctypes_ipc` | stable (world_size=2) |
| pytorch_ipc / peer_access_pointer_exchange | — | — | 诊断/bring-up only |

## 公开 API

高层入口（优先使用）：

```
xtile.ops.gemm_allscatter(...)
xtile.ops.gemm_allgather(...)
xtile.ops.gemm_reducescatter(...)
xtile.ops.allgather(...)
xtile.ops.allreduce(...)
xtile.ops.reduce_scatter(...)
```

Plan builders（可复用执行计划）：

```
xtile.ops.build_gemm_allscatter_plan(...)
xtile.ops.build_gemm_allgather_plan(...)
xtile.ops.build_allreduce_plan(...)
```

`pattern.execute(...)` 定位为 expert/internal surface。

## Contract 支持矩阵

| Contract | 状态 |
|----------|------|
| `gemm_allscatter.full/full` | supported |
| `gemm_allscatter.shard/shard` | supported (via `gemm_allscatter_sharded`) |
| `gemm_allscatter.full/shard` | supported (wrapper 先分配 full 再返回 shard) |
| `gemm_allscatter.shard/full` | intentionally unsupported |
| `gemm_allgather.shard/full` | supported |
| `gemm_reducescatter.full/shard` | supported |
| `allreduce.in_place` | supported |

## 内存模型

- Allocator-first 设计已落地，默认 allocator: `torch_bump`。
- 已实现：`BaseSymmetricAllocator` ABC + `TorchBumpAllocator`，segment metadata，peer export/import descriptors，`AllocatorMemoryModelDescriptor`，`import_external_tensor()` / `as_symmetric()`。
- 未完成：canonical segmented peer import/map；`fd passing + DMA-BUF` 零拷贝 external mapping。

## Canonical 索引

| 内容 | 路径 |
|------|------|
| 总状态文档 | `docs/XTile现状流程_修订版.md` |
| Benchmark 摘要 | `docs/generated/benchmark_runtime_summary.md` |
| Runtime support matrix | `xtile/support.py` |
| 高层 ops | `xtile/ops.py` |
| 内存子系统 | `xtile/memory/` |
| 通信原语 | `xtile/primitives/` |
| Overlap patterns | `xtile/patterns/` |
| 同步原语 | `xtile/sync/` |
| 后端抽象 | `xtile/backends/` |
| 工具/feature gates | `xtile/utils/` |
| GEMM 内核 | `xtile/kernels/` |

## 下一步

1. 内存底座收口：推进到 canonical `segmented peer import/map` + `fd passing / DMA-BUF` 零拷贝。
2. 收缩公共语义面：所有默认路径通过 `xtile.ops.*`，减少 pattern 细节暴露。
3. 扩大验证面：从 `world_size=2 + ctypes_ipc` 扩到更完整的 multiprocess/world-size/transport 矩阵。
