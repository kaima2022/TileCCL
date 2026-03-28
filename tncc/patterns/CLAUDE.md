# tncc/patterns/ - Overlap Pattern Library

## 四种模式（源自 Iris taxonomy）

1. **Bulk-Synchronous** (`bulk_sync.py`) — 无重叠，baseline
2. **Fused Sequential** (`fused_sequential.py`) — 单 kernel，tile 级顺序重叠
3. **Producer-Consumer** (`producer_consumer.py`) — 双 kernel/stream，tile 级并行重叠
4. **WG Specialization** (`wg_specialized.py`) — 单 kernel，SM 级并行重叠

## 结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | `Pattern` 基类，benchmark helpers |
| `contracts.py` | `PatternExecutionSpec`, `PatternTensorSpec`, `resolve_pattern_execution()` |
| `_helpers.py` | 共享 `scatter_tile_to_peer()` 原语 |
| `auto_select.py` | 数据驱动 pattern 选择（M, N/W, K, SM count, tile 数量） |

## 统一接口

```python
pattern.execute(A, B, C, *, spec=... | full_N=... | b_layout="full|shard" | c_layout="full|shard")
```

## ctx 接口

Pattern 通过 `ctx` 对象接收分布式上下文：`ctx.rank`, `ctx.world_size`, `ctx.heap_bases`, `ctx.backend`。

## 执行契约

- 高层入口（推荐）: `tncc.ops.gemm_allscatter(...)` / `build_gemm_allscatter_plan(...)`
- Expert 入口: `pattern.execute(A, B, C, spec=...)`
- 正式支持: `full/full`, `shard/shard`, `full/shard`
- 不支持: `shard/full`

## 优化

- K-loop 双缓冲软件流水线（所有 4 个 pattern）
- Cache modifier: `scatter_tile_to_peer` 可选 `.wt` write-through
- `eviction_policy="evict_last"` 保留 prefetch 数据

## 约束

- 每个 pattern 独立文件，`@triton.jit` kernel 作为静态方法。
- Triton 不支持 `continue`/`break`，使用 if-guard 替代。
- Scatter helper 必须显式接收 layout metadata。
