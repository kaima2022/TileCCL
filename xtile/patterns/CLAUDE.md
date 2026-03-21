# xtile/patterns/ - Overlap Pattern Library

## 模式 Taxonomy（来自 Iris）
1. Bulk-Synchronous: 无重叠，baseline
2. Fused Sequential: 单 kernel，tile 级顺序重叠
3. Producer-Consumer: 双 kernel/stream，tile 级并行重叠
4. WG Specialization: 单 kernel，CU/SM 级并行重叠

## 参考
- Iris Listing 3: bulk-sync
- Iris Listing 4: fused sequential
- Iris Section 4.1.2: producer-consumer
- Iris Listing 5: WG specialization
- Auto-select 逻辑基于 Iris Section 5 实验观察

## 架构
- `_helpers.py`: 共享 @triton.jit scatter 工具（scatter_tile_to_peer）
- `contracts.py`: 显式 shape/layout contract（PatternExecutionSpec / PatternTensorSpec）
- 每个 pattern 是独立文件，通过 _helpers.py 复用 scatter 逻辑
- scatter 内部调用 translate_ptr 做对称堆指针翻译
- ProducerConsumer / WGSpecialized 使用 tile_signal/tile_wait 做 release-acquire 同步

## ctx 接口
Pattern 通过 ctx 对象接收分布式上下文：
- `ctx.rank`: 当前 GPU rank
- `ctx.world_size`: 总 GPU 数
- `ctx.heap_bases`: `SymmetricHeap.get_heap_bases()` 返回的 (world_size,) int64 张量
- `ctx.backend`: BackendInterface 实例

## 执行契约
- 单 GPU：可继续直接 `pattern.execute(A, B, C)`
- 多 GPU：不要再靠 `B.shape[1]` / `C.shape[1]` 猜 full-vs-shard 语义
- 推荐：
  - 高层入口：`xtile.ops.gemm_allscatter(...)`
  - 显式计划：`xtile.ops.build_gemm_allscatter_plan(...)`
  - shard/shard expert 入口：`xtile.ops.gemm_allscatter_sharded(...)`
  - 低层入口：`pattern.execute(A, B, C, spec=...)`
  - 或显式 `full_N=...`, `b_layout="full|shard"`, `c_layout="full|shard"`
- 当前 API 收口方向：外部单一契约，内部计划执行；`pattern.execute(...)` 继续保留，但定位为 expert/internal surface
- 当前正式支持两种 multi-rank 合同：
  - `B(K, N)` + `C(M, N)` with `full/full`
  - `B(K, N_per_rank)` + `C(M, N_per_rank)` with `shard/shard`
- `full/shard` 或 `shard/full` 混合合同暂不支持，必须由 host-side wrapper 先规范化

## 优化 (Phase 2)
- **K-loop 软件流水线化**：所有 4 个 pattern 的 GEMM 内循环使用双缓冲
  - 预取第一个 K-tile，然后 compute 当前 + prefetch 下一个交替执行
  - 使用 `eviction_policy="evict_last"` 保留 prefetch 数据在 cache
- **Cache modifier 支持**：`scatter_tile_to_peer` 可选 `.wt` write-through
- **auto_select v1**：数据驱动阈值，考虑 M, N/W, K, SM count, tile 数量

## 约束
- 每个 pattern 是独立文件
- 统一接口: `Pattern.execute(A, B, C, *, spec|full_N|b_layout|c_layout)`
- @triton.jit kernel 在 pattern 类内作为静态方法
- Triton 不支持 continue/break，使用 if-guard 替代
- scatter helper 必须显式接收 layout metadata，不能再把“full buffer / shard buffer”语义写死在 helper 里
