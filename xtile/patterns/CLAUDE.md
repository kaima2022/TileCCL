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

## 约束
- 每个 pattern 是独立文件
- 统一接口: Pattern.execute(A, B, C)
- @triton.jit kernel 在 pattern 类内作为静态方法
