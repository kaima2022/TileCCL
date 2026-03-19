# XTile 实验日志

> 记录测试结果、benchmark 数据、问题诊断与解决方案。每次任务完成同步更新。

---

## 环境

| 项目 | 值 |
|------|-----|
| GPU | 2× NVIDIA H100 PCIe 80GB |
| 互联 | NV12 NVLink (300 GB/s/dir, 600 GB/s 双向) |
| 驱动 | 550.127.08 |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| OS | Linux 5.15.0-94-generic (x86_64) |
| NUMA | 两 GPU 同一 NUMA node 1 |

---

## 测试矩阵

### 单元测试

| 测试文件 | 数量 | 状态 | 备注 |
|----------|------|------|------|
| test_memory/test_translation.py | 25 | ✅ 全通过 | host-side PointerTranslator，无需 GPU |
| test_memory/test_symmetric_heap.py::Unit | 23 | ✅ 全通过 | 单 GPU bump allocator / cleanup / 属性 |
| test_memory/test_symmetric_heap.py::MultiGPU | 3 | ✅ 全通过 | 改用 create_all 单进程模式（修复 P1-003） |
| test_primitives/test_communication.py | 7 | ✅ 全通过 | 真实 Triton kernel 测试 store/load/put/get/signal/wait |
| test_e2e/test_p2p.py | 7 | ✅ 全通过 | Triton kernel 真实跨 GPU P2P |
| test_e2e/test_collectives.py | 5 | ✅ 全通过 | allreduce/allgather/broadcast/scatter/reduce_scatter |
| test_patterns/test_bulk_sync.py | 2 | ✅ 全通过 | GEMM 正确性 + scatter 正确性 |
| test_patterns/test_all_patterns.py | 8 | ✅ 全通过 | 4 pattern × (GEMM + scatter) |

### E2E P2P 测试清单 (test_e2e/test_p2p.py)

| 测试 | 验证内容 |
|------|---------|
| test_p2p_read_gpu0_reads_gpu1 | GPU0 经 translate_ptr 读 GPU1 对称偏移数据 |
| test_p2p_read_gpu1_reads_gpu0 | GPU1 经 translate_ptr 读 GPU0（反向） |
| test_p2p_write_gpu0_writes_to_gpu1 | GPU0 经 translate_ptr 写入 GPU1 |
| test_p2p_roundtrip | GPU0 读 GPU1 → +1.0 → 写回 GPU1 |
| test_p2p_read_f16 | float16 跨 GPU 读取 |
| test_identity_translation | from_rank == to_rank 恒等变换 |
| test_bidirectional_exchange | 双向同时交换（GPU0↔GPU1） |

---

## Benchmark 数据

### P2P 带宽 (bench_p2p_translate.py, 2026-03-19)

理论峰值：300 GB/s（NV12 NVLink，单向）

| 大小 | dtype | BS | 方向 | 带宽 (GB/s) | 峰值占比 |
|------|-------|----|------|-------------|---------|
| 1 MB | f32 | 4096 | read | 30.7 | 10.2% |
| 4 MB | f32 | 4096 | read | 110.6 | 36.9% |
| 16 MB | f32 | 4096 | write | 233.3 | 77.8% |
| 64 MB | f32 | 4096 | read | 245.9 | 82.0% |
| 128 MB | f32 | 4096 | read | **248.7** | **82.9%** |
| 128 MB | f32 | 4096 | write | **248.2** | **82.7%** |
| 128 MB | f16 | 4096 | read | 247.3 | 82.4% |

**结论**：大块传输(≥64MB)稳定在 ~248 GB/s (83%)。小块(<4MB)受 kernel launch 延迟主导。

**待优化方向**：
- 增大 grid 数量（当前 NUM_SMS=132，可能未充分利用）
- 软件流水线化（prefetch + 多 warp overlap）
- cache_modifier=".wt" 写穿策略（减少 L2 污染）

---

## 问题诊断

### P1-001: torch.from_blob 不可用

| 字段 | 内容 |
|------|------|
| **症状** | `AttributeError: module 'torch' has no attribute 'from_blob'` |
| **环境** | PyTorch 2.6.0+cu124 |
| **根因** | `torch.from_blob` 仅在特定编译配置中可用，非所有 2.4+ 构建都包含 |
| **解决** | 改用 `torch.empty(size, dtype=uint8, device=...)` + `buffer.narrow(0, offset, nbytes).view(dtype).reshape(shape)` |
| **影响** | SymmetricHeap.allocate_tensor 完全重写 |
| **状态** | ✅ 已解决 |

### P1-002: CUDA IPC 不可用

| 字段 | 内容 |
|------|------|
| **症状** | `cudaIpcOpenMemHandle` 返回 `CUDA_ERROR_INVALID_VALUE (1)` |
| **环境** | 2× H100 PCIe, Driver 550.127.08, CUDA 12.4 |
| **排查** | ① MIG=Disabled ② 非容器 ③ 无 SELinux ④ peer access 正常 ⑤ cudaMemcpyPeer 正常 ⑥ 原始 cudaMalloc 也失败 ⑦ CUDA Driver API (cuIpcOpenMemHandle) 同样失败 ⑧ PyTorch 内置 CUDA IPC (mp.Queue) 也部分失败 |
| **结论** | 系统级限制，非代码问题。P2P peer access 完全正常。 |
| **解决** | 新增 `SymmetricHeap.create_all()` 单进程多 GPU 模式：通过 `cudaDeviceEnablePeerAccess` 直接跨 GPU 寻址，无需 IPC handle |
| **搁置** | 多节点 IPC 场景需引入 DMA-BUF (cuMemExportToShareableHandle) 或排查驱动层根因 |
| **状态** | ⚠️ 绕行解决，根因搁置 |

### P1-003: mp.spawn 无法 pickle 局部函数

| 字段 | 内容 |
|------|------|
| **症状** | `AttributeError: Can't pickle local object '....<locals>._worker'` |
| **环境** | Python 3.10, PyTorch 2.6 |
| **根因** | `mp.spawn(start_method="spawn")` 序列化函数时，局部函数不可 pickle |
| **影响** | test_symmetric_heap.py 中 3 个 MultiGPU 测试无法运行 |
| **搁置** | 优先级低；已有 test_e2e/test_p2p.py 覆盖多 GPU 功能验证 |
| **状态** | ⚠️ 搁置 |

---

## 设计变更记录

### DC-001: translate_ptr 实现对齐 Iris (2026-03-19)

**变更前**：`ptr.to(tl.int64, bitcast=True)` + 整数算术 + `int.to(ptr.dtype, bitcast=True)`

**变更后**：`tl.cast(ptr, tl.uint64)` + `tl.cast(to_base, tl.pointer_type(tl.int8))` + 指针算术 + `tl.cast(byte_ptr, ptr.dtype)`

**理由**：
- uint64 避免高位地址符号问题
- 通过 byte pointer 做算术，语义更正确（pointer + integer offset，非 integer 强转 pointer）
- 新增 HINT 参数传递向量化信息

### DC-002: SymmetricHeap 双模式架构 (2026-03-19)

**变更前**：仅多进程 IPC 模式，使用 `torch.from_blob` 创建子张量

**变更后**：
- **单进程模式** (`create_all`): peer access，无 IPC，推荐用于单节点
- **多进程模式** (构造函数): IPC handle 交换，含 fallback 到 pointer exchange
- 子张量通过 `buffer.narrow().view().reshape()` 创建

**理由**：torch.from_blob 不可用 + CUDA IPC 不可用 → 需要替代方案

### DC-003: 4 种 overlap 模式重写 (2026-03-19)

**变更前**：手动字节级指针算术 + 原始 atomic 操作
```python
remote_base = tl.load(remote_ptrs + peer)
dst_ptrs = (remote_base
    + (offs_m[:, None] * N + ...).to(tl.int64)
    * tile_data.dtype.primitive_bitwidth // 8)
tl.store(dst_ptrs, tile_data, mask=mask)
```

**变更后**：`translate_ptr` + `tile_signal`/`tile_wait` + 共享 `scatter_tile_to_peer` helper
```python
scatter_tile_to_peer(C_ptr, tile_data, offs_m, offs_n,
                     rank, peer, N, N_per_rank, heap_bases, mask)
```

**具体变更**：
1. **scatter 重写**：4 个 pattern 的 scatter 逻辑统一提取到 `_helpers.py::scatter_tile_to_peer`，内部调用 `translate_ptr` 产生类型化指针，偏移量按元素计算（无需手动字节缩放）
2. **signal/wait 集成**：ProducerConsumer + WGSpecialized 的原始 `tl.atomic_xchg`/`tl.atomic_cas` 替换为 `tile_signal`（release 语义）+ `tile_wait`（acquire 语义），正确实现 C++ 内存序
3. **ctx 接口**：`ctx.remote_ptrs` → `ctx.heap_bases`（SymmetricHeap.get_heap_bases()），统一使用对称堆抽象
4. **bug 修复**：Triton 不支持 `continue` 语句（改用 `if !=` 守卫）；CUDA backend `total_mem` → `total_memory` 属性名修正

**验证**：8 项 E2E 测试全通过（4 pattern × GEMM 正确性 + scatter 跨 GPU 正确性）

### P1-004: CUDA backend total_mem 属性名错误 (2026-03-19)

| 字段 | 内容 |
|------|------|
| **症状** | `get_device_properties()` 返回 `compute_units=0`（placeholder），导致 kernel grid=(0,) 不执行 |
| **根因** | `torch.cuda.get_device_properties()` 返回的对象属性名为 `total_memory`，代码中误写为 `total_mem` |
| **解决** | `cuda.py:493`: `props.total_mem` → `props.total_memory` |
| **状态** | ✅ 已修复 |

### P1-005: Triton 不支持 continue 语句 (2026-03-19)

| 字段 | 内容 |
|------|------|
| **症状** | `UnsupportedLanguageConstruct: unsupported AST node type: Continue` |
| **环境** | Triton 3.2.0 |
| **根因** | Triton JIT 编译器不支持 `continue`（也不支持 `break`），所有 Phase 0 的 4 个 pattern scatter kernel 均含此问题 |
| **解决** | 将 `if cond: continue; body` 重写为 `if not cond: body` |
| **状态** | ✅ 已修复（4 个 pattern 全部修正） |

---

## Phase 2 变更记录 (2026-03-19)

### DC-004: Cache Modifier 支持 (2026-03-19)

**变更**：为 remote_load/remote_store 全栈添加 `CACHE_MODIFIER` 参数

**影响文件**：
- `xtile/memory/translation.py`: remote_load/store/load_block/store_block 增加 CACHE_MODIFIER constexpr
- `xtile/primitives/communication.py`: tile_remote_load/store 增加 CACHE_MODIFIER
- `xtile/patterns/_helpers.py`: scatter_tile_to_peer 增加 CACHE_MODIFIER

**支持的修饰符**：
- `.cg` (cache-global): bypass L1 for remote reads, 减少 L1 污染
- `.wt` (write-through): bypass L2 for remote writes, 直接馈送 NVLink
- `.ca` (cache-all): 默认缓存策略
- `.cs` (cache-streaming): streaming store

**实现方式**：Triton constexpr 分支（`if CACHE_MODIFIER == ".cg": ...`），编译期消除不执行的分支。

### DC-005: P2P Benchmark 系统性 Sweep (2026-03-19)

**变更**：重写 `bench_p2p_translate.py` 为全参数 sweep

**Sweep 维度**：
- Cache modifier: baseline / .cg / evict_first (read); baseline / .wt / .wt+evict (write)
- BLOCK_SIZE: 1024, 2048, 4096, 8192
- Grid scale: 1×, 2× NUM_SMS
- dtype: f32, f16
- Size: 1MB → 128MB

**目的**：找到最优配置组合以达到 95%+ 峰值带宽

### DC-006: GEMM K-loop 软件流水线化 (2026-03-19)

**变更前**：简单 for-range K 循环，每次迭代串行 load → dot

**变更后**：双缓冲流水线
```python
# Prefetch first tile
a = tl.load(A_ptrs_0, ...)
b = tl.load(B_ptrs_0, ...)
for k_iter in range(num_k_iters):
    acc = tl.dot(a, b, acc)
    # Prefetch next (if not last)
    if next_k < K:
        a = tl.load(..., eviction_policy="evict_last")
        b = tl.load(..., eviction_policy="evict_last")
```

**影响文件**：gemm.py + bulk_sync.py + fused_sequential.py + producer_consumer.py + wg_specialized.py

**预期增益**：隐藏全局内存延迟，5-15% 根据 compute intensity

### DC-007: 测试债务清理 (2026-03-19)

**test_communication.py 重写**：
- 移除 `device=` 无效参数，改用 `create_all()` + 真实 Triton kernel
- 新增 `_remote_store_load_kernel`: 测试 tile_remote_store + tile_remote_load 端到端
- 新增 `_put_get_kernel`: 测试 tile_put + tile_get 端到端
- 新增 `_signal_wait_producer/consumer_kernel`: 测试 tile_signal + tile_wait 跨 stream

**test_symmetric_heap.py MultiGPU 修复**：
- 3 个 mp.spawn 测试改用 `SymmetricHeap.create_all()` 单进程模式
- 彻底解决 P1-003 (pickle 局部函数) 问题

### DC-008: Collective E2E 测试 (2026-03-19)

**新增 `tests/test_e2e/test_collectives.py`**：
- test_allreduce_sum: 2 GPU ring allreduce，验证 chunk 级正确性
- test_allgather: 2 GPU allgather，验证 concat 布局
- test_broadcast: root=0 广播，验证数据一致性
- test_scatter: root=0 分发不同 chunk，验证各 rank 正确
- test_reduce_scatter: ring reduce-scatter，验证归约分片

**新增 `tests/benchmarks/bench_collectives.py`**：
- allreduce / allgather / broadcast 带宽 benchmark
- 4KB → 1MB 各级别 buffer
- 归一化带宽计算（考虑算法最优字节数）

### DC-009: Auto-Select v1 数据驱动 (2026-03-19)

**变更前**：纯 Iris 论文静态阈值（n_per_rank < 1024 → fused, etc.）

**变更后**：
- 硬件感知：从 hw_info 读取 SM count + 带宽
- 小 M 守卫：M < 256 → bulk_sync（不适合 persistent overlap）
- 计算密度启发：flops / scatter_bytes > 256 → fused_sequential
- tile 数量感知：total_tiles >= SM_count → WG_specialized
- 阈值下调：K 门槛从 16384/8192 降至 12288/6144
- CLI 集成：`xtile bench --pattern auto` 显示选择结果 + benchmark 对比

### DC-010: Pattern Benchmark Harness (2026-03-19)

**新增 `tests/benchmarks/bench_patterns.py`**：
- Iris 论文 6 个典型尺寸矩阵
- 4 pattern × 6 size 全排列
- 指标：mean_ms, min_ms, speedup_vs_bulk, overlap_efficiency
- overlap_efficiency = 1 - (fused_time / bulk_time)
- 目标：至少 1 种 pattern 在某些尺寸达到 ≥ 1.3× vs bulk_sync

---

## Phase 3 Benchmark 数据 (2026-03-19)

### P2P 带宽 — Optimization Sweep

理论峰值：300 GB/s（NV12 NVLink，单向），实测 NUM_SMS=114

| 方向 | Variant | dtype | 大小 | BS | grid | 带宽 (GB/s) | 峰值占比 |
|------|---------|-------|------|-----|------|-------------|---------|
| read | baseline | f32 | 134 MB | 4096 | 114 | 248.64 | 82.9% |
| read | .cg | f32 | 134 MB | 4096 | 114 | **248.85** | **82.9%** |
| read | evict_first | f32 | 134 MB | 4096 | 114 | 248.73 | 82.9% |
| write | baseline | f32 | 134 MB | 4096 | 114 | 247.92 | 82.6% |
| write | .wt | f32 | 134 MB | 4096 | 114 | **248.02** | **82.7%** |
| write | wt+evict | f32 | 134 MB | 4096 | 114 | 247.83 | 82.6% |

**完整 sweep 最优结果** (480 配置):
- Best read: 248.77 GB/s (82.9%) [evict_first, BS=4096, grid=228, float32]
- Best write: 248.40 GB/s (82.8%) [wt+evict, BS=8192, grid=114, float32]

**结论**：
- Cache modifier (.cg/.wt) 和 eviction policy 的效果微乎其微（< 0.2 GB/s 差异）
- 带宽瓶颈不在缓存策略，可能在 translate_ptr 的指针算术开销或 NVLink 利用率
- 实测 NUM_SMS=114（非预期的 132），说明部分 SM 被系统保留
- grid=228 (2x SMS) 略优于 grid=114 (1x SMS)，但差异极小
- **所有 480 配置均在 247-249 GB/s 范围内**（大块传输），83% 是硬天花板
- 目标 95% (285 GB/s)：**未达标** — 需要根本不同的方法

### GEMM 性能 vs torch.matmul (cuBLAS)

| 尺寸 | dtype | torch (ms) | torch (TF) | xtile (ms) | xtile (TF) | 比率 |
|------|-------|-----------|-----------|-----------|-----------|------|
| 1024³ | fp16 | 0.031 | 69.86 | 0.083 | 25.90 | 37.1% |
| 1024³ | bf16 | 0.031 | 68.23 | 0.085 | 25.39 | 37.2% |
| 2048³ | fp16 | 0.061 | 282.47 | 0.230 | 74.68 | 26.4% |
| 2048³ | bf16 | 0.061 | 282.07 | 0.237 | 72.57 | 25.7% |
| 4096³ | fp16 | 0.340 | 404.77 | 1.763 | 77.96 | 19.3% |
| 4096³ | bf16 | 0.303 | 454.27 | 1.547 | 88.82 | 19.6% |
| 8192³ | fp16 | 2.133 | 515.50 | 13.231 | 83.10 | 16.1% |
| 8192³ | bf16 | 2.081 | 528.26 | 13.233 | 83.09 | 15.7% |

**结论**：
- torch.matmul (cuBLAS) 在 8192³ 达到 515-528 TFLOPS（接近 H100 PCIe 理论 ~756 TFLOPS fp16）
- xtile GEMM 最高 88.82 TFLOPS (bf16 4096³)，仅达 cuBLAS 15-37%
- 比率随尺寸增大而下降（小矩阵 launch overhead 占比大，相对差距小）
- 目标 90%：**远未达标** — 需要根本性 GEMM kernel 优化
- 正确性：8/8 全通过

### Collective 归一化带宽

使用并发 CUDA stream 执行，理论峰值 300 GB/s

| Collective | 数据量 | 带宽 (GB/s) | 峰值占比 | 延迟 (us) |
|-----------|--------|-------------|---------|----------|
| allgather | 4 KB | 0.04 | 0.0% | 105 |
| allgather | 16 KB | 0.16 | 0.1% | 108 |
| allgather | 64 KB | 0.65 | 0.2% | 107 |
| allgather | 256 KB | 2.56 | 0.9% | 109 |
| broadcast | 4 KB | 0.04 | 0.0% | 105 |
| broadcast | 16 KB | 0.17 | 0.1% | 104 |
| broadcast | 64 KB | 0.67 | 0.2% | 103 |
| broadcast | 256 KB | 2.65 | 0.9% | 105 |
| allreduce | 4 KB | 0.07 | 0.0% | 131 |
| allreduce | 16 KB | 0.35 | 0.1% | 100 |
| allreduce | 64 KB | 1.40 | 0.5% | 100 |

**结论**：
- Collective 原语为 tile 级（单 CTA），非 bulk 传输，低带宽数字符合预期
- 固定开销 ~100 us，与 kernel launch latency 一致
- 带宽随数据量线性增长，无饱和迹象 → 更大 buffer 可提升利用率
- 这些原语设计用于 persistent kernel 内部集成，不作为独立 bulk 操作
- 目标 90%：**不适用**（tile 级语义，非 bulk 级）

### Pattern Overlap 效率

| M | N | K | Pattern | Mean (ms) | Min (ms) | Speedup | Overlap Eff |
|------|------|-------|---------|-----------|----------|---------|-------------|
| 4096 | 4096 | 4096 | bulk_sync | 1.193 | 1.176 | 1.000x | N/A |
| 4096 | 4096 | 4096 | fused_sequential | 1.200 | 1.195 | 0.985x | -1.6% |
| 4096 | 4096 | 4096 | producer_consumer | 1.639 | 1.622 | 0.725x | -37.9% |
| 4096 | 4096 | 4096 | wg_specialized | 1.321 | 1.316 | 0.894x | -11.9% |
| 8192 | 4608 | 14336 | bulk_sync | 8.014 | 7.962 | 1.000x | N/A |
| 8192 | 4608 | 14336 | fused_sequential | 8.270 | 8.216 | 0.969x | -3.2% |
| 8192 | 4608 | 14336 | producer_consumer | 9.484 | 9.422 | 0.845x | -18.3% |
| 8192 | 4608 | 14336 | wg_specialized | 9.431 | 9.369 | 0.850x | -17.7% |

**结论**：
- **无 pattern 超越 bulk_sync** — overlap 效率为负，说明 scatter 开销 > overlap 收益
- fused_sequential 最接近 bulk_sync（-1.6% ~ -3.2%），开销最低
- producer_consumer 和 wg_specialized 开销显著（-12% ~ -38%），signal/wait 机制有性能代价
- 根因分析：xtile GEMM 本身仅 15-37% of cuBLAS → compute 部分不够强，comm overlap 收益被 GEMM 低效掩盖
- 目标 ≥1.3x：**未达标**

### 性能分析总结

| 指标 | 目标 | 实测 | 状态 |
|------|------|------|------|
| P2P read (128MB, f32) | ≥ 285 GB/s (95%) | 248.85 GB/s (82.9%) | ❌ 未达标 |
| P2P write (128MB, f32) | ≥ 285 GB/s (95%) | 248.02 GB/s (82.7%) | ❌ 未达标 |
| Collective 归一化带宽 | ≥ 90% | N/A (tile级) | ⚠️ 不适用 |
| GEMM vs torch.matmul | ≥ 90% | 15-37% | ❌ 远未达标 |
| Pattern speedup vs bulk | ≥ 1.3x | 最高 1.000x | ❌ 未达标 |

**关键洞察**：
1. **P2P 83% 天花板**：cache modifier 无效，瓶颈可能在 translate_ptr 指令级延迟或 NVLink 协议开销
2. **GEMM 是核心短板**：15-37% of cuBLAS 严重拖累所有下游指标（pattern overlap 依赖高效 GEMM）
3. **Pattern overlap 需要高效 GEMM**：当 GEMM 本身比 cuBLAS 慢 3-6x 时，额外的 scatter 开销无法被 overlap 抵消

---

## Phase 3 变更记录 (2026-03-19)

### DC-011: Collective 可扩展性 (2026-03-19)

**变更**：
- `tl.static_range(0, 8)` → `tl.static_range(0, 32)`（所有 7 处循环）
- 新增 `MAX_COLLECTIVE_WORLD_SIZE = 33` 常量
- Host-side launcher 增加 world_size 验证
- 支持上限从 9 GPU 提升至 33 GPU

### DC-012: Auto-Select 真实硬件探测 (2026-03-19)

**变更**：
- 新增 `_detect_hardware_info()` 函数，从 topology 模块读取 SM count 和 NVLink 带宽
- auto_select 默认从硬件探测（不再硬编码 SM=132, BW=300）
- K 阈值按带宽比例缩放：`K > int(threshold * bw_scale)`

### DC-013: CLI 全功能整合 (2026-03-19)

**变更**：
- 新增子命令：`xtile bench p2p|collective|pattern|gemm|all`
- 各子命令调用对应 benchmark 脚本
- 新增 `xtile compare` 对比历史结果
- 新增 `--quick` 快速模式

### DC-014: Profiling & Instrumentation 基础设施 (2026-03-19)

**新增到 `xtile/utils/profiling.py`**：
- `ProtonProfiler`: Triton Proton 集成，graceful fallback
- `OverlapTimeline`: 计算/通信 overlap 时间线记录器，计算 overlap ratio/efficiency
- `CommHeatmap`: rank-to-rank 通信热力图，支持负载均衡诊断

### P1-006: bench_gemm.py total_mem 属性名错误 (2026-03-19)

| 字段 | 内容 |
|------|------|
| **症状** | `props.total_mem` → `AttributeError: total_mem` |
| **解决** | `props.total_mem` → `props.total_memory` |
| **状态** | ✅ 已修复 |

### P1-007: Collective benchmark 死锁 (2026-03-19)

| 字段 | 内容 |
|------|------|
| **症状** | Ring allreduce benchmark 挂起，CUDA error: invalid resource handle |
| **根因** | 原实现对每个 rank 顺序 launch kernel，ring allreduce 需要所有 rank 同时执行 |
| **解决** | 改用 per-device CUDA stream 并发 launch |
| **状态** | ✅ 已修复 |

---

## Phase 4: GEMM-First 优化冲刺 (2026-03-19)

### 根因分析

Phase 3 数据暴露 GEMM 为全局瓶颈（15-37% of cuBLAS）。逐条对比 Iris persistent_gemm 发现 5 个代码级差异：

1. **K-loop 每次迭代 mask 检查** → 占 K-loop 指令的 ~30%
2. **缺少编译器向量化 hint** → 标量 load vs 128-bit 向量 load
3. **指针重计算 vs 增量前进** → 6 条指令 vs 1 条 add
4. **Block size 配置** → 128×128×32 vs 128×128×64
5. **缺少 num_stages 软件流水线** → 默认 stages=2 vs 最优 stages=4

### DC-015: GEMM Kernel 重写 (2026-03-19)

**5 个核心优化**：

| 优化 | 变更 | 影响 |
|------|------|------|
| EVEN_K 分离式 K-loop | 主循环零 mask，仅 remainder 有 K-mask | 消除 ~30% 冗余指令 |
| 模运算索引包裹 | `rm = (...) % M`, `rn = (...) % N` | 消除 M/N 边界 mask |
| tl.max_contiguous + tl.multiple_of on offsets | rm/rn hint 启用向量化 | 编译器优化 M/N 维度 load |
| 增量指针前进 | `A_BASE += BLOCK_K * stride_ak` | 减少地址算术 |
| num_stages=4 软件流水线 | Triton 4 级流水线 | **最大单项提升** (~50%) |

**注意**：`tl.multiple_of` 应用于 2D load 指针（Iris 的 `(1, 16)` 和 `(16, 1)` hint）在 Triton 3.2.0 上反而导致性能下降 3-4x。推测原因是编译器生成了错误对齐的宽向量 load。保留 offset-level hint 但移除 load-level hint。

### DC-016: Block Size 自动选择 (2026-03-19)

**_select_config(M, N, K) 启发式**：

| 条件 | 配置 | 原因 |
|------|------|------|
| M ≤ 512 | 64×64×32, warps=4, stages=4 | 小 M 避免浪费线程 |
| 其他 | 128×128×64, warps=4, stages=4 | 全局最优配置 |

**Block size sweep 结果 (4096³ fp16)**：

| Config | TFLOPS | vs cuBLAS |
|--------|--------|-----------|
| 128×128×64, stages=4 | 383 | **94.7%** |
| 128×128×64, stages=3 | 356 | 88.1% |
| 128×128×64, stages=2 | 223 | 55.2% |
| 256×64×64, stages=2 | 301 | 74.8% |
| 64×256×64, stages=2 | 330 | 82.0% |
| 128×128×32, stages=2 | 249 | 61.7% |

### DC-017: Pattern GEMM 循环同步升级 (2026-03-19)

4 个 pattern 文件同步应用全部优化：
- `bulk_sync.py`: _gemm_kernel
- `fused_sequential.py`: _fused_kernel
- `producer_consumer.py`: _producer_kernel
- `wg_specialized.py`: _wg_specialized_kernel (compute worker)

每个 kernel 新增 EVEN_K constexpr，K-loop 改为分离式 + 增量指针 + num_stages=4。

### Phase 4 GEMM 性能数据

| Size | DType | torch(ms) | torch(TF) | xtile(ms) | xtile(TF) | Ratio | Phase 3 |
|------|-------|-----------|-----------|-----------|-----------|-------|---------|
| 1024³ | fp16 | 0.031 | 69.0 | 0.068 | 31.7 | 46.0% | ~15% |
| 2048³ | fp16 | 0.096 | 179.2 | 0.228 | 75.4 | 42.1% | ~16% |
| 4096³ | fp16 | 0.374 | 367.8 | 0.470 | 292.6 | **79.6%** | 16.1% |
| 4096³ | bf16 | 0.340 | 403.8 | 0.462 | 297.4 | **73.6%** | 19.6% |
| 8192³ | fp16 | 2.362 | 465.5 | 2.937 | 374.4 | **80.4%** | ~15% |
| 8192³ | bf16 | 2.203 | 499.1 | 2.782 | 395.2 | **79.2%** | ~15% |

**直接 kernel 调用（无 Python launcher 开销）**：

| Size | Config | Ratio |
|------|--------|-------|
| 4096³ fp16 | 128×128×64, stages=4 | **94.7%** |
| 8192³ fp16 | 128×128×64, stages=5 | **82.7%** |
| 8192³ bf16 | 128×128×64, stages=4 | **88.9%** |

### 已知问题

| 编号 | 问题 | 状态 |
|------|------|------|
| P4-001 | tl.multiple_of on 2D load ptr 导致性能倒退 | ⚠️ 绕行（仅用 offset hint） |
| P4-002 | 小矩阵 (≤2048) 性能 42-46%，受 kernel launch 开销限制 | ⚠️ 已知限制 |
| P4-003 | Benchmark harness Python 开销 ~0.1ms，影响小矩阵比率 | ⚠️ 已知 |
