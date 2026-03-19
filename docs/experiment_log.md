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
