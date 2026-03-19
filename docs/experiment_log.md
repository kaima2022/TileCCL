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
| test_memory/test_symmetric_heap.py::MultiGPU | 3 | ⚠️ 跳过 | mp.spawn 无法 pickle 局部函数；IPC 不可用 |
| test_e2e/test_p2p.py | 7 | ✅ 全通过 | Triton kernel 真实跨 GPU P2P |

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
