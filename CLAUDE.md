# XTile - 下一代跨平台 Tile 通信库

## 项目定位
集各家之长的 Tile 通信库。**第一个同时实现"跨硬件可移植 + 编译器全可见 + 多尺度统一 + 多节点通信"的 tile 通信库。**

## 技术栈
- 主语言：Python + Triton（编译器全可见，不包装 xSHMEM 为不透明字节码）
- 目标硬件：NVIDIA (Hopper/Blackwell) + AMD (CDNA3/CDNA4)
- 通信模型：Symmetric Memory（one-sided, IPC-based），不依赖 NCCL/RCCL
- 内存语义：C++/HIP/CUDA 内存模型（acquire/release/acq_rel），非 SHMEM quiet/wait_until
- 双 API：value-based（寄存器 <-> 远端内存）+ pointer-based（内存 <-> 内存）
- 依赖：triton>=3.0, torch>=2.4
- 测试：pytest + 多GPU benchmark harness

## 架构层次（自上而下，6 层）
```
1. User API         xtile/__init__.py       init(), XTileContext, SymmetricHeap
2. Pattern Library   xtile/patterns/         BulkSync / FusedSeq / PC / WGSpec / AutoSelect
3. Core Primitives   xtile/primitives/       compute / memory / communication（三位一体）
4. Synchronization   xtile/sync/             atomic_* + tile_signal/wait + barrier
5. Memory Mgmt       xtile/memory/           SymmetricHeap + translate_ptr（热路径）
6. HAL               xtile/backends/         HIP (AMD) + CUDA (NVIDIA) via ctypes
```

## 模块依赖拓扑
```
patterns → primitives → sync → memory/translation → backends/base
patterns → kernels/gemm
patterns → sync/primitives (tile_signal/tile_wait)
memory/symmetric_heap → backends/{hip,cuda}
```

## 关键设计决策
1. **纯 Triton 实现**：所有 device-side 函数必须 @triton.jit，保证编译器全可见
2. **Symmetric Memory**：每个 GPU 分配等大堆，IPC 建立后 translate_ptr 做零拷贝远端访问
3. **C++ 内存模型语义**：atomic 操作统一使用 sem(relaxed/acquire/release/acq_rel) + scope(block/gpu/sys)
4. **translate_ptr 是热路径**：5 条指令（2 load + 1 sub + 1 add + 1 bitcast），不可膨胀
5. **信号原语分两类**：
   - tile_signal/tile_wait = 本地原语（同 GPU 内 compute/comm worker 同步，无翻译）
   - tile_atomic_* = 远端原语（跨 GPU，需 target_rank + caller_rank + heap_bases）
6. **Persistent kernel 风格**：所有 GEMM 和 pattern 内核使用 round-robin `range(pid, total, NUM_SMS)`

## 关键参考实现
- **Iris** (github.com/ROCm/iris): 核心算法参考。Listing 1(translate), 3(bulk-sync), 4(fused), 5(WG-spec)
- **TileScale**: 多尺度抽象 + communication 一等原语设计哲学
- **TileLink/Triton-Distributed**: tile_signal/tile_wait 信号原语启发
- **ThunderKittens**: 16×16 tile 尺寸 + Load-Store-Compute-Finish 流水线

## 代码规范
- 所有 device-side 函数：@triton.jit，参数类型用 tl.constexpr 标注编译期常量
- 所有 host-side API：兼容 PyTorch tensor，返回类型明确
- 命名：snake_case，模块级公开 API 带 docstring（NumPy-style）
- Backend 层：所有底层 C API 调用必须检查返回码，不使用 assert 做运行时检查
- 错误处理：raise RuntimeError/ValueError，附带上下文（rank、指针地址、尺寸等）
- 资源管理：支持 context manager（__enter__/__exit__），__del__ 做 best-effort 清理

## 当前阶段
**Phase 0 完成 → Phase 1 进行中**：核心原语层实现

### Phase 0 交付物（已完成 2026-03-19）
- [x] 完整项目脚手架（51 文件，~8800 行）
- [x] HAL 层：HIP + CUDA backend（ctypes 封装，IPC handle 交换）
- [x] SymmetricHeap：bump allocator + IPC 建立 + heap_bases 构建
- [x] translate_ptr：device-side 5指令翻译 + host-side PointerTranslator
- [x] 通信原语：tile_remote_load/store + tile_put/get
- [x] 同步原语：8 atomic + 4 signal/wait + 1 barrier（全部 @triton.jit）
- [x] 4 种 overlap 模式骨架：BulkSync / FusedSeq / PC / WGSpec
- [x] 标准 Triton persistent GEMM 内核
- [x] Auto-select 引擎 v0
- [x] CI/CD：GitHub Actions（lint + AMD/NVIDIA self-hosted）
- [x] 测试框架：pytest fixtures + benchmark harness
- [x] Iris 审计脚本：scripts/audit_iris.py

### Phase 1 进展（2026-03-19 ~）
**硬件：2× NVIDIA H100 PCIe 80GB, NV12 NVLink (300 GB/s/dir)**
**软件：PyTorch 2.6.0, CUDA 12.4, Triton 3.2.0**

已完成：
- [x] translate_ptr 重写：匹配 Iris 实现（tl.cast uint64 + byte-pointer 算术 + 向量化 hint）
- [x] SymmetricHeap 重写：torch.empty 分配 + buffer.narrow().view() 子张量
- [x] 新增 `SymmetricHeap.create_all()` 单进程多 GPU 模式（peer access，无需 IPC）
- [x] 多进程 IPC 模式保留（含 fallback）
- [x] BackendInterface 新增 close_ipc_handle / enable_peer_access
- [x] 7 项 Triton P2P 端到端测试全部通过（read/write/roundtrip/bidirectional/f16/identity）
- [x] P2P 带宽 benchmark：248.7 GB/s read, 248.2 GB/s write（83% NVLink 峰值）

进行中：
- [ ] 4 种 overlap 模式重写（使用 translate_ptr + heap_bases 替代手动指针算术）
- [ ] P2P 带宽优化至 95%+ 峰值（当前 83%）

已知限制：
- 当前系统 CUDA IPC (cudaIpcOpenMemHandle) 不可用，使用 peer access 模式代替
- 多节点场景需要修复 IPC 或引入 DMA-BUF 支持

### Phase 1 目标
- [ ] 在 AMD MI300X 上 P2P 原语达到 95%+ 归一化带宽
- [x] 在 NVIDIA H100 上 P2P 原语初步可用（已达 83% 峰值带宽）
- [x] Microbenchmark 矩阵完整（P2P read/write × f16/f32 × 1~128MB × BS 1024/4096）
- [x] Symmetric Heap 多 GPU 端到端验证（7 项 Triton kernel 测试）

## 性能目标
| 指标 | 目标值 |
|------|--------|
| P2P 归一化带宽 | ≥ 95%（对比理论峰值） |
| Collective 归一化带宽 | ≥ 90% |
| GEMM+AllScatter vs bulk-sync | ≥ 1.3× 加速 |
| NVIDIA vs AMD 性能差 | ≤ 15% |
| 通信 API 行数开销 | ≤ 5 行/kernel |

## 重要约束
- **不** 包装 xSHMEM/NVSHMEM 为不透明字节码
- **不** 引入 NCCL/RCCL 作为运行时依赖
- **不** 在 @triton.jit 函数中使用 Python 对象或动态分派
- 测试必须同时覆盖 NVIDIA 和 AMD（可分阶段，用 pytest markers）
- Persistent kernel 必须使用 round-robin 调度（不用 block-partition）
