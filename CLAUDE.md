# XTile - 下一代跨平台 Tile 通信库

## 项目定位
集各家之长的 Tile 通信库。**第一个同时实现"跨硬件可移植 + 编译器全可见 + 多尺度统一 + 多节点通信"的 tile 通信库。**

## 技术栈
- 主语言：Python + Triton（编译器全可见，不包装 xSHMEM 为不透明字节码）
- 目标硬件：NVIDIA (Hopper/Blackwell) + AMD (CDNA3/CDNA4)
- 通信模型：Symmetric Memory（one-sided），不依赖 NCCL/RCCL
- 内存语义：C++/HIP/CUDA 内存模型（acquire/release/acq_rel），非 SHMEM quiet/wait_until
- 双 API：value-based（寄存器 <-> 远端内存）+ pointer-based（内存 <-> 内存）
- 依赖：triton>=3.0, torch>=2.4
- 测试：pytest + 多GPU benchmark harness

## 架构层次（6 层）
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
2. **Symmetric Memory**：每个 GPU 分配等大堆，translate_ptr 做零拷贝远端访问
3. **C++ 内存模型语义**：atomic 统一使用 sem(relaxed/acquire/release/acq_rel) + scope(block/gpu/sys)
4. **translate_ptr 是热路径**：5 条核心指令，匹配 Iris Listing 1 实现
   - `tl.cast(ptr, tl.uint64)` → uint64 避免符号问题
   - 通过 `tl.pointer_type(tl.int8)` 做 byte-pointer 算术
   - HINT 参数传递 `tl.max_contiguous`/`tl.multiple_of` 向量化信息
5. **SymmetricHeap 双模式**：
   - 单进程模式 (`create_all`): peer access 直接跨 GPU 寻址，推荐单节点
   - 多进程模式: IPC handle 交换，含 fallback
   - 子张量通过 `buffer.narrow().view().reshape()` 创建
6. **信号原语分两类**：
   - tile_signal/tile_wait = 本地原语（同 GPU 内 compute/comm worker 同步，无翻译）
   - tile_atomic_* = 远端原语（跨 GPU，需 target_rank + caller_rank + heap_bases）
7. **Persistent kernel 风格**：所有 GEMM 和 pattern 内核使用 round-robin `range(pid, total, NUM_SMS)`

## 关键参考实现
- **Iris** (github.com/ROCm/iris): 核心算法参考（Listing 1/3/4/5），本地副本 `/home/makai/iris`
- **Triton**: 本地源码 `/home/makai/triton`（API 参考，非运行时依赖）
- **TileScale**: 多尺度抽象 + communication 一等原语设计哲学
- **TileLink/Triton-Distributed**: tile_signal/tile_wait 信号原语启发

## 代码规范
- device-side：@triton.jit，编译期常量用 tl.constexpr
- host-side：兼容 PyTorch tensor，返回类型明确
- 命名：snake_case，公开 API 带 docstring（NumPy-style）
- Backend 层：所有 C API 调用必须检查返回码，不使用 assert
- 错误处理：raise RuntimeError/ValueError，附带上下文
- 资源管理：context manager（__enter__/__exit__），__del__ best-effort

## 实验文档
> 测试结果、benchmark 数据、问题诊断与解决方案详见 **[docs/experiment_log.md](docs/experiment_log.md)**
> 每次任务完成同步更新 CLAUDE.md 和 experiment_log.md

---

## 当前阶段
**Phase 0 完成 → Phase 1 进行中**

### Phase 0 交付物（已完成 2026-03-19）
- [x] 项目脚手架（~8800 行）
- [x] HAL 层（CUDA + HIP backend）
- [x] SymmetricHeap + translate_ptr
- [x] 通信原语（tile_remote_load/store + tile_put/get）
- [x] 同步原语（8 atomic + 4 signal/wait + 1 barrier）
- [x] 4 种 overlap 模式骨架
- [x] Persistent GEMM 内核 + Auto-select v0
- [x] 测试框架 + Iris 审计脚本

### Phase 1（2026-03-19 ~）
**硬件**：2× H100 PCIe 80GB, NV12 NVLink (300 GB/s/dir)
**软件**：PyTorch 2.6.0, CUDA 12.4, Triton 3.2.0

| 任务 | 状态 | 备注 |
|------|------|------|
| translate_ptr 重写（匹配 Iris） | ✅ | tl.cast uint64 + byte-pointer + HINT |
| SymmetricHeap 重写 | ✅ | torch.empty + narrow/view + create_all |
| BackendInterface 扩展 | ✅ | close_ipc_handle / enable_peer_access |
| E2E P2P Triton 测试 | ✅ | 7 项全通过 |
| P2P 带宽 benchmark | ✅ | 248.7 GB/s read (83% peak) |
| 4 种 overlap 模式重写 | ⬜ | 用 translate_ptr 替代手动指针算术 |
| P2P 带宽优化至 95%+ | ⬜ | 当前 83%，需优化 kernel |
| AMD MI300X P2P 95%+ | ⬜ | 待硬件 |

### 已知问题（详见 docs/experiment_log.md）
| 编号 | 问题 | 状态 |
|------|------|------|
| P1-001 | torch.from_blob 不可用 | ✅ 已解决 |
| P1-002 | CUDA IPC 系统级不可用 | ⚠️ 绕行（peer access） |
| P1-003 | mp.spawn pickle 局部函数 | ⚠️ 搁置 |

## 性能基线
| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| P2P read (128MB, f32) | 248.7 GB/s (83%) | ≥ 95% |
| P2P write (128MB, f32) | 248.2 GB/s (83%) | ≥ 95% |
| Collective 归一化带宽 | — | ≥ 90% |
| GEMM+AllScatter vs bulk-sync | — | ≥ 1.3× |

## 重要约束
- **不** 包装 xSHMEM/NVSHMEM 为不透明字节码
- **不** 引入 NCCL/RCCL 作为运行时依赖
- **不** 在 @triton.jit 函数中使用 Python 对象或动态分派
- Persistent kernel 必须使用 round-robin 调度
