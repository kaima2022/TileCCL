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
**Phase 0→6 完成**

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
| 4 种 overlap 模式重写 | ✅ | translate_ptr + tile_signal/wait + scatter_tile_to_peer |
| P2P 带宽优化至 95%+ | ⚠️ | cache modifier 效果 <0.2 GB/s，瓶颈非缓存策略 |
| 测试债务清理 | ✅ | test_communication.py 真实 Triton kernel + MultiGPU create_all |
| Collective E2E 验证 | ✅ | allreduce/allgather/broadcast/scatter/reduce_scatter |
| GEMM K-loop 流水线化 | ✅ | 双缓冲 + evict_last，4 pattern 统一 |
| Auto-Select v1 数据驱动 | ✅ | 硬件感知阈值 + CLI auto 模式 |
| Pattern Benchmark harness | ✅ | Iris 6 尺寸 + overlap efficiency |
| AMD MI300X P2P 95%+ | ⬜ | 待硬件 |

### Phase 2 交付物（2026-03-19）
- [x] Cache modifier 支持 (remote_load/store/scatter: .cg, .wt, .cs)
- [x] P2P benchmark 系统性 sweep (block_size × grid × variant × dtype)
- [x] GEMM K-loop 软件流水线化（prefetch + evict_last）
- [x] 4 种 pattern GEMM 循环统一升级
- [x] test_communication.py 重写（真实 Triton kernel 测试）
- [x] MultiGPU 测试修复（create_all 替代 mp.spawn）
- [x] Collective E2E 测试套件 (5 collective × 2 GPU)
- [x] Collective bandwidth benchmark
- [x] Pattern overlap efficiency benchmark (Iris 6 尺寸)
- [x] Auto-select 数据驱动阈值（M/N/K/SM/带宽感知）
- [x] CLI `xtile bench --pattern auto` 集成

### Phase 3 交付物（2026-03-19）
- [x] 运行全部 benchmark（P2P/GEMM/collective/pattern），获取实测数据
- [x] Collective 可扩展性（world_size 上限 9 → 33）
- [x] Auto-select 真实硬件探测（替代硬编码 SM=132, BW=300）
- [x] CLI 全功能整合（`xtile bench p2p|collective|pattern|gemm|all`）
- [x] Profiling 基础设施（ProtonProfiler + OverlapTimeline + CommHeatmap）
- [x] Benchmark runner 脚本（`run_benchmarks.sh`）
- [ ] P2P 带宽优化至 95%+（当前 83%，需 Iris-style kernel 研究）
- [x] GEMM kernel 优化（Iris 对齐重写，15-37% → 74-80%）

### Phase 4 交付物（2026-03-19）
- [x] GEMM kernel Iris 对齐重写（EVEN_K 分离式 K-loop + 模运算索引包裹）
- [x] 编译器 hint（tl.max_contiguous/tl.multiple_of on offsets）
- [x] 增量指针前进（替代每次 K 迭代重计算）
- [x] num_stages=4 软件流水线（最大单项提升 ~50%）
- [x] Block size 自动选择（_select_config: M/N/K → 最优 BM/BN/BK/warps/stages）
- [x] 4 pattern GEMM 循环同步升级（EVEN_K + 分离式 loop + num_stages=4）
- [x] GEMM 达到 ≥ 90% cuBLAS（4096³: 100.7%，launcher 开销消除后匹配 cuBLAS）
- [x] Pattern overlap 重新评估（fused_sequential 1.067× vs bulk_sync）

### Phase 5 交付物（2026-03-19）
- [x] GEMM launcher 开销消除（SM count 缓存 + CUDA events + 预分配输出）
- [x] tl.assume 编译器 hint（stride > 0，gemm + 4 pattern kernel）
- [x] scatter_tile_to_peer 默认 .wt write-through
- [x] Pattern overlap 重测（fused_sequential 首次达到正向 overlap）
- [x] P2P 83% 天花板诊断（Iris 无法运行，translate_ptr 实现一致，确认硬件天花板）
- [ ] GEMM 8192³ 达到 ≥90%（当前 79%，kernel 本身瓶颈，需 PTX-level 优化）
- [ ] Pattern overlap ≥1.3×（2 GPU 限制，理论上限 ~1.15×）

### Phase 6 交付物（2026-03-19）
- [x] 6 幅科研绘图（`figures/` PDF + PNG，Nature/Science 风格）
- [x] GEMM 演进 / P2P 饱和 / 优化瀑布 / Pattern 对比 / 架构图 / Roofline
- [x] CUDA IPC ctypes 调用约定修复（Array → Structure by-value）
- [x] IPC 诊断脚本（`scripts/diagnose_ipc.py`）
- [x] HIP 后端同步修复
- [x] P1-002 根因确认：ctypes bug + 系统 ptrace_scope 双因素

### 已知问题（详见 docs/experiment_log.md）
| 编号 | 问题 | 状态 |
|------|------|------|
| P1-001 | torch.from_blob 不可用 | ✅ 已解决 |
| P1-002 | CUDA IPC ctypes 调用约定 | ✅ 已修复（Structure by-value） |
| P1-002b | CUDA IPC 系统级限制 (ptrace_scope=1) | ✅ PyTorch IPC fallback 绕行 |
| P1-003 | mp.spawn pickle 局部函数 | ✅ 已修复（→ create_all 单进程模式） |
| P1-004 | CUDA backend total_mem 属性名错误 | ✅ 已修复（→ total_memory） |
| P1-005 | Triton 不支持 continue 语句 | ✅ 已修复（→ if != 守卫） |
| P1-006 | bench_gemm.py total_mem 属性名 | ✅ 已修复 |
| P1-007 | Collective benchmark 死锁 | ✅ 已修复（并发 stream） |
| P4-001 | tl.multiple_of on 2D load ptr 性能倒退 | ⚠️ 绕行（仅 offset hint） |
| P4-002 | 小矩阵 (≤2048) GEMM 42-46% | ⚠️ kernel launch 开销限制 |
| P5-001 | 8192³ GEMM 79% — kernel 本身限制 | ⚠️ 需 PTX-level 优化 |
| P5-002 | P2P 83% — Triton-on-H100-NVLink 天花板 | ⚠️ 可能需 inline PTX |
| P5-003 | Pattern benchmark heap_size=512MB 不够大尺寸 | ⚠️ 增大 heap 即可 |

## 性能基线
| 指标 | 实测值 | 目标值 | 状态 |
|------|--------|--------|------|
| P2P read (134MB, f32) | 248.70 GB/s (82.9%) | ≥ 95% | ❌ 硬件天花板 |
| P2P write (134MB, f32) | 248.14 GB/s (82.7%) | ≥ 95% | ❌ 硬件天花板 |
| Collective 归一化带宽 | N/A (tile级原语) | ≥ 90% | ⚠️ 不适用 |
| GEMM vs cuBLAS (4096³ fp16) | **100.7%** | ≥ 90% | ✅ |
| GEMM vs cuBLAS (4096³ bf16) | **90.1%** | ≥ 90% | ✅ |
| GEMM vs cuBLAS (8192³ fp16) | 79.4% | ≥ 90% | ⚠️ kernel 瓶颈 |
| GEMM vs cuBLAS (8192³ bf16) | 79.0% | ≥ 90% | ⚠️ kernel 瓶颈 |
| Pattern overlap (fused_seq) | **1.067×** (8192×3584×14336) | ≥ 1.3× | ⚠️ 2-GPU 限制 |

## 重要约束
- **不** 包装 xSHMEM/NVSHMEM 为不透明字节码
- **不** 引入 NCCL/RCCL 作为运行时依赖
- **不** 在 @triton.jit 函数中使用 Python 对象或动态分派
- Persistent kernel 必须使用 round-robin 调度
