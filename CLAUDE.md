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
1. User API         xtile/__init__.py       init(), init_local(), XTileContext, SymmetricHeap, xtile.ops
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
8. **禁止手工拼装伪 ctx**：CLI / tests / benchmarks 应复用 `XTileContext`，不得再用 `_Ctx`/`DistCtx` 临时对象绕过真实 runtime 契约
9. **Pattern 多 GPU 调用必须显式声明 contract**：
   - `pattern.execute(...)` 在多 GPU 下不应再靠 `B.shape[1]` / `C.shape[1]` 猜 full-vs-shard 语义
   - 推荐默认入口：`xtile.ops.gemm_allscatter(...)`
   - API 收口方向：外部单一契约，内部计划执行（`build_*_plan(...)`）
   - 若直接调用 pattern，则显式传 `spec=...` 或 `full_N + b_layout + c_layout`
10. **benchmark / figures 共享结构化数据源**：
   - pattern / GEMM / P2P benchmark 分别输出到 `figures/data/pattern_overlap_latest.json`、`gemm_latest.json`、`p2p_latest.json`
   - 图表脚本优先读取这些 JSON；quick smoke run 不应污染正式 figures
   - 写入 `figures/data/` 的 canonical benchmark 必须经由全局锁串行执行，禁止并发 official rerun 污染基线

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
**Phase 0→9 完成**

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
- [x] Pattern overlap 重新评估（历史阶段曾出现 fused_sequential 1.067× vs bulk_sync）

### Phase 5 交付物（2026-03-19）
- [x] GEMM launcher 开销消除（SM count 缓存 + CUDA events + 预分配输出）
- [x] tl.assume 编译器 hint（stride > 0，gemm + 4 pattern kernel）
- [x] scatter_tile_to_peer 默认 .wt write-through
- [x] Pattern overlap 重测（fused_sequential 首次达到正向 overlap）
- [x] P2P 83% 天花板诊断（Iris 无法运行，translate_ptr 实现一致，确认硬件天花板）
- [ ] GEMM 8192³ 达到 ≥90%（当前 79%，kernel 本身瓶颈，需 PTX-level 优化）
- [x] Pattern overlap ≥1.3×（2026-03-21 latest canonical rerun best = 1.667×）

### Phase 6 交付物（2026-03-19）
- [x] 6 幅科研绘图（`figures/` PDF + PNG，Nature/Science 风格）
- [x] GEMM 演进 / P2P 饱和 / 优化瀑布 / Pattern 对比 / 架构图 / Roofline
- [x] CUDA IPC ctypes 调用约定修复（Array → Structure by-value）
- [x] IPC 诊断脚本（`scripts/diagnose_ipc.py`）
- [x] HIP 后端同步修复
- [x] P1-002 根因确认：ctypes bug + 系统 ptrace_scope 双因素

### Phase 7 交付物（2026-03-20）
- [x] Runtime context 统一：`XTileContext` 现持有 `backend` 实例、`backend_name`、可选 `heap`
- [x] `XTileContext.heap_bases` 属性：pattern/collective 不再依赖外部手工塞字段
- [x] `xtile.init(..., heap=...)`：可绑定外部 `SymmetricHeap`
- [x] `xtile.init(..., heap_size=...)`：单 GPU / 分布式 rank-local 自动建堆
- [x] `xtile.init_local(world_size, heap_size)`：单进程多 GPU 统一入口
- [x] `XTileContext.empty/zeros/randn/barrier/auto_select_pattern` 主机侧便捷 API
- [x] CLI pattern bench 切换到真实 `XTileContext`
- [x] pattern 测试移除手工 `_Ctx`，覆盖真实 runtime 入口
- [x] 新增 `tests/test_context.py`，回归验证 context + heap 绑定路径
- [x] `xtile.kernels.gemm.gemm(..., num_warps=..., num_stages=...)`：公开实验覆盖参数，避免 benchmark 误判 wrapper 配置
- [x] Pattern benchmark 动态 heap sizing：按 `(M, N, K)` 自动估算每 rank 对称堆需求，不再固定 512 MiB
- [x] `bench_patterns.py` 切换到 `xtile.init_local(...)`：benchmark 与真实 runtime 入口对齐
- [x] CLI `xtile bench pattern` 新增 `--warmup/--iters/--heap-size-mb`
- [x] `producer_consumer` / `wg_specialized` 复用 lock buffer 与 stream，削减每次 `execute()` 的主机端辅助开销

### Phase 8 交付物（2026-03-21）
- [x] Pattern execution contract 显式化：新增 `xtile.patterns.contracts`
- [x] `PatternExecutionSpec` / `PatternTensorSpec`：统一 full/shard layout metadata
- [x] 4 个 overlap pattern 改为消费显式 contract，不再在多 GPU 下隐式猜 `N`
- [x] 新增 `xtile.ops.gemm_allscatter(...)` 高层入口
- [x] 新增 `xtile.ops.GemmAllScatterPlan` / `build_gemm_allscatter_plan(...)`
- [x] 新增 `xtile.ops.gemm_allscatter_sharded(...)` expert 入口，显式区分 shard/shard 合同
- [x] 新增 `xtile.ops.AllGatherPlan` / `build_allgather_plan(...)` / `xtile.ops.allgather(...)`
- [x] benchmark / helper 开始复用统一 plan-builder 主链
- [x] Pattern benchmark 结构化 JSON 输出：`figures/data/pattern_overlap_latest.json`
- [x] GEMM / P2P benchmark 结构化 JSON 输出：`figures/data/gemm_latest.json` / `figures/data/p2p_latest.json`
- [x] `scripts/plot_figures.py` 改为优先读取结构化 benchmark 数据，并过滤 quick smoke run
- [x] `SymmetricHeap.mode` / `transport_strategy` 元数据暴露给 benchmark 结果
- [x] Pattern overlap full 6-size rerun 重跑：best speedup vs bulk_sync = **1.667×**

### Phase 9 交付物（2026-03-21）
- [x] runtime support matrix：新增 `xtile.describe_runtime_support(ctx)` / `ctx.support_matrix()`
- [x] support matrix 测试：新增 `tests/test_support.py`
- [x] 文档计划收紧：`gemm_reducescatter(...)` 不再被写成“可直接补齐”，而是明确依赖 stable `reduce_scatter` host contract
- [x] baseline `reduce_scatter(...)` host launcher：新增 `xtile.primitives.reduce_scatter(...)` + `tests/test_collectives_host.py`
- [x] 高层 `reduce_scatter` API：新增 `xtile.ops.ReduceScatterPlan` / `build_reduce_scatter_plan(...)` / `xtile.ops.reduce_scatter(...)`
- [x] 正式状态出口：新增 `xtile support --json`
- [x] benchmark 结构化结果内嵌 `runtime_support` 快照：新增 `tests/test_benchmark_results.py`
- [x] plot / Markdown 导出统一消费 `runtime_support`：新增 `scripts/_benchmark_reporting.py` / `scripts/export_benchmark_summary.py`
- [x] `reduce_scatter(..., implementation="device")` 在 `single_process` 模式下显式拒绝，避免暴露未验证错误路径
- [x] canonical benchmark 全局锁：写入 `figures/data/` 的 benchmark 现在会自动串行化，避免并发污染正式 artifact

### Phase 10 交付物（2026-03-21）
- [x] `gemm_allscatter(full/shard)` host wrapper：高层 API 现可自动 materialize heap-backed full output，再返回本 rank shard
- [x] `XTileContext.workspace(...)`：新增可复用 heap-backed scratch buffer，避免 mixed wrapper 重复调用导致 bump allocator 持续增长
- [x] support matrix 更新：`gemm_allscatter.full/shard` 从 unsupported 调整为 supported
- [x] mixed-layout multigpu 回归：`tests/test_ops.py` 新增 full/shard 真实 2-GPU 校验
- [x] 连续两次真实调用验收：确认 workspace 复用，`after_first_bytes == after_second_bytes`
- [x] `gemm_allscatter.shard/full` 保持显式拒绝：真实 2-GPU 诊断确认当前 shard/shard gemm_allscatter 输出是 peer-scatter ownership contract，不是稳定 local-shard basis，因此该方向应转入 future `gemm_allgather` 风格 API，而不是继续伪装成 allscatter wrapper

### Phase 11 交付物（2026-03-21）
- [x] CUDA / HIP IPC handle 序列化修复：`get_ipc_handle()` 现在返回完整 64-byte payload，而不是被 `bytes(c_char_array)` 截断
- [x] multiprocess `reduce_scatter(device)` 真实诊断：新增 `tests/test_e2e/_run_reduce_scatter_multiprocess.py`
- [x] multiprocess device collective 默认 gate：新增 `XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES`
- [x] public 路径安全收口：multiprocess `reduce_scatter` 默认不再自动落到未验证甚至会崩进程的 device path
- [x] backend / support 回归：新增 `tests/test_backend_ipc.py`，并补 multiprocess gate 相关单测

### Phase 12 交付物（2026-03-21）
- [x] `tile_reduce_scatter` multiprocess correctness 修复：device 路径改为只远端读 peer chunk、只本地写 output 的 correctness-first 实现，避免 peer 覆盖未归约本地 chunk 的 data race
- [x] multiprocess IPC bring-up 复测闭环：`python -m tests.test_e2e._run_ipc_test` 真实通过
- [x] multiprocess `reduce_scatter(device)` 2-GPU 真机验收：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 python -m tests.test_e2e._run_reduce_scatter_multiprocess` 真实通过，`rank0=4.0`、`rank1=6.0`
- [x] 分布式诊断脚本硬化：`init_process_group(..., device_id=torch.device("cuda", rank))`，消除 NCCL barrier 设备未知警告
- [x] opt-in multiprocess 回归：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_backend_ipc.py tests/test_ops.py tests/test_support.py tests/test_collectives_host.py tests/test_reduce_scatter_multiprocess.py tests/test_cli_support.py tests/test_context.py tests/test_patterns/test_contracts.py` → `33 passed`

### Phase 13 交付物（2026-03-21）
- [x] multiprocess transport forcing：新增 `XTILE_FORCE_MULTIPROCESS_TRANSPORT`，用于受控诊断 `ctypes_ipc` / `pytorch_ipc` / `peer_access_pointer_exchange`
- [x] `SymmetricHeap._setup_multiprocess()` 重构：拆成按 transport 分派的独立 helper，默认行为不变，但实验可控性显著提高
- [x] multiprocess reduce_scatter 矩阵 benchmark：新增 `tests/benchmarks/bench_reduce_scatter_multiprocess.py`，产出 `docs/generated/reduce_scatter_multiprocess_matrix.json`
- [x] 2-GPU dtype × transport 真机矩阵：`auto/ctypes_ipc` 在 `fp16/bf16/fp32` 全通过；`pytorch_ipc` / `peer_access_pointer_exchange` 当前全部失败
- [x] transport-aware gate：experimental device reduce_scatter 现仅允许 `transport_strategy='ctypes_ipc'`，其他 transport 在 host 侧提前显式拒绝
- [x] transport-aware 回归：`pytest -q tests/test_feature_gates.py tests/test_support.py tests/test_ops.py tests/test_backend_ipc.py` → `29 passed`

### Phase 14 交付物（2026-03-21）
- [x] 最小 multiprocess Triton remote-access 诊断：新增 `tests/test_e2e/_run_triton_remote_access_multiprocess.py`
- [x] 最小 transport 矩阵 benchmark：新增 `tests/benchmarks/bench_triton_remote_access_multiprocess.py`，产出 `docs/generated/triton_remote_access_multiprocess_matrix.json`
- [x] 2-GPU 最小 remote load/store 真机矩阵：`auto/ctypes_ipc` 在 `fp16/bf16/fp32` 全通过；`pytorch_ipc` / `peer_access_pointer_exchange` 全部失败
- [x] auto multiprocess transport 收窄：`SymmetricHeap._setup_multiprocess()` 默认只尝试 `ctypes_ipc`；其他 transport 仅保留 forced diagnostics
- [x] support matrix 新增 `memory["symmetric_heap.device_remote_access"]`，显式暴露“当前 transport 是否真能被 Triton device-side 远端解引用”
- [x] 收口回归：`pytest -q tests/test_feature_gates.py tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_ops.py` → `57 passed`

### Phase 15 交付物（2026-03-21）
- [x] multiprocess `allgather` 真机诊断：新增 `tests/test_e2e/_run_allgather_multiprocess.py`
- [x] multiprocess `allgather` 矩阵 benchmark：新增 `tests/benchmarks/bench_allgather_multiprocess.py`，产出 `docs/generated/allgather_multiprocess_matrix.json`
- [x] 2-GPU allgather 真机矩阵：`auto/ctypes_ipc` 在 `fp16/bf16/fp32` 的 primitive / kernel / high-level API 全通过
- [x] multiprocess `gemm_allscatter` 真机诊断：新增 `tests/test_e2e/_run_gemm_allscatter_multiprocess.py`
- [x] multiprocess `gemm_allscatter` 矩阵 benchmark：新增 `tests/benchmarks/bench_gemm_allscatter_multiprocess.py`，产出 `docs/generated/gemm_allscatter_multiprocess_matrix.json`
- [x] 2-GPU `gemm_allscatter` public baseline：`auto/ctypes_ipc` 在 `full/full + full/shard × fp16/bf16/fp32` 的 plan / high-level API 全通过；forced `pytorch_ipc` / `peer_access_pointer_exchange` 都在 host 侧明确拒绝
- [x] multiprocess `gemm_allscatter` auto pattern coverage：新增 `tests/benchmarks/bench_gemm_allscatter_multiprocess_auto_patterns.py`，产出 `docs/generated/gemm_allscatter_multiprocess_auto_patterns.json`
- [x] 2-GPU `gemm_allscatter` representative auto correctness：`auto/ctypes_ipc` 已覆盖 `bulk_sync / fused_sequential / producer_consumer / wg_specialized` 四个分支，且 `full/full + full/shard` 全通过
- [x] public surface 收口：`xtile.primitives.allgather/allreduce/broadcast`、`xtile.ops.build_allgather_plan(...)`、`xtile.ops.build_gemm_allscatter_plan(...)`、4 个 pattern `execute(...)` 现都会在 unsupported multiprocess transport 下 host 侧提前失败
- [x] support matrix 收紧：`gemm_allscatter` 无 heap 时不再宣称 `supported`；multiprocess `allgather/gemm_allscatter` 改为 mode/transport 感知状态
- [x] 回归：`pytest -q tests/test_feature_gates.py tests/test_collectives_host.py tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_ops.py tests/test_benchmark_results.py tests/test_cli_support.py tests/test_allgather_multiprocess.py tests/test_gemm_allscatter_multiprocess.py tests/test_gemm_allscatter_auto_patterns_multiprocess.py` → `80 passed`

### Phase 16 交付物（2026-03-22）
- [x] `xtile.ops.GemmReduceScatterContract` / `GemmReduceScatterPlan`：补齐高层 GEMM + reduce-scatter 的稳定 public contract
- [x] `xtile.ops.build_gemm_reducescatter_plan(...)` / `xtile.ops.gemm_reducescatter(...)`：不再是占位符，当前实现主链固定为 `local GEMM materialize -> column-pack -> reduce_scatter plan`
- [x] contract 收紧：当前显式要求 `A(M,K)`、`B(K,N)`、`C(M,N/world_size)`，并要求 `C` 位于 attached symmetric heap
- [x] single-process 顺序调用协同：新增 staged finalize 逻辑，支持单进程多 GPU 下按 rank 顺序调用后在最后一个 rank 到达时完成 collective
- [x] support matrix 更新：`gemm_reducescatter` 从 placeholder/unsupported 口径升级为 mode-aware 状态
- [x] `gemm_reducescatter` heap 契约回归：新增“仅 `C` 必须在 heap 上”的测试，明确 `A/B` 可为普通 device tensor
- [x] multiprocess 真机验收：新增 `tests/test_e2e/_run_gemm_reducescatter_multiprocess.py` / `tests/test_gemm_reducescatter_multiprocess.py`
- [x] 2-GPU `gemm_reducescatter` public baseline：opt-in `ctypes_ipc` 下 `plan` / 高层 `ops` 已完成 float32 真机 correctness 校验
- [x] multiprocess `gemm_reducescatter` 结构化矩阵：新增 `tests/benchmarks/bench_gemm_reducescatter_multiprocess.py`，产出 `docs/generated/gemm_reducescatter_multiprocess_matrix.json`
- [x] 2-GPU dtype × transport 真机矩阵：`auto/ctypes_ipc` 在 `fp16/bf16/fp32` 全通过；`pytorch_ipc` / `peer_access_pointer_exchange` 当前全部失败
- [x] 默认基础回归：`pytest -q tests/test_ops.py tests/test_support.py tests/test_cli_support.py tests/test_benchmark_results.py tests/test_collectives_host.py` → `40 passed`
- [x] opt-in multiprocess 回归：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 17 交付物（2026-03-22）
- [x] allocator-first substrate v1：新增 `xtile.memory.allocators`，建立 `BaseSymmetricAllocator` / `TorchBumpAllocator` 边界
- [x] `SymmetricHeap` allocator 化：分配、ownership、metadata、cleanup 不再全部硬编码在 heap 内部
- [x] external import surface：新增 `SymmetricHeap.import_external_tensor(...)` / `as_symmetric(...)` 与 `XTileContext.is_symmetric(...)` / `as_symmetric(...)`
- [x] support matrix 更新：`symmetric_heap_allocator_first_import_map` 升级为 `partial`，新增 `symmetric_heap.external_import`
- [x] `xtile.ops.GemmAllGatherContract` / `GemmAllGatherPlan`：补齐独立 `gemm_allgather` public contract
- [x] `xtile.ops.build_gemm_allgather_plan(...)` / `xtile.ops.gemm_allgather(...)`：主链固定为 `local GEMM materialize -> allgather -> full-output materialize`
- [x] single-process 顺序调用协同：新增 staged finalize 逻辑，支持单进程多 GPU 下按 rank 顺序调用 `gemm_allgather(...)`
- [x] support matrix 更新：新增 `ops["gemm_allgather"]` 与 `contracts["gemm_allgather.shard/full"]`
- [x] allocator-first / `gemm_allgather` 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_ops.py tests/test_support.py tests/test_cli_support.py tests/test_benchmark_results.py` → `73 passed`
- [x] multiprocess `gemm_allgather` 真机验收：`pytest -q tests/test_gemm_allgather_multiprocess.py` → `1 passed`
- [x] multiprocess `gemm_allgather` 结构化矩阵：`python -m tests.benchmarks.bench_gemm_allgather_multiprocess --M 128 --N 256 --K 128 --warmup 0 --iters 1 --timeout-sec 180 --output-json docs/generated/gemm_allgather_multiprocess_matrix.json`
- [x] 2-GPU dtype × transport 真机矩阵：`auto/ctypes_ipc` 与 forced `ctypes_ipc` 在 `fp16/bf16/fp32` 全通过；`pytorch_ipc` / `peer_access_pointer_exchange` 当前全部失败
- [x] 共享基础路径复测：`pytest -q tests/test_allgather_multiprocess.py` → `1 passed`
- [x] opt-in collective 侧向回归：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 18 交付物（2026-03-22）
- [x] `tests/benchmarks/bench_gemm_allgather_multiprocess.py` 新增 `--shapes` shape-grid 入口，可一次性跑多组 `MxNxK`
- [x] backward compatibility 验证：旧 `--M/--N/--K` 单 shape 入口仍可用
- [x] `gemm_allgather` 支持面内扩验：`python -m tests.benchmarks.bench_gemm_allgather_multiprocess --dtypes float16,bfloat16,float32 --transports auto,ctypes_ipc --shapes 128x256x128,256x512x256 --warmup 1 --iters 3 --timeout-sec 240 --output-json docs/generated/gemm_allgather_multiprocess_ctypes_shapes.json`
- [x] 当前正式支持面 `auto/ctypes_ipc + forced ctypes_ipc` 的 `2 shapes × 3 dtypes × 2 transport selections` 真机结果为 `12/12` 全通过
- [x] 结构化 artifact：新增 `docs/generated/gemm_allgather_multiprocess_ctypes_shapes.json`

### Phase 19 交付物（2026-03-22）
- [x] allocator-owned peer export/import surface：新增 `PeerMemoryExportDescriptor` / `ImportedPeerMemory`
- [x] `BaseSymmetricAllocator` / `TorchBumpAllocator` 新增 `export_peer_memory(...)` / `import_peer_memory(...)`
- [x] `SymmetricHeap` multiprocess 建堆收口：`ctypes_ipc` / `pytorch_ipc` / `peer_access_pointer_exchange` 的导出/导入不再直接散落在 heap 内部，而是统一走 allocator surface
- [x] allocator 单测：`tests/test_memory/test_symmetric_heap.py` 新增 structured export/import 回归
- [x] allocator-first 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_ops.py tests/test_support.py tests/test_cli_support.py tests/test_benchmark_results.py` → `75 passed`
- [x] multiprocess 主路径回归：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径回归：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`
- [x] `gemm_allgather` 全 transport 官方矩阵复测：`docs/generated/gemm_allgather_multiprocess_matrix.json` 仍保持 `12 cases = 6 pass / 6 fail`，行为与收口前一致

### Phase 20 交付物（2026-03-22）
- [x] allocator-owned peer mapping metadata：新增 `PeerMemoryMapEntry`
- [x] `SymmetricHeap.peer_export_descriptors()` / `peer_memory_map()` / `peer_memory_map_metadata()`：peer import/map 元数据正式对外可见
- [x] single-process / local-only peer map 回归：`tests/test_memory/test_symmetric_heap.py` 新增 local-only 与 peer-access metadata 测试
- [x] support matrix 更新：新增 `memory["symmetric_heap.peer_mapping_metadata"]`
- [x] allocator/support 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_cli_support.py tests/test_benchmark_results.py` → `48 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 21 交付物（2026-03-22）
- [x] unified runtime metadata surface：新增 `SymmetricHeap.metadata()`、`XTileContext.heap_metadata()`、`XTileContext.runtime_metadata()`
- [x] context 回归：`tests/test_context.py` 新增 runtime/heap metadata 验证
- [x] context/memory/support 基础回归：`pytest -q tests/test_context.py tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_cli_support.py tests/test_benchmark_results.py` → `50 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`

### Phase 22 交付物（2026-03-22）
- [x] benchmark artifact 统一补齐 `runtime_metadata`：`tests/benchmarks/bench_gemm.py`、`tests/benchmarks/bench_p2p_translate.py`、`tests/benchmarks/bench_patterns.py`
- [x] pattern benchmark 额外按 size 持久化 `runtime_metadata`，避免不同 problem size 下 heap_size 变化被顶层单一快照误导
- [x] benchmark/support 基础回归：`pytest -q tests/test_context.py tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_cli_support.py tests/test_benchmark_results.py` → `52 passed`
- [x] benchmark reporting / export 回归：`pytest -q tests/test_benchmark_reporting.py tests/test_export_benchmark_summary.py` → `4 passed`
- [x] 真实 benchmark smoke（H100 PCIe x2）：
  - `python -m tests.benchmarks.bench_gemm --device cuda:0 --repeats 1 --output-json /tmp/xtile_gemm_runtime_metadata_smoke.json`
  - `python -m tests.benchmarks.bench_p2p_translate --quick --output-json /tmp/xtile_p2p_runtime_metadata_smoke.json`
  - `python -m tests.benchmarks.bench_patterns --quick --warmup 1 --iters 1 --output-json /tmp/xtile_patterns_runtime_metadata_smoke.json`
- [x] smoke 结果确认：三类 artifact 均已写出 `runtime_metadata`；P2P quick `best read=248.71 GB/s`、`best write=247.80 GB/s`；pattern quick `best speedup vs bulk_sync=1.210x`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 23 交付物（2026-03-22）
- [x] allocator-owned segment metadata：新增 `MemorySegmentDescriptor`
- [x] peer export / peer map 补齐 `segment_id` / `segment_kind`，把 local segment 与 peer import-map 对齐到同一结构化语义
- [x] `SymmetricHeap.segment_descriptors()` / `segment_metadata()`：heap 现在可显式暴露本地 segment catalog
- [x] support matrix 更新：新增 `memory["symmetric_heap.segment_metadata"]`
- [x] allocator/context/support 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_support.py tests/test_benchmark_results.py` → `50 passed`
- [x] support CLI 回归：`pytest -q tests/test_cli_support.py` → `3 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`
- [x] 真实 benchmark smoke（H100 PCIe x2）：`python -m tests.benchmarks.bench_p2p_translate --quick --output-json /tmp/xtile_p2p_segment_metadata_smoke.json`
- [x] smoke 结果确认：artifact 已写出 `segments`、`segment_id`、`segment_kind`，P2P quick `best read=248.74 GB/s`、`best write=248.23 GB/s`

### Phase 24 交付物（2026-03-22）
- [x] structured peer import state：`ImportedPeerMemory` 现在不再只是 `mapped_ptr + cleanup_resource`，而是带完整 segment / transport / cleanup metadata
- [x] `SymmetricHeap` import-map 内部状态收口：peer import 现在成为 heap 的单一结构化状态源，`_remote_ptrs` / IPC cleanup bookkeeping 改为从 imported-peer records 派生
- [x] `SymmetricHeap.peer_imports()` / `peer_import_metadata()`：peer import records 正式对外可见
- [x] heap/runtime metadata 更新：新增 `peer_imports`
- [x] support matrix 更新：新增 `memory["symmetric_heap.peer_import_metadata"]`
- [x] allocator/context/support 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_support.py tests/test_benchmark_results.py` → `50 passed`
- [x] support CLI 回归：`pytest -q tests/test_cli_support.py` → `3 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`
- [x] 真实 benchmark smoke（H100 PCIe x2）：`python -m tests.benchmarks.bench_p2p_translate --quick --output-json /tmp/xtile_p2p_peer_imports_smoke.json`
- [x] smoke 结果确认：artifact 已写出 `peer_imports`，P2P quick `best read=248.68 GB/s`、`best write=248.12 GB/s`

### Phase 25 交付物（2026-03-22）
- [x] allocator capability surface：`BaseSymmetricAllocator.capabilities()` 已接入，allocator metadata 现显式带 `multi_segment / peer_export / peer_import / external_import_copy / external_mapping / fd_passing / dmabuf_mapping`
- [x] support matrix 更新：新增 `memory["symmetric_heap.external_mapping"]`，明确当前 runtime 仍不支持 zero-copy external mapping
- [x] allocator/context/support 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_support.py tests/test_benchmark_results.py` → `50 passed`
- [x] 真实 benchmark smoke（H100 PCIe x2）：`python -m tests.benchmarks.bench_p2p_translate --quick --output-json /tmp/xtile_p2p_allocator_capabilities_smoke.json`
- [x] smoke 结果确认：artifact 已写出 allocator `capabilities`，其中 `external_mapping=false`、`fd_passing=false`、`dmabuf_mapping=false`；P2P quick `best read=248.74 GB/s`、`best write=248.04 GB/s`

### Phase 26 交付物（2026-03-22）
- [x] import-map 单一状态源：`SymmetricHeap` 已去掉 `_remote_ptrs` / `_peer_map` 这类派生缓存，`heap_bases`、`translate()`、`peer_memory_map()` 现直接从 `peer_imports` 派生
- [x] `heap_bases` 刷新链路收口：不再在 `create_all(...)` / multiprocess transport setup 各分支手工覆写 `_heap_bases`，统一经 `_refresh_heap_bases()` 从 `peer_imports` 重建
- [x] 这一步是内部 substrate 收口，不改变 transport 支持面，也不改变 public contract
- [x] allocator/context/support 基础回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_context.py tests/test_benchmark_results.py` → `50 passed`
- [x] support CLI 回归：`pytest -q tests/test_cli_support.py` → `3 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 27 交付物（2026-03-22）
- [x] peer-mapping state validator：`SymmetricHeap._validate_peer_mapping_state(...)` 已接入，对 `peer_exports` / `peer_imports` / `world_size` / local-rank base pointer / segment metadata 对齐进行 fail-closed 校验
- [x] `_apply_peer_mapping_state(...)` 现先校验再发布状态，异常输入不会污染 `_peer_exports` / `_peer_imports` / `heap_bases`
- [x] focused unit tests：新增长度不一致、local import pointer drift、export/import metadata mismatch、invalid apply fail-closed 覆盖
- [x] 定向回归：`pytest -q tests/test_memory/test_symmetric_heap.py` → `40 passed`
- [x] allocator/context/support/CLI 回归：`pytest -q tests/test_support.py tests/test_context.py tests/test_benchmark_results.py tests/test_cli_support.py` → `17 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 28 交付物（2026-03-22）
- [x] self-describing peer records：`PeerMemoryExportDescriptor` / `ImportedPeerMemory` 已显式带 `peer_rank`
- [x] runtime metadata 收口：`peer_exports` / `peer_imports` 不再只靠列表位置隐含 rank，文档/benchmark/context 现在都可直接消费自描述记录
- [x] validator 增强：`_validate_peer_mapping_state(...)` 现在也校验 record 内的 `peer_rank` 与列表位置一致
- [x] allocator/export/import 定向回归：`pytest -q tests/test_memory/test_symmetric_heap.py` → `41 passed`
- [x] allocator/context/support/CLI 回归：`pytest -q tests/test_support.py tests/test_context.py tests/test_benchmark_results.py tests/test_cli_support.py` → `17 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 29 交付物（2026-03-22）
- [x] export metadata symmetry：`SymmetricHeap.peer_export_metadata()` 已接入，peer export 现在与 `segment_metadata()` / `peer_import_metadata()` / `peer_memory_map_metadata()` 一样有直接可序列化 surface
- [x] heap metadata 收口：`SymmetricHeap.metadata()["peer_exports"]` 现统一走 `peer_export_metadata()`，避免 artifact / docs 侧再手写 `to_dict()` 展开
- [x] 定向回归：`pytest -q tests/test_memory/test_symmetric_heap.py` → `41 passed`
- [x] context/benchmark metadata 回归：`pytest -q tests/test_context.py tests/test_benchmark_results.py` → `8 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 30 交付物（2026-03-22）
- [x] rank-normalized apply path：`SymmetricHeap._apply_peer_mapping_state(...)` 现先按 `peer_rank` 归一化 `peer_exports` / `peer_imports`，再做一致性校验与发布
- [x] fail-closed canonicalization：重复 `peer_rank`、缺失 rank、越界 `peer_rank` 会在发布前直接失败
- [x] internal contract 收紧：incoming peer records 现在只需要“自描述且完整”，不再额外要求调用方先按 rank 排序
- [x] 定向回归：`pytest -q tests/test_memory/test_symmetric_heap.py` → `43 passed`
- [x] allocator/context/support/CLI 回归：`pytest -q tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `17 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 31 交付物（2026-03-22）
- [x] consumer-side metadata regression：`tests/test_context.py` / `tests/test_benchmark_results.py` 现显式断言 `peer_exports`、`peer_imports`、`peer_memory_map` 的 `peer_rank` 可见性
- [x] runtime artifact contract 收紧：context/runtime metadata 层现在不只验证 heap 存在，还验证 peer-mapping metadata surface 是否完整暴露
- [x] metadata 回归：`pytest -q tests/test_context.py tests/test_benchmark_results.py` → `8 passed`
- [x] substrate/support 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_support.py tests/test_cli_support.py` → `52 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 32 交付物（2026-03-22）
- [x] rank-addressed peer accessors：`SymmetricHeap.peer_export_descriptor(rank)` / `peer_import(rank)` 已接入
- [x] host-side translate 收口：`translate(...)` 现统一走 `peer_import(rank)`，而不是直接依赖内部列表索引
- [x] explicit accessor regression：新增 accessor 成功路径与 invalid-rank 失败路径覆盖
- [x] 定向回归：`pytest -q tests/test_memory/test_symmetric_heap.py` → `44 passed`
- [x] allocator/context/support/CLI 回归：`pytest -q tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `17 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 33 交付物（2026-03-22）
- [x] external import mode metadata：allocator metadata 现显式带 `external_tensor_import_mode`
- [x] 当前真实语义落账：`torch_bump` allocator 现在明确声明 `external_tensor_import_mode = "copy"`
- [x] allocator/context/runtime artifact 回归：`tests/test_memory/test_symmetric_heap.py`、`tests/test_context.py`、`tests/test_benchmark_results.py` 现显式断言该字段
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `61 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 34 交付物（2026-03-22）
- [x] peer import access semantics：`ImportedPeerMemory` / `PeerMemoryMapEntry` 现显式带 `access_kind`
- [x] transport vs access 解耦：当前 runtime 现在可区分 `transport`（bring-up 路径）与 `access_kind`（local / peer_direct / mapped_remote / remote_pointer）
- [x] validator 增强：`_validate_peer_mapping_state(...)` 现在也校验 import record 的 `access_kind` 是否与 transport + local/remote 角色一致
- [x] allocator/context/runtime artifact 回归：`tests/test_memory/test_symmetric_heap.py`、`tests/test_context.py`、`tests/test_benchmark_results.py` 已补 `access_kind` 断言
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `62 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 35 交付物（2026-03-22）
- [x] allocator peer semantics catalog：allocator metadata 现显式带 `peer_transport_modes` 与 `peer_import_access_kinds`
- [x] allocator surface 与 public support 解耦：runtime artifact 现在能表达“allocator 实现了哪些 transport/access semantics”，但不把这自动等同于 public-supported transport matrix
- [x] allocator/context/runtime artifact 回归：`tests/test_memory/test_symmetric_heap.py`、`tests/test_context.py`、`tests/test_benchmark_results.py` 已显式断言这两个字段
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `62 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 36 交付物（2026-03-22）
- [x] allocator memory-model descriptor：allocator metadata 现新增结构化 `memory_model`
- [x] schema 收口：当前 `memory_model` 已显式描述 `local_segment_layout`、`peer_import_model`、`peer_mapping_model`、`external_tensor_import_mode`、`external_mapping_mode`
- [x] support matrix 更新：新增 `memory["symmetric_heap.allocator_memory_model"]`
- [x] allocator/context/runtime artifact/support 回归：`tests/test_memory/test_symmetric_heap.py`、`tests/test_context.py`、`tests/test_benchmark_results.py`、`tests/test_support.py` 已显式断言该 surface
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `62 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 37 交付物（2026-03-22）
- [x] heap-level memory-model accessor：`SymmetricHeap.allocator_memory_model_descriptor()` / `allocator_memory_model()` 已接入
- [x] surface 收口：消费方现在可以直接从 heap surface 取 allocator `memory_model`，不必了解 allocator metadata 嵌套结构
- [x] accessor 回归：`tests/test_memory/test_symmetric_heap.py` / `tests/test_context.py` 已显式断言 heap-level accessor
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `63 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 38 交付物（2026-03-22）
- [x] allocator segment-layout descriptor：allocator metadata 现显式带结构化 `segment_layout`
- [x] heap-level segment-layout accessor：`SymmetricHeap.segment_layout_descriptor()` / `segment_layout()` 已接入
- [x] support matrix 更新：新增 `memory["symmetric_heap.segment_layout"]`
- [x] segmented-import-map 预备接口：当前 runtime 现可显式表达 `layout_kind`、`primary_segment_id`、`exportable_segment_ids`，而不是只靠“只有一个 heap segment”这种隐式前提
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `64 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 39 交付物（2026-03-22）
- [x] exportable-segment split：allocator metadata 现显式带 `exportable_segments`，并与 full `segments` 分开表达
- [x] heap-level exportable segment surface：`SymmetricHeap.exportable_segment_descriptors()` / `primary_segment_descriptor()` / `exportable_segment_metadata()` 已接入
- [x] support matrix 更新：新增 `memory["symmetric_heap.exportable_segment_metadata"]`
- [x] multi-segment 预备接口：当前 runtime 现已正式区分 “allocator-owned segments” 与 “runtime-exportable segments”
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `65 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### Phase 40 交付物（2026-03-22）
- [x] external-memory interface descriptor：allocator metadata 现显式带结构化 `external_memory_interface`
- [x] heap-level external-memory interface surface：`SymmetricHeap.external_memory_interface_descriptor()` / `external_memory_interface()` 已接入
- [x] support matrix 更新：新增 `memory["symmetric_heap.external_memory_interface"]`
- [x] external-mapping 预备接口：当前 runtime 现在能正式表达 `import_mode`、`mapping_mode`、`copy_import_supported`、`zero_copy_mapping_supported`、`fd_passing`、`dmabuf_mapping`
- [x] substrate/support/CLI 回归：`pytest -q tests/test_memory/test_symmetric_heap.py tests/test_context.py tests/test_benchmark_results.py tests/test_support.py tests/test_cli_support.py` → `66 passed`
- [x] multiprocess 主路径复测：`pytest -q tests/test_allgather_multiprocess.py tests/test_gemm_allgather_multiprocess.py` → `2 passed`
- [x] opt-in collective 主路径复测：`XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES=1 pytest -q tests/test_reduce_scatter_multiprocess.py tests/test_gemm_reducescatter_multiprocess.py` → `4 passed`

### 已知问题（详见 docs/experiment_log.md）
| 编号 | 问题 | 状态 |
|------|------|------|
| P1-001 | torch.from_blob 不可用 | ✅ 已解决 |
| P1-002 | CUDA IPC ctypes 调用约定 | ✅ 已修复（Structure by-value） |
| P1-002b | CUDA IPC 系统级限制 (ptrace_scope=1) | ✅ 历史问题已由 64B handle 修复 + fallback 双重收口；当前 H100 复测下 `ctypes_ipc` 也已恢复可用 |
| P1-003 | mp.spawn pickle 局部函数 | ✅ 已修复（→ create_all 单进程模式） |
| P1-004 | CUDA backend total_mem 属性名错误 | ✅ 已修复（→ total_memory） |
| P1-005 | Triton 不支持 continue 语句 | ✅ 已修复（→ if != 守卫） |
| P1-006 | bench_gemm.py total_mem 属性名 | ✅ 已修复 |
| P1-007 | Collective benchmark 死锁 | ✅ 已修复（并发 stream） |
| P4-001 | tl.multiple_of on 2D load ptr 性能倒退 | ⚠️ 绕行（仅 offset hint） |
| P4-002 | 小矩阵 (≤2048) GEMM 42-46% | ⚠️ kernel launch 开销限制 |
| P5-001 | 8192³ GEMM 仍仅 83.0-83.5% of cuBLAS（latest canonical rerun） | ⚠️ 仍需 kernel-level 优化 |
| P5-002 | P2P 83% — Triton-on-H100-NVLink 天花板 | ⚠️ 可能需 inline PTX |
| P5-003 | Pattern benchmark heap_size=512MB 不够大尺寸 | ✅ 已修复（动态估算 + CLI override） |
| P8-001 | pattern `execute()` 曾依赖 `B.shape[1]` 隐式猜 full/shard 语义 | ✅ 已修复（显式 execution contract） |
| P8-002 | plot_figures 可能被 quick benchmark 最新结果污染 | ✅ 已修复（结构化 metadata + canonical result gate） |
| P8-003 | pattern benchmark shard path 曾把 `N` 语义隐式缩两次，导致 overlap 结论失真 | ✅ 已修复（显式 `full_N + shard/full layout` contract） |
| P9-001 | `reduce_scatter(...)` 的 single_process reference 与 2-GPU multiprocess/device correctness 已闭环，但 multiprocess public/performance contract 仍未正式放开 | ⚠️ 当前真实支持面已精确收窄为 `ctypes_ipc`；`pytorch_ipc` / `peer_access_pointer_exchange` 尚未通过 device-collective 矩阵，因此默认 gate 仍关闭，仅保留 transport-aware experimental opt-in |
| P13-001 | multiprocess/device 传输面目前只在 `ctypes_ipc` 上通过真实矩阵，其他 transport 仍未修复 | ⚠️ auto contract 已正式收窄为 `ctypes_ipc only`；下一优先级是修 `pytorch_ipc` / `peer_access_pointer_exchange` 的最小 Triton remote-access 正确性，再决定是否重新放开 |
| P10-001 | `gemm_allscatter.shard/full` 不应作为 allscatter wrapper 继续补；该需求已独立收口到 `gemm_allgather.shard/full` public contract | ⚠️ contract 已落地，但 `gemm_allgather` 的 broader multiprocess/performance/world-size validation 仍未闭环 |
| P15-001 | multiprocess `gemm_allscatter` 已完成 2-GPU `ctypes_ipc` public baseline correctness，并补齐 representative auto-selected coverage（4 个 pattern、`full/full + full/shard`），但 broader larger-shape / world-size / stress / performance contract 仍未闭环 | ⚠️ 继续保持 `partial`；下一优先级转为更大 shape、长时压力和 world-size 扩展验证 |
| P17-001 | allocator-first substrate 已落地 v1，但仍缺 Iris 风格 canonical export/import/map/access substrate | ⚠️ 当前已有 `torch_bump`、allocator capability metadata、local segment metadata、allocator-owned peer export、peer import 与 peer-map metadata；但 FD/DMA-BUF external mapping、segmented import-map 和统一 access 语义仍待实现 |

## 性能基线
| 指标 | 实测值 | 目标值 | 状态 |
|------|--------|--------|------|
| P2P read (134MB, f32) | 248.74 GB/s (82.9%) | ≥ 95% | ❌ 未达标 |
| P2P write (134MB, f32) | 248.43 GB/s (82.8%) | ≥ 95% | ❌ 未达标 |
| Collective 归一化带宽 | N/A (tile级原语) | ≥ 90% | ⚠️ 不适用 |
| GEMM vs cuBLAS (4096³ fp16) | **94.9%** (latest canonical rerun) | ≥ 90% | ✅ |
| GEMM vs cuBLAS (4096³ bf16) | **91.1%** (latest canonical rerun) | ≥ 90% | ✅ |
| GEMM vs cuBLAS (8192³ fp16) | 83.0% (latest canonical rerun) | ≥ 90% | ⚠️ 未达标 |
| GEMM vs cuBLAS (8192³ bf16) | 83.5% (latest canonical rerun) | ≥ 90% | ⚠️ 未达标 |
| Pattern overlap (full 6-size rerun) | **1.667×** best speedup（`wg_specialized`, 8192×4608×36864） | ≥ 1.3× | ✅ contract 修正后已达标 |

## 重要约束
- **不** 包装 xSHMEM/NVSHMEM 为不透明字节码
- **不** 引入 NCCL/RCCL 作为运行时依赖
- **不** 在 @triton.jit 函数中使用 Python 对象或动态分派
- Persistent kernel 必须使用 round-robin 调度
