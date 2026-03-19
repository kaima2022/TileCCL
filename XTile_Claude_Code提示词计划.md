# XTile 项目：Claude Code 提示词工程计划

## 使用原则与工作流方法论

---

## 一、Claude Code 工作模式设计

### 1.1 项目记忆体系：CLAUDE.md 文件

Claude Code 通过 `CLAUDE.md` 文件保持项目上下文。这是整个高效协作的基石。

**在项目根目录创建 `CLAUDE.md`，内容如下：**

```markdown
# XTile - 下一代跨平台 Tile 通信库

## 项目定位
集各家之长的 Tile 通信库，目标：跨硬件可移植 + 编译器全可见 + 多尺度统一 + 多节点通信。

## 技术栈
- 主语言：Python + Triton（编译器全可见）
- 目标硬件：NVIDIA (Hopper/Blackwell) + AMD (CDNA3/CDNA4)
- 依赖：triton, torch, hip-runtime / cuda-runtime
- 测试：pytest + 多GPU benchmark harness

## 架构层次（自上而下）
1. User API 层 (xtile/) - Python 高级接口
2. Pattern Library 层 (xtile/patterns/) - overlap 模式
3. Core Primitive 层 (xtile/primitives/) - compute/memory/communication
4. Synchronization 层 (xtile/sync/) - acquire/release + memory scope
5. Memory Management 层 (xtile/memory/) - symmetric heap + 指针翻译
6. HAL 层 (xtile/backends/) - NVIDIA / AMD 硬件抽象

## 关键参考实现
- Iris (github.com/ROCm/iris): 核心算法参考，纯 Triton 实现的 symmetric memory
- TileScale: 多尺度抽象参考
- TileLink/Triton-Distributed: tile 级信号原语参考

## 代码规范
- 所有 device-side 函数用 @triton.jit 装饰
- 所有 host-side API 兼容 PyTorch tensor
- 双 API 模式：value-based (寄存器↔远端) + pointer-based (内存↔内存)
- Atomic 语义：acquire/release/acq_rel + scope: block/gpu/sys
- 命名：snake_case，公开 API 带 docstring

## 当前阶段
Phase 0 - 基础设施搭建

## 重要约束
- 不包装 xSHMEM 为不透明字节码（保持编译器可见性）
- 不引入 NCCL/RCCL 作为依赖（纯 Triton 实现通信）
- 测试必须同时覆盖 NVIDIA 和 AMD（可分阶段）
```

### 1.2 子目录 CLAUDE.md

在关键子目录放置局部 CLAUDE.md，提供模块级上下文：

```
xtile/
├── CLAUDE.md                    # 全局
├── backends/
│   └── CLAUDE.md               # "HAL 层。每个 backend 必须实现 BackendInterface。
│                                #  当前有 hip.py 和 cuda.py。添加新 backend 时
│                                #  参考 hip.py 的实现模式。"
├── memory/
│   └── CLAUDE.md               # "Symmetric heap 管理。核心是 pointer translation。
│                                #  参考 Iris 的 __translate 函数。"
├── primitives/
│   └── CLAUDE.md               # "三大原语：compute/memory/communication。
│                                #  所有 device-side 函数必须 @triton.jit。"
├── patterns/
│   └── CLAUDE.md               # "Overlap 模式库。每个 pattern 是独立文件。
│                                #  参考 Iris 论文 Listing 3/4/5。"
└── tests/
    └── CLAUDE.md               # "测试规范。microbenchmark 输出归一化带宽%。
                                 #  目标：P2P ≥ 95%。"
```

---

## 二、分阶段提示词计划

### Phase 0：基础设施搭建

---

#### Prompt 0-1：项目脚手架

```
初始化 XTile 项目结构。创建以下目录和文件：

xtile/
├── __init__.py          # 公开 API: init, Tile, SymmetricHeap
├── backends/
│   ├── __init__.py
│   ├── base.py          # BackendInterface 抽象基类
│   ├── hip.py           # AMD HIP backend (stub)
│   └── cuda.py          # NVIDIA CUDA backend (stub)
├── memory/
│   ├── __init__.py
│   ├── symmetric_heap.py  # SymmetricHeap 类
│   └── translation.py     # 指针翻译引擎
├── primitives/
│   ├── __init__.py
│   ├── compute.py       # tile_dot, tile_reduce 等
│   ├── memory.py        # tile_load, tile_store 等
│   └── communication.py # remote_load, remote_store, get, put 等
├── sync/
│   ├── __init__.py
│   └── primitives.py    # tile_signal, tile_wait, atomic_* 等
├── patterns/
│   ├── __init__.py
│   ├── bulk_sync.py
│   ├── fused_sequential.py
│   ├── producer_consumer.py
│   └── wg_specialized.py
├── utils/
│   ├── __init__.py
│   ├── topology.py      # 硬件拓扑检测
│   └── profiling.py     # 性能分析工具
tests/
├── conftest.py          # pytest fixtures (多GPU设备)
├── test_memory/
├── test_primitives/
├── test_patterns/
└── benchmarks/
setup.py / pyproject.toml

BackendInterface 抽象基类需定义：
- init_ipc() -> None
- allocate(size) -> ptr
- get_ipc_handle(ptr) -> handle
- open_ipc_handle(handle) -> ptr
- get_heap_bases() -> tensor
- detect_topology() -> TopologyInfo
- synchronize() -> None

每个 stub 文件写清楚 TODO 注释说明该模块的职责。
```

---

#### Prompt 0-2：Iris 源码审计脚本

```
我需要系统性审计 Iris 代码库 (github.com/ROCm/iris)。
帮我写一个审计脚本 scripts/audit_iris.py，功能：

1. 克隆 Iris 仓库到 vendor/iris/
2. 遍历所有 .py 文件，统计：
   - 所有 @triton.jit 函数列表（名称、参数、行数）
   - 所有使用 tl.atomic_* 的位置
   - 所有 hipIpc* 调用位置
   - 所有 tl.load/tl.store 用于远端访问的模式
3. 输出结构化的 JSON 报告到 docs/iris_audit.json
4. 生成 markdown 摘要到 docs/iris_audit_summary.md

这个审计是为了精确定位需要移植/参考的核心函数。
重点关注：
- iris 的 __translate 函数实现
- symmetric heap 的建立流程
- load/store/get/put/copy 的完整实现
- atomic 操作的 sem 和 scope 参数处理
```

---

#### Prompt 0-3：构建系统与 CI

```
为 XTile 项目配置构建系统：

1. pyproject.toml:
   - 依赖：triton>=3.0, torch>=2.4
   - 可选依赖组：[nvidia], [amd], [dev], [benchmark]
   - 入口点：xtile CLI 工具（用于 benchmark 运行）

2. Makefile:
   - make install: pip install -e .[dev]
   - make test: pytest tests/ -v
   - make bench: 运行 benchmarks
   - make lint: ruff + mypy
   - make audit: 运行 Iris 审计脚本

3. .github/workflows/ci.yml:
   - lint job: ruff + mypy（无需 GPU）
   - test-amd job: 在 AMD 环境运行（标记为 self-hosted）
   - test-nvidia job: 在 NVIDIA 环境运行（标记为 self-hosted）

4. 添加 .gitignore, LICENSE (Apache 2.0), README.md 骨架
```

---

### Phase 1：核心原语层

---

#### Prompt 1-1：Symmetric Heap（AMD 先行）

```
实现 xtile/memory/symmetric_heap.py 的 AMD HIP backend。

参考 Iris 的实现模式：
1. 每个 rank 用 hipMalloc 分配设备内存
2. hipIpcGetMemHandle 导出 handle
3. 通过 PyTorch distributed all_gather 交换 handle
4. hipIpcOpenMemHandle 打开远端 handle
5. 构建 heap_bases tensor（所有 rank 的基地址）

关键类设计：
class SymmetricHeap:
    def __init__(self, size: int, backend: str = "auto"):
        ...
    def allocate_tensor(self, shape, dtype) -> torch.Tensor:
        """在 symmetric heap 中分配 tensor"""
    def get_heap_bases(self) -> torch.Tensor:
        """返回所有 rank 的堆基址 tensor（用于 device-side 翻译）"""
    def translate(self, local_ptr, to_rank: int):
        """host-side 指针翻译（调试用）"""
    def barrier(self):
        """全局同步"""
    def cleanup(self):
        """释放所有 IPC 资源"""

请确保：
- 使用 ctypes 调用 HIP runtime API
- 错误处理完善（每个 HIP 调用检查返回值）
- 资源清理用 __del__ 和 context manager
- 写对应的单元测试 tests/test_memory/test_symmetric_heap.py
  （单 GPU 可测分配/释放，多 GPU 测 IPC 建立）
```

---

#### Prompt 1-2：指针翻译（Device-Side）

```
实现 xtile/memory/translation.py 的 Triton device-side 指针翻译。

这是 Iris 最核心的函数，参考 Iris 论文 Listing 1：

@triton.jit
def translate(ptr, from_rank, to_rank, heap_bases):
    """
    将 from_rank 的本地指针转换为 to_rank 的远端指针。
    
    步骤：
    1. 加载 from_rank 的 heap base
    2. 加载 to_rank 的 heap base
    3. 计算 offset = ptr - from_base
    4. 远端指针 = to_base + offset
    5. 类型转换回原始指针类型
    """

然后基于 translate 实现：
- remote_load(ptr, to_rank, from_rank, heap_bases, mask=None)
- remote_store(ptr, value, src_rank, dst_rank, heap_bases, mask=None)

所有函数必须是 @triton.jit。
写 microbenchmark: benchmarks/bench_translation.py
- 测试不同 buffer 大小的 P2P load/store 带宽
- 输出归一化带宽百分比（对比理论峰值）
- 参考 Iris 论文 Figure 6 的热力图格式
```

---

#### Prompt 1-3：完整 Device-Side 原语集

```
在 xtile/primitives/communication.py 中实现完整的通信原语。

Value-based 操作（寄存器↔远端内存）：
- tile_remote_load(ptr, to_rank, from_rank, heap_bases, mask)
- tile_remote_store(ptr, value, src_rank, dst_rank, heap_bases, mask)

Pointer-based 操作（内存↔内存）：
- tile_get(dst_ptr, src_ptr, src_rank, dst_rank, heap_bases, mask)
- tile_put(src_ptr, dst_ptr, src_rank, dst_rank, heap_bases, mask)
- tile_copy(src_ptr, dst_ptr, from_rank, to_rank, heap_bases, mask)

Atomic 操作（全部支持 sem 和 scope 参数）：
- tile_atomic_add(ptr, value, rank, heap_bases, sem="relaxed", scope="gpu")
- tile_atomic_cas(ptr, expected, desired, rank, heap_bases, sem, scope)
- tile_atomic_xchg(ptr, value, rank, heap_bases, sem, scope)
- tile_atomic_and / or / xor / min / max

信号原语（借鉴 TileLink，基于 atomic 实现）：
- tile_signal(locks, tile_id, sem="release", scope="gpu")
  # 内部: atomic_cas(locks + tile_id, 0, 1, sem, scope)
- tile_wait(locks, tile_id, sem="acquire", scope="gpu")
  # 内部: while atomic_cas(locks + tile_id, 1, 0, sem, scope) == 0: pass

所有函数都是 @triton.jit。
每个函数写 docstring 说明用途和参数。
写对应测试覆盖每个原语的正确性。
```

---

#### Prompt 1-4：NVIDIA Backend 适配

```
现在将 xtile 适配到 NVIDIA CUDA。需要修改的模块：

1. xtile/backends/cuda.py:
   - 用 ctypes 封装 cudaIpcGetMemHandle / cudaIpcOpenMemHandle
   - 注意 CUDA IPC handle 大小可能不同于 HIP
   - NVLink 拓扑检测（nvmlDeviceGetNvLinkState 等）

2. xtile/memory/symmetric_heap.py:
   - 已有 AMD 实现，添加 CUDA 分支
   - 统一接口，差异在 backend 类内部

3. xtile/backends/base.py:
   - 确保 BackendInterface 覆盖两个 backend 的所有差异点

4. 关键差异处理：
   - warp size: AMD=64, NVIDIA=32
     → 在 triton.jit 中用 tl.constexpr 参数化
   - memory scope 命名：AMD agent = NVIDIA device
     → 在 sync 层做映射
   - CU vs SM 数量：影响 workgroup specialization 的分配

5. 测试：
   - 复用 AMD 的测试用例，参数化 backend
   - 用 pytest.mark.nvidia / pytest.mark.amd 标记

目标：在 NVIDIA H100 上 P2P load/store 达到 95%+ 归一化带宽。
```

---

### Phase 2：Pattern Library

---

#### Prompt 2-1：GEMM 基础内核

```
实现 xtile/kernels/gemm.py，提供标准 Triton GEMM 作为 pattern 的计算基础。

参考 Iris 论文 Listing 2 的 gemm_loop：
- 输入 A[M,K], B[K,N]，输出 C[M,N]
- persistent kernel 风格（for tile_id in range(pid, total, NUM_SMS)）
- 支持编译期参数：BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_SMS
- swizzle 支持：GROUP_SIZE_M 参数
- chiplet_swizzle 支持（AMD XCD 感知）

这个 GEMM 不包含通信逻辑，是纯计算内核。
Pattern 层会在此基础上插入通信操作。

写 benchmark 对比 torch.matmul 的性能，
确保在 bf16 上达到理论 TFLOPS 的 90%+。
```

---

#### Prompt 2-2：四种 Overlap 模式实现

```
在 xtile/patterns/ 下实现四种 GEMM+AllScatter 的 overlap 模式。

每个模式一个文件，统一接口：
class Pattern:
    def __init__(self, ctx: XTileContext, **kwargs): ...
    def execute(self, A, B, C) -> None: ...
    def benchmark(self, A, B, C, warmup=10, iters=100) -> dict: ...

模式 1: bulk_sync.py（参考 Iris Listing 3）
- 两个独立 kernel：gemm_kernel → barrier → scatter_kernel
- scatter_kernel 用 iris.put 风格的 tile_put

模式 2: fused_sequential.py（参考 Iris Listing 4）
- 单 kernel，主循环内 gemm_loop 后直接 tile_remote_store
- for tile_id: acc = gemm_loop(...); scatter(acc)

模式 3: producer_consumer.py（参考 Iris Section 4.1.2）
- 两个 kernel 在不同 stream 上并发
- producer 用 COMPUTE_SMS 个 CU/SM 做 GEMM
- consumer 用 COMM_SMS 个 CU/SM 做 scatter
- 用 atomic_cas 做 tile 级同步

模式 4: wg_specialized.py（参考 Iris Listing 5）
- 单 fused kernel
- if pid < GEMM_SMS: 做 GEMM + signal
- else: wait + scatter
- 需要 locks tensor 做 tile 级同步

每个模式必须有：
1. 完整实现代码
2. 单元测试（正确性验证，对比 torch.matmul + manual scatter）
3. Benchmark 脚本（输出 ms 时间 + 对比 baseline 的加速比）
```

---

#### Prompt 2-3：Auto-Select 引擎

```
实现 xtile/patterns/auto_select.py：

def auto_select(
    op: str,           # "gemm_allscatter", "gemm_allgather", ...
    M: int, N: int, K: int,
    world_size: int,
    hw_info: HardwareInfo,
) -> Pattern:
    """
    基于 Iris 论文 Section 5 的实验观察：
    
    规则（初始版本，后续用 benchmark 数据校准）：
    1. N/world_size < 1024 且 K > 16384 → fused_sequential
       理由：通信小，GEMM 需要全部资源
    2. N/world_size < 2048 且 K > 8192 → producer_consumer
       理由：可完全隐藏通信
    3. N > 4096 且 K > 8192 → wg_specialized
       理由：通信量大，需要专用资源
    4. 默认 → bulk_synchronous
       理由：最安全，适合未知工况
    """

同时实现 benchmark_all_patterns() 函数：
- 对给定 (M,N,K,world_size) 运行所有模式
- 输出对比表格（pattern, time_ms, speedup_vs_baseline）
- 用这个函数的输出来校准 auto_select 的阈值

写测试验证 auto_select 在已知 case 上选择正确的 pattern。
```

---

### Phase 3：高级特性

---

#### Prompt 3-1：Collective 通信原语

```
实现 xtile/primitives/collectives.py：

tile 级集体通信（全部 @triton.jit，纯 Triton 实现）：

1. tile_allreduce(tile, op="sum", heap_bases, rank, world_size)
   - ring allreduce 实现
   - 每个 step 发送一个 tile 分块到邻居
   - 支持 op: sum, max, min

2. tile_allgather(tile, heap_bases, rank, world_size)
   - 每个 rank 将自己的 tile 写到所有远端的对应位置
   - 结果：world_size 倍大小的 gathered tile

3. tile_scatter(tile, heap_bases, rank, world_size)
   - rank 0 将不同 tile 分块发到不同 rank
   
4. tile_all2all(tile, heap_bases, rank, world_size)
   - 每个 rank 的第 i 块发给 rank i

每个 collective 写两个版本：
a) 简单版：用 for 循环 + remote_store（先保证正确）
b) 优化版：利用拓扑感知的通信顺序（减少链路竞争）

microbenchmark 对比 NCCL/RCCL 的等效操作。
目标：大 buffer（>64MB）达到 90%+ 的 NCCL/RCCL 性能。
```

---

#### Prompt 3-2：Profiling 与可视化

```
实现 xtile/utils/profiling.py：

1. TileProfiler 类：
   - 在 kernel 内部插入计时点（用 Triton 的 clock 指令）
   - 记录每个 tile 的 compute time 和 comm time
   - 计算 overlap 效率 = 1 - (idle_time / total_time)

2. 集成 Triton Proton：
   - 自动给 xtile 原语添加 proton region 标记
   - 输出 chrome tracing 格式的 profile

3. 可视化工具 scripts/visualize_overlap.py：
   - 输入：profiler 数据
   - 输出：类似 Iris 论文 Figure 4/5 的时间线图
     - X 轴：时间
     - Y 轴：CU/SM
     - 颜色：compute tile / comm tile / idle
   - 用 matplotlib 生成 PNG/SVG

4. 带宽热力图 scripts/visualize_bandwidth.py：
   - 输入：microbenchmark 结果
   - 输出：类似 Iris 论文 Figure 6 的 GPU×GPU 热力图
   - 显示归一化带宽百分比
```

---

#### Prompt 3-3：文档与示例

```
创建完整的文档和示例体系：

1. docs/getting_started.md：
   - 安装（AMD / NVIDIA 分别说明）
   - 5 分钟快速开始：hello world P2P
   - 第一个 fused kernel 教程

2. docs/api_reference.md：
   - 从代码 docstring 自动生成
   - Host-Side API 表格（类似 Iris 论文 Table 2）
   - Device-Side API 表格（类似 Iris 论文 Table 3）

3. docs/patterns_guide.md：
   - 四种 overlap 模式的原理图（ASCII art）
   - 每种模式的适用场景和代码示例
   - 如何选择正确的模式
   - 如何编写自定义 pattern

4. examples/ 目录：
   - 01_p2p_basic.py: 最简 P2P load/store
   - 02_allreduce.py: 8-GPU ring allreduce
   - 03_gemm_scatter_bulk.py: bulk-sync 模式
   - 04_gemm_scatter_fused.py: fused sequential 模式
   - 05_gemm_scatter_specialized.py: WG specialization
   - 06_auto_pattern.py: auto-select 演示

每个示例必须：
- 可独立运行（torchrun --nproc_per_node=N example.py）
- 有详细注释
- 输出性能数据
```

---

## 三、日常工作流提示词模板

### 3.1 功能开发模板

```
【任务】实现 xtile/{module}/{file}.py 中的 {function_name}

【功能描述】
{一句话描述}

【参考】
- Iris 对应实现：{文件路径或论文 Listing 编号}
- 关键差异：{与 Iris 的不同点}

【接口定义】
{函数签名 + 参数说明}

【实现要求】
- {具体要求 1}
- {具体要求 2}

【测试】
写 tests/{对应测试文件} 覆盖：
- 正确性：{边界条件}
- 性能：{benchmark 指标}
```

### 3.2 Bug 修复模板

```
【问题】{现象描述}

【复现】
运行命令：{command}
错误信息：{error message}

【相关代码】xtile/{file}:{line_range}

【期望行为】{正确结果}

请定位根因并修复，同时添加回归测试防止复现。
```

### 3.3 性能优化模板

```
【目标】将 {benchmark_name} 的性能从 {current} 提升到 {target}

【当前 profile 数据】
{粘贴 profiling 输出}

【瓶颈分析】
我认为瓶颈在 {分析}，但需要你验证。

请：
1. 分析 profile 数据确认瓶颈
2. 提出 2-3 个优化方案及预期收益
3. 实现最有希望的方案
4. 运行 benchmark 验证效果
```

### 3.4 代码审查模板

```
请审查 xtile/{file} 的最新改动，关注：

1. 正确性：内存序是否正确？scope 是否匹配？
2. 性能：是否有不必要的 global memory 访问？
3. 可移植性：NVIDIA/AMD 两个 backend 是否都能工作？
4. 代码风格：是否符合项目规范？
5. 测试覆盖：是否有遗漏的边界条件？

给出具体的改进建议和代码修改。
```

---

## 四、关键策略

### 4.1 上下文管理策略

**问题**：XTile 项目涉及多个复杂模块，单次对话无法覆盖全部上下文。

**策略**：

| 场景 | 做法 |
|------|------|
| 开始新模块 | 先让 Claude Code 读 CLAUDE.md + 该模块的 CLAUDE.md + 相关参考代码 |
| 跨模块改动 | 在 prompt 中明确列出涉及的所有文件路径 |
| 性能调优 | 附上 benchmark 输出和 profile 数据 |
| 参考 Iris | 让 Claude Code 先读 vendor/iris/ 下的对应源码 |

### 4.2 渐进式开发策略

**核心原则：每个 prompt 产出可测试的增量**

```
❌ 错误方式："实现整个 Pattern Library"
✅ 正确方式："实现 bulk_sync.py 的 GEMM+AllScatter 模式，
              写正确性测试，运行 benchmark"
```

**每个 prompt 的结束条件**：
1. 代码能通过 `make lint`
2. 新测试全部通过
3. 如果是性能相关，benchmark 数据已输出

### 4.3 并行工作流

Claude Code 支持多个会话。推荐同时开两条线：

```
会话 A（主线）：核心功能开发
  → Prompt 1-1 → 1-2 → 1-3 → ...

会话 B（辅助线）：测试 + 文档 + CI
  → 0-3 → 测试框架 → benchmark harness → 文档
```

### 4.4 检查点与回顾

每完成一个 Phase，用一次总结性 prompt：

```
Phase {N} 完成回顾。请：

1. 阅读当前代码库的所有文件
2. 检查 CLAUDE.md 是否需要更新
3. 列出所有 TODO 和已知问题
4. 评估当前代码对 Phase {N+1} 的就绪程度
5. 建议 Phase {N+1} 的优先级调整（如有）

输出更新后的 CLAUDE.md（包括"当前阶段"字段）。
```

---

## 五、预期产出时间线

| 周 | Prompt 序列 | 预期产出 |
|----|------------|---------|
| 1 | 0-1, 0-3 | 项目脚手架 + 构建系统 |
| 2 | 0-2 | Iris 审计报告 |
| 3-4 | 1-1 | AMD Symmetric Heap 可用 |
| 5-6 | 1-2 | 指针翻译 + P2P benchmark |
| 7-8 | 1-3 | 完整 device-side 原语集 |
| 9-10 | 1-4 | NVIDIA backend 可用 |
| 11-14 | 2-1, 2-2 | GEMM + 四种 pattern |
| 15-16 | 2-3 | Auto-select 引擎 |
| 17-20 | 3-1, 3-2 | Collectives + Profiling |
| 21-24 | 3-3 | 文档 + 示例 + 开源准备 |
