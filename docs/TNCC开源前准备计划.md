# TNCC 开源前准备计划

日期：`2026-03-28`

## 1. 目标

TNCC 的首个开源版本不追求“大而全”，而追求三件事：

1. **一眼能看懂**
   - 用户在 3 分钟内看明白 TNCC 是什么、不是什么、当前支持什么。
2. **一跑就有结果**
   - 用户可以按照 README 跑通最小示例、最小测试、最小 benchmark。
3. **特点足够鲜明**
   - 公开定位必须从一开始就与 Iris、NCCL/RCCL、普通 Triton kernel library 区分开。

建议把首发版本定义为：

> 一个面向 Triton 的 tile-native collective 通信库预览版。它把 collective 作为编译器可见的 tile 级原语、同步机制和执行模式来组织，重点展示 GEMM + collective 融合、pattern 化 overlap、以及稳定的高层公共 contract。

## 2. 首发版本定位

### 2.1 建议版本口径

- 版本号：`v0.1.0-preview` 或 `v0.1.0`
- 对外口径：`research-preview` / `developer-preview`
- 不要使用的口径：
  - “替代 NCCL”
  - “通用分布式训练通信栈”
  - “已支持大规模多节点生产环境”

### 2.2 首发主叙事

TNCC 的首发版本应明确聚焦这条主线：

- **Tile-native collective**
- **纯 Triton、编译器可见**
- **collective-first，而不是 generic RMA-first**
- **支持 fused compute-communication overlap**
- **公共入口是 `tncc.ops.*`，而不是要求用户直接拼底层 primitive**

### 2.3 首发边界

首发时建议主动收缩，只承诺当前最清晰、最能体现特色的支持面：

- 单机多 GPU
- 优先强调 single-process multi-GPU 主路径
- multiprocess 只公开当前 validated surface
- 重点 op：
  - `gemm_allscatter`
  - `allgather`
  - `allreduce`
  - `reduce_scatter`
  - `gemm_allgather`
  - `gemm_reducescatter`
- 重点 pattern：
  - `bulk_sync`
  - `fused_sequential`
  - `producer_consumer`
  - `wg_specialized`

不要在首发版本里扩张到：

- 多节点 scale-out
- 复杂网络传输叙事
- 太多尚未验证的 transport/path
- 过多“未来将支持”但没有最小验证闭环的能力

## 3. 首发版本完成定义

如果下面 8 项都达标，就可以开源：

1. README 首页完整
2. 一张专业清晰的架构图可用
3. 一个 30 行以内的最小示例可运行
4. 高层 API 和 runtime 支持面写清楚
5. 最小测试矩阵能在 CI 跑通
6. License、NOTICE、第三方依赖口径清楚
7. Benchmark/性能图只保留可信结论
8. 仓库名称、包名、文档名全部完成 `TNCC` 收口

## 4. 工作分解

### 4.1 文档与叙事

这是开源前最优先的工作流。因为首发版本首先卖的是“认知清晰度”。

#### 必做

- 重写 README 首页，结构建议如下：
  - 项目一句话定位
  - TNCC 与 Iris / NCCL-RCCL 的区别
  - 核心特性
  - 一张架构图
  - Quick Start
  - 最小示例
  - 当前支持矩阵
  - Benchmark/现状说明
  - 路线图
- 新增 `docs/OPEN_SOURCE_STATUS.md` 或直接在 README 中写清：
  - 当前 validated public surface
  - 当前硬件验证范围
  - 不承诺的范围
- 新增 `docs/ARCHITECTURE.md`
  - 把当前 WG 工作组模式文档中的主链抽象成仓库总架构说明
- 新增 `docs/CONCEPTS.md`
  - 解释这些词：
    - symmetric heap
    - pointer translation
    - tile collective
    - pattern
    - contract
    - plan
    - single-process / multiprocess

#### 建议的 README 首页标题文案

> TNCC is a tile-native collective communication library for Triton.  
> It makes collective communication compiler-visible and fuses communication with tiled compute through reusable patterns, synchronization primitives, and stable high-level contracts.

### 4.2 架构图与视觉材料

这是最容易快速提升项目专业感的部分，必须做。

#### 必做图 1：README 首页主架构图

来源建议基于：

- [`docs/TNCC WG工作组模式完整代码流程报告.md`](/home/makai/XTile/docs/TNCC%20WG工作组模式完整代码流程报告.md)

但不要直接搬文档里的 ASCII 图。建议单独产出一张可放首页的 SVG/PNG。

#### 这张图的目标

让读者 10 秒内看懂：

- 用户写的是 `tncc.ops.*`
- 中间有 plan / contract / pattern engine
- 下层是 tile communication / sync / memory translation
- 执行发生在 Triton kernel 内部
- 通信与计算在同一个 device-side 程序里 overlap

#### 图的视觉要求

- 使用矢量格式：首选 `SVG`
- 风格：白底、论文风、轻量工程风，不要花哨渐变
- 色彩建议：
  - Host/API：蓝色系
  - Pattern/Plan：青色或绿色系
  - Kernel 内 Compute：橙色系
  - Kernel 内 Comm：红色系
  - Memory/Sync substrate：灰色系
- 图中文字全部英文，便于国际读者理解
- 图不要超过 6 个主层级

#### 图的结构建议

```text
User Code / tncc.ops.*
        |
Contract + Plan Resolution
        |
Pattern Engine
        |
Fused Triton Kernel
   | Compute Workers
   | Comm Workers
        |
Sync Primitives + Pointer Translation + Symmetric Heap
        |
CUDA / HIP Backend
```

#### 建议交付物

- `figures/fig_architecture_tncc.svg`
- `figures/fig_architecture_tncc.png`
- README 嵌图
- `docs/ARCHITECTURE.md` 中补充一版可点击放大的高清图

#### 可选补图

- 图 2：`gemm_allscatter` 执行时序图
- 图 3：四种 overlap pattern 对比图
- 图 4：TNCC vs Iris vs NCCL/RCCL 的定位图

### 4.3 公共 API 收口

TNCC 首发最关键的代码工作，不是继续加 primitive，而是把 tile-native collective 变成更稳定、更可选型、更可验证的公共层。

#### 必做

- 明确 `tncc.ops.*` 是默认 public surface
- 明确 `build_*_plan(...)` 是 expert/public reusable surface
- 明确 `pattern.execute(...)` 是 expert/internal surface
- 在 README 和 docstring 里统一口径，不再让三种入口平级出现
- 给每个高层 op 写清公共 contract：
  - 输入 shape
  - layout 约定
  - storage 要求
  - 当前支持的 dtype
  - 当前不支持的情况

#### 首发建议保留的公共 API

- `tncc.init(...)`
- `tncc.init_local(...)`
- `tncc.current_context()`
- `ctx.empty(...)`
- `ctx.zeros(...)`
- `ctx.randn(...)`
- `ctx.workspace(...)`
- `tncc.ops.gemm_allscatter(...)`
- `tncc.ops.gemm_allgather(...)`
- `tncc.ops.gemm_reducescatter(...)`
- `tncc.ops.allgather(...)`
- `tncc.ops.allreduce(...)`
- `tncc.ops.reduce_scatter(...)`

#### 首发建议降级为 expert/internal 的 API

- pattern class 直接暴露的底层细节
- 诊断 transport/path 选择入口
- 还没有稳定 contract 的内部 helper

#### API 收口原则

- 一个能力只保留一个默认入口
- 所有默认入口都能映射到明确 contract
- 所有 contract 都能对应测试和错误信息
- 所有错误都尽量在 host-side 提前报，而不是 device-side 崩

### 4.4 支持矩阵与验证口径

开源前必须把“支持什么、不支持什么”写得非常清楚。

#### 必做

- 在 README 加一个简短 support matrix
- 在 `docs/` 保留完整 support matrix
- 将下面几类信息统一到一个口径：
  - backend
  - heap mode
  - transport strategy
  - op availability
  - hardware validation status

#### 建议公开口径

- Backend：
  - CUDA: code path available, validated on H100
  - HIP: code path available, validation status explicitly marked
- Runtime：
  - single-process multi-GPU: primary path
  - multiprocess: limited validated surface only
- Hardware：
  - 明确写出实际测试机器
- World size：
  - 明确写出当前公开保证的 world size

#### 一定不要做

- 用模糊话术掩盖当前边界
- 把 bring-up path 写成稳定能力
- 把尚未持续验证的 benchmark 数据写成 headline

### 4.5 示例与教程

首发版至少需要 3 个示例，少而精。

#### 必做示例

1. `examples/01_allreduce_minimal.py`
   - 展示初始化、heap、张量、`tncc.ops.allreduce`
2. `examples/02_gemm_allscatter_minimal.py`
   - 展示 TNCC 的主卖点
3. `examples/03_pattern_auto_select.py`
   - 展示 pattern engine 和 auto-select

#### 示例要求

- 每个示例不超过 80 行
- 运行命令直接写在文件头注释或 README
- 输出结果可验证
- 不依赖仓库外隐藏脚本

### 4.6 测试、CI 与可验证性

首发版本必须给用户“它不是只会画图”的信号。

#### 必做测试层次

- 单元测试：
  - contract 解析
  - translation
  - heap metadata
  - feature gates
- 集成测试：
  - `allgather`
  - `allreduce`
  - `reduce_scatter`
  - `gemm_allscatter`
- CLI/smoke test：
  - `tncc --help`
  - `tncc bench pattern --quick`

#### 首发 CI 建议

- `lint`
- `unit tests`
- `CPU-only import smoke`
- GPU CI 如果资源有限，至少保留：
  - nightly/self-hosted GPU correctness

#### 对外口径

如果 GPU CI 资源有限，就诚实说明：

- PR 上跑基础检查
- GPU correctness 在 self-hosted runner / nightly 上跑

不要因为 CI 不完整而阻塞首发，但必须把口径写清。

### 4.7 性能材料与 benchmark 策略

首发一定要有性能材料，但不能让性能口径反噬项目可信度。

#### 必做

- README 中只保留 1 张最能说明问题的性能图
- 图的结论必须与当前 public surface 对齐
- 所有 headline 都必须可追溯到 `figures/data/*.json`

#### 首发建议保留的性能叙事

- `pattern overlap` 图：
  - 说明 TNCC 的 value 在于 fused overlap pattern
- `comm-only collective` 图：
  - 只在结论足够稳时保留
- `p2p translate / remote access` 图：
  - 用于说明底层机制不是黑盒库

#### 首发不建议作为 headline 的叙事

- “全面超越 NCCL”
- “全场景性能领先”
- “跨所有 collective 全面最优”

#### 更稳妥的首发性能句式

- TNCC demonstrates compiler-visible fused compute-communication patterns.
- TNCC already shows meaningful overlap benefits on selected tiled workloads.
- Current collective kernels are functional and measurable on the validated public surface.

### 4.8 仓库卫生与开源合规

这是正式开源前必须一次性清理的部分。

#### 必做

- 仓库根目录补齐：
  - `README.md`
  - `LICENSE`
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
  - `ROADMAP.md`
  - `CHANGELOG.md` 或首发 release notes
- 检查第三方引用：
  - Iris
  - ThunderKittens
  - TileScale
  - Triton-distributed
- 检查 LICENSE 与引用兼容性
- 检查是否存在不应公开的：
  - 机器名
  - 私有路径
  - 内部备注
  - 临时调试脚本
  - 非必要 benchmark 噪声文件

#### 命名清理

- 确保 `XTile` 老名字不再出现在 public 文档、包路径、命令、图题中
- README、docs、figure title、json metadata、CLI 帮助统一成 `TNCC`

### 4.9 Packaging 与安装体验

首发版安装体验必须简洁，否则用户第一步就流失。

#### 必做

- `pip install -e .` 可用
- `pip install -e ".[dev]"` 可用
- `python -c "import tncc"` 可用
- `tncc --help` 可用
- README 中给出最短安装路径

#### 可选

- Dockerfile / dev container
- `make install-dev`
- 一键 smoke 命令，例如：
  - `make smoke`

## 5. 建议的最小首发内容

如果目标是尽快出雏形，建议把首发压缩成下面这套“小而完整”的组合：

### 必须有

- 一个清晰 README
- 一张架构图
- 三个最小示例
- 六个高层 op 的公共说明
- 一页 support matrix
- 一套最小测试
- 一个最小 benchmark 图
- 基本开源文件

### 可以暂缓

- 大而全教程
- 多节点叙事
- 大量性能图
- 太多 transport/path 文档
- 太多 benchmark 变体

### 首发鲜明特点

首发时只强调 3 个记忆点：

1. TNCC is collective-first.
2. TNCC is compiler-visible.
3. TNCC supports tile-native overlap patterns.

## 6. 两周执行计划

### 第 1 阶段：第 1-3 天，收口叙事与公共面

- 完成 README 首页重写
- 明确首发口径和支持边界
- 清理 `XTile` 残留命名
- 明确 public/expert/internal API 分层

交付物：

- README 初稿
- support matrix 初稿
- 首发版本 scope 清单

### 第 2 阶段：第 4-6 天，补图与示例

- 画 README 主架构图
- 补三个最小示例
- 补 `ARCHITECTURE.md`
- 补 `CONCEPTS.md`

交付物：

- `fig_architecture_tncc.svg/png`
- `examples/*`
- 架构说明文档

### 第 3 阶段：第 7-10 天，补验证闭环

- 对公共 op 补最小 correctness 回归
- 补 import/CLI/smoke 测试
- 明确 nightly / self-hosted GPU 流程
- 清理不稳定 benchmark 叙事

交付物：

- 最小 CI 闭环
- 开源前 checklist
- 可公开 benchmark 图

### 第 4 阶段：第 11-14 天，开源清理与发布

- 清理许可证和仓库根文件
- 检查依赖引用和第三方归属
- 准备 GitHub 首发 release notes
- 准备 issue template / PR template

交付物：

- 开源可发布仓库
- `v0.1.0-preview` release notes

## 7. 开源前检查清单

### 文档

- [ ] README 首页已重写
- [ ] 一句话定位清晰
- [ ] TNCC 与 Iris / NCCL-RCCL 的区别已写明
- [ ] support matrix 已写明
- [ ] 已知边界已写明

### 视觉

- [ ] README 架构图已完成
- [ ] SVG 源文件已入库
- [ ] 图题和图中文字统一为 TNCC

### 代码

- [ ] `tncc.ops.*` 为默认 public surface
- [ ] contract 说明齐全
- [ ] 错误提示可读
- [ ] package/import/CLI 可用

### 验证

- [ ] 单元测试可跑
- [ ] 最小集成测试可跑
- [ ] 至少一个 GPU correctness 路径可重复验证
- [ ] benchmark headline 与数据一致

### 合规

- [ ] LICENSE 明确
- [ ] 第三方引用明确
- [ ] 无私有路径/敏感信息
- [ ] 仓库命名已统一为 TNCC

## 8. 首发版本的“不要做”

- 不要试图一次性把所有 pattern、transport、硬件、world size 讲全
- 不要把 research prototype 包装成 production-ready system
- 不要为了“看起来强”写超出验证面的 headline
- 不要在 README 首页堆过多 benchmark 图
- 不要把底层 primitive 暴露成用户默认使用方式

## 9. 建议的开源顺序

建议按这个顺序推进，而不是平均用力：

1. README + 主架构图
2. 公共 API 收口
3. 最小示例
4. support matrix
5. 最小 correctness + CI
6. benchmark headline 清理
7. 开源合规与仓库卫生

## 10. 对首发版本的建议判断

如果你想尽快放出一个“麻雀虽小，五脏俱全，且特点鲜明”的版本，我建议：

- **现在就可以开始准备开源，不必等到所有路径都成熟**
- 但必须主动把首发定义成：
  - 小范围
  - 高辨识度
  - 强公共语义
  - 强文档
  - 强边界说明

最适合 TNCC 首发的姿态不是“我已经做完了一个完整通信系统”，而是：

> 我已经做出了一个有清晰方向、有稳定公共层、有最小验证闭环、并且在 tile-native collective 这件事上足够有新意的开源雏形。

---

## 附：建议的首发仓库首页结构

```text
Title
One-line positioning
Architecture figure
Why TNCC
Key features
Quick start
Minimal examples
Supported runtime surface
Performance snapshot
Roadmap
Citation / License / Contributing
```
