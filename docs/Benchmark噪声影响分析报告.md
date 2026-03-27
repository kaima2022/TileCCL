# Benchmark 噪声影响分析报告（2026-03-26）

原始汇总见：

- [noise_study summary](../figures/data/noise_study/20260326T192227Z/summary.json)
- [noise_study latest](../figures/data/noise_study/latest.json)
- [analyze_benchmark_noise.py](../scripts/analyze_benchmark_noise.py)

## 目的

回答三个问题：

1. `NCCL` 在当前共享 GPU 环境下会不会抖。
2. 共享 GPU 到底影响什么，是 correctness、绝对性能，还是相对排序。
3. `figures/` 对应的各类实验，哪些还能用于结构性结论，哪些不能用于定量结论。

## 实验环境

- 时间：`2026-03-26 UTC`
- GPU：`2 x NVIDIA H100 PCIe`
- 共享状态不是猜测，而是实测污染：
  - `GPU0` 常驻 `VLLM::EngineCore`、`llama-box`、`python`，显存占用约 `34 GiB`
  - `GPU1` 常驻 `llama-box`、`python`，并且在多次采样中持续 `100%` GPU utilization
- 本次 study 在当前工作树上执行；工作树包含未提交的 collective 相关改动。

最后一条很重要：本文不把“当前数值”和历史 `*_latest.json` 的绝对值直接做因果归因。本文只分析同一代码快照下的 run-to-run 噪声敏感性。

## 当前共享 GPU 画像

基于 `2026-03-26 21:48 UTC` 的实时快照：

- `GPU0`
  - `34197 MiB / 81559 MiB`
  - `SM = 0%`
  - 常驻进程：
    - `VLLM::EngineCore`：`33134 MiB`
    - `python (pid=1054772)`：`590 MiB`
    - `llama-box`：`448 MiB`
- `GPU1`
  - `1055 MiB / 81559 MiB`
  - `SM = 99~100%`
  - 常驻进程：
    - `python (pid=1054772)`：`590 MiB`
    - `llama-box`：`448 MiB`

这两个 GPU 的压力形态完全不同：

- `GPU0` 是“大显存常驻，但基本不跑计算”
- `GPU1` 是“小显存占用，但长期满 SM”

因此当前最值得警惕的污染源，不是 `GPU0` 的大显存占用，而是 `GPU1` 上长期持续的执行压力。

## GPU1 为何长期 99~100% SM

目前能确定的事实：

- `SM` 压力几乎持续归因到同一个进程：`python -`，`pid=1054772`
- 该进程属于当前用户 `makai`
- 启动时间是 `2026-03-25`
- 当前工作目录是 `/home/makai/XTile`
- 父进程已经退出，`PPID=1`
- 进程命令行只有 `python -`，说明它是通过 stdin 启动的匿名 Python 任务
- 该进程长期打开大量 `/dev/nvidia0`、`/dev/nvidia1`、`/dev/nvidiactl`、`/dev/nvidia-uvm`
- 其地址空间加载了：
  - `torch`
  - `triton`
  - `nccl`
  - `cublas`
  - `cuda runtime`
- 进程共有 `67` 个线程，只有主线程处于 `R`，绝大多数线程在 `futex_wait_queue_me`

因此，当前最稳妥的判断是：

- `GPU1` 的长期 `99~100% SM` 主要是 `pid=1054772` 这个长生命周期的 PyTorch/Triton 任务在持续向 `GPU1` 提交 compute kernel
- 它不是“显存占着没动”的空闲进程
- 它也不是 `llama-box` 或 `VLLM::EngineCore` 直接造成的那条持续 `99~100% SM`

目前不能完全确定的部分：

- 因为该进程是 `python -`，父进程已退出，无法从 `cmdline` 恢复原始脚本内容
- 在不 attach 调试器/不打断进程的前提下，无法把它精确还原到某一段 Python 源码

所以这里的结论是“高置信度定位到具体进程类型和运行形态”，但不是“已经精确还原出原始脚本文本”。

## 方法

- `fig1 / fig2 / fig3`：各重复 `5` 轮
- `fig6 allreduce / exchange / reduce_scatter(4 KiB)`：各重复 `3` 轮
- `fig6 reduce_scatter @ 256 KiB`：做 `1` 个单点探针
- `fig7`：重复 `3` 轮
- 每轮 benchmark 前后都抓一份环境污染快照
- 重点统计：
  - 跨轮 `median latency / bandwidth` 的 `spread`
  - 组内 timed-iteration 的最坏 slowdown factor
  - 最优方案 / 最佳 pattern 是否翻转
  - benchmark 是否超时或 correctness 失效

## 结论摘要

1. `NCCL` 也会抖，但不是所有实验都同样抖。
2. 当前共享环境主要污染的是 performance stability，不是普遍破坏 correctness。
3. 当前更关键的干扰源是“参与通信的 GPU 被持续占 SM”，不是“只有显存被占”。
4. 真正最受害的是“短时、同步密集、分阶段握手多”的实验面，不是所有 GPU benchmark。
5. `fig2 P2P` 和 `fig3 patterns` 基本稳定；`fig6 allreduce` 和 `fig6 reduce_scatter @ 256 KiB` 明显失真。
6. `fig7` 当前不能拿来做噪声归因，因为它在当前工作树上已经不满足“先 correctness，再 performance”的前提。

## 先回答：NCCL 也抖动吗

会，但程度依赖 benchmark surface。

- `fig6 allreduce @ 256 KiB`：`NCCL` 基本稳定。
  - `NCCL latency spread = 1.17%`
  - `XTile latency spread = 1546.61%`
  - 3 轮里 `XTile median latency` 分别是 `7.78 ms / 8.74 ms / 142.97 ms`
  - 同一组里 `NCCL median latency` 只有 `2.355 ms ~ 2.383 ms`
- `fig6 broadcast @ 4 KiB`：`NCCL` 明显抖动。
  - `NCCL latency spread = 58.92%`
  - `XTile latency spread = 4.60%`
- `fig6 scatter @ 4 KiB`：两边都抖，但 `XTile` 更大。
  - `NCCL latency spread = 11.27%`
  - `XTile latency spread = 94.88%`
- `fig6 reduce_scatter @ 4 KiB`：两边都比较稳定。
  - `NCCL latency spread = 0.17%`
  - `XTile latency spread = 6.23%`

所以结论不是“只有 XTile 抖、NCCL 完全不抖”，也不是“只要是 NCCL 就完全可靠”。正确表述是：

- `NCCL` 也会受共享 GPU 影响
- 但在本次实验里，`NCCL` 对 `allreduce` 和多数大消息 case 的抗扰动能力明显强于当前 `XTile`
- `NCCL` 的抖动更容易出现在小消息、极短时、固定开销占主导的 case 上

## 为什么非 NVLink 任务也会影响通信

这部分单独做了 controlled study。原始数据在：

- [comm_interference_study summary](../figures/data/comm_interference_study/20260326T213552Z/summary.json)
- [investigate_comm_interference.py](../scripts/investigate_comm_interference.py)

实验设计很简单：

- 只在参与通信的 `GPU0` 上额外加本地负载
- 不增加额外 `GPU0 <-> GPU1` 通信流量
- 对比：
  - `none`
  - `resident_only`
  - `sm_burn`
- collective 使用 `host_wall`
- 每个条件重复 `3` 次

### 1. 显存常驻本身不是主因

`resident_only` 相比 `none`：

- `P2P read`：`+0.01%`
- `P2P write`：`-0.05%`
- `broadcast 4 KiB / 256 KiB` 基本不变
- `NCCL allreduce @ 256 KiB`：`2.405 ms -> 2.405 ms`

因此，当前证据支持：

- 单纯“显存被别人占着”不是主要干扰源

### 2. 本地 SM 压力会明显拖慢 collective

`sm_burn` 相比 `none`：

- `P2P read bandwidth`
  - `248.71 -> 247.79 GB/s`
  - 仅 `-0.37%`
- `P2P write bandwidth`
  - `247.85 -> 247.03 GB/s`
  - 仅 `-0.33%`

但 collective 端到端延迟明显上升：

- `allreduce @ 4 KiB`
  - `XTile: 7.32 -> 23.43 ms`，约 `3.20x`
  - `NCCL: 2.40 -> 2.78 ms`，约 `1.16x`
- `allreduce @ 256 KiB`
  - `XTile: 20.01 -> 30.84 ms`，约 `1.54x`
  - `NCCL: 2.41 -> 2.80 ms`，约 `1.16x`
- `broadcast @ 256 KiB`
  - `XTile: 2.39 -> 2.95 ms`，约 `1.23x`
  - `NCCL: 2.40 -> 2.70 ms`，约 `1.12x`

这说明：

- 不是只有“抢 NVLink 带宽”才会影响通信
- 只要参与 collective 的 GPU 没法及时跑通信 kernel、及时推进协议，通信 benchmark 就会变慢

### 3. 根因不是链路被抢，而是协议推进被拖慢

纯 `P2P` 大吞吐面在 [bench_p2p_translate.py](/home/makai/XTile/tests/benchmarks/bench_p2p_translate.py#L50) 更接近长 steady-state 的单边 remote load/store，因此对 peer 的“协议配合推进”依赖小。

但 `allreduce` 在 [collectives.py](/home/makai/XTile/xtile/primitives/collectives.py#L621) 到 [collectives.py](/home/makai/XTile/xtile/primitives/collectives.py#L685) 以及 [collectives.py](/home/makai/XTile/xtile/primitives/collectives.py#L733) 到 [collectives.py](/home/makai/XTile/xtile/primitives/collectives.py#L782) 有显式：

- staging
- `published_epoch`
- 轮询等待
- peer 读取
- ack / consume 握手

这类协议路径要求双方 GPU 都持续提供进度。peer GPU 即使没有抢 `NVLink` 带宽，只要 SM 被长期占着，协议推进就会慢。

### 4. 回答“到底是 SM 占用还是显存占用”

当前证据支持的准确表述是：

- 主要问题是“活跃的执行压力”，尤其是 `SM` 持续占用
- 单纯显存常驻不是主要问题
- 不能把这句话过度简化成“只有 SM 会影响、显存永远没影响”

原因是：

- `resident_only` 已经表明显存常驻本身影响很小
- 但如果任务真的持续打本地 HBM/L2/atomic，仍然可能影响通信
- 只是本次最强、最稳定的证据，确实来自 `SM` 压力，而不是显存占用

## 每类实验受影响情况

### 1. `fig1` GEMM

影响是“部分明显，部分很小”。

- 大尺寸、计算主导的配置相对稳定：
  - `4096^3 fp16` 比值 spread `2.57%`
  - `8192^3 bf16` 比值 spread `3.56%`
- 小中尺寸配置波动很大：
  - `1024^3 bf16` 比值 spread `62.78%`
  - `2048^3 bf16` 比值 spread `51.80%`
  - `2048^3 bf16` 的 `XTile TFLOPS spread = 147.04%`

代表性原始值：

- `1024^3 bf16` 的 `XTile TFLOPS` 在 5 轮里是 `38.68 / 15.39 / 38.82 / 39.18 / 39.03`
- `2048^3 bf16` 的 `XTile TFLOPS` 在 5 轮里是 `92.01 / 92.26 / 269.70 / 121.03 / 269.97`

解释：

- 长、算力主导的 kernel 更像 steady-state compute，外界污染不容易放大
- 小中尺寸更容易被 launch / clock / 短时调度扰动放大

结论：

- `fig1` 还能用来讨论大尺寸趋势
- 不要在共享 GPU 上拿小中尺寸 GEMM 直接下 headline

### 2. `fig2` P2P

这是本次最稳定的一类实验。

- `best read bandwidth spread = 0.036%`
- `best write bandwidth spread = 0.018%`
- 5 轮都能稳定跑在 `248.7 GB/s` 左右

影响点不在绝对带宽，而在“谁是最优微变体”：

- `best read` 在 `baseline / cg / evict_first` 之间切换
- `best write` 在 `baseline / wt / wt+evict` 之间切换

解释：

- 大块 persistent P2P streaming kernel 的 steady-state 足够长，外部噪声不容易改变总吞吐
- 但几个微变体之间本来就只差很小，winner label 会在噪声地板内翻转

结论：

- `fig2` 的绝对带宽结论仍然可信
- 不要在共享 GPU 上放大解释“哪个 cache modifier 一定更优”

### 3. `fig3` Patterns

这类实验在当前环境下也很稳定。

- 两个测试尺寸在 `5/5` 轮里最佳 pattern 都没有翻转，都是 `fused_sequential`
- `best speedup vs bulk_sync spread`
  - `4096 x 4096 x 4096`：`1.19%`
  - `8192 x 4608 x 14336`：`1.49%`

结论：

- `fig3` 的结构性结论目前可信
- 在当前共享环境下，它仍可用于回答“哪个 overlap pattern 更好”

### 4. `fig6` Comm-Only Collectives

这是最需要小心的一类。

#### `allreduce`

`XTile` 明显受污染，`NCCL` 只有轻微波动。

- `4 KiB`
  - `NCCL latency spread = 0.05%`
  - `XTile latency spread = 67.37%`
- `256 KiB`
  - `NCCL latency spread = 1.17%`
  - `XTile latency spread = 1546.61%`
  - 单轮组内最坏 slowdown factor：`XTile = 59.18x`，`NCCL = 1.02x`

这说明当前共享环境会把 `XTile allreduce` 的同步/分块协议代价极大放大，而 `NCCL` 在同一硬件上没有出现同数量级失真。

#### `allgather / scatter / broadcast`

影响是“case by case”的，不是一刀切。

- `allgather`
  - `NCCL` 很稳定
  - `XTile` 有可见波动，但不至于像 allreduce 那样崩
- `scatter`
  - `4 KiB` 波动很大，`XTile latency spread = 94.88%`
  - `256 KiB` 反而比较稳定，`XTile latency spread = 2.15%`
- `broadcast`
  - `4 KiB` 是本次最能说明“`NCCL` 也会抖”的 case
  - `NCCL latency spread = 58.92%`
  - `256 KiB` 仍有明显波动，`NCCL latency spread = 23.21%`

解释：

- 小消息里，固定开销、同步点和短时调度扰动占主导
- 大消息里，如果实现是稳定的 steady-state 数据流，波动会小一些
- 但如果协议本身同步很重，外部干扰会被成倍放大

#### `reduce_scatter`

必须分开看。

- `4 KiB`：稳定
  - `NCCL latency spread = 0.17%`
  - `XTile latency spread = 6.23%`
- `256 KiB`：单点探针在 `180s` 内超时，没有拿到有效结果

这意味着：

- 小消息 `reduce_scatter` 还能测
- 大消息 `reduce_scatter` 在当前共享环境下已经不是“数字会抖”这么简单，而是 benchmark surface 直接不可用

结论：

- `fig6` 不能再被当成一张“当前共享 GPU 上随手跑一下也能定量比较”的图
- 其中 `allreduce` 和大消息 `reduce_scatter` 最不能信
- 如果一定要在共享环境下看 `fig6`，只能看 very limited 的结构性信息，不能拿绝对数字下结论

### 5. `fig7` Collective vs Bulk-Sync

这类实验当前不适合拿来做噪声归因。

原因不是单纯“数值抖”，而是 benchmark surface 自身已经不干净：

- 本次 `3/3` 轮 `4 KiB` 结果里出现 correctness failure
- 但历史 canonical [collective_bulk_sync_latest.json](../figures/data/collective_bulk_sync_latest.json) 是全对的

在这个前提下，`fig7` 的性能数字不能直接拿来解释“共享 GPU 到底造成了多少噪声”。

这里不能把 correctness failure 直接归因给共享 GPU，因为当前工作树本身就含有未提交的 collective 相关改动；`fig7` 在本次 study 里是一个混杂面，不是干净的归因面。

如果只看 timing 现象，仍然能看到一个很强的信号：

- `bulk_sync scatter @ 4 KiB` latency spread `205.82%`
- `bulk_sync broadcast @ 4 KiB` latency spread `221.00%`
- 同时对应的 `xtile` device path 却几乎不动

这说明 host-orchestrated、多阶段、同步密集的 baseline 对外部干扰极其敏感。

但因为 correctness 已经失效，这里只能作为 cautionary note，不能作为正式性能证据。

## 这次噪声到底影响了什么

### 影响很小

- 长 steady-state 的 P2P 大带宽测量
- overlap pattern 的 winner 排序
- 大尺寸、算力主导的 GEMM

### 影响中等

- 小消息 broadcast / scatter 的 latency
- 小中尺寸 GEMM 的绝对 TFLOPS 与相对比值
- 微优化 winner label

### 影响极大

- 大消息 `allreduce`
- 大消息 `reduce_scatter`
- host-orchestrated / bulk-sync 类基准
- 任何依赖多阶段 barrier、slot 握手、短 critical section 的 benchmark

## Correctness 与 Performance 要分开看

- 已完成的 `fig6` comm-only case 没有 correctness failure
- `fig2` 和 `fig3` 也没有暴露 correctness 问题
- `fig7` 当前工作树上的 benchmark surface 出现 correctness failure，因此它的 performance 数字不应继续解释

这意味着共享 GPU 当前最主要污染的是：

- latency / bandwidth 的稳定性
- speedup / ratio 的可解释性
- benchmark completion time

但它也可能把一个原本边界就脆弱的 benchmark surface 直接推到不可用状态。

## 后续使用建议

1. 共享 GPU 上仍可保留：
   - `fig2` 的吞吐级判断
   - `fig3` 的 pattern 排序
   - `fig1` 的大尺寸趋势判断
2. 共享 GPU 上不要直接拿来下 headline：
   - `fig1` 的小中尺寸比值
   - `fig6` 的绝对 latency / bandwidth 对比
   - 任何“谁比谁快 X%”的 collective headline
3. 必须等 GPU 隔离后再跑：
   - `fig6 allreduce`
   - `fig6 reduce_scatter @ >= 256 KiB`
   - 任何要上图、要对外引用的 collective 定量结论
4. `fig7` 先恢复 correctness，再谈噪声分析。

## 一句话结论

共享 GPU 确实会影响我们，而且不是“所有实验一视同仁”。

- `NCCL` 也会抖，但通常比当前 `XTile` 的脆弱 collective surface 更抗扰动
- 真正被打穿的是 `fig6` 大消息 collective 和 host-orchestrated baseline
- `fig2`、`fig3`、以及 `fig1` 的大尺寸部分，当前仍可用于有限结论
