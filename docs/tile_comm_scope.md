# XTile Tile 通信适用范围说明

日期：`2026-03-27`

## 目的

回答两个容易混在一起的问题：

1. 当前 XTile 的 tile 通信在 device 侧是不是本质上基于 `tl.load` / `tl.store`。
2. 这套机制是不是“所有 GPU 都适用”，还是只适用于某一类 GPU。

## 核心结论

- 是。当前 XTile 的 device-side tile 通信，本质上是：
  `translate_ptr(...) + Triton tl.load / tl.store / tl.atomic_*`
- 它不是通过 `NCCL` / `RCCL` / `NVSHMEM` 这类外部黑盒通信库发数据面。
- 它也不是“所有 GPU 都适用”。
- 更准确地说，它适用于：
  - Triton 能生成内核的 GPU
  - XTile 已实现 backend 的 GPU
  - 运行时能把对端 symmetric heap 建成“当前 GPU 可直接解引用”的地址映射
- 就当前仓库状态看，设计目标覆盖 NVIDIA 和 AMD；但“公开验证通过的 runtime surface”比这个目标更窄。

## 现在到底基于什么

XTile 的通信原语直接写在 Triton kernel 里。

- [`xtile/primitives/communication.py`](../xtile/primitives/communication.py)
  - `tile_remote_load(...)` 先 `translate_ptr(...)`，再做 `tl.load(...)`
  - `tile_remote_store(...)` 先 `translate_ptr(...)`，再做 `tl.store(...)`
- [`xtile/memory/translation.py`](../xtile/memory/translation.py)
  - `translate_ptr(...)` 只是基址加载和指针平移：
    `offset = ptr - heap_bases[from_rank]`
    `remote_ptr = heap_bases[to_rank] + offset`
- [`xtile/primitives/collectives.py`](../xtile/primitives/collectives.py)
  - 模块说明明确写了 collectives 是 pure `@triton.jit`
  - 远端访问也走 `translate_ptr`

所以，从编译器视角看，XTile 当前通信表达不是“调用一个通信 runtime API”，而是“翻译后的远端地址 + 标准 Triton memory op”。

## 它适用于哪些 GPU

### 1. 不是所有 GPU

它不是一个脱离 Triton 的通用 GPU 通信层。没有 Triton 后端、没有可用 backend、或者运行时无法建立 peer-dereferenceable 映射时，这条路径就不成立。

仓库的硬件抽象层只实现了两类 backend：

- [`xtile/backends/cuda.py`](../xtile/backends/cuda.py)
- [`xtile/backends/hip.py`](../xtile/backends/hip.py)

硬件检测也只在 `cuda` / `hip` 两类里选：

- [`xtile/backends/__init__.py`](../xtile/backends/__init__.py)

因此，当前代码语义上支持的是：

- NVIDIA GPU，经 CUDA backend
- AMD GPU，经 HIP backend

不在这两类 backend 里的 GPU，不属于当前实现范围。

### 2. 也不是“任何 Triton 支持的 GPU 自动都已验证”

这里要区分两层：

- 设计目标 / 代码路径：
  - README 与 `xtile/__init__.py` 都把目标写成 NVIDIA + AMD
- 当前公开验证范围：
  - 仓库中的真实 multiprocess remote-access 矩阵当前只公开证明了更小的 surface

已有文档已明确写出当前边界：

- [`docs/XTile现状流程_修订版.md`](./XTile现状流程_修订版.md)
  - “硬件验证范围：仅在 NVIDIA H100 上实测；AMD 待硬件验证，代码路径已就绪”
- [`docs/generated/triton_remote_access_multiprocess_matrix.json`](./generated/triton_remote_access_multiprocess_matrix.json)
  - 当前生成矩阵的 GPU 也是 `NVIDIA H100 PCIe`

所以，准确表述应该是：

- 代码结构上：面向 Triton 支持的 NVIDIA / AMD GPU
- 当前已公开验证上：主要收敛在 NVIDIA H100 的真实矩阵

## 运行时支持边界

### 单进程多卡

单进程多卡走的是 peer access 路径。

- [`xtile/memory/symmetric_heap.py`](../xtile/memory/symmetric_heap.py)
  - `create_all()` / 初始化逻辑会把 mode 设为 `single_process`
  - transport 记为 `peer_access`
- [`xtile/support.py`](../xtile/support.py)
  - `single_process` 下，`device_remote_access` 被描述为 `supported`

这意味着：如果当前机器和 backend 支持 peer access，那么单进程多卡的 remote dereference 是当前支持面的一部分。

### 多进程

多进程要更保守。当前自动 public path 只保留了一个 transport：

- [`xtile/memory/symmetric_heap.py`](../xtile/memory/symmetric_heap.py)
  - `_setup_multiprocess()` 默认策略只有 `ctypes_ipc`
  - `pytorch_ipc` 和 `peer_access_pointer_exchange` 只保留为 forced diagnostic path
- [`xtile/utils/feature_gates.py`](../xtile/utils/feature_gates.py)
  - `_VALIDATED_MULTIPROCESS_DEVICE_TRANSPORTS = {"ctypes_ipc"}`
  - `_VALIDATED_MULTIPROCESS_DEVICE_WORLD_SIZES = {2}`
- [`xtile/ops.py`](../xtile/ops.py)
  - 高层 op 在 launch 前会检查这个 runtime gate，不满足就直接报错

所以当前仓库的公开结论不是“multiprocess 普遍可用”，而是：

- `world_size=2`
- `transport_strategy='ctypes_ipc'`

这才是当前 validated public surface。

## 一张表说清楚

| 层次 | 当前结论 |
|------|------|
| device 表达 | `translate_ptr + tl.load/store/atomic` |
| 是否依赖外部通信库数据面 | 否 |
| 理论硬件范围 | Triton + XTile backend 可覆盖的 NVIDIA / AMD GPU |
| 是否对所有 GPU 适用 | 否 |
| 单进程多卡 | 支持面包含 peer access 路径 |
| 多进程 public surface | 仅 `2 GPU + ctypes_ipc` |
| `pytorch_ipc` / `peer_access_pointer_exchange` | 仅诊断面，不是 public device path |
| 当前公开实测硬件 | NVIDIA H100 |
| AMD 当前状态 | 代码路径已就绪，仓库内未见同等级公开真机矩阵 |

## 推荐表述

如果要对外一句话描述，建议写成：

> XTile 当前的 tile 通信不是独立通信指令或黑盒 runtime，而是基于 Triton 的远端地址翻译加标准 `tl.load` / `tl.store` / `tl.atomic_*`。因此它并不对“所有 GPU”通用，而是面向 Triton 支持且已由 XTile backend 接入的 NVIDIA / AMD GPU；其中当前公开验证通过的 multiprocess device-side surface 仅为 `world_size=2 + ctypes_ipc`。

## 延伸阅读

- 详细机制：[`docs/XTile远端访存底层机制报告.md`](./XTile远端访存底层机制报告.md)
- 当前流程与边界：[`docs/XTile现状流程_修订版.md`](./XTile现状流程_修订版.md)
