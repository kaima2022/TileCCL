# XTile - 下一代跨平台 Tile 通信库

## 项目定位
集各家之长的 Tile 通信库，目标：跨硬件可移植 + 编译器全可见 + 多尺度统一 + 多节点通信。

## 技术栈
- 主语言：Python + Triton（编译器全可见）
- 目标硬件：NVIDIA (Hopper/Blackwell) + AMD (CDNA3/CDNA4)
- 依赖：triton>=3.0, torch>=2.4
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
- ThunderKittens: 硬件感知 tile 尺寸设计

## 代码规范
- 所有 device-side 函数用 @triton.jit 装饰
- 所有 host-side API 兼容 PyTorch tensor
- 双 API 模式：value-based (寄存器<->远端) + pointer-based (内存<->内存)
- Atomic 语义：acquire/release/acq_rel + scope: block/gpu/sys
- 命名：snake_case，公开 API 带 docstring

## 当前阶段
Phase 0 - 基础设施搭建

## 重要约束
- 不包装 xSHMEM 为不透明字节码（保持编译器可见性）
- 不引入 NCCL/RCCL 作为依赖（纯 Triton 实现通信）
- 测试必须同时覆盖 NVIDIA 和 AMD（可分阶段）
