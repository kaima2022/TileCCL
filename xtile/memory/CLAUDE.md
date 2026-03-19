# xtile/memory/ - Symmetric Heap & Pointer Translation

## 核心概念
Symmetric heap: 每个 GPU 分配相同大小的堆，通过 IPC 互相可见。
指针翻译: local_ptr → offset → remote_base + offset = remote_ptr。

## 关键参考
- Iris 论文 Listing 1: translate 函数
- Iris 的 __translate 实现（Python host-side + Triton device-side）

## 约束
- heap_bases tensor 必须在所有 @triton.jit 函数中作为参数传递
- translate_ptr 是热路径，必须最小化指令数
- 所有 device-side 函数必须是 @triton.jit
