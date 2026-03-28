# tncc/memory/ - Symmetric Heap & Memory Abstraction

## 结构

| 文件 | 职责 |
|------|------|
| `symmetric_heap.py` | `SymmetricHeap`：堆分配、IPC 交换、peer mapping |
| `allocators.py` | `BaseSymmetricAllocator` ABC, `TorchBumpAllocator`；segment/export descriptor 体系 |
| `translation.py` | 指针翻译：`translate_ptr`, `remote_load`, `remote_store` |

## 核心概念

- **Symmetric heap**: 每个 GPU 分配相同大小的堆，通过 IPC 互相可见。
- **指针翻译**: `local_ptr -> offset -> remote_base + offset = remote_ptr`。
- **Allocator-first**: `BaseSymmetricAllocator` 定义分配/元数据/导出接口，`TorchBumpAllocator` 是当前默认实现。

## SymmetricHeap 主要能力

- `create_all()` — 单进程多 GPU 快速建堆
- `allocate_tensor()` / `get_heap_bases()` / `barrier()`
- `import_external_tensor()` / `as_symmetric()` — 外部张量导入
- `segment_metadata()` / `export_metadata()` / `allocator_memory_model()` — 结构化元数据
- `peer_import_metadata()` / `peer_segment_catalog()` / `peer_mapping_metadata()` — peer 映射查询

## 约束

- `heap_bases` tensor 必须在所有 `@triton.jit` 函数中作为参数传递。
- `translate_ptr` 是热路径，必须最小化指令数。
- 所有 device-side 函数必须是 `@triton.jit`。
