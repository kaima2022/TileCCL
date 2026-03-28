# tncc/primitives/ - Core Primitive Layer

三大原语 compute / memory / communication 地位平等，加 collectives 作为高层组合。

## 结构

| 文件 | 核心函数 |
|------|---------|
| `compute.py` | `tile_dot`, `tile_reduce`, `tile_reduce_max/min`, `tile_elementwise`, `tile_cast`, `tile_zeros`, `tile_full` |
| `memory.py` | `tile_load`, `tile_store`, `tile_copy`, `make_block_offsets` |
| `communication.py` | `tile_remote_load`, `tile_remote_store`, `tile_put`, `tile_get` |
| `collectives.py` | Device: `tile_allreduce`, `tile_allgather`, `tile_scatter`, `tile_reduce_scatter`, `tile_broadcast`; Host launchers: `allreduce`, `allgather`, `broadcast`, `scatter`, `reduce_scatter` |

## 约束

- 所有 device-side 函数必须 `@triton.jit`。
- Communication 原语内部调用 `translate_ptr` 做指针翻译。
- Value-based（寄存器到远端内存，细粒度）vs Pointer-based（内存到内存，粗粒度）。
- 命名统一 `tile_*` 前缀。
