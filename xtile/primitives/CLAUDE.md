# xtile/primitives/ - Core Primitive Layer

三大原语：compute / memory / communication，地位平等。

## 约束
- 所有 device-side 函数必须 @triton.jit
- Communication 原语内部调用 translate_ptr 做指针翻译
- Value-based: 寄存器直接到远端内存（细粒度）
- Pointer-based: 内存到内存批量传输（粗粒度）
- 命名统一: tile_xxx 前缀

## 参考
- Iris 论文 Table 3: Device-side API
- TileLink: tile_push_data / tile_pull_data 概念
