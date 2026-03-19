# xtile/backends/ - Hardware Abstraction Layer (HAL)

HAL 层抽象 NVIDIA 和 AMD 的硬件差异。

## 关键约束
- 每个 backend 必须完整实现 BackendInterface（base.py）
- 添加新 backend 时参考 hip.py 的实现模式
- IPC handle 交换通过 torch.distributed（不依赖 MPI）
- 所有底层 API 调用必须检查返回值

## 硬件差异对照
| 方面 | AMD (HIP) | NVIDIA (CUDA) |
|------|-----------|---------------|
| Warp size | 64 (wavefront) | 32 (warp) |
| Compute units | CU | SM |
| IPC API | hipIpc* | cudaIpc* |
| 互联 | Infinity Fabric | NVLink/NVSwitch |
| Memory scope | agent | device |
