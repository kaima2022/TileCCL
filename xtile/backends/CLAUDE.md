# xtile/backends/ - Hardware Abstraction Layer (HAL)

HAL 层抽象 NVIDIA 和 AMD 的硬件差异。

## 关键约束
- 每个 backend 必须完整实现 BackendInterface（base.py）
- 添加新 backend 时参考 hip.py 的实现模式
- IPC handle 交换通过 torch.distributed（不依赖 MPI）
- **所有底层 API 调用必须检查返回值**，不使用 `assert` 做运行时检查
- ctypes wrapper 是进程单例（模块级 `_hip` / `_cuda` 实例）
- peer access 启用失败时 "already enabled" 错误码可安全忽略（HIP=704, CUDA=704）

## 硬件差异对照
| 方面 | AMD (HIP) | NVIDIA (CUDA) |
|------|-----------|---------------|
| Warp size | 64 (wavefront) | 32 (warp) |
| Compute units | CU (304 on MI300X) | SM (132 on H100) |
| IPC handle | hipIpcMemHandle_t (64B) | cudaIpcMemHandle_t (64B) |
| IPC 标志 | flags=0 | flags=1 (LazyEnablePeerAccess) |
| 互联 | Infinity Fabric | NVLink / NVSwitch |
| Memory scope | agent scope | device scope |
| 拓扑检测 | peer-access matrix | nvidia-smi topo -m / NVML |
| 运行时库 | libamdhip64.so | libcudart.so |
