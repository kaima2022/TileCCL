# tncc/backends/ - Hardware Abstraction Layer

## 结构

| 文件 | 职责 |
|------|------|
| `base.py` | `BackendInterface` ABC, `TopologyInfo`, `DeviceProperties` |
| `cuda.py` | NVIDIA CUDA 实现 |
| `hip.py` | AMD HIP 实现 |

## 关键约束

- 每个 backend 必须完整实现 `BackendInterface`（IPC / Memory / Topology / DeviceProperties）。
- ctypes wrapper 是进程单例（模块级 `_hip` / `_cuda` 实例）。
- IPC handle 交换通过 `torch.distributed`，不依赖 MPI。
- 所有底层 API 调用必须检查返回值，不用 `assert` 做运行时检查。
- Peer access 启用失败时 "already enabled" 错误码可安全忽略（HIP=704, CUDA=704）。
- 添加新 backend 参考 `hip.py` 实现模式。

## 硬件差异

| 方面 | AMD (HIP) | NVIDIA (CUDA) |
|------|-----------|---------------|
| Warp size | 64 (wavefront) | 32 (warp) |
| Compute units | CU | SM |
| IPC handle | `hipIpcMemHandle_t` (64B) | `cudaIpcMemHandle_t` (64B) |
| IPC flags | `flags=0` | `flags=1` (LazyEnablePeerAccess) |
| Interconnect | Infinity Fabric | NVLink / NVSwitch |
| Memory scope | agent scope | device scope |
| Runtime lib | `libamdhip64.so` | `libcudart.so` |
