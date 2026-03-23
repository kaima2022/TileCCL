# XTile 当前实验状态

本文件不是时间线日志，只保留当前有效实验结论、canonical 数据位置。

## 当前基线环境

- GPU SKU：`2 x NVIDIA H100 PCIe`
- GPU-GPU interconnect：`NVLink (NV12)`
- 当前多 GPU baseline：`world_size=2`
- 最稳定单进程路径：`heap_mode=single_process` + `peer_access`
- 当前多进程公开 baseline：`ctypes_ipc`

## Canonical 实验产物

| 类别 | 结论含义 | 路径 |
|------|----------|------|
| GEMM | XTile GEMM 与 `torch.matmul` 的比值基线 | `figures/data/gemm_latest.json` |
| P2P | 对称堆 + `translate_ptr` 的跨 GPU 读写带宽 | `figures/data/p2p_latest.json` |
| Pattern overlap | overlap pattern 相对 `bulk_sync` 的收益 | `figures/data/pattern_overlap_latest.json` |
| Comm-only collectives | 纯通信 collective 相对 NCCL 的当前差距 | `figures/data/collective_comm_only_latest.json` |
| Collective vs bulk_sync | XTile collective 相对本仓库 `bulk_sync` 的收益 | `figures/data/collective_bulk_sync_latest.json` |
| 自动摘要 | 汇总运行时、拓扑和 headline 指标 | `docs/generated/benchmark_runtime_summary.md` |

## 当前有效结论

- GEMM：
  - 当前 best ratio `0.9542x`
  - 大尺寸 `8192` 级别仍约 `0.82x ~ 0.84x`
- P2P：
  - best read `248.76 GB/s`
  - best write `248.40 GB/s`
  - 当前最强路径来自 `single_process + peer_access`
- Pattern overlap：
  - best speedup vs `bulk_sync` 为 `1.635x`
  - 说明 tile-level overlap 已有结构性收益，但不是所有尺寸都同样占优
- Comm-only collectives vs NCCL：
  - 当前 best case 是 `scatter 0.781x`
  - `allreduce` 仅 `0.124x`
  - 结论是“公共 contract 已有、性能仍明显落后 NCCL”
- Collective vs `bulk_sync`：
  - best case 是 `reduce_scatter 2.221x`
  - `allreduce 2.049x`
  - `broadcast` 当前仍落后 `bulk_sync`

## 当前实验边界

- 图和 headline 应只信 `figures/data/*.json` 的 canonical 结果。
- `H100 PCIe` 指 GPU SKU，不代表双卡之间只走 PCIe；本机双卡互联实际是 `NVLink (NV12)`。
- multiprocess 正式 baseline 目前只应写 `world_size=2 + ctypes_ipc`。
- `pytorch_ipc` 和 `peer_access_pointer_exchange` 不能写成“已支持”，当前仍是诊断路径。
- comm-only collective 的性能结论应与 `single_process + peer_access` 的 overlap/GEMM 结论分开写，不能混为同一条 headline。

