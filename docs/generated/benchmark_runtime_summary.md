# TNCC Benchmark Runtime Summary

> Generated at UTC: `2026-03-25T18:21:01.551576+00:00`
> This file is auto-generated from canonical benchmark JSON artifacts.

## Artifact Status

| Benchmark | Status | File |
|------|------|------|
| GEMM | available | `figures/data/gemm_latest.json` |
| P2P | available | `figures/data/p2p_latest.json` |
| Pattern | available | `figures/data/pattern_overlap_latest.json` |
| Comm-only Collectives | available | `figures/data/collective_comm_only_latest.json` |
| Collective vs bulk_sync | available | `figures/data/collective_bulk_sync_latest.json` |

## Host Topology

- GPU SKU: `2 x NVIDIA H100 PCIe`
- GPU-GPU interconnect for multi-GPU runs: `NVLink (NV12)`

## Runtime Support Snapshots

- GEMM: source=gemm_latest.json | run=2026-03-23 | backend=cuda, ws=1, heap=none | cmd=/home/makai/TNCC/tests/benchmarks/bench_gemm.py --repeats 3 --output-json figures/data/gemm_latest.json
- P2P: source=p2p_latest.json | run=2026-03-23 | backend=cuda, ws=2, heap=single_process, transport=peer_access, reduce_scatter=supported | cmd=/home/makai/TNCC/tests/benchmarks/bench_p2p_translate.py --output-json figures/data/p2p_latest.json
- Pattern: source=pattern_overlap_latest.json | run=2026-03-23 | backend=cuda, ws=2, heap=single_process, transport=peer_access, gemm_allscatter=supported | cmd=/home/makai/TNCC/tests/benchmarks/bench_patterns.py --warmup 3 --iters 10 --output-json figures/data/pattern_overlap_latest.json
- Comm-only collectives: source=collective_comm_only_latest.json | run=2026-03-25 | backend=cuda, ws=2, heap=multiprocess, transport=ctypes_ipc | cmd=fig6 mixed-source dataset: small-message steady host-wall + unified 256 KiB anchors + allreduce 1/2 MiB sweep
- Collective vs bulk_sync: source=collective_bulk_sync_latest.json | run=2026-03-23 | backend=cuda, ws=2, heap=single_process, transport=peer_access | cmd=/home/makai/TNCC/tests/benchmarks/bench_collective_bulk_sync.py

## Execution Paths

- P2P/collective runtime: reduce_scatter.reference=supported, reduce_scatter.device=unsupported

## Headline Metrics

### GEMM

- `4096³ fp16`: 95.4% of torch.matmul
- `4096³ bf16`: 91.9% of torch.matmul
- `8192³ fp16`: 82.2% of torch.matmul
- `8192³ bf16`: 84.4% of torch.matmul

### P2P

- best read: 248.76 GB/s, variant=evict_first, block_size=8192, grid=228
- best write: 248.40 GB/s, variant=wt+evict, block_size=8192, grid=114

### Pattern Overlap

- best speedup vs bulk_sync: 1.635×
- best size: 8192×4608×36864, pattern=wg_specialized, speedup=1.635×

### Comm-only Collectives

- allreduce: TNCC peak 0.03 GB/s, NCCL peak 0.44 GB/s, best ratio 0.997×
- allgather: TNCC peak 0.06 GB/s, NCCL peak 0.04 GB/s, best ratio 1.471×
- scatter: TNCC peak 0.06 GB/s, NCCL peak 0.04 GB/s, best ratio 1.466×
- reduce_scatter: TNCC peak 0.03 GB/s, NCCL peak 0.11 GB/s, best ratio 1.002×
- broadcast: TNCC peak 0.11 GB/s, NCCL peak 0.11 GB/s, best ratio 1.004×

### Collective vs bulk_sync

- best speedup vs bulk_sync: 2.221× (reduce_scatter, 64 KiB)
- allreduce: best speedup 2.049×
- allgather: best speedup 1.782×
- scatter: best speedup 1.001×
- reduce_scatter: best speedup 2.221×
- broadcast: best speedup 0.652×
