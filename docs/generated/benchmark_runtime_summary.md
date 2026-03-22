# XTile Benchmark Runtime Summary

> Generated at UTC: `2026-03-21T20:34:05.569066+00:00`
> This file is auto-generated from canonical benchmark JSON artifacts.

## Artifact Status

| Benchmark | Status | File |
|------|------|------|
| GEMM | available | `figures/data/gemm_latest.json` |
| P2P | available | `figures/data/p2p_latest.json` |
| Pattern | available | `figures/data/pattern_overlap_latest.json` |

## Runtime Support Snapshots

- GEMM: source=gemm_latest.json | run=2026-03-21 | backend=cuda, ws=1, heap=none | cmd=tests/benchmarks/bench_gemm.py --repeats 3 --output-json figures/data/gemm_latest.json
- P2P: source=p2p_latest.json | run=2026-03-21 | backend=cuda, ws=2, heap=single_process, transport=peer_access, reduce_scatter=supported | cmd=tests/benchmarks/bench_p2p_translate.py --output-json figures/data/p2p_latest.json
- Pattern: source=pattern_overlap_latest.json | run=2026-03-21 | backend=cuda, ws=2, heap=single_process, transport=peer_access, gemm_allscatter=supported | cmd=tests/benchmarks/bench_patterns.py --warmup 3 --iters 10 --output-json figures/data/pattern_overlap_latest.json

## Execution Paths

- P2P/collective runtime: reduce_scatter.reference=supported, reduce_scatter.device=unsupported

## Headline Metrics

### GEMM

- `4096³ fp16`: 94.9% of torch.matmul
- `4096³ bf16`: 91.1% of torch.matmul
- `8192³ fp16`: 83.0% of torch.matmul
- `8192³ bf16`: 83.5% of torch.matmul

### P2P

- best read: 248.74 GB/s, variant=evict_first, block_size=4096, grid=114
- best write: 248.43 GB/s, variant=wt, block_size=8192, grid=114

### Pattern Overlap

- best speedup vs bulk_sync: 1.667×
- best size: 8192×4608×36864, pattern=wg_specialized, speedup=1.667×
