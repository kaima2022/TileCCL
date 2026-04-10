# SPDX-License-Identifier: Apache-2.0
"""Tests for the benchmark-summary Markdown exporter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_export_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "export_benchmark_summary.py"
    spec = importlib.util.spec_from_file_location("export_benchmark_summary_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault("export_benchmark_summary_test", module)
    spec.loader.exec_module(module)
    return module


def test_build_summary_document_contains_runtime_and_headlines() -> None:
    """Markdown export should include runtime support and headline metrics."""
    exporter = _load_export_module()

    gemm_payload = {
        "generated_at_utc": "2026-03-21T18:00:00+00:00",
        "command": "python bench_gemm.py --repeats 3",
        "runtime_support": {
            "context": {"backend": "cuda", "world_size": 1, "has_heap": False},
            "ops": {},
        },
        "results": [
            {"M": 4096, "dtype": "fp16", "ratio_pct": 97.8},
            {"M": 4096, "dtype": "bf16", "ratio_pct": 92.0},
            {"M": 8192, "dtype": "fp16", "ratio_pct": 83.4},
            {"M": 8192, "dtype": "bf16", "ratio_pct": 80.8},
        ],
    }
    p2p_payload = {
        "generated_at_utc": "2026-03-21T18:00:00+00:00",
        "command": "python bench_p2p.py",
        "runtime_support": {
            "context": {
                "backend": "cuda",
                "world_size": 2,
                "has_heap": True,
                "heap_mode": "single_process",
                "transport_strategy": "peer_access",
            },
            "ops": {"reduce_scatter": {"state": "supported"}},
            "execution_paths": {
                "reduce_scatter.reference": {"state": "supported"},
                "reduce_scatter.device": {"state": "unsupported"},
            },
        },
        "summary": {
            "best_read": {
                "bandwidth_gbps": 248.65,
                "variant": "evict_first",
                "block_size": 4096,
                "grid": 114,
            },
            "best_write": {
                "bandwidth_gbps": 247.90,
                "variant": "baseline",
                "block_size": 4096,
                "grid": 114,
            },
        },
    }
    pattern_payload = {
        "generated_at_utc": "2026-03-21T18:00:00+00:00",
        "command": "python bench_patterns.py",
        "runtime_support": {
            "context": {
                "backend": "cuda",
                "world_size": 2,
                "has_heap": True,
                "heap_mode": "single_process",
                "transport_strategy": "peer_access",
            },
            "ops": {"gemm_allscatter": {"state": "supported"}},
        },
        "summary": {"best_speedup_vs_bulk": 1.619},
        "sizes": [
            {
                "M": 8192,
                "N": 8192,
                "K": 30720,
                "best_pattern": "wg_specialized",
                "best_speedup_vs_bulk": 1.619,
            }
        ],
    }

    document = exporter.build_summary_document(
        gemm_payload=gemm_payload,
        p2p_payload=p2p_payload,
        pattern_payload=pattern_payload,
    )

    assert "# TNCC Benchmark Runtime Summary" in document
    assert "backend=cuda, ws=1, heap=none" in document
    assert "reduce_scatter=supported" in document
    assert "reduce_scatter.reference=supported, reduce_scatter.device=unsupported" in document
    assert "`4096³ fp16`: 97.8% of torch.matmul" in document
    assert "best read: 248.65 GB/s" in document
    assert "best speedup vs bulk_sync: 1.619×" in document


def test_build_summary_document_flags_collective_publication_gaps() -> None:
    """Collective sections should surface contaminated or unverified latest artifacts."""
    exporter = _load_export_module()

    base_runtime = {
        "runtime_support": {
            "context": {
                "backend": "cuda",
                "world_size": 2,
                "has_heap": True,
                "heap_mode": "single_process",
                "transport_strategy": "peer_access",
            },
            "ops": {},
        },
        "generated_at_utc": "2026-03-21T18:00:00+00:00",
        "command": "python bench.py --output-json latest.json",
    }

    collective_payload = {
        **base_runtime,
        "environment_health": {"status": "contaminated"},
        "summary": {
            "peak_by_collective": {},
        },
    }
    collective_bulk_payload = {
        **base_runtime,
        "summary": {
            "peak_by_collective": {},
        },
    }

    document = exporter.build_summary_document(
        gemm_payload={},
        p2p_payload={},
        pattern_payload={},
        collective_payload=collective_payload,
        collective_bulk_payload=collective_bulk_payload,
    )

    assert "| Comm-only Collectives | contaminated |" in document
    assert "| Collective vs bulk_sync | unverified |" in document
    assert "captured under a contaminated GPU environment" in document
    assert "missing benchmark-environment health metadata" in document
    assert "env=contaminated" in document
    assert "status=contaminated" in document
