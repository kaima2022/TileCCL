# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for plot_figures comm-only payload loading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_plot_module_with_payload(payload: dict) -> object:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    data_dir = repo_root / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    comm_path = data_dir / "collective_comm_only_latest.json"
    comm_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_name = "plot_figures_test_module"
    if module_name in sys.modules:
        del sys.modules[module_name]
    script_path = scripts_dir / "plot_figures.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _base_comm_only_payload(*, include_execution: bool) -> dict:
    tncc_block = {
        "median_ms": 0.20,
        "median_bandwidth_gbps": 123.4,
        "correct_all_ranks": True,
        "rank_times_ms": [[0.20, 0.21], [0.22, 0.23]],
        "aggregate_times_ms": [0.22, 0.23],
    }
    if include_execution:
        tncc_block["execution"] = {
            "collective": "allgather",
            "path": "staged",
            "protocol": "ws2_slot_epoch_pipeline",
            "message_regime": "staged",
            "root_mode": "no_root",
            "chunk_elems": 4096,
            "num_chunks": 1,
            "pipeline_slots": 1,
            "chunk": {"elems": 4096, "count": 1},
            "pipeline": {"slots": 1},
        }
    return {
        "schema_version": 1,
        "benchmark": "collective_comm_only",
        "generated_at_utc": "2026-04-11T00:00:00+00:00",
        "environment": {
            "quick_mode": False,
            "world_size": 2,
        },
        "environment_health": {
            "status": "clean",
        },
        "cases": [
            {
                "collective": "allgather",
                "size_bytes": 65536,
                "size_mib": 0.0625,
                "tncc": tncc_block,
                "nccl": {
                    "median_ms": 0.19,
                    "median_bandwidth_gbps": 118.0,
                    "correct_all_ranks": True,
                    "rank_times_ms": [[0.19, 0.20], [0.20, 0.21]],
                    "aggregate_times_ms": [0.20, 0.21],
                },
                "tncc_vs_nccl_bandwidth_ratio": 1.0457627118644068,
            }
        ],
        "summary": {
            "peak_by_collective": {
                "allgather": {
                    "peak_tncc_bandwidth_gbps": 123.4,
                    "peak_nccl_bandwidth_gbps": 118.0,
                    "best_tncc_vs_nccl_ratio": 1.0457627118644068,
                }
            }
        },
    }


def test_plot_collective_loader_accepts_tncc_execution_metadata() -> None:
    """Plot consumer should keep working when tncc.execution is present."""
    module = _load_plot_module_with_payload(_base_comm_only_payload(include_execution=True))

    series, summary = module._load_collective_comm_only()

    assert "allgather" in series
    bucket = series["allgather"]
    assert bucket["size_bytes"] == [65536]
    assert bucket["tncc_ms"] == [0.2]
    assert bucket["nccl_ms"] == [0.19]
    assert bucket["tncc_bw"] == [123.4]
    assert bucket["nccl_bw"] == [118.0]
    assert bucket["bandwidth_ratio"] == [1.0457627118644068]
    assert (
        summary["peak_by_collective"]["allgather"]["best_tncc_vs_nccl_ratio"] == 1.0457627118644068
    )


def test_plot_collective_loader_accepts_payload_without_tncc_execution_metadata() -> None:
    """Plot consumer should keep working when tncc.execution is absent."""
    module = _load_plot_module_with_payload(_base_comm_only_payload(include_execution=False))

    series, _ = module._load_collective_comm_only()

    assert "allgather" in series
    bucket = series["allgather"]
    assert bucket["size_bytes"] == [65536]
    assert bucket["tncc_ms"] == [0.2]
    assert bucket["bandwidth_ratio"] == [1.0457627118644068]
