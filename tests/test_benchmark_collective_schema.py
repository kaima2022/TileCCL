# SPDX-License-Identifier: Apache-2.0
"""Fig6 comm-only producer/consumer JSON contract tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_plot_module_with_payload(payload: dict[str, Any]) -> object:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    data_dir = repo_root / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    comm_path = data_dir / "collective_comm_only_latest.json"
    comm_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_name = "plot_figures_collective_schema_contract_test_module"
    if module_name in sys.modules:
        del sys.modules[module_name]

    script_path = scripts_dir / "plot_figures.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _assert_number(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"Invalid type at {path}: expected number, got {type(value).__name__}")


def _required_key(mapping: dict[str, Any], key: str, *, path: str) -> Any:
    if key not in mapping:
        raise AssertionError(f"Missing required field: {path}.{key}")
    return mapping[key]


def _assert_collective_case_contract(case: dict[str, Any], *, index: int) -> None:
    base = f"cases[{index}]"
    collective = _required_key(case, "collective", path=base)
    if not isinstance(collective, str):
        raise AssertionError(
            f"Invalid type at {base}.collective: expected str, got {type(collective).__name__}"
        )

    _assert_number(_required_key(case, "size_bytes", path=base), path=f"{base}.size_bytes")
    _assert_number(_required_key(case, "size_mib", path=base), path=f"{base}.size_mib")

    tncc = _required_key(case, "tncc", path=base)
    if not isinstance(tncc, dict):
        raise AssertionError(
            f"Invalid type at {base}.tncc: expected dict, got {type(tncc).__name__}"
        )

    nccl = _required_key(case, "nccl", path=base)
    if not isinstance(nccl, dict):
        raise AssertionError(
            f"Invalid type at {base}.nccl: expected dict, got {type(nccl).__name__}"
        )

    # Old flat fields are still required for plot consumer compatibility.
    _assert_number(
        _required_key(tncc, "median_ms", path=f"{base}.tncc"),
        path=f"{base}.tncc.median_ms",
    )
    _assert_number(
        _required_key(tncc, "median_bandwidth_gbps", path=f"{base}.tncc"),
        path=f"{base}.tncc.median_bandwidth_gbps",
    )
    _assert_number(
        _required_key(nccl, "median_ms", path=f"{base}.nccl"),
        path=f"{base}.nccl.median_ms",
    )
    _assert_number(
        _required_key(nccl, "median_bandwidth_gbps", path=f"{base}.nccl"),
        path=f"{base}.nccl.median_bandwidth_gbps",
    )

    ratio_path = f"{base}.tncc_vs_nccl_bandwidth_ratio"
    if "tncc_vs_nccl_bandwidth_ratio" in case:
        ratio_value = case["tncc_vs_nccl_bandwidth_ratio"]
    elif "xtile_vs_nccl_bandwidth_ratio" in case:
        ratio_path = f"{base}.xtile_vs_nccl_bandwidth_ratio"
        ratio_value = case["xtile_vs_nccl_bandwidth_ratio"]
    else:
        raise AssertionError(
            "Missing required field: "
            f"{base}.tncc_vs_nccl_bandwidth_ratio "
            f"(or {base}.xtile_vs_nccl_bandwidth_ratio)"
        )
    _assert_number(ratio_value, path=ratio_path)

    # New metadata is append-only and optional.
    execution = tncc.get("execution")
    if execution is not None and not isinstance(execution, dict):
        raise AssertionError(
            f"Invalid type at {base}.tncc.execution: expected dict, got {type(execution).__name__}"
        )


def _assert_fig6_collective_comm_contract(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise AssertionError(
            f"Invalid type at payload: expected dict, got {type(payload).__name__}"
        )

    schema_version = _required_key(payload, "schema_version", path="payload")
    if not isinstance(schema_version, int):
        raise AssertionError(
            "Invalid type at payload.schema_version: "
            f"expected int, got {type(schema_version).__name__}"
        )

    benchmark = _required_key(payload, "benchmark", path="payload")
    if benchmark != "collective_comm_only":
        raise AssertionError(
            "Invalid value at payload.benchmark: "
            f"expected 'collective_comm_only', got {benchmark!r}"
        )

    environment_health = _required_key(payload, "environment_health", path="payload")
    if not isinstance(environment_health, dict):
        raise AssertionError(
            "Invalid type at payload.environment_health: "
            f"expected dict, got {type(environment_health).__name__}"
        )
    status = _required_key(environment_health, "status", path="payload.environment_health")
    if not isinstance(status, str):
        raise AssertionError(
            "Invalid type at payload.environment_health.status: "
            f"expected str, got {type(status).__name__}"
        )

    cases = _required_key(payload, "cases", path="payload")
    if not isinstance(cases, list):
        raise AssertionError(
            f"Invalid type at payload.cases: expected list, got {type(cases).__name__}"
        )
    if not cases:
        raise AssertionError("Invalid value at payload.cases: expected non-empty list")

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise AssertionError(
                f"Invalid type at cases[{index}]: expected dict, got {type(case).__name__}"
            )
        _assert_collective_case_contract(case, index=index)


def _base_comm_only_payload(*, include_execution: bool) -> dict[str, Any]:
    tncc_block: dict[str, Any] = {
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


@pytest.mark.parametrize(
    "include_execution",
    [False, True],
    ids=["without_optional_tncc_execution", "with_optional_tncc_execution"],
)
def test_fig6_contract_happy_path_validates_and_plot_loader_accepts_payload(
    include_execution: bool,
) -> None:
    payload = _base_comm_only_payload(include_execution=include_execution)

    _assert_fig6_collective_comm_contract(payload)
    module = _load_plot_module_with_payload(payload)
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


def test_fig6_contract_missing_tncc_median_ms_reports_required_field_path() -> None:
    payload = _base_comm_only_payload(include_execution=True)
    del payload["cases"][0]["tncc"]["median_ms"]

    with pytest.raises(
        AssertionError,
        match=r"Missing required field: cases\[0\]\.tncc\.median_ms",
    ):
        _assert_fig6_collective_comm_contract(payload)


def test_fig6_contract_old_flat_fields_remain_required_with_execution_present() -> None:
    payload = _base_comm_only_payload(include_execution=True)
    del payload["cases"][0]["tncc"]["median_bandwidth_gbps"]

    with pytest.raises(
        AssertionError,
        match=r"Missing required field: cases\[0\]\.tncc\.median_bandwidth_gbps",
    ):
        _assert_fig6_collective_comm_contract(payload)
