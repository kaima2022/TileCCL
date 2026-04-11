# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/validate_fig6_gates.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_gate_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "validate_fig6_gates.py"
    module_name = "validate_fig6_gates_test_module"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _comm_case(
    *,
    collective: str,
    size_bytes: int,
    tncc_median_ms: float,
    tncc_median_bandwidth_gbps: float,
    tncc_correct_all_ranks: bool = True,
    nccl_correct_all_ranks: bool = True,
) -> dict[str, Any]:
    return {
        "collective": collective,
        "size_bytes": size_bytes,
        "size_mib": float(size_bytes / (1024**2)),
        "tncc": {
            "median_ms": tncc_median_ms,
            "median_bandwidth_gbps": tncc_median_bandwidth_gbps,
            "correct_all_ranks": tncc_correct_all_ranks,
        },
        "nccl": {
            "median_ms": max(1e-6, tncc_median_ms * 0.95),
            "median_bandwidth_gbps": max(1e-6, tncc_median_bandwidth_gbps * 0.95),
            "correct_all_ranks": nccl_correct_all_ranks,
        },
        "tncc_vs_nccl_bandwidth_ratio": 1.05,
    }


def _baseline_payload() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for collective in ("allgather", "broadcast", "reduce_scatter", "scatter"):
        cases.extend(
            [
                _comm_case(
                    collective=collective,
                    size_bytes=4096,
                    tncc_median_ms=0.20,
                    tncc_median_bandwidth_gbps=40.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=16384,
                    tncc_median_ms=0.24,
                    tncc_median_bandwidth_gbps=52.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=65536,
                    tncc_median_ms=0.40,
                    tncc_median_bandwidth_gbps=100.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=262144,
                    tncc_median_ms=0.80,
                    tncc_median_bandwidth_gbps=110.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=1048576,
                    tncc_median_ms=1.20,
                    tncc_median_bandwidth_gbps=120.0,
                ),
            ]
        )

    cases.extend(
        [
            _comm_case(
                collective="allreduce",
                size_bytes=4096,
                tncc_median_ms=0.16,
                tncc_median_bandwidth_gbps=45.0,
            ),
            _comm_case(
                collective="allreduce",
                size_bytes=262144,
                tncc_median_ms=0.60,
                tncc_median_bandwidth_gbps=130.0,
            ),
        ]
    )

    return {
        "schema_version": 1,
        "benchmark": "collective_comm_only",
        "environment": {
            "world_size": 2,
            "quick_mode": False,
        },
        "environment_health": {
            "status": "clean",
        },
        "cases": cases,
        "summary": {
            "peak_by_collective": {},
        },
    }


def _candidate_payload(*, degrade_large_ratio: bool = False) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for collective in ("allgather", "broadcast", "reduce_scatter", "scatter"):
        cases.extend(
            [
                _comm_case(
                    collective=collective,
                    size_bytes=4096,
                    tncc_median_ms=0.21,
                    tncc_median_bandwidth_gbps=42.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=16384,
                    tncc_median_ms=0.25,
                    tncc_median_bandwidth_gbps=50.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=65536,
                    tncc_median_ms=0.42,
                    tncc_median_bandwidth_gbps=100.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=262144,
                    tncc_median_ms=0.85,
                    tncc_median_bandwidth_gbps=95.0,
                ),
                _comm_case(
                    collective=collective,
                    size_bytes=1048576,
                    tncc_median_ms=1.20,
                    tncc_median_bandwidth_gbps=(80.0 if degrade_large_ratio else 100.0),
                ),
            ]
        )

    return {
        "schema_version": 1,
        "benchmark": "collective_comm_only",
        "environment": {
            "world_size": 2,
            "quick_mode": False,
        },
        "environment_health": {
            "status": "clean",
        },
        "cases": cases,
        "summary": {
            "peak_by_collective": {},
        },
    }


def _set_candidate_correctness(
    payload: dict[str, Any],
    *,
    collective: str,
    size_bytes: int,
    tncc_correct_all_ranks: bool,
    nccl_correct_all_ranks: bool,
) -> None:
    for case in payload["cases"]:
        if case["collective"] == collective and int(case["size_bytes"]) == int(size_bytes):
            case["tncc"]["correct_all_ranks"] = tncc_correct_all_ranks
            case["nccl"]["correct_all_ranks"] = nccl_correct_all_ranks
            return
    raise AssertionError(
        f"expected case not found for correctness override: {collective}@{size_bytes}"
    )


def _summary_case(
    *, collective: str, size_bytes: int, median_ms: float, cv_pct: float
) -> dict[str, Any]:
    return {
        "collective": collective,
        "size_bytes": size_bytes,
        "tncc_latency_ms": {
            "median": median_ms,
            "cv_pct": cv_pct,
            "spread_pct": cv_pct,
        },
        "tncc_bandwidth_gbps": {
            "median": 100.0,
            "cv_pct": cv_pct,
            "spread_pct": cv_pct,
        },
    }


def _summary_payload(*, success_count: int = 10) -> dict[str, Any]:
    def _records() -> list[dict[str, Any]]:
        failed = max(0, 10 - success_count)
        return [{"status": "ok"} for _ in range(success_count)] + [
            {"status": "failed"} for _ in range(failed)
        ]

    return {
        "schema_version": 1,
        "benchmark": "noise_study",
        "experiments": {
            "fig6_comm_allreduce": {
                "records": _records(),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="allreduce",
                            size_bytes=4096,
                            median_ms=0.165,
                            cv_pct=4.0,
                        ),
                        _summary_case(
                            collective="allreduce",
                            size_bytes=262144,
                            median_ms=0.615,
                            cv_pct=5.0,
                        ),
                    ]
                },
            },
            "fig6_comm_exchange": {
                "records": _records(),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="allgather",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=6.0,
                        ),
                        _summary_case(
                            collective="broadcast",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=6.5,
                        ),
                        _summary_case(
                            collective="scatter",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=7.0,
                        ),
                    ]
                },
            },
            "fig6_comm_reduce_scatter": {
                "records": _records(),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="reduce_scatter",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=6.0,
                        )
                    ]
                },
            },
            "fig6_comm_reduce_scatter_256k_probe": {
                "records": _records(),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="reduce_scatter",
                            size_bytes=262144,
                            median_ms=0.80,
                            cv_pct=8.0,
                        )
                    ]
                },
            },
        },
    }


def _summary_payload_with_record_overrides(
    *,
    allreduce_records: list[str],
    exchange_records: list[str],
    reduce_scatter_records: list[str],
    reduce_scatter_probe_records: list[str],
) -> dict[str, Any]:
    def _records(statuses: list[str]) -> list[dict[str, Any]]:
        return [{"status": status} for status in statuses]

    return {
        "schema_version": 1,
        "benchmark": "noise_study",
        "experiments": {
            "fig6_comm_allreduce": {
                "records": _records(allreduce_records),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="allreduce",
                            size_bytes=4096,
                            median_ms=0.165,
                            cv_pct=4.0,
                        ),
                        _summary_case(
                            collective="allreduce",
                            size_bytes=262144,
                            median_ms=0.615,
                            cv_pct=5.0,
                        ),
                    ]
                },
            },
            "fig6_comm_exchange": {
                "records": _records(exchange_records),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="allgather",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=6.0,
                        ),
                        _summary_case(
                            collective="broadcast",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=6.5,
                        ),
                        _summary_case(
                            collective="scatter",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=7.0,
                        ),
                    ]
                },
            },
            "fig6_comm_reduce_scatter": {
                "records": _records(reduce_scatter_records),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="reduce_scatter",
                            size_bytes=4096,
                            median_ms=0.20,
                            cv_pct=6.0,
                        )
                    ]
                },
            },
            "fig6_comm_reduce_scatter_256k_probe": {
                "records": _records(reduce_scatter_probe_records),
                "analysis": {
                    "cases": [
                        _summary_case(
                            collective="reduce_scatter",
                            size_bytes=262144,
                            median_ms=0.80,
                            cv_pct=8.0,
                        )
                    ]
                },
            },
        },
    }


def _sparse_baseline_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "collective_comm_only",
        "environment": {
            "world_size": 2,
            "quick_mode": False,
        },
        "environment_health": {
            "status": "clean",
        },
        "cases": [
            _comm_case(
                collective="allgather",
                size_bytes=65536,
                tncc_median_ms=0.40,
                tncc_median_bandwidth_gbps=100.0,
            ),
            _comm_case(
                collective="allreduce",
                size_bytes=262144,
                tncc_median_ms=0.60,
                tncc_median_bandwidth_gbps=130.0,
            ),
        ],
        "summary": {
            "peak_by_collective": {},
        },
    }


def _gate_argv(*, candidate: Path, baseline: Path, summary: Path) -> list[str]:
    return [
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--summary",
        str(summary),
        "--large-bw-ratio-256k-vs-64k-min",
        "0.90",
        "--large-bw-ratio-1m-vs-256k-min",
        "1.00",
        "--small-latency-regression-max",
        "0.10",
        "--allreduce-regression-max",
        "0.05",
        "--max-cv",
        "0.10",
        "--require-success",
        "10",
    ]


def test_validate_fig6_gates_passes_with_valid_candidate_baseline_and_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())
    _write_json(baseline_path, _baseline_payload())
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "OVERALL: PASS" in captured.out
    assert "[PASS] large-bw-ratio-1m-vs-256k" in captured.out
    assert "[PASS] small-latency-regression" in captured.out
    assert "[PASS] allreduce-regression" in captured.out
    assert "[PASS] stability-max-cv-latency" in captured.out
    assert "[PASS] stability-require-success" in captured.out


def test_validate_fig6_gates_fails_thresholds_for_degraded_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate_degraded.json"
    baseline_path = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload(degrade_large_ratio=True))
    _write_json(baseline_path, _baseline_payload())
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "OVERALL: FAIL" in captured.out
    assert "[FAIL] large-bw-ratio-1m-vs-256k: collective=allgather" in captured.out


def test_validate_fig6_gates_fails_on_candidate_correctness_flag_false(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate_correctness_fail.json"
    baseline_path = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary.json"

    candidate_payload = _candidate_payload()
    _set_candidate_correctness(
        candidate_payload,
        collective="allgather",
        size_bytes=65536,
        tncc_correct_all_ranks=False,
        nccl_correct_all_ranks=True,
    )

    _write_json(candidate_path, candidate_payload)
    _write_json(baseline_path, _baseline_payload())
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "OVERALL: FAIL" in captured.out
    assert (
        "[FAIL] candidate-correctness: collective=allgather size_bytes=65536 "
        "tncc.correct_all_ranks=False nccl.correct_all_ranks=True"
    ) in captured.out


def test_validate_fig6_gates_fail_closed_on_missing_required_baseline_points(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline_sparse.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())
    _write_json(baseline_path, _sparse_baseline_payload())
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "INPUT_CONTRACT_FAIL:" in captured.err
    assert (
        "missing required benchmark case in baseline: collective='allgather', size_bytes=4096"
        in captured.err
    )


def test_validate_fig6_gates_allreduce_regression_only_requires_256k_baseline_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline_missing_allreduce_4k.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())

    baseline_payload = _baseline_payload()
    baseline_payload["cases"] = [
        case
        for case in baseline_payload["cases"]
        if not (case["collective"] == "allreduce" and int(case["size_bytes"]) == 4096)
    ]
    _write_json(baseline_path, baseline_payload)
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "OVERALL: PASS" in captured.out
    assert "[PASS] allreduce-regression: collective=allreduce size_bytes=262144" in captured.out


def test_validate_fig6_gates_fail_closed_when_summary_missing_required_allreduce_baseline_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline_requires_non_summary_allreduce_size.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())

    baseline_payload = _baseline_payload()
    baseline_payload["cases"] = [
        case for case in baseline_payload["cases"] if case["collective"] != "allreduce"
    ]
    baseline_payload["cases"].append(
        _comm_case(
            collective="allreduce",
            size_bytes=524288,
            tncc_median_ms=0.90,
            tncc_median_bandwidth_gbps=120.0,
        )
    )
    _write_json(baseline_path, baseline_payload)
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "INPUT_CONTRACT_FAIL:" in captured.err
    assert (
        "summary.experiments.fig6_comm_allreduce.analysis.cases must include allreduce size_bytes=524288"
        in captured.err
    )


def test_validate_fig6_gates_uses_preconditioning_checkpoint_when_baseline_has_no_allreduce_cases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline_no_allreduce.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())

    baseline_payload = _baseline_payload()
    baseline_payload["cases"] = [
        case for case in baseline_payload["cases"] if case["collective"] != "allreduce"
    ]
    baseline_payload["environment"]["preconditioning"] = {
        "collective": "allreduce",
        "size_bytes": 262144,
        "iterations": 4,
    }
    _write_json(baseline_path, baseline_payload)
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "OVERALL: PASS" in captured.out
    assert "[PASS] allreduce-regression: collective=allreduce size_bytes=262144" in captured.out


def test_validate_fig6_gates_fail_closed_when_no_allreduce_case_and_no_preconditioning_reference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline_no_allreduce_reference.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())

    baseline_payload = _baseline_payload()
    baseline_payload["cases"] = [
        case for case in baseline_payload["cases"] if case["collective"] != "allreduce"
    ]
    baseline_payload["environment"]["preconditioning"] = None
    for rank_payload in baseline_payload.get("rank_payloads", []):
        if isinstance(rank_payload, dict):
            rank_payload["preconditioning"] = None
    _write_json(baseline_path, baseline_payload)
    _write_json(summary_path, _summary_payload(success_count=10))

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "INPUT_CONTRACT_FAIL:" in captured.err
    assert (
        "baseline must provide allreduce checkpoints via baseline.cases or environment/rank_payload preconditioning metadata"
        in captured.err
    )


def test_validate_fig6_gates_requires_absolute_success_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary_scaled_success.json"
    _write_json(candidate_path, _candidate_payload())
    _write_json(baseline_path, _baseline_payload())

    summary_payload = _summary_payload_with_record_overrides(
        allreduce_records=["ok"] * 10,
        exchange_records=["ok"] * 10,
        reduce_scatter_records=["ok"] * 10,
        reduce_scatter_probe_records=["failed"],
    )
    _write_json(summary_path, summary_payload)

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert (
        "[FAIL] stability-require-success: experiment=fig6_comm_reduce_scatter_256k_probe success=0/1 required>=10"
        in captured.out
    )

    summary_payload["experiments"]["fig6_comm_reduce_scatter_256k_probe"]["records"] = [
        {"status": "ok"} for _ in range(10)
    ]
    _write_json(summary_path, summary_payload)

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert (
        "[PASS] stability-require-success: experiment=fig6_comm_reduce_scatter_256k_probe success=10/10 required>=10"
        in captured.out
    )


@pytest.mark.parametrize(
    "baseline_mode, expected_message",
    [
        ("missing", "baseline file not found"),
        (
            "invalid_json",
            "baseline is not valid JSON",
        ),
    ],
)
def test_validate_fig6_gates_fail_closed_on_missing_or_malformed_baseline(
    baseline_mode: str,
    expected_message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_module()

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary.json"
    _write_json(candidate_path, _candidate_payload())
    _write_json(summary_path, _summary_payload(success_count=10))

    if baseline_mode == "invalid_json":
        baseline_path.write_text("{not valid json", encoding="utf-8")
    elif baseline_mode == "missing":
        baseline_path = tmp_path / "baseline_missing.json"
    else:
        raise AssertionError(f"unexpected baseline mode: {baseline_mode}")

    rc = gate.main(
        _gate_argv(candidate=candidate_path, baseline=baseline_path, summary=summary_path)
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "INPUT_CONTRACT_FAIL:" in captured.err
    assert expected_message in captured.err
