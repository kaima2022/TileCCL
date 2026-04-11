# SPDX-License-Identifier: Apache-2.0
"""Tests for fail-closed Figure-6 publication workflow."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_publish_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "publish_fig6_artifacts.py"
    module_name = "publish_fig6_artifacts_test_module"
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


def _load_noise_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "analyze_benchmark_noise.py"
    module_name = "analyze_benchmark_noise_test_module"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_publish_fig6_artifacts_blocks_on_gate_failure_and_preserves_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    publisher = _load_publish_module()

    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    summary = tmp_path / "summary.json"
    canonical = tmp_path / "collective_comm_only_latest.json"
    validate_script = tmp_path / "validate_fig6_gates.py"
    plot_script = tmp_path / "plot_figures.py"
    blocker_report = tmp_path / "blockers" / "report.txt"

    _write_json(candidate, {"kind": "candidate", "value": 2})
    _write_json(canonical, {"kind": "canonical", "value": 1})
    _write_json(baseline, {"kind": "baseline"})
    _write_json(summary, {"kind": "summary"})
    validate_script.write_text("# stub\n", encoding="utf-8")
    plot_script.write_text("# stub\n", encoding="utf-8")
    canonical_before = canonical.read_text(encoding="utf-8")

    commands: list[list[str]] = []

    def _fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == publisher.REPO_ROOT
        assert capture_output is True
        assert text is True
        assert check is False
        commands.append(command)
        if str(validate_script) in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="[FAIL] stability-max-cv-latency: ...\nOVERALL: FAIL\n",
                stderr="",
            )
        if str(plot_script) in command:
            raise AssertionError("plot should not run when gate validation fails")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(publisher.subprocess, "run", _fake_run)

    rc = publisher.main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--summary",
            str(summary),
            "--canonical-output",
            str(canonical),
            "--validate-script",
            str(validate_script),
            "--plot-script",
            str(plot_script),
            "--blocker-report",
            str(blocker_report),
            "--python-executable",
            sys.executable,
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert len(commands) == 1
    assert canonical.read_text(encoding="utf-8") == canonical_before
    assert blocker_report.exists()
    report = blocker_report.read_text(encoding="utf-8")
    assert "FIG6_PUBLICATION_BLOCKER_REPORT" in report
    assert "stage=gate_failed" in report
    assert "gate_exit_code=1" in report
    assert "Gate validation failed. Canonical artifact was not modified." in report
    assert "FIG6 publication blocked because validate_fig6_gates failed." in captured.err
    assert "OVERALL: FAIL" in captured.out


def test_publish_fig6_artifacts_promotes_candidate_after_gate_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    publisher = _load_publish_module()

    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    summary = tmp_path / "summary.json"
    canonical = tmp_path / "collective_comm_only_latest.json"
    validate_script = tmp_path / "validate_fig6_gates.py"
    plot_script = tmp_path / "plot_figures.py"
    blocker_report = tmp_path / "blockers" / "report.txt"

    _write_json(candidate, {"kind": "candidate", "value": 2})
    _write_json(canonical, {"kind": "canonical", "value": 1})
    _write_json(baseline, {"kind": "baseline"})
    _write_json(summary, {"kind": "summary"})
    validate_script.write_text("# stub\n", encoding="utf-8")
    plot_script.write_text("# stub\n", encoding="utf-8")

    call_order: list[str] = []

    def _fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == publisher.REPO_ROOT
        assert capture_output is True
        assert text is True
        assert check is False
        if str(validate_script) in command:
            call_order.append("gate")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="OVERALL: PASS\n",
                stderr="",
            )
        if str(plot_script) in command:
            call_order.append("plot")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="figures generated\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(publisher.subprocess, "run", _fake_run)

    rc = publisher.main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--summary",
            str(summary),
            "--canonical-output",
            str(canonical),
            "--validate-script",
            str(validate_script),
            "--plot-script",
            str(plot_script),
            "--blocker-report",
            str(blocker_report),
            "--python-executable",
            sys.executable,
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert call_order == ["gate", "plot"]
    assert canonical.read_text(encoding="utf-8") == candidate.read_text(encoding="utf-8")
    assert not blocker_report.exists()
    assert "OVERALL: PASS" in captured.out
    assert "FIG6 publication succeeded: promoted" in captured.out


def test_publish_fig6_artifacts_fail_closed_when_plot_and_rollback_both_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    publisher = _load_publish_module()

    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    summary = tmp_path / "summary.json"
    canonical = tmp_path / "collective_comm_only_latest.json"
    validate_script = tmp_path / "validate_fig6_gates.py"
    plot_script = tmp_path / "plot_figures.py"
    blocker_report = tmp_path / "blockers" / "report.txt"

    _write_json(candidate, {"kind": "candidate", "value": 2})
    _write_json(canonical, {"kind": "canonical", "value": 1})
    _write_json(baseline, {"kind": "baseline"})
    _write_json(summary, {"kind": "summary"})
    validate_script.write_text("# stub\n", encoding="utf-8")
    plot_script.write_text("# stub\n", encoding="utf-8")

    def _fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == publisher.REPO_ROOT
        assert capture_output is True
        assert text is True
        assert check is False
        if str(validate_script) in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="OVERALL: PASS\n",
                stderr="",
            )
        if str(plot_script) in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=7,
                stdout="",
                stderr="plot failed\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    def _failing_restore(_self: Path, _data: bytes) -> int:
        raise OSError("simulated rollback write failure")

    monkeypatch.setattr(publisher.subprocess, "run", _fake_run)
    monkeypatch.setattr(publisher.Path, "write_bytes", _failing_restore)

    rc = publisher.main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--summary",
            str(summary),
            "--canonical-output",
            str(canonical),
            "--validate-script",
            str(validate_script),
            "--plot-script",
            str(plot_script),
            "--blocker-report",
            str(blocker_report),
            "--python-executable",
            sys.executable,
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert blocker_report.exists()
    report = blocker_report.read_text(encoding="utf-8")
    assert "stage=plot_failed_after_gate_pass" in report
    assert "plot_exit_code=7" in report
    assert "rollback_status=rollback_failed" in report
    assert "rollback_error=simulated rollback write failure" in report
    assert "rollback failed" in captured.err


def test_analyze_benchmark_noise_fig6_probe_uses_comm_repeats() -> None:
    noise = _load_noise_module()

    specs = noise._build_specs(repeats=3, comm_repeats=10, bulk_repeats=4)
    probe_spec = next(spec for spec in specs if spec.name == "fig6_comm_reduce_scatter_256k_probe")

    assert probe_spec.repeats == 10
