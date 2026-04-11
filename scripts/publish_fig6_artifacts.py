#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed publication gate for canonical Figure-6 artifacts.

Workflow:
1. Run ``scripts/validate_fig6_gates.py``.
2. Promote candidate JSON to canonical latest only when all gates pass.
3. Regenerate figures via ``python scripts/plot_figures.py``.

Any gate failure blocks publication, preserves the existing canonical artifact,
and writes a blocker report.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tncc.utils.benchmark_results import (
    default_collective_comm_only_benchmark_path,
    project_root,
)

REPO_ROOT = project_root()
DEFAULT_CANDIDATE = REPO_ROOT / ".sisyphus" / "evidence" / "fig6-v2-candidate.json"
DEFAULT_SUMMARY = REPO_ROOT / ".sisyphus" / "evidence" / "fig6-v2-stability-summary.json"
DEFAULT_CANONICAL = default_collective_comm_only_benchmark_path()
DEFAULT_VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_fig6_gates.py"
DEFAULT_PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_figures.py"
DEFAULT_BLOCKER_REPORT = (
    REPO_ROOT / ".sisyphus" / "evidence" / "task-20-publication-blocker-report.txt"
)
_EXIT_PUBLICATION_BLOCKED = 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=str,
        default=str(DEFAULT_CANDIDATE),
        help="Candidate comm-only benchmark JSON path.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(DEFAULT_CANONICAL),
        help="Baseline comm-only benchmark JSON path passed to gate validation.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(DEFAULT_SUMMARY),
        help="Stability summary JSON path passed to gate validation.",
    )
    parser.add_argument(
        "--canonical-output",
        type=str,
        default=str(DEFAULT_CANONICAL),
        help="Canonical comm-only JSON path to promote on gate success.",
    )
    parser.add_argument(
        "--validate-script",
        type=str,
        default=str(DEFAULT_VALIDATE_SCRIPT),
        help="Path to validate_fig6_gates.py.",
    )
    parser.add_argument(
        "--plot-script",
        type=str,
        default=str(DEFAULT_PLOT_SCRIPT),
        help="Path to plot_figures.py.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke validation and plotting scripts.",
    )
    parser.add_argument(
        "--blocker-report",
        type=str,
        default=str(DEFAULT_BLOCKER_REPORT),
        help="Blocker report path written when publication is blocked.",
    )

    parser.add_argument(
        "--large-bw-ratio-256k-vs-64k-min",
        type=float,
        default=0.90,
        help="Forwarded gate threshold: BW(256KiB)/BW(64KiB) lower bound.",
    )
    parser.add_argument(
        "--large-bw-ratio-1m-vs-256k-min",
        type=float,
        default=1.00,
        help="Forwarded gate threshold: BW(1MiB)/BW(256KiB) lower bound.",
    )
    parser.add_argument(
        "--small-latency-regression-max",
        type=float,
        default=0.10,
        help="Forwarded gate threshold for small-message latency regression.",
    )
    parser.add_argument(
        "--allreduce-regression-max",
        type=float,
        default=0.05,
        help="Forwarded gate threshold for allreduce regression.",
    )
    parser.add_argument(
        "--max-cv",
        type=float,
        default=0.10,
        help="Forwarded gate threshold for stability CV ratio.",
    )
    parser.add_argument(
        "--require-success",
        type=int,
        default=10,
        help="Forwarded minimum successful-run count for stability experiments.",
    )

    args = parser.parse_args(argv)
    for name in (
        "large_bw_ratio_256k_vs_64k_min",
        "large_bw_ratio_1m_vs_256k_min",
        "small_latency_regression_max",
        "allreduce_regression_max",
        "max_cv",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.require_success <= 0:
        parser.error("--require-success must be > 0")
    return args


def _command_string(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _render_output_block(text: str) -> str:
    stripped = text.rstrip()
    return stripped if stripped else "<empty>"


def _write_blocker_report(
    *,
    report_path: Path,
    stage: str,
    gate_command: Sequence[str],
    gate_result: subprocess.CompletedProcess[str],
    candidate: Path,
    baseline: Path,
    summary: Path,
    canonical_output: Path,
    plot_command: Sequence[str] | None = None,
    plot_result: subprocess.CompletedProcess[str] | None = None,
    detail: str | None = None,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "FIG6_PUBLICATION_BLOCKER_REPORT",
        f"generated_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"stage={stage}",
        f"candidate={candidate}",
        f"baseline={baseline}",
        f"summary={summary}",
        f"canonical_output={canonical_output}",
        f"gate_command={_command_string(gate_command)}",
        f"gate_exit_code={gate_result.returncode}",
        "gate_stdout:",
        _render_output_block(gate_result.stdout),
        "gate_stderr:",
        _render_output_block(gate_result.stderr),
    ]
    if plot_command is not None:
        lines.append(f"plot_command={_command_string(plot_command)}")
    if plot_result is not None:
        lines.extend(
            (
                f"plot_exit_code={plot_result.returncode}",
                "plot_stdout:",
                _render_output_block(plot_result.stdout),
                "plot_stderr:",
                _render_output_block(plot_result.stderr),
            )
        )
    if detail:
        lines.append(f"detail={detail}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _gate_command(
    *,
    python_executable: str,
    validate_script: Path,
    candidate: Path,
    baseline: Path,
    summary: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python_executable,
        str(validate_script),
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--summary",
        str(summary),
        "--large-bw-ratio-256k-vs-64k-min",
        str(float(args.large_bw_ratio_256k_vs_64k_min)),
        "--large-bw-ratio-1m-vs-256k-min",
        str(float(args.large_bw_ratio_1m_vs_256k_min)),
        "--small-latency-regression-max",
        str(float(args.small_latency_regression_max)),
        "--allreduce-regression-max",
        str(float(args.allreduce_regression_max)),
        "--max-cv",
        str(float(args.max_cv)),
        "--require-success",
        str(int(args.require_success)),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    candidate = Path(args.candidate)
    baseline = Path(args.baseline)
    summary = Path(args.summary)
    canonical_output = Path(args.canonical_output)
    validate_script = Path(args.validate_script)
    plot_script = Path(args.plot_script)
    blocker_report_path = Path(args.blocker_report)

    gate_command = _gate_command(
        python_executable=str(args.python_executable),
        validate_script=validate_script,
        candidate=candidate,
        baseline=baseline,
        summary=summary,
        args=args,
    )
    gate_result = _run_command(gate_command)
    if gate_result.stdout:
        print(gate_result.stdout, end="")
    if gate_result.stderr:
        print(gate_result.stderr, end="", file=sys.stderr)
    if gate_result.returncode != 0:
        report_path = _write_blocker_report(
            report_path=blocker_report_path,
            stage="gate_failed",
            gate_command=gate_command,
            gate_result=gate_result,
            candidate=candidate,
            baseline=baseline,
            summary=summary,
            canonical_output=canonical_output,
            detail="Gate validation failed. Canonical artifact was not modified.",
        )
        print(
            "FIG6 publication blocked because validate_fig6_gates failed. "
            f"Blocker report: {report_path}",
            file=sys.stderr,
        )
        return gate_result.returncode

    canonical_output.parent.mkdir(parents=True, exist_ok=True)
    canonical_previous = canonical_output.read_bytes() if canonical_output.exists() else None
    try:
        shutil.copy2(candidate, canonical_output)
    except OSError as exc:
        report_path = _write_blocker_report(
            report_path=blocker_report_path,
            stage="promotion_copy_failed",
            gate_command=gate_command,
            gate_result=gate_result,
            candidate=candidate,
            baseline=baseline,
            summary=summary,
            canonical_output=canonical_output,
            detail=f"Failed to promote candidate to canonical artifact: {exc}",
        )
        print(
            "FIG6 publication blocked because canonical promotion failed. "
            f"Blocker report: {report_path}",
            file=sys.stderr,
        )
        return _EXIT_PUBLICATION_BLOCKED

    plot_command = [str(args.python_executable), str(plot_script)]
    plot_result = _run_command(plot_command)
    if plot_result.stdout:
        print(plot_result.stdout, end="")
    if plot_result.stderr:
        print(plot_result.stderr, end="", file=sys.stderr)
    if plot_result.returncode != 0:
        rollback_status = "restored_previous_canonical"
        rollback_error: OSError | None = None
        try:
            if canonical_previous is None:
                canonical_output.unlink(missing_ok=True)
                rollback_status = "removed_new_canonical"
            else:
                canonical_output.write_bytes(canonical_previous)
        except OSError as exc:
            rollback_status = "rollback_failed"
            rollback_error = exc

        detail = (
            f"Plot generation failed after gate pass; canonical rollback_status={rollback_status}."
        )
        if rollback_error is not None:
            detail += f" rollback_error={rollback_error}"

        report_path = _write_blocker_report(
            report_path=blocker_report_path,
            stage="plot_failed_after_gate_pass",
            gate_command=gate_command,
            gate_result=gate_result,
            candidate=candidate,
            baseline=baseline,
            summary=summary,
            canonical_output=canonical_output,
            plot_command=plot_command,
            plot_result=plot_result,
            detail=detail,
        )
        if rollback_error is not None:
            print(
                "FIG6 publication blocked because plot regeneration failed after gate pass "
                "and canonical rollback failed. "
                f"Blocker report: {report_path}",
                file=sys.stderr,
            )
            return _EXIT_PUBLICATION_BLOCKED

        print(
            "FIG6 publication blocked because plot regeneration failed after gate pass. "
            f"Canonical rollback succeeded ({rollback_status}). "
            f"Blocker report: {report_path}",
            file=sys.stderr,
        )
        return plot_result.returncode

    print(f"FIG6 publication succeeded: promoted {candidate} -> {canonical_output}")
    print(f"Regenerated figures with: {_command_string(plot_command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
