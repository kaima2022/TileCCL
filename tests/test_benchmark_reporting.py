# SPDX-License-Identifier: Apache-2.0
"""Tests for shared benchmark reporting helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_reporting_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "_benchmark_reporting.py"
    spec = importlib.util.spec_from_file_location("_benchmark_reporting_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault("_benchmark_reporting_test", module)
    spec.loader.exec_module(module)
    return module


def test_runtime_support_brief_formats_heap_and_op_state() -> None:
    """Runtime support helper should surface key context and op states."""
    reporting = _load_reporting_module()
    payload = {
        "runtime_support": {
            "context": {
                "backend": "cuda",
                "world_size": 2,
                "has_heap": True,
                "heap_mode": "single_process",
                "transport_strategy": "peer_access",
            },
            "ops": {
                "reduce_scatter": {"state": "supported"},
            },
        }
    }

    summary = reporting.runtime_support_brief(
        payload,
        highlight_ops=("reduce_scatter",),
    )

    assert summary == (
        "backend=cuda, ws=2, heap=single_process, transport=peer_access, "
        "reduce_scatter=supported"
    )


def test_benchmark_footer_text_includes_source_date_and_command() -> None:
    """Figure/export footers should keep provenance and support together."""
    reporting = _load_reporting_module()
    payload = {
        "generated_at_utc": "2026-03-21T18:00:00+00:00",
        "command": "python bench.py --quick",
        "runtime_support": {
            "context": {
                "backend": "cuda",
                "world_size": 1,
                "has_heap": False,
            },
            "ops": {},
        },
    }

    footer = reporting.benchmark_footer_text(payload, source_name="sample.json")

    assert "source=sample.json" in footer
    assert "run=2026-03-21" in footer
    assert "backend=cuda, ws=1, heap=none" in footer
    assert "cmd=python bench.py --quick" in footer


def test_benchmark_footer_text_includes_environment_health() -> None:
    """Footers should surface recorded environment-health provenance."""
    reporting = _load_reporting_module()
    payload = {
        "generated_at_utc": "2026-03-21T18:00:00+00:00",
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
        "environment_health": {
            "status": "contaminated",
        },
    }

    footer = reporting.benchmark_footer_text(payload, source_name="sample.json")

    assert "env=contaminated" in footer


def test_benchmark_publication_status_flags_unverified_and_quick_mode() -> None:
    """Publication status should reject contaminated, quick, or unverified payloads."""
    reporting = _load_reporting_module()

    assert reporting.benchmark_publication_status(
        {
            "environment_health": {"status": "contaminated"},
        },
        require_environment_health=True,
    ) == "contaminated"
    assert reporting.benchmark_publication_status(
        {
            "environment": {"quick_mode": True},
        }
    ) == "quick_mode"
    assert reporting.benchmark_publication_status(
        {
            "environment": {"quick_mode": False},
        },
        require_environment_health=True,
    ) == "unverified"


def test_execution_path_brief_formats_selected_paths() -> None:
    """Execution-path helper should surface implementation-level states."""
    reporting = _load_reporting_module()
    payload = {
        "runtime_support": {
            "execution_paths": {
                "reduce_scatter.reference": {"state": "supported"},
                "reduce_scatter.device": {"state": "unsupported"},
            }
        }
    }

    summary = reporting.execution_path_brief(
        payload,
        names=("reduce_scatter.reference", "reduce_scatter.device"),
    )

    assert summary == (
        "reduce_scatter.reference=supported, "
        "reduce_scatter.device=unsupported"
    )
