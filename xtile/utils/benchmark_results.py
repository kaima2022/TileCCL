"""Helpers for benchmark result artifacts shared by scripts and docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def figures_data_dir() -> Path:
    """Return the structured data directory used by plotting scripts."""
    path = project_root() / "figures" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_pattern_benchmark_path() -> Path:
    """Return the canonical latest pattern-benchmark JSON path."""
    return figures_data_dir() / "pattern_overlap_latest.json"


def default_gemm_benchmark_path() -> Path:
    """Return the canonical latest GEMM benchmark JSON path."""
    return figures_data_dir() / "gemm_latest.json"


def default_p2p_benchmark_path() -> Path:
    """Return the canonical latest P2P benchmark JSON path."""
    return figures_data_dir() / "p2p_latest.json"


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write benchmark JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a structured benchmark JSON payload."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
