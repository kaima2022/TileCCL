# SPDX-License-Identifier: Apache-2.0
"""Public smoke tests for the CLI entrypoint."""

from __future__ import annotations

import subprocess
import sys


def test_python_module_help_smoke() -> None:
    """``python -m tileccl --help`` should succeed."""
    completed = subprocess.run(
        [sys.executable, "-m", "tileccl", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower()
    assert "tileccl" in completed.stdout.lower()
