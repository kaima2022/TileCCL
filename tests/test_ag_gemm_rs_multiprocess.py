# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess validation for allgather_gemm_reducescatter."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.multigpu
@pytest.mark.parametrize("dtype_name", ["float32"])
def test_ag_gemm_rs_multiprocess_default_transport(
    skip_no_multigpu,
    dtype_name: str,
) -> None:
    """The three-stage AG→GEMM→RS pipeline should be correct on the current runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_ag_gemm_rs_multiprocess",
            "--M", "128",
            "--K", "256",
            "--N", "128",
            "--dtype", dtype_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Multiprocess AG→GEMM→RS validation failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    payloads = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    assert payloads, f"No JSON payloads found in stdout:\n{result.stdout}"
    for payload in payloads:
        assert payload["dtype"] == dtype_name
        assert payload["mode"] == "multiprocess"
        assert payload["plan_ok"] is True
        assert payload["high_level_ok"] is True
