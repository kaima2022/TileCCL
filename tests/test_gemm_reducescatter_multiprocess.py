# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess validation for gemm_reducescatter."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.multigpu
@pytest.mark.parametrize("dtype_name", ["float32"])
def test_gemm_reducescatter_multiprocess_default_transport(
    skip_no_multigpu,
    dtype_name: str,
) -> None:
    """The default multiprocess gemm_reducescatter path should be correct on the current runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_gemm_reducescatter_multiprocess",
            "--M",
            "128",
            "--N",
            "256",
            "--K",
            "128",
            "--dtype",
            dtype_name,
            "--warmup",
            "0",
            "--iters",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Multiprocess gemm_reducescatter validation failed.\n"
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
        assert payload["transport_strategy"] == "ctypes_ipc"
        assert payload["plan_implementation"] == "device"
        assert payload["plan_ok"] is True
        assert payload["high_level_ok"] is True
