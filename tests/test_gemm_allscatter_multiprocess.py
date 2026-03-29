# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess validation for gemm_allscatter public contracts."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.multigpu
@pytest.mark.parametrize("contract", ["full_full", "full_shard"])
def test_gemm_allscatter_multiprocess_default_transport(
    skip_no_multigpu,
    contract: str,
) -> None:
    """The default multiprocess gemm_allscatter path should be correct on the current runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_gemm_allscatter_multiprocess",
            "--dtype",
            "float32",
            "--contract",
            contract,
            "--warmup",
            "1",
            "--iters",
            "2",
            "--pattern",
            "bulk_sync",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Multiprocess gemm_allscatter validation failed.\n"
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
        assert payload["contract"] == contract
        assert payload["mode"] == "multiprocess"
        assert payload["transport_strategy"] == "ctypes_ipc"
        assert payload["plan_pattern_name"] == "bulk_sync"
        assert payload["plan_ok"] is True
        assert payload["high_level_ok"] is True
