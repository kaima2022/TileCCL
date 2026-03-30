# SPDX-License-Identifier: Apache-2.0
"""Representative multiprocess auto-pattern validation for gemm_allscatter."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_AUTO_CASES = (
    ("bulk_sync", 128, 512, 256, "bulk_sync"),
    ("fused_sequential", 512, 1024, 16384, "fused_sequential"),
    ("producer_consumer", 512, 3072, 8192, "producer_consumer"),
    ("wg_specialized", 2048, 4096, 8192, "wg_specialized"),
)


@pytest.mark.multigpu
@pytest.mark.parametrize(
    "case_name,M,N,K,expected_pattern",
    _AUTO_CASES,
    ids=[case[0] for case in _AUTO_CASES],
)
def test_gemm_allscatter_auto_patterns_multiprocess(
    skip_no_multigpu,
    case_name: str,
    M: int,
    N: int,
    K: int,
    expected_pattern: str,
) -> None:
    """Default auto selection should be correct for each representative branch."""
    del case_name
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_gemm_allscatter_multiprocess",
            "--M",
            str(M),
            "--N",
            str(N),
            "--K",
            str(K),
            "--dtype",
            "float16",
            "--contract",
            "full_full",
            "--pattern",
            "auto",
            "--expect-pattern",
            expected_pattern,
            "--warmup",
            "1",
            "--iters",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Multiprocess auto-pattern gemm_allscatter validation failed.\n"
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
        assert payload["mode"] == "multiprocess"
        assert payload["transport_strategy"] == "ctypes_ipc"
        assert payload["pattern"] == "auto"
        assert payload["plan_pattern_name"] == expected_pattern
        assert payload["plan_ok"] is True
        assert payload["high_level_ok"] is True
