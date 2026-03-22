"""Real multiprocess validation for reduce_scatter(device)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from xtile.utils.feature_gates import MULTIPROCESS_DEVICE_COLLECTIVES_ENV


@pytest.mark.multigpu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_reduce_scatter_device_path_multiprocess(
    skip_no_multigpu,
    dtype_name: str,
) -> None:
    """The multiprocess device path should be correct on the current runtime."""
    if os.getenv(MULTIPROCESS_DEVICE_COLLECTIVES_ENV, "").strip() != "1":
        pytest.skip(
            "Real multiprocess device-path diagnostics are opt-in only. "
            f"Set {MULTIPROCESS_DEVICE_COLLECTIVES_ENV}=1 to run this unsafe test."
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_reduce_scatter_multiprocess",
            "--dtype",
            dtype_name,
            "--warmup",
            "2",
            "--iters",
            "4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Multiprocess reduce_scatter(device) validation failed.\n"
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
        assert payload["primitive_ok"] is True
        assert payload["high_level_ok"] is True
