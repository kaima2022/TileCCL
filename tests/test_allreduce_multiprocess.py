"""Real multiprocess validation for allreduce."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.multigpu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_allreduce_multiprocess_default_transport(
    skip_no_multigpu,
    dtype_name: str,
) -> None:
    """The default multiprocess allreduce path should be correct on the current runtime."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_allreduce_multiprocess",
            "--dtype",
            dtype_name,
            "--block-size",
            "2049",
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
            "Multiprocess allreduce validation failed.\n"
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
        assert payload["primitive_ok"] is True
        assert payload["high_level_ok"] is True
        assert payload["kernel_ok"] is None
        assert payload["total_elements"] == 4098
        assert payload["primitive_execution"]["implementation"] == "device_staged_pipeline"
        assert payload["primitive_execution"]["protocol"] == "slot_epoch_pipeline"
        assert payload["primitive_execution"]["kernel_family"] == "ws2_specialized"
        assert payload["primitive_execution"]["reuse_handshake"] == "ws2_epoch_ack"
        assert payload["primitive_execution"]["pipeline_slots"] >= 2
        assert payload["high_level_plan"]["implementation"] == "device_staged_pipeline"
        assert payload["high_level_plan"]["protocol"] == "slot_epoch_pipeline"
        assert payload["high_level_plan"]["kernel_family"] == "ws2_specialized"
        assert payload["high_level_plan"]["reuse_handshake"] == "ws2_epoch_ack"
        assert payload["high_level_plan"]["pipeline_slots"] >= 2
