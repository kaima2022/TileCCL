# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess validation for broadcast."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.multigpu
@pytest.mark.parametrize("root", [0, 1])
@pytest.mark.parametrize(
    ("dtype_name", "block_size", "expected_path", "expected_regime", "expected_protocol"),
    [
        ("float16", 128, "legacy", "legacy", "flat_root_broadcast"),
        ("bfloat16", 128, "legacy", "legacy", "flat_root_broadcast"),
        ("float32", 128, "legacy", "legacy", "flat_root_broadcast"),
        ("float32", 4097, "staged", "staged", "ws2_slot_epoch_pipeline"),
    ],
)
def test_broadcast_multiprocess_default_transport(
    skip_no_multigpu,
    dtype_name: str,
    block_size: int,
    expected_path: str,
    expected_regime: str,
    expected_protocol: str,
    root: int,
) -> None:
    """Broadcast should be correct for root=0/1 across legacy and staged sizes."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_e2e._run_broadcast_multiprocess",
            "--dtype",
            dtype_name,
            "--block-size",
            str(block_size),
            "--warmup",
            "2",
            "--iters",
            "4",
            "--root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Multiprocess broadcast validation failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    payloads = [
        json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")
    ]
    assert payloads, f"No JSON payloads found in stdout:\n{result.stdout}"
    for payload in payloads:
        assert payload["dtype"] == dtype_name
        assert payload["block_size"] == block_size
        assert payload["root"] == root
        assert payload["mode"] == "multiprocess"
        assert payload["transport_strategy"] == "ctypes_ipc"
        assert payload["primitive_ok"] is True
        assert payload["high_level_ok"] is True
        assert payload["kernel_ok"] is True
        execution = payload["primitive_execution"]
        assert execution["path"] == expected_path
        assert execution["message_regime"] == expected_regime
        assert execution["protocol"] == expected_protocol
        assert execution["root_mode"] == f"explicit_root_rank_{root}"
        assert execution["chunk_elems"] <= 4096
        assert execution["pipeline_slots"] <= 8
