"""Backend-level IPC handle sanity checks."""

from __future__ import annotations

import subprocess
import sys

import torch

from tncc.backends.cuda import CUDA_IPC_HANDLE_SIZE, CUDABackend


def test_cuda_ipc_handle_has_full_64_bytes(skip_no_gpu, device_info) -> None:
    """CUDA IPC handle serialization must preserve the full by-value payload."""
    if device_info.backend != "cuda":
        return

    torch.cuda.set_device(0)
    tensor = torch.empty(1, device="cuda:0", dtype=torch.uint8)
    backend = CUDABackend()
    handle = backend.get_ipc_handle(tensor.data_ptr())

    assert isinstance(handle, bytes)
    assert len(handle) == CUDA_IPC_HANDLE_SIZE


def test_cuda_ipc_open_handle_fails_cleanly_in_subprocess(
    skip_no_gpu,
    device_info,
) -> None:
    """IPC open should fail with a Python exception, not a segfault."""
    if device_info.backend != "cuda":
        return

    code = """
import torch
from tncc.backends.cuda import CUDABackend

torch.cuda.set_device(0)
tensor = torch.empty(1, device='cuda:0', dtype=torch.uint8)
backend = CUDABackend()
handle = backend.get_ipc_handle(tensor.data_ptr())
try:
    backend.open_ipc_handle(handle)
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(3)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Expected open_ipc_handle(self-handle) to raise RuntimeError cleanly, "
        f"got rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
