# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the TNCC test suite.

Provides GPU detection, backend parametrization, and resource fixtures
that other test modules consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import pytest

# ---------------------------------------------------------------------------
# Device / backend detection
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """Lightweight summary of the available GPU environment."""

    has_gpu: bool
    """True if at least one CUDA/ROCm GPU is visible."""

    num_gpus: int
    """Number of visible GPU devices (0 on CPU-only machines)."""

    backend: str
    """``'cuda'``, ``'hip'``, or ``'cpu'``."""

    device: str
    """Torch device string for the first GPU, e.g. ``'cuda:0'``."""


def _detect_device_info() -> DeviceInfo:
    """Probe the runtime and return a :class:`DeviceInfo`."""
    try:
        import torch
    except ImportError:
        return DeviceInfo(has_gpu=False, num_gpus=0, backend="cpu", device="cpu")

    if not torch.cuda.is_available():
        return DeviceInfo(has_gpu=False, num_gpus=0, backend="cpu", device="cpu")

    num_gpus = torch.cuda.device_count()
    hip_version = getattr(torch.version, "hip", None)
    backend = "hip" if hip_version is not None else "cuda"

    return DeviceInfo(
        has_gpu=True,
        num_gpus=num_gpus,
        backend=backend,
        device="cuda:0",
    )


# Cache once per session
_DEVICE_INFO: Optional[DeviceInfo] = None


def _get_device_info() -> DeviceInfo:
    global _DEVICE_INFO
    if _DEVICE_INFO is None:
        _DEVICE_INFO = _detect_device_info()
    return _DEVICE_INFO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device_info() -> DeviceInfo:
    """Session-scoped fixture returning a :class:`DeviceInfo` snapshot."""
    return _get_device_info()


@pytest.fixture
def skip_no_gpu(device_info: DeviceInfo) -> None:
    """Skip the test if no GPU is available."""
    if not device_info.has_gpu:
        pytest.skip("No GPU available")


@pytest.fixture
def skip_no_multigpu(device_info: DeviceInfo) -> None:
    """Skip the test if fewer than 2 GPUs are available."""
    if device_info.num_gpus < 2:
        pytest.skip("Requires >= 2 GPUs")


@pytest.fixture(params=["cuda", "hip"])
def backend(request: pytest.FixtureRequest, device_info: DeviceInfo) -> str:
    """Parametrize over available backends, skipping unavailable ones.

    Yields ``'cuda'`` or ``'hip'`` -- only the variant that matches the
    current system is actually run.
    """
    requested = request.param
    if not device_info.has_gpu:
        pytest.skip("No GPU available")
    if requested != device_info.backend:
        pytest.skip(f"{requested} backend not available (have {device_info.backend})")
    return requested


@pytest.fixture
def symmetric_heap(device_info: DeviceInfo) -> Generator:
    """Create a small SymmetricHeap for testing and clean up afterwards.

    Yields the heap instance.  If no GPU is available the test is skipped.
    """
    if not device_info.has_gpu:
        pytest.skip("No GPU available -- cannot create SymmetricHeap")

    from tncc.memory.symmetric_heap import SymmetricHeap

    heap = SymmetricHeap(
        size=1024 * 1024,  # 1 MB
        rank=0,
        world_size=1,
        backend=device_info.backend,
    )
    try:
        yield heap
    finally:
        heap.cleanup()
