"""
tncc.backends - Hardware Abstraction Layer (HAL).

This package abstracts over NVIDIA (CUDA) and AMD (HIP) GPUs so that the
rest of TNCC can be written in a hardware-agnostic way.

Public API
----------
.. autoclass:: BackendInterface
.. autofunction:: get_backend
.. autofunction:: detect_hardware
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch

from tncc.backends.base import BackendInterface, DeviceProperties, TopologyInfo

logger = logging.getLogger(__name__)

__all__ = [
    "BackendInterface",
    "DeviceProperties",
    "TopologyInfo",
    "get_backend",
    "detect_hardware",
]


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> Literal["cuda", "hip", "none"]:
    """Detect which GPU backend is available on the current machine.

    Detection order:

    1. If ``torch.cuda.is_available()`` is ``True`` **and** the PyTorch
       build targets ROCm (``torch.version.hip`` is set), return ``"hip"``.
    2. If ``torch.cuda.is_available()`` is ``True`` and the build targets
       CUDA, return ``"cuda"``.
    3. Otherwise return ``"none"``.

    Returns:
        One of ``"cuda"``, ``"hip"``, or ``"none"``.
    """
    if not torch.cuda.is_available():
        logger.info("No GPU detected (torch.cuda.is_available() == False)")
        return "none"

    # On ROCm builds, torch.version.hip is a non-empty string (e.g. "5.7")
    hip_version = getattr(torch.version, "hip", None)
    if hip_version is not None and hip_version:
        logger.info("AMD GPU detected (torch.version.hip = %s)", hip_version)
        return "hip"

    logger.info("NVIDIA GPU detected (torch.version.cuda = %s)", torch.version.cuda)
    return "cuda"


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

# Module-level cache so repeated calls to get_backend() reuse the same object.
_backend_instance: Optional[BackendInterface] = None


def get_backend(backend: Optional[str] = None, *, force: bool = False) -> BackendInterface:
    """Return a :class:`BackendInterface` for the requested (or detected) GPU.

    Args:
        backend: Explicit backend name -- ``"cuda"`` or ``"hip"``.
            If ``None`` (default), :func:`detect_hardware` is called
            to pick automatically.
        force: When ``True``, create a fresh backend even if one has
            already been cached.  Useful in tests.

    Returns:
        A concrete :class:`BackendInterface` instance.

    Raises:
        RuntimeError: If no GPU is available or the requested backend
            cannot be loaded.
    """
    global _backend_instance

    if _backend_instance is not None and not force:
        return _backend_instance

    if backend is None:
        backend = detect_hardware()

    if backend == "cuda":
        from tncc.backends.cuda import CUDABackend

        _backend_instance = CUDABackend()
        logger.info("Initialised CUDA backend")
        return _backend_instance

    if backend == "hip":
        from tncc.backends.hip import HIPBackend

        _backend_instance = HIPBackend()
        logger.info("Initialised HIP backend")
        return _backend_instance

    raise RuntimeError(
        f"No usable GPU backend found (requested={backend!r}). "
        "Ensure that either CUDA or ROCm is properly installed and "
        "that PyTorch can see a GPU (torch.cuda.is_available())."
    )
