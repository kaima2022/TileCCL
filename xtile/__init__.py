"""XTile: Cross-platform tile communication library with full compiler visibility.

XTile combines the best ideas from Iris, TileScale, TileLink, and ThunderKittens
to provide a unified tile communication library that works across NVIDIA and AMD GPUs.

Key features:
- Communication as first-class primitive (alongside compute and memory)
- Full compiler visibility (pure Triton implementation)
- Hardware portability (NVIDIA Hopper/Blackwell + AMD CDNA3/CDNA4)
- Multi-scale unified abstraction
- Built-in compute-communication overlap patterns
"""

from __future__ import annotations

__version__ = "0.1.0"

import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from xtile.backends.base import TopologyInfo


# ---------------------------------------------------------------------------
# XTileContext -- returned by init()
# ---------------------------------------------------------------------------

@dataclass
class XTileContext:
    """Runtime context returned by :func:`init`.

    Carries the distributed configuration and hardware topology so that
    all higher-level APIs (patterns, memory, primitives) can query it
    without re-detecting.
    """

    rank: int
    """This GPU's rank in the process group (0-indexed)."""

    world_size: int
    """Total number of GPUs in the process group."""

    device: str
    """Torch device string, e.g. ``"cuda:0"`` or ``"hip:0"``."""

    backend: str
    """Backend identifier: ``"cuda"`` or ``"hip"``."""

    topology: Optional["TopologyInfo"] = field(default=None, repr=False)
    """Hardware topology information (populated lazily on first access)."""


# ---------------------------------------------------------------------------
# Global context singleton
# ---------------------------------------------------------------------------

_ctx: Optional[XTileContext] = None


def _get_ctx() -> XTileContext:
    """Return the global context, raising if :func:`init` was not called."""
    if _ctx is None:
        raise RuntimeError(
            "XTile has not been initialised. Call xtile.init() first."
        )
    return _ctx


# ---------------------------------------------------------------------------
# init()
# ---------------------------------------------------------------------------

def init(
    backend: str = "auto",
    rank: int | None = None,
    world_size: int | None = None,
) -> XTileContext:
    """Initialize the XTile runtime.

    Detects (or accepts) the GPU backend, rank, and world size, then
    returns an :class:`XTileContext` that all subsequent API calls use.

    Args:
        backend: ``"cuda"``, ``"hip"``, or ``"auto"`` (detect from
            ``torch.version``).
        rank: GPU rank.  Defaults to the ``RANK`` environment variable, or
            ``torch.distributed.get_rank()`` if distributed is initialised,
            or ``0``.
        world_size: Total number of GPUs.  Defaults to the ``WORLD_SIZE``
            environment variable, or ``torch.distributed.get_world_size()``
            if distributed is initialised, or ``1``.

    Returns:
        An :class:`XTileContext` carrying rank, world_size, device, backend,
        and topology information.

    Raises:
        RuntimeError: If the requested backend is unavailable.
    """
    global _ctx  # noqa: PLW0603

    import torch

    # -- backend detection --------------------------------------------------
    if backend == "auto":
        backend = _detect_backend()
    if backend not in ("cuda", "hip"):
        raise ValueError(f"Unsupported backend: {backend!r}. Use 'cuda', 'hip', or 'auto'.")

    # -- rank / world_size --------------------------------------------------
    if rank is None:
        rank = _resolve_rank()
    if world_size is None:
        world_size = _resolve_world_size()

    # -- device string ------------------------------------------------------
    device = f"cuda:{rank}" if backend == "cuda" else f"cuda:{rank}"
    # Note: PyTorch ROCm builds still use "cuda" device strings.

    # -- topology (lazy -- may fail on CPU-only machines) -------------------
    topology: Optional[TopologyInfo] = None
    try:
        from xtile.utils.topology import detect_topology
        topology = detect_topology(backend)
    except Exception:
        pass  # topology is optional; patterns fall back to defaults

    _ctx = XTileContext(
        rank=rank,
        world_size=world_size,
        device=device,
        backend=backend,
        topology=topology,
    )
    return _ctx


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Return the current rank (requires prior :func:`init` call)."""
    return _get_ctx().rank


def get_world_size() -> int:
    """Return the world size (requires prior :func:`init` call)."""
    return _get_ctx().world_size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Auto-detect ``'cuda'`` or ``'hip'`` from the installed PyTorch."""
    try:
        import torch
        hip_version = getattr(torch.version, "hip", None)
        if hip_version is not None:
            return "hip"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    raise RuntimeError(
        "Cannot auto-detect GPU backend. "
        "Install PyTorch with CUDA or ROCm support, or pass backend='cuda'/'hip' explicitly."
    )


def _resolve_rank() -> int:
    """Resolve rank from env vars or torch.distributed."""
    env_rank = os.environ.get("RANK")
    if env_rank is not None:
        return int(env_rank)
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _resolve_world_size() -> int:
    """Resolve world_size from env vars or torch.distributed."""
    env_ws = os.environ.get("WORLD_SIZE")
    if env_ws is not None:
        return int(env_ws)
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

# Tile is a forward-reference alias; the real class lives in xtile.memory.
# We defer import to avoid circular dependencies at startup.
def __getattr__(name: str):
    """Lazy import for heavy sub-modules to keep ``import xtile`` fast."""
    if name == "Tile":
        # Placeholder alias -- will resolve to a proper Tile class once
        # xtile.memory.tile is implemented.
        from xtile.memory.symmetric_heap import SymmetricHeap as _SH
        return _SH  # Tile == SymmetricHeap for now
    if name == "SymmetricHeap":
        from xtile.memory.symmetric_heap import SymmetricHeap
        return SymmetricHeap
    if name == "patterns":
        from xtile import patterns as _patterns
        return _patterns
    raise AttributeError(f"module 'xtile' has no attribute {name!r}")


__all__ = [
    "__version__",
    "init",
    "get_rank",
    "get_world_size",
    "XTileContext",
    "Tile",
    "SymmetricHeap",
    "patterns",
]
