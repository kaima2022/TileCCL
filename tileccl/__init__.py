# SPDX-License-Identifier: Apache-2.0
"""TileCCL -- Tile-Native Collective Communication.

Collective communication as compiler-visible tile primitives in Triton,
with built-in compute-communication overlap patterns.

Public entry points::

    tileccl.init(...)          # distributed or single-rank initialization
    tileccl.init_local(...)    # single-process multi-GPU setup
    tileccl.ops.*              # high-level fused operations

See the README for usage examples.
"""

from __future__ import annotations

__version__ = "0.1.0"

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

    from tileccl.backends.base import BackendInterface, TopologyInfo
    from tileccl.memory.symmetric_heap import SymmetricHeap


# ---------------------------------------------------------------------------
# TileCCLContext -- returned by init()
# ---------------------------------------------------------------------------

@dataclass
class TileCCLContext:
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

    backend_name: str
    """Backend identifier: ``"cuda"`` or ``"hip"``."""

    backend: "BackendInterface" = field(repr=False)
    """Concrete backend implementation used by patterns and utilities."""

    topology: Optional["TopologyInfo"] = field(default=None, repr=False)
    """Hardware topology information (populated lazily on first access)."""

    heap: Optional["SymmetricHeap"] = field(default=None, repr=False)
    """Optional symmetric heap attached to this context."""

    _workspace_cache: dict[tuple[str, tuple[int, ...], str], "torch.Tensor"] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    """Reusable heap-backed scratch buffers keyed by logical purpose + shape."""

    @property
    def has_heap(self) -> bool:
        """Whether a symmetric heap is attached to this context."""
        return self.heap is not None

    @property
    def heap_bases(self) -> "torch.Tensor":
        """Return the attached heap's ``heap_bases`` tensor.

        Raises:
            RuntimeError: If no symmetric heap is attached.
        """
        return self.require_heap().get_heap_bases()

    def require_heap(self) -> "SymmetricHeap":
        """Return the attached heap or raise with an actionable error."""
        if self.heap is None:
            raise RuntimeError(
                "No SymmetricHeap is attached to this TileCCLContext. "
                "Pass heap=... or heap_size=... to tileccl.init(), call "
                "tileccl.init_local(...), or attach an existing heap via "
                "ctx.attach_heap(heap)."
            )
        return self.heap

    def attach_heap(self, heap: "SymmetricHeap") -> "TileCCLContext":
        """Attach an existing :class:`~tileccl.memory.symmetric_heap.SymmetricHeap`.

        The heap must match this context's rank and world size.
        Returns ``self`` for fluent usage.
        """
        if heap.rank != self.rank:
            raise ValueError(
                f"Heap rank {heap.rank} does not match context rank {self.rank}"
            )
        if heap.world_size != self.world_size:
            raise ValueError(
                "Heap world_size does not match context world_size: "
                f"{heap.world_size} != {self.world_size}"
            )
        heap_backend = getattr(heap, "_backend_name", self.backend_name)
        if heap_backend != self.backend_name:
            raise ValueError(
                "Heap backend does not match context backend: "
                f"{heap_backend!r} != {self.backend_name!r}"
            )
        self.heap = heap
        self._workspace_cache.clear()
        return self

    def barrier(self) -> None:
        """Synchronize pending work associated with this context."""
        if self.heap is not None:
            self.heap.barrier()
        else:
            self.backend.synchronize()

    def allocate_tensor(self, shape: tuple[int, ...], dtype: "torch.dtype") -> "torch.Tensor":
        """Allocate a tensor from the attached symmetric heap."""
        return self.require_heap().allocate_tensor(shape, dtype)

    def is_symmetric(self, tensor: "torch.Tensor") -> bool:
        """Return ``True`` when *tensor* resides in the attached symmetric heap."""
        return self.require_heap().is_symmetric(tensor)

    def as_symmetric(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Materialize one external tensor inside the attached symmetric heap."""
        return self.require_heap().as_symmetric(tensor)

    def empty(self, *size: int, dtype: "torch.dtype") -> "torch.Tensor":
        """Allocate an uninitialised tensor from the attached symmetric heap."""
        return self.allocate_tensor(_normalize_shape(size), dtype)

    def zeros(self, *size: int, dtype: "torch.dtype") -> "torch.Tensor":
        """Allocate a zero-filled tensor from the attached symmetric heap."""
        tensor = self.allocate_tensor(_normalize_shape(size), dtype)
        tensor.zero_()
        return tensor

    def randn(self, *size: int, dtype: "torch.dtype") -> "torch.Tensor":
        """Allocate a tensor from the heap and fill it with normal samples."""
        tensor = self.allocate_tensor(_normalize_shape(size), dtype)
        tensor.normal_()
        return tensor

    def workspace(
        self,
        name: str,
        *size: int,
        dtype: "torch.dtype",
        zero: bool = False,
    ) -> "torch.Tensor":
        """Return a reusable heap-backed workspace tensor.

        The first request for a given ``(name, shape, dtype)`` key allocates
        from the attached symmetric heap; subsequent requests reuse the same
        buffer so high-level wrappers do not monotonically consume heap space.
        """
        shape = _normalize_shape(size)
        key = (name, shape, str(dtype))
        tensor = self._workspace_cache.get(key)
        if tensor is None:
            tensor = self.allocate_tensor(shape, dtype)
            self._workspace_cache[key] = tensor
        if zero:
            tensor.zero_()
        return tensor

    def auto_select_pattern(
        self,
        op: str,
        M: int,
        N: int,
        K: int,
        *,
        hw_info: object | None = None,
    ):
        """Return the auto-selected pattern bound to this context."""
        from tileccl.patterns.auto_select import auto_select

        return auto_select(
            op,
            M=M,
            N=N,
            K=K,
            world_size=self.world_size,
            hw_info=hw_info,
            ctx=self,
        )

    def support_matrix(self):
        """Return the current structured runtime support matrix."""
        from tileccl.support import describe_runtime_support

        return describe_runtime_support(self)

    def heap_metadata(self) -> dict[str, object]:
        """Return structured metadata for the attached symmetric heap."""
        return self.require_heap().metadata()

    def runtime_metadata(self) -> dict[str, object]:
        """Return one structured runtime snapshot for docs and diagnostics."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "device": self.device,
            "backend": self.backend_name,
            "has_heap": self.has_heap,
            "heap": self.heap_metadata() if self.has_heap else None,
        }


# ---------------------------------------------------------------------------
# Global context singleton
# ---------------------------------------------------------------------------

_ctx: Optional[TileCCLContext] = None


def _get_ctx() -> TileCCLContext:
    """Return the global context, raising if :func:`init` was not called."""
    if _ctx is None:
        raise RuntimeError(
            "TileCCL has not been initialised. Call tileccl.init() first."
        )
    return _ctx


def describe_runtime_support(ctx: TileCCLContext | None = None):
    """Return the current structured runtime support matrix."""
    from tileccl.support import describe_runtime_support as _describe_runtime_support

    return _describe_runtime_support(ctx)


# ---------------------------------------------------------------------------
# init()
# ---------------------------------------------------------------------------

def init(
    backend: str = "auto",
    rank: int | None = None,
    world_size: int | None = None,
    *,
    heap: "SymmetricHeap | None" = None,
    heap_size: int | None = None,
    force_backend: bool = False,
) -> TileCCLContext:
    """Initialize the TileCCL runtime.

    Detects (or accepts) the GPU backend, rank, and world size, then
    returns an :class:`TileCCLContext` that all subsequent API calls use.

    Args:
        backend: ``"cuda"``, ``"hip"``, or ``"auto"`` (detect from
            ``torch.version``).
        rank: GPU rank.  Defaults to the ``RANK`` environment variable, or
            ``torch.distributed.get_rank()`` if distributed is initialised,
            or ``0``.
        world_size: Total number of GPUs.  Defaults to the ``WORLD_SIZE``
            environment variable, or ``torch.distributed.get_world_size()``
            if distributed is initialised, or ``1``.
        heap: Existing symmetric heap to attach to the returned context.
            Useful when heaps are created externally via
            :meth:`tileccl.memory.symmetric_heap.SymmetricHeap.create_all`.
        heap_size: When provided, create and attach a symmetric heap
            automatically. For ``world_size > 1`` this requires
            ``torch.distributed`` to be initialised; otherwise use
            :func:`init_local` or create/attach heaps manually.
        force_backend: When ``True``, bypass the cached backend instance and
            create a fresh backend object.

    Returns:
        An :class:`TileCCLContext` carrying rank, world_size, device, backend,
        and topology information.

    Raises:
        RuntimeError: If the requested backend is unavailable.
    """
    global _ctx  # noqa: PLW0603

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

    if heap is not None and heap_size is not None:
        raise ValueError("Pass at most one of heap=... and heap_size=...")

    _ctx = _build_context(
        backend_name=backend,
        rank=rank,
        world_size=world_size,
        force_backend=force_backend,
    )
    if heap is not None:
        _ctx.attach_heap(heap)
    elif heap_size is not None:
        _ctx.attach_heap(_create_heap_for_context(_ctx, heap_size))
    return _ctx


def init_local(
    world_size: int,
    heap_size: int,
    *,
    backend: str = "auto",
    force_backend: bool = False,
) -> list[TileCCLContext]:
    """Initialise one attached context per GPU in a single process.

    This is the recommended entry point for single-node, single-process
    multi-GPU experiments and benchmarks.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if heap_size <= 0:
        raise ValueError(f"heap_size must be positive, got {heap_size}")

    if backend == "auto":
        backend = _detect_backend()
    if backend not in ("cuda", "hip"):
        raise ValueError(f"Unsupported backend: {backend!r}. Use 'cuda', 'hip', or 'auto'.")

    from tileccl.memory.symmetric_heap import SymmetricHeap

    heaps = SymmetricHeap.create_all(size=heap_size, world_size=world_size, backend=backend)
    contexts: list[TileCCLContext] = []
    for rank, heap in enumerate(heaps):
        ctx = _build_context(
            backend_name=backend,
            rank=rank,
            world_size=world_size,
            force_backend=force_backend and rank == 0,
        )
        ctx.attach_heap(heap)
        contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Return the current rank (requires prior :func:`init` call)."""
    return _get_ctx().rank


def get_world_size() -> int:
    """Return the world size (requires prior :func:`init` call)."""
    return _get_ctx().world_size


def current_context() -> TileCCLContext:
    """Return the process-global context set by :func:`init`."""
    return _get_ctx()


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


def _build_context(
    *,
    backend_name: str,
    rank: int,
    world_size: int,
    force_backend: bool = False,
) -> TileCCLContext:
    """Create a context object without mutating global state."""
    from tileccl.backends import get_backend

    backend_impl = get_backend(backend_name, force=force_backend)
    topology = _detect_topology_safe(backend_name)
    device = f"cuda:{rank}"  # PyTorch ROCm builds still use CUDA device strings.
    return TileCCLContext(
        rank=rank,
        world_size=world_size,
        device=device,
        backend_name=backend_name,
        backend=backend_impl,
        topology=topology,
    )


def _detect_topology_safe(backend: str) -> Optional["TopologyInfo"]:
    """Best-effort topology detection used by context construction."""
    try:
        from tileccl.utils.topology import detect_topology

        return detect_topology(backend)
    except Exception:
        return None


def _normalize_shape(size: tuple[object, ...]) -> tuple[int, ...]:
    """Normalise ``torch``-style ``*size`` arguments to a shape tuple."""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = size[0]
    else:
        shape = size
    try:
        normalized = tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError(f"Invalid shape specification: {size!r}") from exc
    if any(dim < 0 for dim in normalized):
        raise ValueError(f"Shape dimensions must be non-negative, got {normalized}")
    return normalized


def _create_heap_for_context(ctx: TileCCLContext, heap_size: int) -> "SymmetricHeap":
    """Create a heap for ``ctx`` using the safest supported mode."""
    if heap_size <= 0:
        raise ValueError(f"heap_size must be positive, got {heap_size}")

    from tileccl.memory.symmetric_heap import SymmetricHeap

    if ctx.world_size == 1:
        return SymmetricHeap(
            size=heap_size,
            rank=ctx.rank,
            world_size=ctx.world_size,
            backend=ctx.backend_name,
        )

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return SymmetricHeap(
                size=heap_size,
                rank=ctx.rank,
                world_size=ctx.world_size,
                backend=ctx.backend_name,
            )
    except Exception:
        pass

    raise RuntimeError(
        "Automatic heap creation for world_size > 1 requires torch.distributed "
        "to be initialised. For single-process multi-GPU, use tileccl.init_local(...); "
        "otherwise create a SymmetricHeap explicitly and attach it via heap=..."
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

# Tile is a forward-reference alias; the real class lives in tileccl.memory.
# We defer import to avoid circular dependencies at startup.
def __getattr__(name: str):
    """Lazy import for heavy sub-modules to keep ``import tileccl`` fast."""
    if name == "Tile":
        # Placeholder alias -- will resolve to a proper Tile class once
        # tileccl.memory.tile is implemented.
        from tileccl.memory.symmetric_heap import SymmetricHeap as _SH
        return _SH  # Tile == SymmetricHeap for now
    if name == "SymmetricHeap":
        from tileccl.memory.symmetric_heap import SymmetricHeap
        return SymmetricHeap
    if name == "patterns":
        import importlib

        _patterns = importlib.import_module("tileccl.patterns")
        return _patterns
    if name == "ops":
        import importlib

        _ops = importlib.import_module("tileccl.ops")
        return _ops
    raise AttributeError(f"module 'tileccl' has no attribute {name!r}")


__all__ = [
    "__version__",
    "init",
    "init_local",
    "get_rank",
    "get_world_size",
    "current_context",
    "describe_runtime_support",
    "TileCCLContext",
    "Tile",
    "SymmetricHeap",
    "patterns",
    "ops",
]
