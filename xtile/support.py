"""Runtime support matrix for public XTile surfaces.

This module centralises the "what is actually supported right now"
answer so docs, tests, and future CLI/status tooling can share one
source of truth instead of manually repeating status tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import xtile
from xtile.utils.feature_gates import (
    multiprocess_device_collectives_detail,
    multiprocess_device_collectives_enabled,
    multiprocess_device_collectives_runtime_supported,
    multiprocess_device_collectives_transport_supported,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_runtime_supported,
    multiprocess_device_remote_access_transport_supported,
)

SupportState = Literal["supported", "partial", "unsupported"]


@dataclass(frozen=True, slots=True)
class SupportStatus:
    """One support-matrix entry."""

    state: SupportState
    detail: str

    @property
    def supported(self) -> bool:
        """Return ``True`` when the entry is fully supported."""
        return self.state == "supported"

    def to_dict(self) -> dict[str, str]:
        """Serialize the entry for JSON/debug output."""
        return {
            "state": self.state,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSupportMatrix:
    """Structured summary of the currently exposed XTile support surface."""

    backend: str
    device: str
    rank: int
    world_size: int
    has_heap: bool
    heap_mode: str | None
    transport_strategy: str | None
    ops: dict[str, SupportStatus]
    contracts: dict[str, SupportStatus]
    execution_paths: dict[str, SupportStatus]
    collectives: dict[str, SupportStatus]
    memory: dict[str, SupportStatus]

    def to_dict(self) -> dict[str, object]:
        """Serialize the matrix for logs, docs, or CLI rendering."""
        return {
            "context": {
                "backend": self.backend,
                "device": self.device,
                "rank": self.rank,
                "world_size": self.world_size,
                "has_heap": self.has_heap,
                "heap_mode": self.heap_mode,
                "transport_strategy": self.transport_strategy,
            },
            "ops": {name: status.to_dict() for name, status in self.ops.items()},
            "contracts": {
                name: status.to_dict() for name, status in self.contracts.items()
            },
            "execution_paths": {
                name: status.to_dict()
                for name, status in self.execution_paths.items()
            },
            "collectives": {
                name: status.to_dict() for name, status in self.collectives.items()
            },
            "memory": {
                name: status.to_dict() for name, status in self.memory.items()
            },
        }


def describe_runtime_support(
    ctx: xtile.XTileContext | None = None,
) -> RuntimeSupportMatrix:
    """Return the current public support matrix.

    The matrix is intentionally conservative: entries are marked
    ``"supported"`` only when there is a stable public path in the
    current repository, ``"partial"`` when code exists but the public
    contract or validation story is still incomplete, and
    ``"unsupported"`` otherwise.
    """

    resolved_ctx = ctx if ctx is not None else xtile.current_context()
    heap = resolved_ctx.heap
    has_heap = heap is not None
    heap_mode = heap.mode if has_heap else None
    transport_strategy = heap.transport_strategy if has_heap else None
    world_size = resolved_ctx.world_size
    (
        reduce_scatter_state,
        reduce_scatter_detail,
        reduce_scatter_paths,
    ) = _describe_reduce_scatter_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    gemm_allscatter_status = _describe_gemm_allscatter_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    gemm_allgather_status = _describe_gemm_allgather_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    gemm_reducescatter_status = _describe_gemm_reducescatter_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    allgather_status = _describe_allgather_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    allreduce_status = _describe_allreduce_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )

    ops = {
        "gemm_allscatter": gemm_allscatter_status,
        "gemm_allgather": gemm_allgather_status,
        "allgather": allgather_status,
        "allreduce": allreduce_status,
        "reduce_scatter": SupportStatus(
            reduce_scatter_state,
            reduce_scatter_detail,
        ),
        "gemm_reducescatter": gemm_reducescatter_status,
    }

    contracts = {
        "gemm_allscatter.full/full": SupportStatus(
            "supported",
            "Stable public contract for full RHS and full output buffers.",
        ),
        "gemm_allscatter.shard/shard": SupportStatus(
            "supported",
            "Expert contract available via gemm_allscatter_sharded(...).",
        ),
        "gemm_allscatter.full/shard": SupportStatus(
            "supported",
            "High-level wrapper materializes a heap-backed full output, then returns the rank-local shard.",
        ),
        "gemm_allscatter.shard/full": SupportStatus(
            "unsupported",
            "Rejected intentionally: current shard/shard gemm_allscatter execution is a peer-scatter contract, not a stable local-shard basis for full-output assembly.",
        ),
        "gemm_allgather.shard/full": SupportStatus(
            gemm_allgather_status.state,
            "Stable public host contract: A(M,K) full LHS, B(K,N/world_size) rank-local "
            "RHS shard, C(M,N) full output. Runtime availability inherits the "
            "validated allgather path for the current heap mode/transport.",
        ),
        "gemm_reducescatter.full/shard": SupportStatus(
            gemm_reducescatter_status.state,
            "Stable public host contract: A(M,K) local contribution, B(K,N) full RHS, "
            "C(M,N/world_size) rank-local shard. Runtime availability inherits the "
            "validated reduce_scatter path for the current heap mode/transport.",
        ),
        "allreduce.in_place": SupportStatus(
            allreduce_status.state,
            "Stable public in-place host contract: tensor.numel must be divisible by "
            "world_size, and the tensor must reside in the attached symmetric heap.",
        ),
    }

    collectives = {
        "collectives.allgather_launcher": allgather_status,
        "collectives.allreduce_launcher": allreduce_status,
        "collectives.reduce_scatter_launcher": SupportStatus(
            reduce_scatter_state,
            reduce_scatter_detail,
        ),
    }

    memory = {
        "symmetric_heap.current_runtime": SupportStatus(
            "supported" if has_heap else "partial",
            f"Attached heap mode={heap_mode!r}, transport_strategy={transport_strategy!r}."
            if has_heap
            else "Context is valid without a heap, but collective/high-level heap-backed ops are unavailable.",
        ),
        "symmetric_heap.device_remote_access": _describe_heap_device_remote_access(
            has_heap=has_heap,
            heap_mode=heap_mode,
            transport_strategy=transport_strategy,
            world_size=world_size,
        ),
        "symmetric_heap_allocator_first_import_map": SupportStatus(
            "partial" if has_heap else "unsupported",
            "Allocator-backed heap runtime is implemented with a torch_bump backend, "
            "allocator-owned local segment metadata, peer export/import descriptors, "
            "and copy-based external import/as_symmetric materialization. Canonical "
            "segmented peer import/map is still not implemented."
            if has_heap
            else "Allocator-first canonical import/map layer requires an attached heap.",
        ),
        "symmetric_heap.external_import": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes import_external_tensor()/as_symmetric() via the active allocator."
            if has_heap
            else "Attach a heap before using external import/as_symmetric helpers.",
        ),
        "symmetric_heap.external_mapping": SupportStatus(
            "unsupported" if has_heap else "partial",
            "The active allocator does not yet implement zero-copy external mapping "
            "(FD passing / DMA-BUF import-map)."
            if has_heap
            else "Attach a heap to inspect allocator external-mapping capability. "
            "The current default torch_bump backend does not implement zero-copy external mapping.",
        ),
        "symmetric_heap.external_memory_interface": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes the allocator's structured external-memory interface descriptor, "
            "including import mode, mapping mode, and zero-copy capability flags."
            if has_heap
            else "Attach a heap before querying allocator external-memory interface metadata.",
        ),
        "symmetric_heap.segment_metadata": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes allocator-owned local segment metadata for the active heap backend."
            if has_heap
            else "Attach a heap before querying allocator-owned segment metadata.",
        ),
        "symmetric_heap.segment_layout": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes the allocator's exportable segment-layout descriptor, including "
            "primary segment id and exportable segment ids."
            if has_heap
            else "Attach a heap before querying allocator segment-layout metadata.",
        ),
        "symmetric_heap.exportable_segment_metadata": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes the current exportable segment descriptors separately from the "
            "full allocator-owned segment catalog."
            if has_heap
            else "Attach a heap before querying exportable segment metadata.",
        ),
        "symmetric_heap.allocator_memory_model": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes the allocator memory-model descriptor, including local segment layout, "
            "peer import/mapping model, and current external import/mapping modes."
            if has_heap
            else "Attach a heap before querying allocator memory-model metadata.",
        ),
        "symmetric_heap.peer_import_metadata": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes structured peer-import records for the active transport strategy."
            if has_heap
            else "Attach a heap before querying peer-import metadata.",
        ),
        "symmetric_heap.peer_segment_catalog": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes segment-scoped peer export/import catalogs keyed by peer rank "
            "and segment id, in addition to the flat per-record metadata."
            if has_heap
            else "Attach a heap before querying segment-scoped peer export/import catalogs.",
        ),
        "symmetric_heap.peer_mapping_metadata": SupportStatus(
            "supported" if has_heap else "partial",
            "Heap exposes allocator-owned peer export descriptors, peer-import records, plus structured peer mapping metadata."
            if has_heap
            else "Attach a heap before querying peer export/import mapping metadata.",
        ),
    }

    return RuntimeSupportMatrix(
        backend=resolved_ctx.backend_name,
        device=resolved_ctx.device,
        rank=resolved_ctx.rank,
        world_size=resolved_ctx.world_size,
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        ops=ops,
        contracts=contracts,
        execution_paths=reduce_scatter_paths,
        collectives=collectives,
        memory=memory,
    )


def _describe_reduce_scatter_support(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> tuple[SupportState, str, dict[str, SupportStatus]]:
    """Describe reduce_scatter support at both API and path granularity."""
    if not has_heap:
        detail = "No attached symmetric heap; execution path is unavailable."
        return (
            "partial",
            detail,
            {
                "reduce_scatter.reference": SupportStatus("partial", detail),
                "reduce_scatter.device": SupportStatus("partial", detail),
            },
        )

    if heap_mode == "single_process":
        return (
            "supported",
            "Single-process reference reduce_scatter path is validated; "
            "the explicit device override is intentionally rejected for this mode until proven correct.",
            {
                "reduce_scatter.reference": SupportStatus(
                    "supported",
                    "Validated on single-process peer-access heaps.",
                ),
                "reduce_scatter.device": SupportStatus(
                    "unsupported",
                    "Single-process device override is intentionally rejected because it is not yet correct.",
                ),
            },
        )

    gate_detail = multiprocess_device_collectives_detail(
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    if multiprocess_device_collectives_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        return (
            "supported",
            "Multiprocess device reduce_scatter is validated on the current public "
            "surface: world_size=2 with transport_strategy='ctypes_ipc'.",
            {
                "reduce_scatter.reference": SupportStatus(
                    "unsupported",
                    "Reference path is only defined for single-process peer-buffer access.",
                ),
                "reduce_scatter.device": SupportStatus(
                    "supported",
                    "Validated on the current public multiprocess surface: world_size=2, transport_strategy='ctypes_ipc'.",
                ),
            },
        )
    if not multiprocess_device_collectives_enabled():
        return (
            "unsupported",
            gate_detail,
            {
                "reduce_scatter.reference": SupportStatus(
                    "unsupported",
                    "Reference path is only defined for single-process peer-buffer access.",
                ),
                "reduce_scatter.device": SupportStatus(
                    "unsupported",
                    gate_detail,
                ),
            },
        )

    if not multiprocess_device_collectives_transport_supported(
        transport_strategy
    ):
        return (
            "unsupported",
            gate_detail,
            {
                "reduce_scatter.reference": SupportStatus(
                    "unsupported",
                    "Reference path is only defined for single-process peer-buffer access.",
                ),
                "reduce_scatter.device": SupportStatus(
                    "unsupported",
                    gate_detail,
                ),
            },
        )

    return (
        "partial",
        "Multiprocess device-path launcher is running outside the validated public "
        "surface under an explicit diagnostic gate.",
        {
            "reduce_scatter.reference": SupportStatus(
                "unsupported",
                "Reference path is only defined for single-process peer-buffer access.",
            ),
            "reduce_scatter.device": SupportStatus(
                "partial",
                "Diagnostic-only path outside the validated public surface.",
            ),
        },
    )


def _describe_heap_device_remote_access(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> SupportStatus:
    """Describe whether the current heap transport is device-dereferenceable by Triton."""
    if not has_heap:
        return SupportStatus(
            "partial",
            "No attached symmetric heap; Triton remote device access is unavailable.",
        )
    if heap_mode == "single_process":
        return SupportStatus(
            "supported",
            "Single-process peer-access heaps are directly device-dereferenceable.",
        )
    if multiprocess_device_remote_access_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        return SupportStatus(
            "supported",
            "Real 2-GPU minimal Triton remote-load/store diagnostics pass on the "
            "current public multiprocess surface.",
        )
    if multiprocess_device_remote_access_transport_supported(transport_strategy):
        return SupportStatus(
            "unsupported",
            "The transport itself is known, but the current world_size is outside the "
            "validated public surface for Triton device-side remote dereference. "
            f"Current transport_strategy={transport_strategy!r}, world_size={world_size}.",
        )
    return SupportStatus(
        "unsupported",
        "Current multiprocess transport is not validated for Triton device-side "
        "remote dereference. Real diagnostics currently public-support only "
        "world_size=2 with transport_strategy='ctypes_ipc'. "
        f"Current transport_strategy={transport_strategy!r}, world_size={world_size}.",
    )


def _describe_gemm_allscatter_support(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> SupportStatus:
    """Describe high-level GEMM + all-scatter support conservatively."""
    if not has_heap:
        return SupportStatus(
            "partial",
            "High-level plan API exists, but current execution still requires an attached symmetric heap.",
        )
    if heap_mode == "single_process":
        return SupportStatus(
            "supported",
            "High-level plan API is validated on single-process peer-access heaps.",
        )
    if not multiprocess_device_remote_access_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        return SupportStatus(
            "unsupported",
            multiprocess_device_remote_access_detail(
                transport_strategy=transport_strategy,
                operation="xtile.ops.gemm_allscatter(...)",
                world_size=world_size,
            ),
        )
    return SupportStatus(
        "supported",
        "Current 2-GPU ctypes_ipc correctness matrix passes for the public "
        "full/full and full/shard contracts, including representative "
        "auto-selected pattern coverage.",
    )


def _describe_gemm_allgather_support(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> SupportStatus:
    """Describe high-level GEMM + allgather support conservatively."""
    allgather_status = _describe_allgather_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    if not has_heap:
        return SupportStatus(
            "partial",
            "High-level host contract exists, but execution requires an attached "
            "symmetric heap for the local shard workspace and allgather output path.",
        )
    if heap_mode == "single_process":
        return SupportStatus(
            "supported",
            "High-level host plan is validated on single-process peer-access heaps "
            "via local GEMM materialization plus allgather execution.",
        )
    if allgather_status.state == "unsupported":
        return SupportStatus(
            "unsupported",
            multiprocess_device_remote_access_detail(
                transport_strategy=transport_strategy,
                operation="xtile.ops.gemm_allgather(...)",
                world_size=world_size,
            ),
        )
    return SupportStatus(
        "supported",
        "Host GEMM + allgather contract is validated on the current public "
        "multiprocess surface: world_size=2 with transport_strategy='ctypes_ipc'.",
    )


def _describe_gemm_reducescatter_support(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> SupportStatus:
    """Describe high-level GEMM + reduce-scatter support conservatively."""
    reduce_scatter_state, _, _ = _describe_reduce_scatter_support(
        has_heap=has_heap,
        heap_mode=heap_mode,
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
    if not has_heap:
        return SupportStatus(
            "partial",
            "High-level host contract exists, but execution requires an attached "
            "symmetric heap for the reduce_scatter output path and workspaces.",
        )
    if heap_mode == "single_process":
        return SupportStatus(
            "supported",
            "High-level host plan is validated on single-process peer-access heaps "
            "via local GEMM materialization plus reduce_scatter reference execution.",
        )
    if reduce_scatter_state == "unsupported":
        return SupportStatus(
            "unsupported",
            multiprocess_device_collectives_detail(
                transport_strategy=transport_strategy,
                world_size=world_size,
            ),
        )
    return SupportStatus(
        "supported",
        "Host GEMM + packed reduce_scatter contract is validated on the current "
        "public multiprocess surface: world_size=2 with transport_strategy='ctypes_ipc'.",
    )


def _describe_allgather_support(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> SupportStatus:
    """Describe allgather support conservatively."""
    if not has_heap:
        return SupportStatus(
            "partial",
            "High-level plan API exists, but execution requires an attached symmetric heap.",
        )
    if heap_mode == "single_process":
        return SupportStatus(
            "supported",
            "Host launcher + high-level op are validated on single-process peer-access heaps.",
        )
    if not multiprocess_device_remote_access_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        return SupportStatus(
            "unsupported",
            multiprocess_device_remote_access_detail(
                transport_strategy=transport_strategy,
                operation="xtile.ops.allgather(...)",
                world_size=world_size,
            ),
        )
    return SupportStatus(
        "supported",
        "Current public multiprocess surface is validated for allgather: "
        "world_size=2 with transport_strategy='ctypes_ipc'.",
    )


def _describe_allreduce_support(
    *,
    has_heap: bool,
    heap_mode: str | None,
    transport_strategy: str | None,
    world_size: int,
) -> SupportStatus:
    """Describe allreduce support conservatively but precisely."""
    if not has_heap:
        return SupportStatus(
            "partial",
            "High-level plan API exists, but execution requires an attached symmetric heap.",
        )
    if heap_mode == "single_process":
        return SupportStatus(
            "supported",
            "Host launcher + high-level op are validated on single-process peer-access heaps.",
        )
    if not multiprocess_device_remote_access_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        return SupportStatus(
            "unsupported",
            multiprocess_device_remote_access_detail(
                transport_strategy=transport_strategy,
                operation="xtile.ops.allreduce(...)",
                world_size=world_size,
            ),
        )
    return SupportStatus(
        "supported",
        "Current public multiprocess surface is validated for allreduce: "
        "world_size=2 with transport_strategy='ctypes_ipc'.",
    )
