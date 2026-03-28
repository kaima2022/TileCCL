"""Execution contracts for overlap patterns.

This module centralises shape/layout validation so that pattern
implementations no longer infer logical semantics from raw tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TensorLayout = Literal["full", "shard"]

_VALID_LAYOUTS = {"full", "shard"}


@dataclass(frozen=True, slots=True)
class PatternTensorSpec:
    """Logical-vs-physical layout metadata for one tensor."""

    full_shape: tuple[int, int]
    local_shape: tuple[int, int]
    rank: int
    world_size: int
    layout_kind: TensorLayout
    storage_kind: str = "symmetric"

    @property
    def cols(self) -> int:
        """Return the physical local column count."""
        return self.local_shape[1]

    def to_dict(self) -> dict[str, object]:
        """Serialize the spec for logs / benchmark JSON."""
        return {
            "full_shape": list(self.full_shape),
            "local_shape": list(self.local_shape),
            "rank": self.rank,
            "world_size": self.world_size,
            "layout_kind": self.layout_kind,
            "storage_kind": self.storage_kind,
        }


@dataclass(frozen=True, slots=True)
class PatternExecutionSpec:
    """Canonical execution contract consumed by all overlap patterns."""

    M: int
    K: int
    full_N: int
    local_N: int
    rank: int
    world_size: int
    rhs: PatternTensorSpec
    output: PatternTensorSpec
    scatter_src_col_offset: int
    scatter_cols: int
    scatter_dst_leading_dim: int
    scatter_dst_col_offset: int

    @property
    def rhs_layout(self) -> TensorLayout:
        """Return the logical layout kind of ``B``."""
        return self.rhs.layout_kind

    @property
    def output_layout(self) -> TensorLayout:
        """Return the logical layout kind of ``C``."""
        return self.output.layout_kind

    def to_dict(self) -> dict[str, object]:
        """Serialize the spec for logs / benchmark JSON."""
        return {
            "M": self.M,
            "K": self.K,
            "full_N": self.full_N,
            "local_N": self.local_N,
            "rank": self.rank,
            "world_size": self.world_size,
            "rhs": self.rhs.to_dict(),
            "output": self.output.to_dict(),
            "scatter": {
                "src_col_offset": self.scatter_src_col_offset,
                "cols": self.scatter_cols,
                "dst_leading_dim": self.scatter_dst_leading_dim,
                "dst_col_offset": self.scatter_dst_col_offset,
            },
        }


def resolve_pattern_execution(
    A,
    B,
    C,
    *,
    rank: int,
    world_size: int,
    full_N: int | None = None,
    b_layout: TensorLayout | None = None,
    c_layout: TensorLayout | None = None,
    storage_kind: str = "symmetric",
) -> PatternExecutionSpec:
    """Resolve a canonical pattern execution contract.

    Supported multi-rank contracts today:
    - ``B(K, N)``, ``C(M, N)`` with explicit ``b_layout="full"``,
      ``c_layout="full"``
    - ``B(K, N_per_rank)``, ``C(M, N_per_rank)`` with explicit
      ``full_N`` and ``*_layout="shard"``

    Mixed ``full``/``shard`` input-output contracts are intentionally
    rejected until there is an explicit host-side wrapper for them.
    """
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError(
            "Pattern tensors must all be 2-D: "
            f"got A.ndim={A.ndim}, B.ndim={B.ndim}, C.ndim={C.ndim}"
        )
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank={rank} out of range for world_size={world_size}")

    M, K = (int(A.shape[0]), int(A.shape[1]))
    b_k, b_cols = (int(B.shape[0]), int(B.shape[1]))
    c_m, c_cols = (int(C.shape[0]), int(C.shape[1]))
    if b_k != K:
        raise ValueError(f"B.shape[0] must equal A.shape[1]: got {b_k} vs {K}")
    if c_m != M:
        raise ValueError(f"C.shape[0] must equal A.shape[0]: got {c_m} vs {M}")

    normalized_b_layout = _normalize_layout(b_layout, "b_layout")
    normalized_c_layout = _normalize_layout(c_layout, "c_layout")
    resolved_full_N = _resolve_full_N(
        b_cols=b_cols,
        c_cols=c_cols,
        world_size=world_size,
        full_N=full_N,
        b_layout=normalized_b_layout,
        c_layout=normalized_c_layout,
    )
    if resolved_full_N <= 0:
        raise ValueError(f"full_N must be positive, got {resolved_full_N}")
    if resolved_full_N % world_size != 0:
        raise ValueError(
            f"full_N must be divisible by world_size: {resolved_full_N} % {world_size} != 0"
        )

    shard_N = resolved_full_N if world_size == 1 else resolved_full_N // world_size
    resolved_b_layout = _resolve_layout(
        cols=b_cols,
        full_N=resolved_full_N,
        shard_N=shard_N,
        explicit=normalized_b_layout,
        name="B",
        world_size=world_size,
    )
    resolved_c_layout = _resolve_layout(
        cols=c_cols,
        full_N=resolved_full_N,
        shard_N=shard_N,
        explicit=normalized_c_layout,
        name="C",
        world_size=world_size,
    )

    if world_size > 1 and resolved_b_layout != resolved_c_layout:
        raise ValueError(
            "Mixed multi-rank layouts are not implemented yet: "
            f"B uses {resolved_b_layout!r}, C uses {resolved_c_layout!r}. "
            "Use matching full/full or shard/shard tensors, or call a "
            "higher-level host wrapper that materializes the intermediate layout."
        )

    expected_local_N = resolved_full_N if resolved_b_layout == "full" else shard_N
    if b_cols != expected_local_N:
        raise ValueError(
            f"B shape/layout mismatch: layout={resolved_b_layout!r} expects "
            f"{expected_local_N} columns, got {b_cols}"
        )
    if c_cols != expected_local_N:
        raise ValueError(
            f"C shape/layout mismatch: layout={resolved_c_layout!r} expects "
            f"{expected_local_N} columns, got {c_cols}"
        )

    rhs_spec = PatternTensorSpec(
        full_shape=(K, resolved_full_N),
        local_shape=(K, b_cols),
        rank=rank,
        world_size=world_size,
        layout_kind=resolved_b_layout,
        storage_kind=storage_kind,
    )
    output_spec = PatternTensorSpec(
        full_shape=(M, resolved_full_N),
        local_shape=(M, c_cols),
        rank=rank,
        world_size=world_size,
        layout_kind=resolved_c_layout,
        storage_kind=storage_kind,
    )

    if resolved_c_layout == "full":
        scatter_cols = resolved_full_N if world_size == 1 else shard_N
        scatter_src_col_offset = 0 if world_size == 1 else rank * shard_N
        scatter_dst_col_offset = 0 if world_size == 1 else rank * shard_N
        scatter_dst_leading_dim = resolved_full_N
    else:
        scatter_cols = c_cols
        scatter_src_col_offset = 0
        scatter_dst_col_offset = 0
        scatter_dst_leading_dim = c_cols

    return PatternExecutionSpec(
        M=M,
        K=K,
        full_N=resolved_full_N,
        local_N=b_cols,
        rank=rank,
        world_size=world_size,
        rhs=rhs_spec,
        output=output_spec,
        scatter_src_col_offset=scatter_src_col_offset,
        scatter_cols=scatter_cols,
        scatter_dst_leading_dim=scatter_dst_leading_dim,
        scatter_dst_col_offset=scatter_dst_col_offset,
    )


def _normalize_layout(layout: str | None, name: str) -> TensorLayout | None:
    if layout is None:
        return None
    if layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"{name} must be one of {sorted(_VALID_LAYOUTS)}, got {layout!r}"
        )
    return layout  # type: ignore[return-value]


def _full_N_from_layout(cols: int, layout: TensorLayout, world_size: int) -> int:
    if layout == "full":
        return cols
    return cols if world_size == 1 else cols * world_size


def _resolve_full_N(
    *,
    b_cols: int,
    c_cols: int,
    world_size: int,
    full_N: int | None,
    b_layout: TensorLayout | None,
    c_layout: TensorLayout | None,
) -> int:
    if full_N is not None:
        return int(full_N)

    candidates: set[int] = set()
    if b_layout is not None:
        candidates.add(_full_N_from_layout(b_cols, b_layout, world_size))
    if c_layout is not None:
        candidates.add(_full_N_from_layout(c_cols, c_layout, world_size))

    if len(candidates) > 1:
        raise ValueError(
            "Provided layout hints disagree on full_N: "
            f"resolved candidates={sorted(candidates)}"
        )
    if len(candidates) == 1:
        return next(iter(candidates))

    if world_size == 1:
        return max(b_cols, c_cols)

    if b_cols != c_cols:
        hi = max(b_cols, c_cols)
        lo = min(b_cols, c_cols)
        if hi == lo * world_size:
            return hi

    raise ValueError(
        "Ambiguous multi-rank pattern contract. Pass full_N together with "
        "explicit b_layout/c_layout, or use tncc.ops.gemm_allscatter(...)."
    )


def _resolve_layout(
    *,
    cols: int,
    full_N: int,
    shard_N: int,
    explicit: TensorLayout | None,
    name: str,
    world_size: int,
) -> TensorLayout:
    if explicit is not None:
        expected = full_N if explicit == "full" else shard_N
        if cols != expected:
            raise ValueError(
                f"{name} layout={explicit!r} expects {expected} columns, got {cols}"
            )
        return explicit

    if world_size == 1:
        if cols != full_N:
            raise ValueError(f"{name} expects {full_N} columns, got {cols}")
        return "full"

    matches: list[TensorLayout] = []
    if cols == full_N:
        matches.append("full")
    if cols == shard_N:
        matches.append("shard")

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Cannot infer {name} layout from cols={cols}, full_N={full_N}, shard_N={shard_N}"
        )
    raise ValueError(
        f"Ambiguous {name} layout for cols={cols}. Pass explicit {name.lower()}_layout."
    )
