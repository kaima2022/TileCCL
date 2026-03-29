# SPDX-License-Identifier: Apache-2.0
"""Unit tests for explicit pattern execution contracts."""

from __future__ import annotations

import pytest
import torch

from tncc.patterns.contracts import resolve_pattern_execution


def test_resolve_full_layout_contract() -> None:
    A = torch.empty((256, 128))
    B = torch.empty((128, 512))
    C = torch.empty((256, 512))

    spec = resolve_pattern_execution(
        A,
        B,
        C,
        rank=0,
        world_size=2,
        full_N=512,
        b_layout="full",
        c_layout="full",
        storage_kind="local",
    )

    assert spec.full_N == 512
    assert spec.local_N == 512
    assert spec.output_layout == "full"
    assert spec.scatter_cols == 256
    assert spec.scatter_src_col_offset == 0
    assert spec.scatter_dst_leading_dim == 512


def test_resolve_shard_layout_contract() -> None:
    A = torch.empty((256, 128))
    B = torch.empty((128, 256))
    C = torch.empty((256, 256))

    spec = resolve_pattern_execution(
        A,
        B,
        C,
        rank=1,
        world_size=2,
        full_N=512,
        b_layout="shard",
        c_layout="shard",
    )

    assert spec.full_N == 512
    assert spec.local_N == 256
    assert spec.output_layout == "shard"
    assert spec.scatter_cols == 256
    assert spec.scatter_src_col_offset == 0
    assert spec.scatter_dst_leading_dim == 256


def test_ambiguous_multi_rank_contract_requires_explicit_metadata() -> None:
    A = torch.empty((256, 128))
    B = torch.empty((128, 256))
    C = torch.empty((256, 256))

    with pytest.raises(ValueError, match="Ambiguous multi-rank pattern contract"):
        resolve_pattern_execution(A, B, C, rank=0, world_size=2)


def test_mixed_layout_contract_is_rejected() -> None:
    A = torch.empty((256, 128))
    B = torch.empty((128, 512))
    C = torch.empty((256, 256))

    with pytest.raises(ValueError, match="Mixed multi-rank layouts are not implemented"):
        resolve_pattern_execution(
            A,
            B,
            C,
            rank=0,
            world_size=2,
            full_N=512,
            b_layout="full",
            c_layout="shard",
        )
