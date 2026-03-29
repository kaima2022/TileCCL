# SPDX-License-Identifier: Apache-2.0
"""Tests for tncc.memory.translation.PointerTranslator.

Host-side pointer translation tests that do NOT require a GPU.
Uses synthetic heap_bases tensors to verify translation arithmetic,
bounds checking, offset computation, and round-trip consistency.
"""

from __future__ import annotations

import pytest
import torch

from tncc.memory.translation import PointerTranslator


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

# Three-rank setup with known base addresses spaced 4096 bytes apart.
_HEAP_SIZE = 4096
# Bases spaced far enough apart that heaps don't overlap.
# Rank 0 = 0x10000 (65536), Rank 1 = 0x20000 (131072), Rank 2 = 0x30000 (196608)
_BASES = torch.tensor([0x10000, 0x20000, 0x30000], dtype=torch.int64)
_B0 = 0x10000
_B1 = 0x20000
_B2 = 0x30000


@pytest.fixture
def translator() -> PointerTranslator:
    """Return a PointerTranslator for rank 0 with three ranks."""
    return PointerTranslator(
        heap_bases=_BASES.clone(),
        heap_size=_HEAP_SIZE,
        local_rank=0,
    )


@pytest.fixture
def translator_rank1() -> PointerTranslator:
    """Return a PointerTranslator for rank 1 with three ranks."""
    return PointerTranslator(
        heap_bases=_BASES.clone(),
        heap_size=_HEAP_SIZE,
        local_rank=1,
    )


# ---------------------------------------------------------------------------
# TestPointerTranslator -- host-side tests (no GPU required)
# ---------------------------------------------------------------------------


class TestPointerTranslator:
    """Host-side PointerTranslator tests (no GPU required)."""

    # ---- basic properties ------------------------------------------------

    def test_world_size(self, translator: PointerTranslator) -> None:
        """world_size reflects the number of ranks."""
        assert translator.world_size == 3

    def test_heap_size(self, translator: PointerTranslator) -> None:
        """heap_size returns the configured value."""
        assert translator.heap_size == _HEAP_SIZE

    def test_local_rank(self, translator: PointerTranslator) -> None:
        """local_rank returns the configured rank."""
        assert translator.local_rank == 0

    def test_base(self, translator: PointerTranslator) -> None:
        """base() returns the correct address for each rank."""
        assert translator.base(0) == _B0
        assert translator.base(1) == _B1
        assert translator.base(2) == _B2

    # ---- translate -------------------------------------------------------

    def test_translate_same_rank(self, translator: PointerTranslator) -> None:
        """Translation to the same rank returns the original pointer."""
        ptr = _B0 + 512  # 512 bytes into rank 0's heap
        result = translator.translate(ptr, from_rank=0, to_rank=0)
        assert result == ptr

    def test_translate_different_rank(self, translator: PointerTranslator) -> None:
        """Translation produces correct offset-based pointer."""
        ptr = _B0 + 256
        result = translator.translate(ptr, from_rank=0, to_rank=1)
        assert result == _B1 + 256

    def test_translate_all_pairs(self, translator: PointerTranslator) -> None:
        """Translation works correctly for every (from, to) pair."""
        offset = 100
        for from_rank in range(3):
            ptr = int(_BASES[from_rank].item()) + offset
            for to_rank in range(3):
                result = translator.translate(ptr, from_rank, to_rank)
                expected = int(_BASES[to_rank].item()) + offset
                assert result == expected, (
                    f"translate({ptr}, {from_rank}, {to_rank}) = {result}, "
                    f"expected {expected}"
                )

    def test_translate_roundtrip(self, translator: PointerTranslator) -> None:
        """Translating A->B then B->A returns the original pointer."""
        original_ptr = _B0 + 768  # within rank 0's heap
        # Translate rank 0 -> rank 2
        remote_ptr = translator.translate(original_ptr, from_rank=0, to_rank=2)
        assert remote_ptr == _B2 + 768
        # Translate rank 2 -> rank 0
        recovered_ptr = translator.translate(remote_ptr, from_rank=2, to_rank=0)
        assert recovered_ptr == original_ptr

    def test_translate_roundtrip_all_pairs(self, translator: PointerTranslator) -> None:
        """Round-trip translation preserves the original pointer for all rank pairs."""
        offset = 42
        for from_rank in range(3):
            original = int(_BASES[from_rank].item()) + offset
            for to_rank in range(3):
                remote = translator.translate(original, from_rank, to_rank)
                back = translator.translate(remote, to_rank, from_rank)
                assert back == original

    def test_translate_at_boundary_start(self, translator: PointerTranslator) -> None:
        """Pointer at the very start of a heap translates correctly."""
        result = translator.translate(_B0, from_rank=0, to_rank=1)
        assert result == _B1

    def test_translate_at_boundary_end(self, translator: PointerTranslator) -> None:
        """Pointer at (base + heap_size - 1) translates correctly."""
        ptr = _B0 + _HEAP_SIZE - 1
        result = translator.translate(ptr, from_rank=0, to_rank=2)
        assert result == _B2 + _HEAP_SIZE - 1

    # ---- validate --------------------------------------------------------

    def test_validate_in_bounds(self, translator: PointerTranslator) -> None:
        """Pointer within heap validates True."""
        assert translator.validate(_B0, rank=0) is True
        assert translator.validate(_B0 + 2048, rank=0) is True
        assert translator.validate(_B0 + _HEAP_SIZE - 1, rank=0) is True

    def test_validate_out_of_bounds(self, translator: PointerTranslator) -> None:
        """Pointer outside heap validates False."""
        # Below the base
        assert translator.validate(_B0 - 1, rank=0) is False
        # At base + heap_size (one past the end)
        assert translator.validate(_B0 + _HEAP_SIZE, rank=0) is False
        # Far outside
        assert translator.validate(0xFFFFF, rank=0) is False

    def test_validate_default_rank(self, translator: PointerTranslator) -> None:
        """validate() without explicit rank uses local_rank."""
        # translator.local_rank == 0, base == _B0
        assert translator.validate(_B0 + 500) is True
        # Outside rank 0's heap (beyond _B0 + 4096)
        assert translator.validate(_B0 + _HEAP_SIZE + 100) is False

    def test_validate_wrong_rank(self, translator: PointerTranslator) -> None:
        """A pointer valid in rank 0 may be invalid in rank 2."""
        ptr = _B0 + 100  # valid for rank 0
        assert translator.validate(ptr, rank=0) is True
        # This ptr is NOT in rank 2's range [_B2, _B2 + 4096)
        assert translator.validate(ptr, rank=2) is False

    def test_validate_invalid_rank(self, translator: PointerTranslator) -> None:
        """validate() with an out-of-range rank returns False (not exception)."""
        assert translator.validate(_B0 + 500, rank=99) is False
        assert translator.validate(_B0 + 500, rank=-1) is False

    # ---- get_offset ------------------------------------------------------

    def test_get_offset(self, translator: PointerTranslator) -> None:
        """Offset computation is correct."""
        assert translator.get_offset(_B0, rank=0) == 0
        assert translator.get_offset(_B0 + 256, rank=0) == 256
        assert translator.get_offset(_B1 + 48, rank=1) == 48
        assert translator.get_offset(_B2 + 1000, rank=2) == 1000

    def test_get_offset_default_rank(self, translator: PointerTranslator) -> None:
        """get_offset() without rank uses local_rank."""
        assert translator.get_offset(_B0 + 512) == 512

    def test_get_offset_out_of_bounds_raises(self, translator: PointerTranslator) -> None:
        """Offset computation raises ValueError for out-of-bounds pointer."""
        with pytest.raises(ValueError, match="outside"):
            translator.get_offset(_B0 - 1, rank=0)
        with pytest.raises(ValueError, match="outside"):
            translator.get_offset(_B0 + _HEAP_SIZE, rank=0)

    # ---- error handling --------------------------------------------------

    def test_invalid_rank_raises(self, translator: PointerTranslator) -> None:
        """Invalid rank raises ValueError."""
        with pytest.raises(ValueError, match="rank="):
            translator.translate(_B0 + 500, from_rank=0, to_rank=5)
        with pytest.raises(ValueError, match="rank="):
            translator.translate(_B0 + 500, from_rank=-1, to_rank=0)
        with pytest.raises(ValueError, match="rank="):
            translator.translate(_B0 + 500, from_rank=99, to_rank=0)

    def test_out_of_bounds_translate_raises(self, translator: PointerTranslator) -> None:
        """Out-of-bounds pointer raises ValueError during translation."""
        # Below rank 0's base
        with pytest.raises(ValueError, match="outside"):
            translator.translate(_B0 - 500, from_rank=0, to_rank=1)
        # Above rank 0's range
        with pytest.raises(ValueError, match="outside"):
            translator.translate(_B0 + _HEAP_SIZE + 100, from_rank=0, to_rank=1)

    # ---- constructor validation ------------------------------------------

    def test_constructor_rejects_2d_bases(self) -> None:
        """heap_bases must be 1-D."""
        bases_2d = torch.tensor([[_B0, _B1]], dtype=torch.int64)
        with pytest.raises(ValueError, match="1-D"):
            PointerTranslator(bases_2d, heap_size=4096, local_rank=0)

    def test_constructor_rejects_wrong_dtype(self) -> None:
        """heap_bases must be int64."""
        bases_f32 = torch.tensor([_B0, _B1, _B2], dtype=torch.float32)
        with pytest.raises(ValueError, match="int64"):
            PointerTranslator(bases_f32, heap_size=4096, local_rank=0)

    # ---- repr ------------------------------------------------------------

    def test_repr(self, translator: PointerTranslator) -> None:
        """__repr__ contains essential info."""
        r = repr(translator)
        assert "PointerTranslator" in r
        assert "world_size=3" in r
        assert "heap_size=4096" in r
        assert "local_rank=0" in r

    # ---- cross-rank translator -------------------------------------------

    def test_different_local_rank(self, translator_rank1: PointerTranslator) -> None:
        """A translator created for rank 1 works correctly."""
        t = translator_rank1
        assert t.local_rank == 1
        assert t.base(1) == _B1
        # default get_offset uses local_rank=1
        assert t.get_offset(_B1 + 100) == 100
        # validate without rank uses local_rank=1
        assert t.validate(_B1 + 500) is True
        assert t.validate(_B0 + 500) is False
