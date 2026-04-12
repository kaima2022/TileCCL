# SPDX-License-Identifier: Apache-2.0
"""
tileccl.memory - Symmetric heap and pointer translation.

This package provides the memory management foundation for TileCCL's multi-GPU
communication layer.  Every GPU allocates an identically-sized heap, exchanges
IPC handles, and can translate local pointers to remote pointers for zero-copy
access.

Public API
----------
SymmetricHeap
    Host-side symmetric heap manager -- allocate, translate, barrier, cleanup.
PointerTranslator
    Host-side pointer translation helper (wraps ``heap_bases``).

Device-side ``@triton.jit`` translation primitives live in
:mod:`tileccl.memory.translation` and should be imported directly::

    from tileccl.memory.translation import translate_ptr, remote_load, remote_store
"""

from tileccl.memory.allocators import (
    BaseSymmetricAllocator,
    ImportedPeerMemory,
    MemorySegmentDescriptor,
    PeerMemoryExportDescriptor,
    TorchBumpAllocator,
)
from tileccl.memory.symmetric_heap import PeerMemoryMapEntry, SymmetricHeap
from tileccl.memory.translation import PointerTranslator

__all__ = [
    "BaseSymmetricAllocator",
    "ImportedPeerMemory",
    "MemorySegmentDescriptor",
    "PeerMemoryExportDescriptor",
    "PeerMemoryMapEntry",
    "SymmetricHeap",
    "PointerTranslator",
    "TorchBumpAllocator",
]
