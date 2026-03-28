"""
tncc.memory - Symmetric heap and pointer translation.

This package provides the memory management foundation for TNCC's multi-GPU
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
:mod:`tncc.memory.translation` and should be imported directly::

    from tncc.memory.translation import translate_ptr, remote_load, remote_store
"""

from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.memory.symmetric_heap import PeerMemoryMapEntry
from tncc.memory.translation import PointerTranslator
from tncc.memory.allocators import (
    BaseSymmetricAllocator,
    ImportedPeerMemory,
    MemorySegmentDescriptor,
    PeerMemoryExportDescriptor,
    TorchBumpAllocator,
)

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
