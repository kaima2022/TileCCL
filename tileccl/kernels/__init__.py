# SPDX-License-Identifier: Apache-2.0
"""tileccl.kernels - Triton kernel implementations.

This sub-package contains JIT-compiled Triton kernels used by the
communication-overlap patterns.  The kernels are written in pure Triton
so that the compiler has full visibility for fusion and scheduling.
"""

from tileccl.kernels.gemm import gemm, gemm_kernel

__all__ = [
    "gemm_kernel",
    "gemm",
]
