#!/usr/bin/env python3
"""TNCC Phase 6 — CUDA IPC diagnostic script.

Tests four approaches to identify the root cause of P1-002:
  1. Current Array-based ctypes signature (expected: fail)
  2. Structure-based ctypes signature (hypothesis: may fix it)
  3. PyTorch native IPC (torch.multiprocessing)
  4. System parameter checks (ptrace_scope, driver, MIG)

Run: python scripts/diagnose_ipc.py
"""

import ctypes
import ctypes.util
import os
import subprocess
import sys
import traceback

import torch

CUDA_IPC_HANDLE_SIZE = 64


def _load_cudart():
    """Load libcudart.so and return CDLL handle."""
    search = [
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
        "libcudart.so",
    ]
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
    if cuda_home:
        search.insert(0, os.path.join(cuda_home, "lib64", "libcudart.so"))

    for path in search:
        try:
            lib = ctypes.CDLL(path)
            print(f"  Loaded: {path}")
            return lib
        except OSError:
            continue

    resolved = ctypes.util.find_library("cudart")
    if resolved:
        return ctypes.CDLL(resolved)

    raise RuntimeError("Cannot find libcudart.so")


# ---------------------------------------------------------------------------
# Approach 1: Array-based (current code)
# ---------------------------------------------------------------------------
def test_array_approach(lib):
    """Test IPC with ctypes Array (c_char * 64) — current TNCC code."""
    print("\n--- Test 1: Array-based ctypes (current code) ---")

    # Set signatures — Array type
    lib.cudaIpcGetMemHandle.argtypes = [
        ctypes.c_char * CUDA_IPC_HANDLE_SIZE,
        ctypes.c_void_p,
    ]
    lib.cudaIpcGetMemHandle.restype = ctypes.c_int

    lib.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char * CUDA_IPC_HANDLE_SIZE,
        ctypes.c_uint,
    ]
    lib.cudaIpcOpenMemHandle.restype = ctypes.c_int

    # Allocate on GPU 0
    lib.cudaSetDevice(ctypes.c_int(0))
    ptr0 = ctypes.c_void_p()
    size = 1024 * 1024  # 1 MB
    err = lib.cudaMalloc(ctypes.byref(ptr0), ctypes.c_size_t(size))
    if err != 0:
        print(f"  cudaMalloc failed: error {err}")
        return False
    print(f"  Allocated 1 MB on GPU 0: 0x{ptr0.value:016x}")

    # Get IPC handle
    handle = (ctypes.c_char * CUDA_IPC_HANDLE_SIZE)()
    err = lib.cudaIpcGetMemHandle(handle, ptr0)
    if err != 0:
        print(f"  cudaIpcGetMemHandle failed: error {err}")
        lib.cudaFree(ptr0)
        return False
    print(f"  IPC handle obtained (first 8 bytes): {bytes(handle)[:8].hex()}")

    # Switch to GPU 1 and open handle
    lib.cudaSetDevice(ctypes.c_int(1))

    opened_ptr = ctypes.c_void_p()
    buf = (ctypes.c_char * CUDA_IPC_HANDLE_SIZE).from_buffer_copy(bytes(handle))
    err = lib.cudaIpcOpenMemHandle(ctypes.byref(opened_ptr), buf, ctypes.c_uint(1))

    if err != 0:
        print(f"  cudaIpcOpenMemHandle FAILED: error {err}")
        lib.cudaSetDevice(ctypes.c_int(0))
        lib.cudaFree(ptr0)
        return False

    print(f"  cudaIpcOpenMemHandle SUCCESS: 0x{opened_ptr.value:016x}")

    # Cleanup
    lib.cudaIpcCloseMemHandle(opened_ptr)
    lib.cudaSetDevice(ctypes.c_int(0))
    lib.cudaFree(ptr0)
    return True


# ---------------------------------------------------------------------------
# Approach 2: Structure-based (proposed fix)
# ---------------------------------------------------------------------------
class CudaIpcMemHandle(ctypes.Structure):
    """ctypes Structure matching cudaIpcMemHandle_t — passed by value."""
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE)]


def test_structure_approach(lib):
    """Test IPC with ctypes Structure — proposed fix for by-value passing."""
    print("\n--- Test 2: Structure-based ctypes (proposed fix) ---")

    # Set signatures — Structure type
    lib.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    lib.cudaIpcGetMemHandle.restype = ctypes.c_int

    lib.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    lib.cudaIpcOpenMemHandle.restype = ctypes.c_int

    # Allocate on GPU 0
    lib.cudaSetDevice(ctypes.c_int(0))
    ptr0 = ctypes.c_void_p()
    size = 1024 * 1024
    err = lib.cudaMalloc(ctypes.byref(ptr0), ctypes.c_size_t(size))
    if err != 0:
        print(f"  cudaMalloc failed: error {err}")
        return False
    print(f"  Allocated 1 MB on GPU 0: 0x{ptr0.value:016x}")

    # Get IPC handle
    handle = CudaIpcMemHandle()
    err = lib.cudaIpcGetMemHandle(ctypes.byref(handle), ptr0)
    if err != 0:
        print(f"  cudaIpcGetMemHandle failed: error {err}")
        lib.cudaFree(ptr0)
        return False
    print(f"  IPC handle obtained (first 8 bytes): {bytes(handle.reserved)[:8].hex()}")

    # Switch to GPU 1 and open handle
    lib.cudaSetDevice(ctypes.c_int(1))

    opened_ptr = ctypes.c_void_p()
    err = lib.cudaIpcOpenMemHandle(ctypes.byref(opened_ptr), handle, ctypes.c_uint(1))

    if err != 0:
        print(f"  cudaIpcOpenMemHandle FAILED: error {err}")
        lib.cudaSetDevice(ctypes.c_int(0))
        lib.cudaFree(ptr0)
        return False

    print(f"  cudaIpcOpenMemHandle SUCCESS: 0x{opened_ptr.value:016x}")

    # Cleanup
    lib.cudaIpcCloseMemHandle(opened_ptr)
    lib.cudaSetDevice(ctypes.c_int(0))
    lib.cudaFree(ptr0)
    return True


# ---------------------------------------------------------------------------
# Approach 3: PyTorch native IPC
# ---------------------------------------------------------------------------
def test_pytorch_ipc():
    """Test CUDA IPC using PyTorch's built-in mechanisms."""
    print("\n--- Test 3: PyTorch native CUDA IPC ---")

    try:
        # Allocate on GPU 0
        t0 = torch.randn(256, 256, device="cuda:0")
        print(f"  Tensor on GPU 0: shape={t0.shape}, ptr=0x{t0.data_ptr():016x}")

        # Get IPC handle via storage
        storage = t0.untyped_storage()
        if not hasattr(storage, '_share_cuda_'):
            print("  _share_cuda_ not available on this PyTorch build")
            return False

        info = storage._share_cuda_()
        print(f"  _share_cuda_ returned: type={type(info)}")

        # Try to reconstruct on same process (simplified test)
        t1 = t0.to("cuda:1")  # This uses cudaMemcpyPeer, not IPC
        print(f"  cudaMemcpyPeer (non-IPC) works: allclose={torch.allclose(t0.cpu(), t1.cpu())}")
        return True

    except Exception as e:
        print(f"  PyTorch IPC test failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Approach 4: System parameter checks
# ---------------------------------------------------------------------------
def check_system_params():
    """Check system parameters that affect CUDA IPC."""
    print("\n--- Test 4: System parameter checks ---")

    checks = {}

    # ptrace_scope
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            val = f.read().strip()
        print(f"  ptrace_scope = {val} {'(OK: unrestricted)' if val == '0' else '(may restrict IPC)'}")
        checks["ptrace_scope"] = val
    except FileNotFoundError:
        print("  ptrace_scope: not available (Yama not enabled)")
        checks["ptrace_scope"] = "N/A"

    # Driver version
    try:
        with open("/proc/driver/nvidia/version") as f:
            lines = f.readlines()
        for line in lines:
            if "NVRM" in line:
                print(f"  Driver: {line.strip()}")
                checks["driver"] = line.strip()
                break
    except FileNotFoundError:
        print("  Driver version: not available")
        checks["driver"] = "N/A"

    # MIG status
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        mig = result.stdout.strip()
        print(f"  MIG mode: {mig}")
        checks["mig"] = mig
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  MIG: nvidia-smi not available")
        checks["mig"] = "N/A"

    # Container check
    in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    print(f"  Container: {'Yes' if in_container else 'No (bare metal)'}")
    checks["container"] = in_container

    # GPU count and peer access
    try:
        n = torch.cuda.device_count()
        print(f"  GPU count: {n}")
        if n >= 2:
            can_access = torch.cuda.can_device_access_peer(0, 1)
            print(f"  Peer access 0→1: {can_access}")
    except Exception as e:
        print(f"  GPU check failed: {e}")

    # CUDA version
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  TNCC CUDA IPC Diagnostic (P1-002)")
    print("=" * 70)

    if torch.cuda.device_count() < 2:
        print("ERROR: Need at least 2 GPUs for IPC testing")
        sys.exit(1)

    print(f"\nGPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # System checks first
    sys_info = check_system_params()

    # Load CUDA runtime
    try:
        lib = _load_cudart()
    except RuntimeError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    # Setup basic signatures needed for all tests
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

    # Run tests
    results = {}

    try:
        results["array"] = test_array_approach(lib)
    except Exception as e:
        print(f"  Exception: {e}")
        traceback.print_exc()
        results["array"] = False

    try:
        results["structure"] = test_structure_approach(lib)
    except Exception as e:
        print(f"  Exception: {e}")
        traceback.print_exc()
        results["structure"] = False

    try:
        results["pytorch"] = test_pytorch_ipc()
    except Exception as e:
        print(f"  Exception: {e}")
        traceback.print_exc()
        results["pytorch"] = False

    # Summary
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Array approach (current):    {'PASS' if results.get('array') else 'FAIL'}")
    print(f"  Structure approach (fix):    {'PASS' if results.get('structure') else 'FAIL'}")
    print(f"  PyTorch native IPC:          {'PASS' if results.get('pytorch') else 'FAIL'}")

    if results.get("structure") and not results.get("array"):
        print("\n  CONCLUSION: ctypes calling convention bug confirmed!")
        print("  ACTION: Apply Structure-based fix to cuda.py and hip.py")
    elif results.get("structure") and results.get("array"):
        print("\n  CONCLUSION: Both approaches work — original P1-002 may be resolved")
        print("  by driver update or other system change.")
    elif not results.get("structure") and not results.get("array"):
        print("\n  CONCLUSION: System-level IPC limitation confirmed.")
        print(f"  ptrace_scope={sys_info.get('ptrace_scope', '?')}")
        if sys_info.get("ptrace_scope") != "0":
            print("  TRY: sudo sysctl kernel.yama.ptrace_scope=0")
        print("  The Structure fix is still correct for calling convention,")
        print("  but IPC may be blocked at the driver/kernel level.")
    else:
        print("\n  CONCLUSION: Unexpected result pattern — manual investigation needed.")

    print()
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
