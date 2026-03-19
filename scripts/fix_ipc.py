#!/usr/bin/env python3
"""XTile — CUDA IPC cross-process fix and verification.

Uses torch.multiprocessing.spawn for proper GPU process management.
Tests both raw ctypes IPC and PyTorch-level IPC.

Run: python scripts/fix_ipc.py
"""

import ctypes
import ctypes.util
import os
import sys
import tempfile
import time

import torch
import torch.multiprocessing as mp

CUDA_IPC_HANDLE_SIZE = 64


class CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE)]


def _load_cudart():
    for p in ["/usr/local/cuda/lib64/libcudart.so", "libcudart.so"]:
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    resolved = ctypes.util.find_library("cudart")
    if resolved:
        return ctypes.CDLL(resolved)
    raise RuntimeError("libcudart.so not found")


# =========================================================================
# Test 1: ctypes IPC via file-based handle exchange
# =========================================================================

def _worker_ctypes(rank, handle_file, result_file):
    """Worker for ctypes IPC test. rank=0 produces, rank=1 consumes."""
    torch.cuda.set_device(rank)
    _ = torch.empty(1, device=f"cuda:{rank}")  # init context

    lib = _load_cudart()
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(CudaIpcMemHandle), ctypes.c_void_p]
    lib.cudaIpcGetMemHandle.restype = ctypes.c_int
    lib.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), CudaIpcMemHandle, ctypes.c_uint
    ]
    lib.cudaIpcOpenMemHandle.restype = ctypes.c_int
    lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

    if rank == 0:
        # Producer: allocate, write pattern, export handle
        ptr = ctypes.c_void_p()
        err = lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4096))
        assert err == 0, f"cudaMalloc: {err}"

        handle = CudaIpcMemHandle()
        err = lib.cudaIpcGetMemHandle(ctypes.byref(handle), ptr)
        if err != 0:
            with open(result_file, "w") as f:
                f.write(f"FAIL:cudaIpcGetMemHandle error {err}")
            return

        with open(handle_file, "wb") as f:
            f.write(bytes(handle.reserved))

        # Wait for consumer
        for _ in range(100):
            time.sleep(0.1)
            if os.path.exists(result_file):
                break

        lib.cudaFree(ptr)

    else:
        # Consumer: wait for handle, open it
        for _ in range(100):
            time.sleep(0.1)
            if os.path.exists(handle_file) and os.path.getsize(handle_file) == CUDA_IPC_HANDLE_SIZE:
                break
        else:
            with open(result_file, "w") as f:
                f.write("FAIL:timeout waiting for handle")
            return

        with open(handle_file, "rb") as f:
            handle_bytes = f.read()

        h = CudaIpcMemHandle()
        ctypes.memmove(h.reserved, handle_bytes, CUDA_IPC_HANDLE_SIZE)

        opened_ptr = ctypes.c_void_p()
        err = lib.cudaIpcOpenMemHandle(ctypes.byref(opened_ptr), h, ctypes.c_uint(1))

        with open(result_file, "w") as f:
            if err == 0:
                f.write(f"PASS:0x{opened_ptr.value:016x}")
                lib.cudaIpcCloseMemHandle(opened_ptr)
            else:
                f.write(f"FAIL:cudaIpcOpenMemHandle error {err}")


def test_ctypes_ipc():
    print("\n--- Test 1: Cross-process ctypes IPC (Structure by-value) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        handle_file = os.path.join(tmpdir, "handle.bin")
        result_file = os.path.join(tmpdir, "result.txt")

        try:
            mp.spawn(_worker_ctypes, args=(handle_file, result_file),
                     nprocs=2, join=True)
        except Exception as e:
            print(f"  spawn error: {e}")
            return False

        if os.path.exists(result_file):
            with open(result_file) as f:
                result = f.read()
            print(f"  Result: {result}")
            return result.startswith("PASS")

        print("  No result file produced")
        return False


# =========================================================================
# Test 2: PyTorch tensor IPC via mp.spawn
# =========================================================================

def _worker_pytorch_ipc(rank, handle_file, result_file):
    """Worker for PyTorch IPC test."""
    torch.cuda.set_device(rank)

    if rank == 0:
        # Producer
        t = torch.arange(64, dtype=torch.float32, device="cuda:0")
        storage = t.untyped_storage()
        try:
            share_info = storage._share_cuda_()
            import pickle
            with open(handle_file, "wb") as f:
                pickle.dump((share_info, t.shape, str(t.dtype), t.storage_offset()), f)
        except Exception as e:
            with open(result_file, "w") as f:
                f.write(f"FAIL:producer {e}")
            return

        for _ in range(100):
            time.sleep(0.1)
            if os.path.exists(result_file):
                break

    else:
        # Consumer
        for _ in range(100):
            time.sleep(0.1)
            if os.path.exists(handle_file) and os.path.getsize(handle_file) > 0:
                break
        else:
            with open(result_file, "w") as f:
                f.write("FAIL:timeout waiting for handle")
            return

        try:
            import pickle
            with open(handle_file, "rb") as f:
                share_info, shape, dtype_str, offset = pickle.load(f)

            storage = torch.UntypedStorage._new_shared_cuda(*share_info)
            dtype = getattr(torch, dtype_str.split(".")[-1])
            t = torch.empty(0, dtype=dtype, device="cuda:0").set_(storage).reshape(shape)
            expected = torch.arange(64, dtype=torch.float32)
            match = torch.allclose(t.cpu(), expected)
            with open(result_file, "w") as f:
                f.write(f"{'PASS' if match else 'MISMATCH'}:shape={t.shape}")
        except Exception as e:
            with open(result_file, "w") as f:
                f.write(f"FAIL:consumer {e}")


def test_pytorch_ipc():
    print("\n--- Test 2: PyTorch _share_cuda_ cross-process ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        handle_file = os.path.join(tmpdir, "pytorch_handle.pkl")
        result_file = os.path.join(tmpdir, "result.txt")

        try:
            mp.spawn(_worker_pytorch_ipc, args=(handle_file, result_file),
                     nprocs=2, join=True)
        except Exception as e:
            print(f"  spawn error: {e}")
            # Check if result was written before the error
            if os.path.exists(result_file):
                with open(result_file) as f:
                    result = f.read()
                print(f"  Result (before crash): {result}")
                return result.startswith("PASS")
            return False

        if os.path.exists(result_file):
            with open(result_file) as f:
                result = f.read()
            print(f"  Result: {result}")
            return result.startswith("PASS")

        print("  No result file produced")
        return False


# =========================================================================
# Test 3: Direct cudaMemcpy peer test (non-IPC baseline)
# =========================================================================

def test_peer_access_baseline():
    print("\n--- Test 3: Peer access baseline (non-IPC) ---")
    if torch.cuda.device_count() < 2:
        print("  Need >= 2 GPUs")
        return False

    torch.cuda.set_device(0)
    t0 = torch.arange(100, dtype=torch.float32, device="cuda:0")

    # Direct copy via cudaMemcpyPeer (not IPC)
    t1 = t0.to("cuda:1")
    match = torch.allclose(t0.cpu(), t1.cpu())
    print(f"  cudaMemcpyPeer: {'PASS' if match else 'FAIL'}")

    # Test peer access pointer arithmetic
    can_access = torch.cuda.can_device_access_peer(0, 1)
    print(f"  Peer access 0→1: {can_access}")

    return match and can_access


# =========================================================================
# Main
# =========================================================================

def main():
    if torch.cuda.device_count() < 2:
        print("ERROR: Need >= 2 GPUs")
        return 1

    print("=" * 60)
    print("  XTile CUDA IPC Cross-Process Fix")
    print("=" * 60)
    print(f"  GPUs: {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            ptrace = f.read().strip()
        print(f"  ptrace_scope: {ptrace}")
    except FileNotFoundError:
        ptrace = "N/A"
        print(f"  ptrace_scope: N/A")

    results = {}
    results["peer_access"] = test_peer_access_baseline()
    results["ctypes_ipc"] = test_ctypes_ipc()
    results["pytorch_ipc"] = test_pytorch_ipc()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL'}")

    print("\n  ANALYSIS:")
    if results["ctypes_ipc"]:
        print("  Structure IPC fix WORKS cross-process.")
    elif results["peer_access"] and not results["ctypes_ipc"]:
        print("  Peer access works but CUDA IPC blocked.")
        print(f"  ptrace_scope={ptrace} is the likely root cause.")
        print()
        print("  For single-node: create_all() (peer access) is the correct path.")
        print("  For multi-node:  CUDA IPC requires ptrace_scope=0 or")
        print("                   use cuMem API (CUDA 11.2+ virtual memory).")
        print()
        print("  To unblock IPC:")
        print("    sudo sysctl kernel.yama.ptrace_scope=0")
        print("    (or add to /etc/sysctl.d/99-ptrace.conf)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
