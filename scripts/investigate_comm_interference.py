#!/usr/bin/env python3
"""Investigate why communication benchmarks are affected without NVLink saturation.

This script runs a focused, controlled study:

- optional background workload on one participating GPU
- one pure P2P bandwidth probe
- one protocol-heavy comm-only collective probe

The goal is to separate:

1. memory residency only
2. local SM pressure
3. local HBM/LDST pressure

from raw inter-GPU link contention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from tests.benchmarks.bench_p2p_translate import benchmark_p2p
from tncc.memory.symmetric_heap import SymmetricHeap
from tncc.utils.benchmark_results import write_json


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "figures" / "data" / "comm_interference_study"
_BG_READY_SENTINEL = "BACKGROUND_READY\n"


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    mode: str


CONDITIONS = (
    Condition("none", "No extra local workload.", "none"),
    Condition("resident_only", "Allocate VRAM only, no active kernels.", "resident_only"),
    Condition("sm_burn", "Continuous GEMM on one participating GPU.", "sm_burn"),
    Condition("dram_burn", "Continuous local memory traffic on one participating GPU.", "dram_burn"),
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _spread_pct(values: list[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    if med == 0.0:
        return 0.0
    return float((max(values) - min(values)) / med * 100.0)


def _run_shell_json(command: list[str], *, timeout_seconds: int) -> tuple[dict[str, Any], dict[str, Any]]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    metadata = {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed with returncode={process.returncode}: {' '.join(command)}\n"
            f"stderr:\n{process.stderr}"
        )
    payload = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
    return payload, metadata


def _nvidia_smi(*args: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.strip()


def _run_p2p_probe() -> dict[str, Any]:
    heap_size = 256 * 1024 * 1024
    heaps = SymmetricHeap.create_all(size=heap_size, world_size=2)
    try:
        size_bytes = 128 * 1024 * 1024
        n_elements = size_bytes // torch.tensor([], dtype=torch.float32).element_size()
        read = benchmark_p2p(
            heaps,
            n_elements=n_elements,
            block_size=4096,
            dtype=torch.float32,
            direction="read",
            variant="evict_first",
            warmup=3,
            iters=20,
            num_sms=114,
        )
        write = benchmark_p2p(
            heaps,
            n_elements=n_elements,
            block_size=4096,
            dtype=torch.float32,
            direction="write",
            variant="wt+evict",
            warmup=3,
            iters=20,
            num_sms=114,
        )
        return {
            "size_bytes": size_bytes,
            "read_bandwidth_gbps": read.bandwidth_gbps,
            "read_time_us": read.time_us,
            "write_bandwidth_gbps": write.bandwidth_gbps,
            "write_time_us": write.time_us,
        }
    finally:
        for heap in heaps:
            heap.cleanup()


def _launch_background_worker(*, gpu: int, mode: str) -> subprocess.Popen[str] | None:
    if mode == "none":
        return None
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--background-worker",
            "--gpu",
            str(gpu),
            "--mode",
            mode,
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    if line != _BG_READY_SENTINEL:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(
            f"background worker for mode={mode!r} failed to initialize.\nstdout={line!r}\nstderr={stderr}"
        )
    return process


def _stop_background_worker(process: subprocess.Popen[str] | None) -> dict[str, Any] | None:
    if process is None:
        return None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _run_collective_probe(output_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _run_shell_json(
        [
            sys.executable,
            "tests/benchmarks/bench_collective_comm_only.py",
            "--collectives",
            "allreduce,broadcast",
            "--message-sizes",
            "4096,262144",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--world-size",
            "2",
            "--timing-mode",
            "host_wall",
            "--output-json",
            str(output_json),
        ],
        timeout_seconds=180,
    )


def _extract_collective_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for case in payload.get("cases", []):
        key = f"{case['collective']}_{case['size_bytes']}"
        cases[key] = {
            "collective": case["collective"],
            "size_bytes": case["size_bytes"],
            "tncc_median_ms": case["tncc"]["median_ms"],
            "tncc_median_bandwidth_gbps": case["tncc"]["median_bandwidth_gbps"],
            "nccl_median_ms": case["nccl"]["median_ms"],
            "nccl_median_bandwidth_gbps": case["nccl"]["median_bandwidth_gbps"],
            "tncc_vs_nccl_bandwidth_ratio": case["tncc_vs_nccl_bandwidth_ratio"],
            "tncc_correct": case["tncc"]["correct_all_ranks"],
            "nccl_correct": case["nccl"]["correct_all_ranks"],
        }
    return cases


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for study outputs.",
    )
    parser.add_argument(
        "--target-gpu",
        type=int,
        default=0,
        help="Participating GPU that receives the extra local workload.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(condition.name for condition in CONDITIONS),
        help="Comma-separated subset of conditions to run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated benchmark samples per condition.",
    )
    parser.add_argument("--background-worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, default="none")
    return parser.parse_args(argv)


def _background_worker(*, gpu: int, mode: str) -> None:
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    if mode == "resident_only":
        bytes_to_hold = 8 * 1024 * 1024 * 1024
        tensor = torch.empty(bytes_to_hold // 4, dtype=torch.float32, device=device)
        tensor.fill_(1.0)
        torch.cuda.synchronize(gpu)
        sys.stdout.write(_BG_READY_SENTINEL)
        sys.stdout.flush()
        while True:
            time.sleep(1.0)

    if mode == "sm_burn":
        size = 4096
        a = torch.randn((size, size), device=device, dtype=torch.float16)
        b = torch.randn((size, size), device=device, dtype=torch.float16)
        c = torch.empty((size, size), device=device, dtype=torch.float16)
        torch.cuda.synchronize(gpu)
        sys.stdout.write(_BG_READY_SENTINEL)
        sys.stdout.flush()
        while True:
            torch.matmul(a, b, out=c)

    if mode == "dram_burn":
        elems = 512 * 1024 * 1024 // 4
        x = torch.randn((elems,), device=device, dtype=torch.float32)
        y = torch.randn((elems,), device=device, dtype=torch.float32)
        z = torch.empty_like(x)
        torch.cuda.synchronize(gpu)
        sys.stdout.write(_BG_READY_SENTINEL)
        sys.stdout.flush()
        while True:
            torch.add(x, y, out=z)
            x, y, z = y, z, x

    raise SystemExit(f"unsupported background worker mode: {mode!r}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.background_worker:
        _background_worker(gpu=int(args.gpu), mode=str(args.mode))
        return

    requested = {item.strip() for item in str(args.conditions).split(",") if item.strip()}
    conditions = [condition for condition in CONDITIONS if condition.name in requested]
    missing = requested - {condition.name for condition in conditions}
    if missing:
        raise SystemExit(f"unknown conditions: {', '.join(sorted(missing))}")

    run_dir = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "comm_interference_study",
        "generated_at_utc": _now_utc(),
        "target_gpu": int(args.target_gpu),
        "repeats": int(args.repeats),
        "conditions": {},
    }

    for condition in conditions:
        condition_dir = run_dir / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {condition.name}: {condition.description} ===", flush=True)
        worker = _launch_background_worker(gpu=int(args.target_gpu), mode=condition.mode)
        try:
            time.sleep(2.0)
            pmon_before = _nvidia_smi("pmon", "-c", "1")
            samples: list[dict[str, Any]] = []
            for repeat_idx in range(1, int(args.repeats) + 1):
                p2p = _run_p2p_probe()
                collective_output = condition_dir / f"collective_probe_{repeat_idx:02d}.json"
                collective_payload, collective_meta = _run_collective_probe(collective_output)
                sample = {
                    "repeat_index": repeat_idx,
                    "p2p_probe": p2p,
                    "collective_probe": _extract_collective_metrics(collective_payload),
                    "collective_command": collective_meta["command"],
                    "collective_stdout": collective_meta["stdout"],
                    "collective_stderr": collective_meta["stderr"],
                }
                samples.append(sample)
                write_json(condition_dir / f"sample_{repeat_idx:02d}.json", sample)
                print(f"[{condition.name}] sample {repeat_idx}/{int(args.repeats)}", flush=True)
            pmon_after = _nvidia_smi("pmon", "-c", "1")
        finally:
            worker_meta = _stop_background_worker(worker)

        p2p_read_values = [sample["p2p_probe"]["read_bandwidth_gbps"] for sample in samples]
        p2p_write_values = [sample["p2p_probe"]["write_bandwidth_gbps"] for sample in samples]
        aggregate_cases: dict[str, dict[str, Any]] = {}
        case_keys = sorted(samples[0]["collective_probe"]) if samples else []
        for key in case_keys:
            tncc_ms = [sample["collective_probe"][key]["tncc_median_ms"] for sample in samples]
            tncc_bw = [
                sample["collective_probe"][key]["tncc_median_bandwidth_gbps"]
                for sample in samples
            ]
            nccl_ms = [sample["collective_probe"][key]["nccl_median_ms"] for sample in samples]
            nccl_bw = [
                sample["collective_probe"][key]["nccl_median_bandwidth_gbps"]
                for sample in samples
            ]
            aggregate_cases[key] = {
                "collective": samples[0]["collective_probe"][key]["collective"],
                "size_bytes": samples[0]["collective_probe"][key]["size_bytes"],
                "tncc_median_ms": _median(tncc_ms),
                "tncc_spread_pct": _spread_pct(tncc_ms),
                "tncc_median_bandwidth_gbps": _median(tncc_bw),
                "nccl_median_ms": _median(nccl_ms),
                "nccl_spread_pct": _spread_pct(nccl_ms),
                "nccl_median_bandwidth_gbps": _median(nccl_bw),
                "tncc_vs_nccl_bandwidth_ratio": _median(
                    [sample["collective_probe"][key]["tncc_vs_nccl_bandwidth_ratio"] for sample in samples]
                ),
            }

        summary["conditions"][condition.name] = {
            "description": condition.description,
            "pmon_before": pmon_before,
            "pmon_after": pmon_after,
            "p2p_probe": {
                "read_bandwidth_gbps_median": _median(p2p_read_values),
                "read_bandwidth_spread_pct": _spread_pct(p2p_read_values),
                "write_bandwidth_gbps_median": _median(p2p_write_values),
                "write_bandwidth_spread_pct": _spread_pct(p2p_write_values),
            },
            "collective_probe": aggregate_cases,
            "samples": samples,
            "background_worker": worker_meta,
        }
        write_json(condition_dir / "condition_summary.json", summary["conditions"][condition.name])

    summary_path = run_dir / "summary.json"
    latest_path = Path(args.output_root) / "latest.json"
    write_json(summary_path, summary)
    write_json(latest_path, summary)
    print(f"Summary written to: {summary_path}")
    print(f"Latest summary updated at: {latest_path}")


if __name__ == "__main__":
    main()
