#!/usr/bin/env python3
"""Run repeated benchmark experiments under the current GPU environment.

This script is intentionally conservative:

- it reuses the existing benchmark entry points that back ``figures/``
- it snapshots environment contamination before every run
- it archives per-run stdout/stderr plus JSON payloads
- it emits one machine-readable summary for later reporting

The goal is not to replace the publication benchmarks. The goal is to answer a
different question: how much do shared-GPU conditions perturb the benchmark
surfaces that drive the figures?
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tncc.utils.benchmark_results import benchmark_environment_health, write_json


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "figures" / "data" / "noise_study"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _cv_pct(values: list[float]) -> float:
    mean = _mean(values)
    if mean == 0.0:
        return 0.0
    return float(_stddev(values) / mean * 100.0)


def _spread_pct(values: list[float]) -> float:
    if not values:
        return 0.0
    median = _median(values)
    if median == 0.0:
        return 0.0
    return float((max(values) - min(values)) / median * 100.0)


def _slowdown_factor(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return 1.0
    return float(max(positive) / min(positive))


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return 0.0


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def _run_status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["status"] for record in records)
    return dict(sorted(counts.items()))


def _summarize_health_snapshots(records: list[dict[str, Any]]) -> dict[str, Any]:
    contaminated = sum(
        1 for record in records if record.get("environment_health_before", {}).get("status") == "contaminated"
    )
    active_gpus: dict[str, int] = defaultdict(int)
    resident_processes: Counter[str] = Counter()

    for record in records:
        health = record.get("environment_health_before", {})
        for gpu in health.get("gpus", []):
            if gpu.get("utilization_gpu_pct", 0) > 0:
                active_gpus[str(gpu.get("index"))] += 1
            for proc in gpu.get("compute_processes", []):
                resident_processes[str(proc.get("process_name", "unknown"))] += 1

    top_processes = [
        {"process_name": name, "occurrences": count}
        for name, count in resident_processes.most_common(8)
    ]
    return {
        "contaminated_runs": contaminated,
        "total_runs": len(records),
        "active_gpu_occurrences": dict(sorted(active_gpus.items())),
        "top_resident_processes": top_processes,
    }


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    visible_gpu_count: int
    repeats: int
    timeout_seconds: int
    command: list[str]
    summarize: Callable[[list[dict[str, Any]]], dict[str, Any]]


def _extract_collective_case_metrics(case: dict[str, Any]) -> dict[str, Any]:
    tncc = case["tncc"]
    nccl = case["nccl"]
    tncc_iter_times = [_safe_float(value) for value in tncc.get("aggregate_times_ms", [])]
    nccl_iter_times = [_safe_float(value) for value in nccl.get("aggregate_times_ms", [])]
    return {
        "collective": case["collective"],
        "size_bytes": int(case["size_bytes"]),
        "size_mib": _safe_float(case.get("size_mib")),
        "tncc_median_ms": _safe_float(tncc.get("median_ms")),
        "tncc_median_bandwidth_gbps": _safe_float(tncc.get("median_bandwidth_gbps")),
        "tncc_iter_slowdown_factor": _slowdown_factor(tncc_iter_times),
        "nccl_median_ms": _safe_float(nccl.get("median_ms")),
        "nccl_median_bandwidth_gbps": _safe_float(nccl.get("median_bandwidth_gbps")),
        "nccl_iter_slowdown_factor": _slowdown_factor(nccl_iter_times),
        "tncc_vs_nccl_bandwidth_ratio": _safe_float(case.get("tncc_vs_nccl_bandwidth_ratio")),
    }


def _summarize_collective_comm(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] != "ok":
            continue
        payload = record["payload"]
        for case in payload.get("cases", []):
            key = (str(case["collective"]), int(case["size_bytes"]))
            grouped[key].append(_extract_collective_case_metrics(case))

    cases_summary: list[dict[str, Any]] = []
    for (collective, size_bytes), samples in sorted(grouped.items()):
        tncc_ms = [sample["tncc_median_ms"] for sample in samples]
        nccl_ms = [sample["nccl_median_ms"] for sample in samples]
        tncc_bw = [sample["tncc_median_bandwidth_gbps"] for sample in samples]
        nccl_bw = [sample["nccl_median_bandwidth_gbps"] for sample in samples]
        ratio = [sample["tncc_vs_nccl_bandwidth_ratio"] for sample in samples]
        cases_summary.append({
            "collective": collective,
            "size_bytes": size_bytes,
            "size_mib": samples[0]["size_mib"],
            "samples": len(samples),
            "tncc_latency_ms": {
                "median": _median(tncc_ms),
                "cv_pct": _cv_pct(tncc_ms),
                "spread_pct": _spread_pct(tncc_ms),
                "max_over_min": _slowdown_factor(tncc_ms),
            },
            "nccl_latency_ms": {
                "median": _median(nccl_ms),
                "cv_pct": _cv_pct(nccl_ms),
                "spread_pct": _spread_pct(nccl_ms),
                "max_over_min": _slowdown_factor(nccl_ms),
            },
            "tncc_bandwidth_gbps": {
                "median": _median(tncc_bw),
                "cv_pct": _cv_pct(tncc_bw),
                "spread_pct": _spread_pct(tncc_bw),
            },
            "nccl_bandwidth_gbps": {
                "median": _median(nccl_bw),
                "cv_pct": _cv_pct(nccl_bw),
                "spread_pct": _spread_pct(nccl_bw),
            },
            "tncc_vs_nccl_ratio": {
                "median": _median(ratio),
                "spread_pct": _spread_pct(ratio),
            },
            "worst_intrarun_iter_slowdown_factor": {
                "tncc": max(sample["tncc_iter_slowdown_factor"] for sample in samples),
                "nccl": max(sample["nccl_iter_slowdown_factor"] for sample in samples),
            },
        })

    worst_tncc = max(
        cases_summary,
        key=lambda item: (
            item["tncc_latency_ms"]["spread_pct"],
            item["worst_intrarun_iter_slowdown_factor"]["tncc"],
        ),
        default=None,
    )
    worst_nccl = max(
        cases_summary,
        key=lambda item: (
            item["nccl_latency_ms"]["spread_pct"],
            item["worst_intrarun_iter_slowdown_factor"]["nccl"],
        ),
        default=None,
    )
    return {
        "status_counts": _run_status_counts(records),
        "environment": _summarize_health_snapshots(records),
        "cases": cases_summary,
        "worst_tncc_case": worst_tncc,
        "worst_nccl_case": worst_nccl,
    }


def _summarize_gemm(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] != "ok":
            continue
        for result in record["payload"].get("results", []):
            key = (
                int(result["M"]),
                int(result["N"]),
                int(result["K"]),
                str(result["dtype"]),
            )
            grouped[key].append(result)

    configs: list[dict[str, Any]] = []
    for (m, n, k, dtype), samples in sorted(grouped.items()):
        torch_tflops = [_safe_float(sample["torch_tflops"]) for sample in samples]
        tncc_tflops = [_safe_float(sample["tncc_tflops"]) for sample in samples]
        ratio_pct = [_safe_float(sample["ratio_pct"]) for sample in samples]
        configs.append({
            "M": m,
            "N": n,
            "K": k,
            "dtype": dtype,
            "samples": len(samples),
            "torch_tflops": {
                "median": _median(torch_tflops),
                "cv_pct": _cv_pct(torch_tflops),
                "spread_pct": _spread_pct(torch_tflops),
            },
            "tncc_tflops": {
                "median": _median(tncc_tflops),
                "cv_pct": _cv_pct(tncc_tflops),
                "spread_pct": _spread_pct(tncc_tflops),
            },
            "ratio_pct": {
                "median": _median(ratio_pct),
                "cv_pct": _cv_pct(ratio_pct),
                "spread_pct": _spread_pct(ratio_pct),
            },
        })

    return {
        "status_counts": _run_status_counts(records),
        "environment": _summarize_health_snapshots(records),
        "configs": configs,
        "worst_tncc_config": max(configs, key=lambda item: item["tncc_tflops"]["spread_pct"], default=None),
        "worst_torch_config": max(configs, key=lambda item: item["torch_tflops"]["spread_pct"], default=None),
        "worst_ratio_config": max(configs, key=lambda item: item["ratio_pct"]["spread_pct"], default=None),
    }


def _summarize_p2p(records: list[dict[str, Any]]) -> dict[str, Any]:
    best_read = []
    best_write = []
    read_variants = Counter()
    write_variants = Counter()
    for record in records:
        if record["status"] != "ok":
            continue
        summary = record["payload"].get("summary", {})
        current_read = summary.get("best_read")
        current_write = summary.get("best_write")
        if isinstance(current_read, dict):
            best_read.append(_safe_float(current_read.get("bandwidth_gbps")))
            read_variants[str(current_read.get("variant", "unknown"))] += 1
        if isinstance(current_write, dict):
            best_write.append(_safe_float(current_write.get("bandwidth_gbps")))
            write_variants[str(current_write.get("variant", "unknown"))] += 1
    return {
        "status_counts": _run_status_counts(records),
        "environment": _summarize_health_snapshots(records),
        "best_read_bandwidth_gbps": {
            "median": _median(best_read),
            "cv_pct": _cv_pct(best_read),
            "spread_pct": _spread_pct(best_read),
        },
        "best_write_bandwidth_gbps": {
            "median": _median(best_write),
            "cv_pct": _cv_pct(best_write),
            "spread_pct": _spread_pct(best_write),
        },
        "best_read_variant_counts": dict(sorted(read_variants.items())),
        "best_write_variant_counts": dict(sorted(write_variants.items())),
    }


def _summarize_patterns(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] != "ok":
            continue
        for size in record["payload"].get("sizes", []):
            key = (int(size["M"]), int(size["N"]), int(size["K"]))
            grouped[key].append(size)

    sizes_summary: list[dict[str, Any]] = []
    for (m, n, k), samples in sorted(grouped.items()):
        best_patterns = Counter(str(sample.get("best_pattern", "unknown")) for sample in samples)
        best_speedups = [_safe_float(sample.get("best_speedup_vs_bulk")) for sample in samples]
        bulk_sync_ms = []
        best_pattern_min_ms = []
        for sample in samples:
            results = sample.get("results", [])
            by_pattern = {result["pattern"]: result for result in results}
            bulk_sync = by_pattern.get("bulk_sync")
            best_pattern = by_pattern.get(str(sample.get("best_pattern")))
            if isinstance(bulk_sync, dict):
                bulk_sync_ms.append(_safe_float(bulk_sync.get("min_ms")))
            if isinstance(best_pattern, dict):
                best_pattern_min_ms.append(_safe_float(best_pattern.get("min_ms")))
        sizes_summary.append({
            "M": m,
            "N": n,
            "K": k,
            "samples": len(samples),
            "best_pattern_counts": dict(sorted(best_patterns.items())),
            "best_speedup_vs_bulk": {
                "median": _median(best_speedups),
                "cv_pct": _cv_pct(best_speedups),
                "spread_pct": _spread_pct(best_speedups),
            },
            "bulk_sync_min_ms": {
                "median": _median(bulk_sync_ms),
                "cv_pct": _cv_pct(bulk_sync_ms),
                "spread_pct": _spread_pct(bulk_sync_ms),
            },
            "best_pattern_min_ms": {
                "median": _median(best_pattern_min_ms),
                "cv_pct": _cv_pct(best_pattern_min_ms),
                "spread_pct": _spread_pct(best_pattern_min_ms),
            },
        })
    return {
        "status_counts": _run_status_counts(records),
        "environment": _summarize_health_snapshots(records),
        "sizes": sizes_summary,
        "most_unstable_size": max(
            sizes_summary,
            key=lambda item: item["best_speedup_vs_bulk"]["spread_pct"],
            default=None,
        ),
    }


def _summarize_bulk_sync(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] != "ok":
            continue
        for case in record["payload"].get("cases", []):
            key = (str(case["collective"]), int(case["size_bytes"]))
            grouped[key].append(case)

    cases_summary: list[dict[str, Any]] = []
    for (collective, size_bytes), samples in sorted(grouped.items()):
        tncc_ms = [_safe_float(sample["tncc"]["median_ms"]) for sample in samples]
        bulk_ms = [_safe_float(sample["bulk_sync"]["median_ms"]) for sample in samples]
        speedups = [_safe_float(sample["speedup_vs_bulk"]) for sample in samples]
        cases_summary.append({
            "collective": collective,
            "size_bytes": size_bytes,
            "samples": len(samples),
            "tncc_latency_ms": {
                "median": _median(tncc_ms),
                "cv_pct": _cv_pct(tncc_ms),
                "spread_pct": _spread_pct(tncc_ms),
            },
            "bulk_sync_latency_ms": {
                "median": _median(bulk_ms),
                "cv_pct": _cv_pct(bulk_ms),
                "spread_pct": _spread_pct(bulk_ms),
            },
            "speedup_vs_bulk": {
                "median": _median(speedups),
                "cv_pct": _cv_pct(speedups),
                "spread_pct": _spread_pct(speedups),
            },
        })
    return {
        "status_counts": _run_status_counts(records),
        "environment": _summarize_health_snapshots(records),
        "cases": cases_summary,
        "most_unstable_case": max(
            cases_summary,
            key=lambda item: item["speedup_vs_bulk"]["spread_pct"],
            default=None,
        ),
    }


def _build_specs(repeats: int, comm_repeats: int, bulk_repeats: int) -> list[ExperimentSpec]:
    python = sys.executable
    return [
        ExperimentSpec(
            name="fig1_gemm",
            description="Figure 1 GEMM benchmark, repeated single full runs on one GPU.",
            visible_gpu_count=1,
            repeats=repeats,
            timeout_seconds=180,
            command=[
                python,
                "tests/benchmarks/bench_gemm.py",
                "--repeats",
                "1",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_gemm,
        ),
        ExperimentSpec(
            name="fig2_p2p",
            description="Figure 2 P2P translate benchmark, quick representative sweep.",
            visible_gpu_count=2,
            repeats=repeats,
            timeout_seconds=180,
            command=[
                python,
                "tests/benchmarks/bench_p2p_translate.py",
                "--quick",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_p2p,
        ),
        ExperimentSpec(
            name="fig3_patterns",
            description="Figure 3 overlap-pattern benchmark, quick representative sweep.",
            visible_gpu_count=2,
            repeats=repeats,
            timeout_seconds=240,
            command=[
                python,
                "tests/benchmarks/bench_patterns.py",
                "--quick",
                "--warmup",
                "1",
                "--iters",
                "3",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_patterns,
        ),
        ExperimentSpec(
            name="fig6_comm_allreduce",
            description="Figure 6 comm-only allreduce benchmark at 4 KiB and 256 KiB.",
            visible_gpu_count=2,
            repeats=comm_repeats,
            timeout_seconds=240,
            command=[
                python,
                "tests/benchmarks/bench_collective_comm_only.py",
                "--collectives",
                "allreduce",
                "--message-sizes",
                "4096,262144",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--world-size",
                "2",
                "--timing-mode",
                "device_event",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_collective_comm,
        ),
        ExperimentSpec(
            name="fig6_comm_exchange",
            description="Figure 6 comm-only exchange collectives at 4 KiB and 256 KiB.",
            visible_gpu_count=2,
            repeats=comm_repeats,
            timeout_seconds=240,
            command=[
                python,
                "tests/benchmarks/bench_collective_comm_only.py",
                "--collectives",
                "allgather,scatter,broadcast",
                "--message-sizes",
                "4096,262144",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--world-size",
                "2",
                "--timing-mode",
                "device_event",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_collective_comm,
        ),
        ExperimentSpec(
            name="fig6_comm_reduce_scatter",
            description="Figure 6 comm-only reduce_scatter latency benchmark at 4 KiB.",
            visible_gpu_count=2,
            repeats=comm_repeats,
            timeout_seconds=180,
            command=[
                python,
                "tests/benchmarks/bench_collective_comm_only.py",
                "--collectives",
                "reduce_scatter",
                "--message-sizes",
                "4096",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--world-size",
                "2",
                "--timing-mode",
                "device_event",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_collective_comm,
        ),
        ExperimentSpec(
            name="fig6_comm_reduce_scatter_256k_probe",
            description="Figure 6 comm-only reduce_scatter 256 KiB single-point probe.",
            visible_gpu_count=2,
            repeats=1,
            timeout_seconds=180,
            command=[
                python,
                "tests/benchmarks/bench_collective_comm_only.py",
                "--collectives",
                "reduce_scatter",
                "--message-sizes",
                "262144",
                "--warmup",
                "0",
                "--iters",
                "1",
                "--world-size",
                "2",
                "--timing-mode",
                "device_event",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_collective_comm,
        ),
        ExperimentSpec(
            name="fig7_bulk_sync",
            description="Figure 7 bulk-sync comparison at 4 KiB only to bound runtime.",
            visible_gpu_count=2,
            repeats=bulk_repeats,
            timeout_seconds=240,
            command=[
                python,
                "tests/benchmarks/bench_collective_bulk_sync.py",
                "--message-sizes",
                "4096",
                "--warmup",
                "1",
                "--iters",
                "1",
                "--world-size",
                "2",
                "--output-json",
                "__OUTPUT_JSON__",
            ],
            summarize=_summarize_bulk_sync,
        ),
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where raw per-run artifacts and the summary JSON are stored.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Default repeat count for fig1/fig2/fig3 representative runs.",
    )
    parser.add_argument(
        "--comm-repeats",
        type=int,
        default=3,
        help="Repeat count for the heavier fig6 collective comparison.",
    )
    parser.add_argument(
        "--bulk-repeats",
        type=int,
        default=3,
        help="Repeat count for the heavier fig7 bulk-sync comparison.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment names to run. Empty means all.",
    )
    return parser.parse_args(argv)


def _replace_output_placeholder(command: list[str], output_json: Path) -> list[str]:
    return [str(output_json) if token == "__OUTPUT_JSON__" else token for token in command]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_experiment(spec: ExperimentSpec, output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for repeat_idx in range(1, spec.repeats + 1):
        run_dir = output_dir / spec.name / f"run_{repeat_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_json = run_dir / "result.json"
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        meta_path = run_dir / "meta.json"

        environment_health_before = benchmark_environment_health(
            visible_gpu_count=spec.visible_gpu_count,
        )
        command = _replace_output_placeholder(spec.command, output_json)
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)
        )

        started_at = _now_utc()
        process: subprocess.Popen[str] | None = None
        returncode: int | None = None
        status = "ok"
        error: str | None = None
        payload: dict[str, Any] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = process.communicate(timeout=spec.timeout_seconds)
            returncode = process.returncode
            stdout_path.write_text(_coerce_text(stdout), encoding="utf-8")
            stderr_path.write_text(_coerce_text(stderr), encoding="utf-8")
            if returncode != 0:
                status = "failed"
                error = f"returncode={returncode}"
            elif not output_json.exists():
                status = "failed"
                error = "benchmark returned success but did not write output JSON"
            else:
                payload = _load_json(output_json)
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = f"timeout after {spec.timeout_seconds}s"
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                returncode = process.returncode
            else:
                stdout = exc.stdout
                stderr = exc.stderr
            stdout_path.write_text(_coerce_text(stdout), encoding="utf-8")
            stderr_path.write_text(_coerce_text(stderr), encoding="utf-8")
        finally:
            finished_at = _now_utc()

        environment_health_after = benchmark_environment_health(
            visible_gpu_count=spec.visible_gpu_count,
        )
        record = {
            "repeat_index": repeat_idx,
            "status": status,
            "error": error,
            "command": command,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "output_json": str(output_json),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "environment_health_before": environment_health_before,
            "environment_health_after": environment_health_after,
            "returncode": returncode,
        }
        if payload is not None:
            record["payload"] = payload
        write_json(meta_path, record)
        records.append(record)
        print(
            f"[{spec.name}] run {repeat_idx}/{spec.repeats}: {status}",
            flush=True,
        )
    return records


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    output_root = Path(args.output_root)
    run_root = output_root / _slug_now()
    run_root.mkdir(parents=True, exist_ok=True)

    requested = {
        item.strip()
        for item in args.only.split(",")
        if item.strip()
    }
    specs = _build_specs(
        repeats=int(args.repeats),
        comm_repeats=int(args.comm_repeats),
        bulk_repeats=int(args.bulk_repeats),
    )
    if requested:
        specs = [spec for spec in specs if spec.name in requested]
        missing = requested - {spec.name for spec in specs}
        if missing:
            raise SystemExit(f"unknown experiment names in --only: {', '.join(sorted(missing))}")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "noise_study",
        "generated_at_utc": _now_utc(),
        "repo_root": str(REPO_ROOT),
        "output_root": str(run_root),
        "experiments": {},
    }

    for spec in specs:
        print(f"=== {spec.name}: {spec.description} ===", flush=True)
        records = _run_experiment(spec, run_root)
        summary["experiments"][spec.name] = {
            "description": spec.description,
            "repeats": spec.repeats,
            "timeout_seconds": spec.timeout_seconds,
            "command_template": spec.command,
            "records": records,
            "analysis": spec.summarize(records),
        }

    latest_path = output_root / "latest.json"
    summary_path = run_root / "summary.json"
    write_json(summary_path, summary)
    write_json(latest_path, summary)
    print(f"Summary written to: {summary_path}")
    print(f"Latest summary updated at: {latest_path}")


if __name__ == "__main__":
    main()
