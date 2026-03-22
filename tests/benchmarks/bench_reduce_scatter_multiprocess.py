#!/usr/bin/env python3
"""Structured multiprocess reduce_scatter(device) benchmark/diagnostic matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from xtile.utils.benchmark_results import write_json
from xtile.utils.feature_gates import MULTIPROCESS_DEVICE_COLLECTIVES_ENV


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the matrix benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float16,bfloat16,float32",
        help="Comma-separated dtype list.",
    )
    parser.add_argument(
        "--transports",
        type=str,
        default="auto,ctypes_ipc,pytorch_ipc,peer_access_pointer_exchange",
        help="Comma-separated transport strategy list.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations per case.")
    parser.add_argument("--block-size", type=int, default=128, help="Reduce-scatter chunk size.")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Subprocess timeout per case.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(Path("docs/generated/reduce_scatter_multiprocess_matrix.json")),
        help="Structured JSON output path.",
    )
    return parser.parse_args(argv)


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated CLI list, dropping empty entries."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _aggregate_rank_payloads(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate rank-local payloads into one case summary."""
    primitive_means = [
        float(item["primitive_timing_ms"]["mean_ms"])  # type: ignore[index]
        for item in payloads
    ]
    high_level_means = [
        float(item["high_level_timing_ms"]["mean_ms"])  # type: ignore[index]
        for item in payloads
    ]
    return {
        "transport_strategy": payloads[0]["transport_strategy"],
        "mode": payloads[0]["mode"],
        "primitive_mean_ms_across_ranks": statistics.mean(primitive_means),
        "high_level_mean_ms_across_ranks": statistics.mean(high_level_means),
        "max_rank_skew_ms": abs(max(high_level_means) - min(high_level_means)),
    }


def main() -> None:
    """Run the structured multiprocess reduce_scatter matrix."""
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 GPUs")

    args = _parse_args(sys.argv[1:])
    dtypes = _split_csv(args.dtypes)
    transports = _split_csv(args.transports)
    repo_root = Path(__file__).resolve().parents[2]

    cases: list[dict[str, object]] = []
    for dtype_name in dtypes:
        for transport in transports:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "tests.test_e2e._run_reduce_scatter_multiprocess",
                "--dtype",
                dtype_name,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--block-size",
                str(args.block_size),
            ]
            if transport != "auto":
                cmd.extend(["--force-transport", transport])

            env = os.environ.copy()
            env[MULTIPROCESS_DEVICE_COLLECTIVES_ENV] = "1"
            try:
                result = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=args.timeout_sec,
                )
                payload = {
                    "dtype": dtype_name,
                    "requested_transport": transport,
                    "command": " ".join(cmd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired as exc:
                payload = {
                    "dtype": dtype_name,
                    "requested_transport": transport,
                    "command": " ".join(cmd),
                    "returncode": -1,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "status": "timed_out",
                }
                cases.append(payload)
                print(
                    f"[FAIL] dtype={dtype_name:8s} requested={transport:28s} rc=timeout"
                    ,
                    flush=True,
                )
                continue
            if result.returncode == 0:
                rank_payloads = [
                    json.loads(line)
                    for line in result.stdout.splitlines()
                    if line.strip().startswith("{")
                ]
                payload["status"] = "passed"
                payload["rank_payloads"] = rank_payloads
                payload["summary"] = _aggregate_rank_payloads(rank_payloads)
            else:
                payload["status"] = "failed"
            cases.append(payload)

            summary = payload.get("summary")
            if payload["status"] == "passed" and isinstance(summary, dict):
                print(
                    f"[PASS] dtype={dtype_name:8s} requested={transport:28s} "
                    f"actual={summary['transport_strategy']:28s} "
                    f"primitive={summary['primitive_mean_ms_across_ranks']:.3f} ms "
                    f"high_level={summary['high_level_mean_ms_across_ranks']:.3f} ms",
                    flush=True,
                )
            else:
                print(
                    f"[FAIL] dtype={dtype_name:8s} requested={transport:28s} "
                    f"rc={result.returncode}",
                    flush=True,
                )

    passed_cases = sum(1 for item in cases if item["status"] == "passed")
    payload = {
        "schema_version": 1,
        "benchmark": "reduce_scatter_multiprocess_matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(),
            "warmup": args.warmup,
            "iters": args.iters,
            "block_size": args.block_size,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_cases": passed_cases,
            "failed_cases": len(cases) - passed_cases,
        },
    }
    output_path = write_json(args.output_json, payload)
    print(f"Structured results written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
