#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structured multiprocess gemm_allgather matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from tncc.utils.benchmark_results import write_json


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
    parser.add_argument("--M", type=int, default=128, help="Rows of A/C.")
    parser.add_argument("--N", type=int, default=256, help="Full output columns.")
    parser.add_argument("--K", type=int, default=128, help="Reduction dimension.")
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Optional comma-separated MxNxK grid (for example: 128x256x128,256x512x256). "
        "When set, overrides --M/--N/--K.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=2, help="Timed iterations per case.")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Subprocess timeout per case.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(Path("docs/generated/gemm_allgather_multiprocess_matrix.json")),
        help="Structured JSON output path.",
    )
    return parser.parse_args(argv)


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated CLI list, dropping empty entries."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_shapes(raw: str, *, default_shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Parse one optional comma-separated shape grid."""
    items = _split_csv(raw)
    if not items:
        return [default_shape]

    shapes: list[tuple[int, int, int]] = []
    for item in items:
        parts = item.lower().split("x")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid shape {item!r}. Expected MxNxK, for example 128x256x128."
            )
        try:
            shape = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"Invalid shape {item!r}. M/N/K must be integers."
            ) from exc
        if any(dim <= 0 for dim in shape):
            raise ValueError(
                f"Invalid shape {item!r}. M/N/K must all be positive."
            )
        shapes.append(shape)
    return shapes


def _aggregate_rank_payloads(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate rank-local payloads into one case summary."""
    plan_means = [
        float(item["plan_timing_ms"]["mean_ms"])  # type: ignore[index]
        for item in payloads
        if item["plan_timing_ms"] is not None
    ]
    high_level_means = [
        float(item["high_level_timing_ms"]["mean_ms"])  # type: ignore[index]
        for item in payloads
        if item["high_level_timing_ms"] is not None
    ]
    return {
        "transport_strategy": payloads[0]["transport_strategy"],
        "mode": payloads[0]["mode"],
        "plan_runtime": payloads[0]["plan_runtime"],
        "plan_mean_ms_across_ranks": statistics.mean(plan_means),
        "high_level_mean_ms_across_ranks": statistics.mean(high_level_means),
        "max_rank_skew_ms": abs(max(high_level_means) - min(high_level_means)),
        "max_plan_abs_diff": max(float(item["plan_max_abs_diff"]) for item in payloads),
        "max_high_level_abs_diff": max(
            float(item["high_level_max_abs_diff"]) for item in payloads
        ),
    }


def main() -> None:
    """Run the structured multiprocess GEMM + allgather matrix."""
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 GPUs")

    args = _parse_args(sys.argv[1:])
    dtypes = _split_csv(args.dtypes)
    transports = _split_csv(args.transports)
    shapes = _parse_shapes(
        args.shapes,
        default_shape=(args.M, args.N, args.K),
    )
    repo_root = Path(__file__).resolve().parents[2]

    cases: list[dict[str, object]] = []
    for M, N, K in shapes:
        for dtype_name in dtypes:
            for transport in transports:
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "tests.test_e2e._run_gemm_allgather_multiprocess",
                    "--M",
                    str(M),
                    "--N",
                    str(N),
                    "--K",
                    str(K),
                    "--dtype",
                    dtype_name,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--launcher",
                    "all",
                ]
                if transport != "auto":
                    cmd.extend(["--force-transport", transport])

                env = os.environ.copy()
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
                        "M": M,
                        "N": N,
                        "K": K,
                        "dtype": dtype_name,
                        "requested_transport": transport,
                        "command": " ".join(cmd),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                except subprocess.TimeoutExpired as exc:
                    payload = {
                        "M": M,
                        "N": N,
                        "K": K,
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
                        f"[FAIL] shape={M}x{N}x{K} dtype={dtype_name:8s} "
                        f"requested={transport:28s} rc=timeout",
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
                        f"[PASS] shape={M}x{N}x{K} dtype={dtype_name:8s} "
                        f"requested={transport:28s} "
                        f"actual={summary['transport_strategy']:28s} "
                        f"runtime={summary['plan_runtime']['execution_model']:24s} "
                        f"plan={summary['plan_mean_ms_across_ranks']:.3f} ms "
                        f"high_level={summary['high_level_mean_ms_across_ranks']:.3f} ms",
                        flush=True,
                    )
                else:
                    print(
                        f"[FAIL] shape={M}x{N}x{K} dtype={dtype_name:8s} "
                        f"requested={transport:28s} rc={result.returncode}",
                        flush=True,
                    )

    passed_cases = sum(1 for item in cases if item["status"] == "passed")
    payload = {
        "schema_version": 1,
        "benchmark": "gemm_allgather_multiprocess_matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(),
            "shapes": [
                {"M": M, "N": N, "K": K}
                for M, N, K in shapes
            ],
            "warmup": args.warmup,
            "iters": args.iters,
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
