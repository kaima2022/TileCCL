#!/usr/bin/env python3
"""Structured multiprocess gemm_allscatter auto-pattern coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from tncc.utils.benchmark_results import write_json


_AUTO_CASES: tuple[dict[str, object], ...] = (
    {
        "case": "bulk_sync_small_m",
        "M": 128,
        "N": 512,
        "K": 256,
        "expect_pattern": "bulk_sync",
    },
    {
        "case": "fused_seq_large_k",
        "M": 512,
        "N": 1024,
        "K": 16384,
        "expect_pattern": "fused_sequential",
    },
    {
        "case": "producer_consumer_mid_n",
        "M": 512,
        "N": 3072,
        "K": 8192,
        "expect_pattern": "producer_consumer",
    },
    {
        "case": "wg_specialized_large_tiles",
        "M": 2048,
        "N": 4096,
        "K": 8192,
        "expect_pattern": "wg_specialized",
    },
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the auto-pattern matrix benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contracts",
        type=str,
        default="full_full,full_shard",
        help="Comma-separated public contracts to validate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Validation dtype.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=2, help="Timed iterations per case.")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=240,
        help="Subprocess timeout per case.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(Path("docs/generated/gemm_allscatter_multiprocess_auto_patterns.json")),
        help="Structured JSON output path.",
    )
    return parser.parse_args(argv)


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated CLI list, dropping empty entries."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_contract(contract: str) -> str:
    """Map user-facing contract spellings onto the e2e runner contract ids."""
    normalized = contract.strip().replace("/", "_")
    if normalized not in {"full_full", "full_shard"}:
        raise ValueError(
            "contract must be one of {'full_full', 'full_shard', 'full/full', 'full/shard'}, "
            f"got {contract!r}"
        )
    return normalized


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
    selected_patterns = sorted({str(item["plan_pattern_name"]) for item in payloads})
    return {
        "transport_strategy": payloads[0]["transport_strategy"],
        "mode": payloads[0]["mode"],
        "heap_size_mb": payloads[0]["heap_size_mb"],
        "selected_patterns": selected_patterns,
        "plan_mean_ms_across_ranks": statistics.mean(plan_means),
        "high_level_mean_ms_across_ranks": statistics.mean(high_level_means),
        "max_rank_skew_ms": abs(max(high_level_means) - min(high_level_means)),
        "max_plan_abs_diff": max(float(item["plan_max_abs_diff"]) for item in payloads),
        "max_high_level_abs_diff": max(
            float(item["high_level_max_abs_diff"]) for item in payloads
        ),
    }


def main() -> None:
    """Run the auto-pattern coverage benchmark."""
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 GPUs")

    args = _parse_args(sys.argv[1:])
    contracts = [_normalize_contract(item) for item in _split_csv(args.contracts)]
    repo_root = Path(__file__).resolve().parents[2]

    cases: list[dict[str, object]] = []
    for contract in contracts:
        for case in _AUTO_CASES:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "tests.test_e2e._run_gemm_allscatter_multiprocess",
                "--M",
                str(case["M"]),
                "--N",
                str(case["N"]),
                "--K",
                str(case["K"]),
                "--dtype",
                args.dtype,
                "--contract",
                contract,
                "--pattern",
                "auto",
                "--expect-pattern",
                str(case["expect_pattern"]),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--launcher",
                "all",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=args.timeout_sec,
                )
                payload = {
                    "case": case["case"],
                    "M": case["M"],
                    "N": case["N"],
                    "K": case["K"],
                    "contract": contract,
                    "dtype": args.dtype,
                    "expected_pattern": case["expect_pattern"],
                    "command": " ".join(cmd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired as exc:
                payload = {
                    "case": case["case"],
                    "M": case["M"],
                    "N": case["N"],
                    "K": case["K"],
                    "contract": contract,
                    "dtype": args.dtype,
                    "expected_pattern": case["expect_pattern"],
                    "command": " ".join(cmd),
                    "returncode": -1,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "status": "timed_out",
                }
                cases.append(payload)
                print(
                    f"[FAIL] contract={contract:10s} case={case['case']:28s} rc=timeout",
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
                patterns = ",".join(summary["selected_patterns"])
                print(
                    f"[PASS] contract={contract:10s} case={case['case']:28s} "
                    f"pattern={patterns:20s} plan={summary['plan_mean_ms_across_ranks']:.3f} ms "
                    f"high_level={summary['high_level_mean_ms_across_ranks']:.3f} ms",
                    flush=True,
                )
            else:
                print(
                    f"[FAIL] contract={contract:10s} case={case['case']:28s} rc={result.returncode}",
                    flush=True,
                )

    passed_cases = sum(1 for item in cases if item["status"] == "passed")
    payload = {
        "schema_version": 1,
        "benchmark": "gemm_allscatter_multiprocess_auto_patterns",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(),
            "dtype": args.dtype,
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
