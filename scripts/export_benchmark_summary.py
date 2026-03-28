#!/usr/bin/env python3
"""Export a structured benchmark/runtime summary into Markdown."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _benchmark_reporting import (
    benchmark_footer_text,
    execution_path_brief,
    load_json_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "generated" / "benchmark_runtime_summary.md"
GEMM_JSON = REPO_ROOT / "figures" / "data" / "gemm_latest.json"
P2P_JSON = REPO_ROOT / "figures" / "data" / "p2p_latest.json"
PATTERN_JSON = REPO_ROOT / "figures" / "data" / "pattern_overlap_latest.json"
COLLECTIVE_JSON = REPO_ROOT / "figures" / "data" / "collective_comm_only_latest.json"
COLLECTIVE_BULK_JSON = REPO_ROOT / "figures" / "data" / "collective_bulk_sync_latest.json"


def _payload_status(payload: dict[str, Any]) -> str:
    """Return a coarse availability status for one benchmark payload."""
    if not payload:
        return "missing"
    return "available"


def _environment_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the environment block when present."""
    environment = payload.get("environment")
    if isinstance(environment, dict):
        return environment
    return {}


def _normalize_interconnect(link: str | None) -> str | None:
    """Normalize topology shorthand into a readable interconnect label."""
    if not link:
        return None
    if link.startswith("NV"):
        return f"NVLink ({link})"
    return link


def _hardware_topology_lines(*payloads: dict[str, Any]) -> list[str]:
    """Return concise host-topology lines inferred from structured payloads."""
    gpu_name = None
    interconnect = None
    visible_gpus = None

    for payload in payloads:
        environment = _environment_dict(payload)
        if gpu_name is None:
            value = environment.get("gpu_name")
            if isinstance(value, str) and value:
                gpu_name = value
        if visible_gpus is None:
            value = environment.get("visible_gpus")
            if isinstance(value, int):
                visible_gpus = value
        if interconnect is None:
            value = environment.get("gpu_interconnect") or environment.get("link")
            if isinstance(value, str) and value:
                interconnect = _normalize_interconnect(value)

    lines: list[str] = []
    if gpu_name is not None and visible_gpus is not None:
        lines.append(f"- GPU SKU: `{visible_gpus} x {gpu_name}`")
    elif gpu_name is not None:
        lines.append(f"- GPU SKU: `{gpu_name}`")

    if interconnect is not None:
        lines.append(f"- GPU-GPU interconnect for multi-GPU runs: `{interconnect}`")

    return lines


def _extract_gemm_headlines(payload: dict[str, Any]) -> list[str]:
    """Return concise GEMM headline metrics."""
    results = payload.get("results")
    if not isinstance(results, list):
        return ["无结构化 GEMM benchmark 数据。"]

    wanted = [
        (4096, "fp16"),
        (4096, "bf16"),
        (8192, "fp16"),
        (8192, "bf16"),
    ]
    lines: list[str] = []
    for size, dtype in wanted:
        match = next(
            (
                item
                for item in results
                if int(item.get("M", -1)) == size and item.get("dtype") == dtype
            ),
            None,
        )
        if match is None:
            continue
        lines.append(
            f"- `{size}³ {dtype}`: {float(match['ratio_pct']):.1f}% of torch.matmul"
        )
    return lines or ["GEMM 结果不足以提取 headline。"]


def _extract_p2p_headlines(payload: dict[str, Any]) -> list[str]:
    """Return concise P2P headline metrics."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return ["无结构化 P2P benchmark 数据。"]

    best_read = summary.get("best_read")
    best_write = summary.get("best_write")
    lines: list[str] = []
    if isinstance(best_read, dict):
        lines.append(
            "- "
            f"best read: {float(best_read['bandwidth_gbps']):.2f} GB/s, "
            f"variant={best_read['variant']}, block_size={best_read['block_size']}, grid={best_read['grid']}"
        )
    if isinstance(best_write, dict):
        lines.append(
            "- "
            f"best write: {float(best_write['bandwidth_gbps']):.2f} GB/s, "
            f"variant={best_write['variant']}, block_size={best_write['block_size']}, grid={best_write['grid']}"
        )
    return lines or ["P2P 结果不足以提取 headline。"]


def _extract_pattern_headlines(payload: dict[str, Any]) -> list[str]:
    """Return concise pattern-overlap headline metrics."""
    summary = payload.get("summary")
    sizes = payload.get("sizes")
    if not isinstance(summary, dict) or not isinstance(sizes, list):
        return ["无结构化 pattern benchmark 数据。"]

    lines = [
        f"- best speedup vs bulk_sync: {float(summary.get('best_speedup_vs_bulk', 0.0)):.3f}×"
    ]
    best_entry = None
    for entry in sizes:
        if not isinstance(entry, dict):
            continue
        current = float(entry.get("best_speedup_vs_bulk", 0.0))
        if best_entry is None or current > float(best_entry.get("best_speedup_vs_bulk", 0.0)):
            best_entry = entry
    if isinstance(best_entry, dict):
        lines.append(
            "- "
            f"best size: {best_entry['M']}×{best_entry['N']}×{best_entry['K']}, "
            f"pattern={best_entry['best_pattern']}, speedup={float(best_entry['best_speedup_vs_bulk']):.3f}×"
        )
    return lines


def _extract_collective_headlines(payload: dict[str, Any]) -> list[str]:
    """Return concise communication-only collective headline metrics."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return ["无结构化 collective benchmark 数据。"]

    peak_by_collective = summary.get("peak_by_collective")
    if not isinstance(peak_by_collective, dict):
        return ["Collective 结果不足以提取 headline。"]

    lines: list[str] = []
    for collective in ("allreduce", "allgather", "scatter", "reduce_scatter", "broadcast"):
        entry = peak_by_collective.get(collective)
        if not isinstance(entry, dict):
            continue
        lines.append(
            "- "
            f"{collective}: TNCC peak {float(entry['peak_tncc_bandwidth_gbps']):.2f} GB/s, "
            f"NCCL peak {float(entry['peak_nccl_bandwidth_gbps']):.2f} GB/s, "
            f"best ratio {float(entry['best_tncc_vs_nccl_ratio']):.3f}×"
        )
    return lines or ["Collective 结果不足以提取 headline。"]


def _extract_collective_bulk_headlines(payload: dict[str, Any]) -> list[str]:
    """Return concise collective-vs-bulk_sync headline metrics."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return ["无结构化 collective-vs-bulk_sync benchmark 数据。"]

    lines: list[str] = []
    best_case = summary.get("best_case")
    if isinstance(best_case, dict):
        lines.append(
            "- "
            f"best speedup vs bulk_sync: {float(best_case['speedup_vs_bulk']):.3f}× "
            f"({best_case['collective']}, {float(best_case['size_kib']):.0f} KiB)"
        )
    peak_by_collective = summary.get("peak_by_collective")
    if isinstance(peak_by_collective, dict):
        for collective in ("allreduce", "allgather", "scatter", "reduce_scatter", "broadcast"):
            entry = peak_by_collective.get(collective)
            if not isinstance(entry, dict):
                continue
            lines.append(
                "- "
                f"{collective}: best speedup {float(entry['best_speedup_vs_bulk']):.3f}×"
            )
    return lines or ["Collective-vs-bulk_sync 结果不足以提取 headline。"]


def build_summary_document(
    *,
    gemm_payload: dict[str, Any],
    p2p_payload: dict[str, Any],
    pattern_payload: dict[str, Any],
    collective_payload: dict[str, Any],
    collective_bulk_payload: dict[str, Any],
) -> str:
    """Render the benchmark/runtime summary as Markdown."""
    generated_at = datetime.now(timezone.utc).isoformat()
    hardware_lines = _hardware_topology_lines(
        p2p_payload,
        pattern_payload,
        collective_payload,
        collective_bulk_payload,
        gemm_payload,
    )
    lines = [
        "# TNCC Benchmark Runtime Summary",
        "",
        f"> Generated at UTC: `{generated_at}`",
        "> This file is auto-generated from canonical benchmark JSON artifacts.",
        "",
        "## Artifact Status",
        "",
        "| Benchmark | Status | File |",
        "|------|------|------|",
        f"| GEMM | {_payload_status(gemm_payload)} | `figures/data/gemm_latest.json` |",
        f"| P2P | {_payload_status(p2p_payload)} | `figures/data/p2p_latest.json` |",
        f"| Pattern | {_payload_status(pattern_payload)} | `figures/data/pattern_overlap_latest.json` |",
        f"| Comm-only Collectives | {_payload_status(collective_payload)} | `figures/data/collective_comm_only_latest.json` |",
        f"| Collective vs bulk_sync | {_payload_status(collective_bulk_payload)} | `figures/data/collective_bulk_sync_latest.json` |",
        "",
        "## Host Topology",
        "",
        *(hardware_lines or ["- unavailable"]),
        "",
        "## Runtime Support Snapshots",
        "",
        f"- GEMM: {benchmark_footer_text(gemm_payload, source_name='gemm_latest.json') or 'unavailable'}",
        f"- P2P: {benchmark_footer_text(p2p_payload, source_name='p2p_latest.json', highlight_ops=('reduce_scatter',)) or 'unavailable'}",
        f"- Pattern: {benchmark_footer_text(pattern_payload, source_name='pattern_overlap_latest.json', highlight_ops=('gemm_allscatter',)) or 'unavailable'}",
        f"- Comm-only collectives: {benchmark_footer_text(collective_payload, source_name='collective_comm_only_latest.json') or 'unavailable'}",
        f"- Collective vs bulk_sync: {benchmark_footer_text(collective_bulk_payload, source_name='collective_bulk_sync_latest.json') or 'unavailable'}",
        "",
        "## Execution Paths",
        "",
        "- "
        f"P2P/collective runtime: "
        f"{execution_path_brief(p2p_payload, names=('reduce_scatter.reference', 'reduce_scatter.device')) or 'unavailable'}",
        "",
        "## Headline Metrics",
        "",
        "### GEMM",
        "",
        *_extract_gemm_headlines(gemm_payload),
        "",
        "### P2P",
        "",
        *_extract_p2p_headlines(p2p_payload),
        "",
        "### Pattern Overlap",
        "",
        *_extract_pattern_headlines(pattern_payload),
        "",
        "### Comm-only Collectives",
        "",
        *_extract_collective_headlines(collective_payload),
        "",
        "### Collective vs bulk_sync",
        "",
        *_extract_collective_bulk_headlines(collective_bulk_payload),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Markdown output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gemm_payload = load_json_payload(GEMM_JSON)
    p2p_payload = load_json_payload(P2P_JSON)
    pattern_payload = load_json_payload(PATTERN_JSON)
    collective_payload = load_json_payload(COLLECTIVE_JSON)
    collective_bulk_payload = load_json_payload(COLLECTIVE_BULK_JSON)
    document = build_summary_document(
        gemm_payload=gemm_payload,
        p2p_payload=p2p_payload,
        pattern_payload=pattern_payload,
        collective_payload=collective_payload,
        collective_bulk_payload=collective_bulk_payload,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(f"Wrote benchmark runtime summary to: {output_path}")


if __name__ == "__main__":
    main()
