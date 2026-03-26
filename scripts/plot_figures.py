#!/usr/bin/env python3
"""XTile — publication-quality figures showing the latest validated status.

Generates 7 figures in Nature/Science style:
  1. GEMM: XTile vs cuBLAS (official helper, median of 3 repeats)
  2. P2P bandwidth vs transfer size (saturation curve)
  3. Pattern speedup vs bulk_sync (full 6-size rerun)
  4. 6-layer architecture diagram
  5. Roofline model (GEMM position analysis)
  6. Communication-only collectives: XTile vs NCCL
  7. Communication-only speedup vs bulk_sync baseline

Output: figures/ directory — PDF (vector) + PNG (300 DPI)
"""

import os
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style — Nature/Science conventions
# ---------------------------------------------------------------------------
sns.set_palette("colorblind")
COLORS = sns.color_palette("colorblind")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUTDIR, exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _benchmark_reporting import benchmark_footer_text, load_json_payload

GEMM_BENCHMARK_JSON = REPO_ROOT / "figures" / "data" / "gemm_latest.json"
P2P_BENCHMARK_JSON = REPO_ROOT / "figures" / "data" / "p2p_latest.json"
PATTERN_BENCHMARK_JSON = REPO_ROOT / "figures" / "data" / "pattern_overlap_latest.json"
COLLECTIVE_BENCHMARK_JSON = REPO_ROOT / "figures" / "data" / "collective_comm_only_latest.json"
COLLECTIVE_BULK_BENCHMARK_JSON = REPO_ROOT / "figures" / "data" / "collective_bulk_sync_latest.json"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

# Fig 1 fallback provenance:
# manually curated experiment-backed values retained as a safe fallback
# when no structured GEMM benchmark JSON is present.
_FALLBACK_GEMM_BARS = {
    "sizes": [
        "1024³\nfp16",
        "1024³\nbf16",
        "2048³\nfp16",
        "2048³\nbf16",
        "4096³\nfp16",
        "4096³\nbf16",
        "8192³\nfp16",
        "8192³\nbf16",
    ],
    "cublas_tflops": [72.21, 72.07, 258.42, 257.78, 440.23, 440.26, 459.40, 481.79],
    "xtile_tflops": [38.34, 37.82, 235.15, 237.25, 428.27, 405.22, 382.96, 398.68],
}

_FALLBACK_GEMM_ROOFLINE = {
    "sizes": [1024, 2048, 4096, 8192],
    "cublas_tflops": [72.21, 258.42, 440.23, 459.40],
    "xtile_tflops": [38.34, 235.15, 428.27, 382.96],
}

# Fig 2 fallback provenance:
# manually curated experiment-backed values retained as a safe fallback
# when no structured P2P benchmark JSON is present.
_FALLBACK_P2P_CURVE = {
    "sizes_mb": [1.0, 4.2, 16.8, 67.1, 134.2],
    "read_bw": [30.454, 105.789, 230.862, 245.914, 248.832],
    "write_bw": [31.969, 106.303, 233.536, 246.347, 248.566],
    "best_read_gbps": 248.832,
}

# Fig 3 fallback provenance:
# latest validated full 6-size rerun from the local H100 machine.
# Used only when no structured benchmark JSON exists yet.
_FALLBACK_PATTERN_SPEEDUPS = {
    "sizes": [
        "4096³",
        "8192×4608\n×36864",
        "8192×3584\n×14336",
        "8192×8192\n×30720",
        "4096×8192\n×8192",
        "2048×16384\n×8192",
    ],
    "bulk_sync": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    "fused_sequential": [1.083, 1.123, 1.201, 1.214, 1.158, 1.190],
    "producer_consumer": [0.944, 1.535, 1.068, 1.489, 1.033, 1.048],
    "wg_specialized": [1.027, 1.660, 1.166, 1.607, 1.116, 1.126],
}

GEMM_PAYLOAD = load_json_payload(GEMM_BENCHMARK_JSON)
P2P_PAYLOAD = load_json_payload(P2P_BENCHMARK_JSON)
PATTERN_PAYLOAD = load_json_payload(PATTERN_BENCHMARK_JSON)
COLLECTIVE_PAYLOAD = load_json_payload(COLLECTIVE_BENCHMARK_JSON)
COLLECTIVE_BULK_PAYLOAD = load_json_payload(COLLECTIVE_BULK_BENCHMARK_JSON)


def _format_size_label(M, N, K):
    if M == N == K:
        return f"{M}³"
    return f"{M}×{N}\n×{K}"


def _load_gemm_bars():
    if not GEMM_PAYLOAD:
        return _FALLBACK_GEMM_BARS, {}
    payload = GEMM_PAYLOAD
    results = payload.get("results", [])
    if len(results) < 8:
        return _FALLBACK_GEMM_BARS, {}

    ordered = sorted(
        results,
        key=lambda item: (int(item["M"]), 0 if item["dtype"] == "fp16" else 1),
    )
    return {
        "sizes": [
            f"{item['M']}³\n{item['dtype']}"
            for item in ordered
        ],
        "cublas_tflops": [float(item["torch_tflops"]) for item in ordered],
        "xtile_tflops": [float(item["xtile_tflops"]) for item in ordered],
    }, payload.get("environment", {})


def _load_gemm_roofline_points():
    if not GEMM_PAYLOAD:
        return _FALLBACK_GEMM_ROOFLINE
    payload = GEMM_PAYLOAD
    results = [
        item for item in payload.get("results", [])
        if item.get("dtype") == "fp16"
    ]
    if len(results) < 4:
        return _FALLBACK_GEMM_ROOFLINE

    ordered = sorted(results, key=lambda item: int(item["M"]))
    return {
        "sizes": [int(item["M"]) for item in ordered],
        "cublas_tflops": [float(item["torch_tflops"]) for item in ordered],
        "xtile_tflops": [float(item["xtile_tflops"]) for item in ordered],
    }


def _load_p2p_curve():
    if not P2P_PAYLOAD:
        return _FALLBACK_P2P_CURVE, {}
    payload = P2P_PAYLOAD
    environment = payload.get("environment", {})
    if environment.get("quick_mode"):
        return _FALLBACK_P2P_CURVE, environment

    points = payload.get("summary", {}).get("float32_by_size", [])
    if len(points) < 5:
        return _FALLBACK_P2P_CURVE, environment

    ordered = sorted(points, key=lambda item: float(item["size_mb"]))
    return {
        "sizes_mb": [float(item["size_mb"]) for item in ordered],
        "read_bw": [float(item["best_read_gbps"]) for item in ordered],
        "write_bw": [float(item["best_write_gbps"]) for item in ordered],
        "best_read_gbps": max(float(item["best_read_gbps"]) for item in ordered),
    }, environment


def _load_pattern_speedups():
    fallback_best_speedup = max(
        max(_FALLBACK_PATTERN_SPEEDUPS["fused_sequential"]),
        max(_FALLBACK_PATTERN_SPEEDUPS["producer_consumer"]),
        max(_FALLBACK_PATTERN_SPEEDUPS["wg_specialized"]),
    )
    if not PATTERN_PAYLOAD:
        return _FALLBACK_PATTERN_SPEEDUPS, {"best_speedup_vs_bulk": fallback_best_speedup}
    payload = PATTERN_PAYLOAD
    environment = payload.get("environment", {})
    if environment.get("quick_mode") or len(payload.get("sizes", [])) < 6:
        return _FALLBACK_PATTERN_SPEEDUPS, {"best_speedup_vs_bulk": fallback_best_speedup}

    sizes = []
    series = {
        "bulk_sync": [],
        "fused_sequential": [],
        "producer_consumer": [],
        "wg_specialized": [],
    }
    for size_entry in payload.get("sizes", []):
        sizes.append(_format_size_label(size_entry["M"], size_entry["N"], size_entry["K"]))
        result_map = {
            result["pattern"]: result["speedup_vs_bulk"]
            for result in size_entry.get("results", [])
        }
        for pattern_name in series:
            series[pattern_name].append(float(result_map.get(pattern_name, 0.0)))

    if not sizes:
        return _FALLBACK_PATTERN_SPEEDUPS, {"best_speedup_vs_bulk": fallback_best_speedup}

    return {
        "sizes": sizes,
        **series,
    }, payload.get("summary", {})


def _load_collective_comm_only():
    if not COLLECTIVE_PAYLOAD:
        return {}, {}

    payload = COLLECTIVE_PAYLOAD
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return {}, payload.get("summary", {})

    per_collective: dict[str, dict[str, list[float]]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        collective = case.get("collective")
        xtile = case.get("xtile")
        nccl = case.get("nccl")
        if not isinstance(collective, str):
            continue
        if not isinstance(xtile, dict) or not isinstance(nccl, dict):
            continue
        bucket = per_collective.setdefault(
            collective,
            {
                "size_bytes": [],
                "size_mib": [],
                "xtile_ms": [],
                "nccl_ms": [],
                "xtile_bw": [],
                "nccl_bw": [],
                "latency_ratio": [],
                "bandwidth_ratio": [],
            },
        )
        xtile_ms = float(xtile["median_ms"])
        nccl_ms = float(nccl["median_ms"])
        bucket["size_bytes"].append(int(case["size_bytes"]))
        bucket["size_mib"].append(float(case["size_mib"]))
        bucket["xtile_ms"].append(xtile_ms)
        bucket["nccl_ms"].append(nccl_ms)
        bucket["xtile_bw"].append(float(xtile["median_bandwidth_gbps"]))
        bucket["nccl_bw"].append(float(nccl["median_bandwidth_gbps"]))
        bucket["latency_ratio"].append(xtile_ms / max(nccl_ms, 1e-9))
        bucket["bandwidth_ratio"].append(float(case["xtile_vs_nccl_bandwidth_ratio"]))

    for bucket in per_collective.values():
        order = np.argsort(bucket["size_bytes"])
        for key in tuple(bucket):
            bucket[key] = [bucket[key][idx] for idx in order]

    return per_collective, payload.get("summary", {})


def _load_collective_bulk_speedups():
    if not COLLECTIVE_BULK_PAYLOAD:
        return {}, {}

    payload = COLLECTIVE_BULK_PAYLOAD
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return {}, payload.get("summary", {})

    by_collective: dict[str, dict[str, list[float]]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        collective = case.get("collective")
        if not isinstance(collective, str):
            continue
        bucket = by_collective.setdefault(
            collective,
            {
                "size_kib": [],
                "speedup": [],
            },
        )
        bucket["size_kib"].append(float(case["size_kib"]))
        bucket["speedup"].append(float(case["speedup_vs_bulk"]))

    for bucket in by_collective.values():
        order = np.argsort(bucket["size_kib"])
        bucket["size_kib"] = [bucket["size_kib"][idx] for idx in order]
        bucket["speedup"] = [bucket["speedup"][idx] for idx in order]

    return by_collective, payload.get("summary", {})


GEMM_BARS, GEMM_ENV = _load_gemm_bars()
GEMM_ROOFLINE = _load_gemm_roofline_points()
P2P_CURVE, P2P_ENV = _load_p2p_curve()
PATTERN_SPEEDUPS, PATTERN_SUMMARY = _load_pattern_speedups()
COLLECTIVE_SERIES, COLLECTIVE_SUMMARY = _load_collective_comm_only()
COLLECTIVE_BULK_SERIES, COLLECTIVE_BULK_SUMMARY = _load_collective_bulk_speedups()
COLLECTIVE_LATENCY_MAX_BYTES = 64 * 1024
COLLECTIVE_BANDWIDTH_MIN_BYTES = 256 * 1024


def _gemm_aggregation_label() -> str:
    repeats = GEMM_ENV.get("repeats")
    aggregation = GEMM_ENV.get("aggregation")
    if isinstance(repeats, int) and repeats > 0:
        if aggregation == "median_of_full_runs":
            return f"median of {repeats} full runs"
        return f"{repeats} runs"
    return "latest benchmark"


def _save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}.pdf + {name}.png")


def _save_with_footer(fig, name, footer: str | None):
    if footer:
        wrapped = textwrap.fill(footer, width=90)
        fig.text(
            0.5,
            0.012,
            wrapped,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, name)


def _format_message_size_mib(value: float) -> str:
    if value < 1.0:
        kib = value * 1024.0
        if kib >= 1:
            return f"{kib:.0f} KiB"
    return f"{value:g} MiB"


def _select_collective_points(
    series: dict[str, list[float]],
    *,
    min_bytes: int | None = None,
    max_bytes: int | None = None,
) -> dict[str, list[float]]:
    indices = [
        idx
        for idx, size_bytes in enumerate(series["size_bytes"])
        if (min_bytes is None or size_bytes >= min_bytes)
        and (max_bytes is None or size_bytes <= max_bytes)
    ]
    return {key: [values[idx] for idx in indices] for key, values in series.items()}


def _collective_point_at_size(
    series: dict[str, list[float]],
    size_bytes: int,
) -> dict[str, float] | None:
    for idx, candidate in enumerate(series["size_bytes"]):
        if int(candidate) == int(size_bytes):
            return {
                key: float(values[idx]) if key != "size_bytes" else int(values[idx])
                for key, values in series.items()
            }
    return None


def _metric_limits(
    metric_keys: tuple[str, str],
    *,
    min_bytes: int | None = None,
    max_bytes: int | None = None,
) -> tuple[float, float]:
    values: list[float] = []
    for series in COLLECTIVE_SERIES.values():
        subset = _select_collective_points(
            series,
            min_bytes=min_bytes,
            max_bytes=max_bytes,
        )
        for key in metric_keys:
            values.extend(float(value) for value in subset.get(key, []))

    if not values:
        return (0.0, 1.0)

    lower = min(values)
    upper = max(values)
    span = max(upper - lower, upper * 0.03, 1e-6)
    return (max(0.0, lower - span * 0.18), upper + span * 0.18)


# ===================================================================
# Figure 1: GEMM — XTile vs cuBLAS (current state)
# ===================================================================
def fig1_gemm_performance():
    sizes = GEMM_BARS["sizes"]
    cublas = GEMM_BARS["cublas_tflops"]
    xtile = GEMM_BARS["xtile_tflops"]
    ratio = [(x / c * 100.0) if c > 0 else 0.0 for c, x in zip(cublas, xtile)]

    x = np.arange(len(sizes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(x - w / 2, cublas, w, label="cuBLAS (torch.matmul)", color=COLORS[0],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, xtile, w, label="XTile", color=COLORS[1],
           edgecolor="white", linewidth=0.5)

    # Annotate ratio above XTile bars
    max_bar = max(max(cublas), max(xtile), 1.0)
    label_offset = max_bar * 0.03
    for i, r in enumerate(ratio):
        color = COLORS[2] if r >= 90 else "#888888"
        weight = "bold" if r >= 90 else "normal"
        ax.text(x[i] + w / 2, xtile[i] + label_offset, f"{r:.0f}%",
                ha="center", va="bottom", fontsize=7, color=color, fontweight=weight)

    ax.set_ylabel("TFLOPS")
    ax.set_title(f"GEMM Performance: XTile vs cuBLAS ({_gemm_aggregation_label()})")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_ylim(0, max_bar * 1.22)
    ax.legend(loc="upper left")

    _save_with_footer(
        fig,
        "fig1_gemm_performance",
        benchmark_footer_text(
            GEMM_PAYLOAD,
            source_name="gemm_latest.json",
            include_command=False,
        ),
    )


# ===================================================================
# Figure 2: P2P Bandwidth vs Transfer Size
# ===================================================================
def fig2_p2p_bandwidth():
    sizes_mb = P2P_CURVE["sizes_mb"]
    read_bw = P2P_CURVE["read_bw"]
    write_bw = P2P_CURVE["write_bw"]
    ceiling_pct = P2P_CURVE["best_read_gbps"] / 300.0 * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(sizes_mb, read_bw, "o-", color=COLORS[0], label="Read", markersize=5, linewidth=1.5)
    ax.plot(sizes_mb, write_bw, "s--", color=COLORS[1], label="Write", markersize=5, linewidth=1.5)

    ax.fill_between(sizes_mb, read_bw, 300, alpha=0.06, color="gray")

    ax.axhline(300, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1.1, 305, "NV12 Peak (300 GB/s)", fontsize=7, color="red", va="bottom")

    ax.axhline(P2P_CURVE["best_read_gbps"], color=COLORS[0], linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(50, 240, f"XTile ceiling\n({ceiling_pct:.1f}%)", fontsize=7, color=COLORS[0], ha="center")

    ax.annotate(
        "Launch latency\ndominated",
        xy=(1.0, 31.8), xytext=(2.5, 80),
        fontsize=7, color="gray", fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Transfer Size (MB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("P2P Bandwidth Saturation")
    ax.set_ylim(0, 330)
    ax.set_xlim(0.8, 200)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"))
    ax.legend(loc="center right")

    _save_with_footer(
        fig,
        "fig2_p2p_bandwidth",
        benchmark_footer_text(
            P2P_PAYLOAD,
            source_name="p2p_latest.json",
            highlight_ops=("reduce_scatter",),
            include_command=False,
        ),
    )


# ===================================================================
# Figure 3: Pattern Overlap Comparison
# ===================================================================
def fig3_pattern_overlap():
    sizes = PATTERN_SPEEDUPS["sizes"]
    bulk = PATTERN_SPEEDUPS["bulk_sync"]
    fused = PATTERN_SPEEDUPS["fused_sequential"]
    prod_cons = PATTERN_SPEEDUPS["producer_consumer"]
    wg_spec = PATTERN_SPEEDUPS["wg_specialized"]

    x = np.arange(len(sizes))
    w = 0.2

    fig, ax = plt.subplots(figsize=(8.2, 3.8))

    ax.bar(x - 1.5 * w, bulk, w, label="bulk_sync", color=COLORS[0], edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5 * w, fused, w, label="fused_sequential", color=COLORS[1], edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5 * w, prod_cons, w, label="producer_consumer", color=COLORS[2], edgecolor="white", linewidth=0.5, alpha=0.6)
    ax.bar(x + 1.5 * w, wg_spec, w, label="wg_specialized", color=COLORS[3], edgecolor="white", linewidth=0.5, alpha=0.6)

    ax.axhline(1.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    pattern_series = [
        ("fused_sequential", fused, -0.5 * w, COLORS[1]),
        ("producer_consumer", prod_cons, 0.5 * w, COLORS[2]),
        ("wg_specialized", wg_spec, 1.5 * w, COLORS[3]),
    ]
    best_name = "fused_sequential"
    best_idx = 0
    best_offset = -0.5 * w
    best_color = COLORS[1]
    best_value = fused[0] if fused else 1.0
    for pattern_name, values, offset, color in pattern_series:
        for idx, value in enumerate(values):
            if value > best_value:
                best_name = pattern_name
                best_idx = idx
                best_offset = offset
                best_color = color
                best_value = value

    ymax = max(max(bulk), max(fused), max(prod_cons), max(wg_spec), best_value) * 1.22
    ymax = max(ymax, 1.30)
    best_on_right = best_idx >= (len(sizes) // 2)
    label_y = min(best_value + 0.12, ymax - 0.18)
    label_dx = -0.58 if best_on_right else 0.58
    label_x = x[best_idx] + best_offset + label_dx
    label_ha = "right" if best_on_right else "left"
    legend_loc = "upper left" if best_on_right else "upper right"
    legend_anchor = (0.01, 0.99) if best_on_right else (0.99, 0.99)

    ax.annotate(
        f"best stable\n{best_value:.3f}×",
        xy=(x[best_idx] + best_offset, best_value),
        xytext=(label_x, label_y),
        fontsize=8,
        fontweight="bold",
        color=best_color,
        arrowprops=dict(arrowstyle="-", color=best_color, lw=0.8),
        ha=label_ha,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
    )

    ax.set_xlabel("Problem Size (M × N × K)")
    ax.set_ylabel("Speedup vs bulk_sync")
    ax.set_title("Pattern Speedup vs bulk_sync (full 6-size rerun)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=7.5)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, ncol=2, fontsize=8, borderaxespad=0.3)

    _save_with_footer(
        fig,
        "fig3_pattern_overlap",
        benchmark_footer_text(
            PATTERN_PAYLOAD,
            source_name="pattern_overlap_latest.json",
            highlight_ops=("gemm_allscatter",),
            include_command=False,
        ),
    )


# ===================================================================
# Figure 4: 6-Layer Architecture Diagram
# ===================================================================
def fig4_architecture():
    # Fig 4 is a schematic architecture diagram, not a benchmark plot.
    layers = [
        ("User API",        "init(), XTileContext, SymmetricHeap"),
        ("Pattern Library", "BulkSync / FusedSeq / PC / WGSpec"),
        ("Core Primitives", "compute / memory / communication"),
        ("Synchronization", "atomic_* + tile_signal/wait"),
        ("Memory Mgmt",     "SymmetricHeap + translate_ptr"),
        ("HAL",             "HIP (AMD) / CUDA (NVIDIA)"),
    ]
    cmap = plt.cm.Blues
    n = len(layers)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.2, n * 1.7 + 0.8)
    ax.axis("off")

    box_w = 4.2
    box_h = 1.0
    x0 = 0.0
    gap = 0.4
    desc_x = x0 + box_w + 0.4

    for i, (name, desc) in enumerate(layers):
        y = (n - 1 - i) * (box_h + gap) + 0.5
        frac = 0.15 + 0.65 * (i / (n - 1))

        rect = mpatches.FancyBboxPatch(
            (x0, y), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=cmap(frac), edgecolor="#2C3E50", linewidth=1.0,
        )
        ax.add_patch(rect)

        text_color = "white" if frac > 0.55 else "#2C3E50"
        ax.text(x0 + box_w / 2, y + box_h / 2, name,
                fontsize=10, fontweight="bold", va="center", ha="center", color=text_color)

        ax.plot([x0 + box_w, desc_x - 0.1], [y + box_h / 2, y + box_h / 2],
                color="#BDC3C7", linewidth=0.6)
        ax.text(desc_x, y + box_h / 2, desc,
                fontsize=7.5, va="center", ha="left", color="#555555", fontstyle="italic")

        if i < n - 1:
            ax.annotate(
                "", xy=(x0 + box_w / 2, y - gap * 0.15),
                xytext=(x0 + box_w / 2, y + 0.02),
                arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.2),
            )

    ax.set_title("XTile 6-Layer Architecture", fontsize=13, pad=12)
    fig.tight_layout()
    _save(fig, "fig4_architecture")


# ===================================================================
# Figure 5: Roofline Model
# ===================================================================
def fig5_roofline():
    # Fig 5 is a derived view: theoretical roofline + fp16 GEMM experiment
    # points from the same benchmark JSON consumed by Fig 1.
    peak_flops = 756e12
    peak_bw = 2.04e12
    ridge_point = peak_flops / peak_bw

    sizes = GEMM_ROOFLINE["sizes"]
    intensity = []
    for n in sizes:
        flops = 2 * n * n * n
        bytes_moved = 2 * (n * n + n * n + n * n)
        intensity.append(flops / bytes_moved)

    cublas_tflops = GEMM_ROOFLINE["cublas_tflops"]
    xtile_tflops = GEMM_ROOFLINE["xtile_tflops"]

    fig, ax = plt.subplots(figsize=(3.5, 3))

    ai_range = np.logspace(-1, 4, 500)
    roofline = np.minimum(peak_flops / 1e12, ai_range * peak_bw / 1e12)
    ax.plot(ai_range, roofline, "-", color="gray", linewidth=1.5, alpha=0.7, label="Roofline")

    ax.axvline(ridge_point, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(ridge_point * 1.15, 20, f"Ridge\n({ridge_point:.0f})", fontsize=6, color="gray")

    ax.scatter(intensity, cublas_tflops, marker="o", s=50, color=COLORS[0],
               zorder=5, label="cuBLAS", edgecolors="white", linewidth=0.5)
    ax.scatter(intensity, xtile_tflops, marker="^", s=50, color=COLORS[1],
               zorder=5, label="XTile", edgecolors="white", linewidth=0.5)

    ratio_by_size = {
        n: (xt / cu * 100.0) if cu > 0 else 0.0
        for n, cu, xt in zip(sizes, cublas_tflops, xtile_tflops)
    }
    best_size = max(ratio_by_size, key=ratio_by_size.get) if ratio_by_size else None

    for i, n in enumerate(sizes):
        ax.annotate(f"{n}", (intensity[i], cublas_tflops[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=6, color=COLORS[0])
        suffix = f"\n({ratio_by_size[n]:.1f}%)" if n == best_size else ""
        ax.annotate(f"{n}{suffix}", (intensity[i], xtile_tflops[i]),
                    textcoords="offset points", xytext=(5, -10), fontsize=6, color=COLORS[1])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title(f"Roofline: GEMM fp16 ({_gemm_aggregation_label()})")
    ax.set_xlim(50, 5000)
    ax.set_ylim(10, 1000)
    ax.legend(loc="lower right", fontsize=7)

    _save_with_footer(
        fig,
        "fig5_roofline",
        benchmark_footer_text(
            GEMM_PAYLOAD,
            source_name="gemm_latest.json",
            include_command=False,
        ),
    )


# ===================================================================
# Figure 6: Communication-only collectives
# ===================================================================
def fig6_collective_comm_only():
    if not COLLECTIVE_SERIES:
        print("  Skipping fig6_collective_comm_only (no structured benchmark JSON)")
        return

    op_order = [
        ("allreduce", "AllReduce"),
        ("allgather", "AllGather"),
        ("scatter", "Scatter"),
        ("reduce_scatter", "ReduceScatter"),
        ("broadcast", "Broadcast"),
    ]
    small_sizes = [4 * 1024, 16 * 1024, 64 * 1024]
    anchor_size = 256 * 1024
    allreduce_large_sizes = [256 * 1024, 1024 * 1024, 2 * 1024 * 1024]

    fig = plt.figure(figsize=(14.0, 6.2))
    grid = fig.add_gridspec(
        2,
        10,
        height_ratios=[1.0, 1.05],
        hspace=0.42,
        wspace=0.42,
    )

    latency_axes = []
    for idx in range(len(op_order)):
        sharey = latency_axes[0] if latency_axes else None
        latency_axes.append(fig.add_subplot(grid[0, idx * 2:(idx + 1) * 2], sharey=sharey))
    anchor_ax = fig.add_subplot(grid[1, :6])
    sweep_ax = fig.add_subplot(grid[1, 6:])

    small_x = np.arange(len(small_sizes))
    small_labels = [_format_message_size_mib(size / (1024.0 * 1024.0)) for size in small_sizes]
    latency_ylim = _metric_limits(("xtile_ms", "nccl_ms"), max_bytes=COLLECTIVE_LATENCY_MAX_BYTES)
    latency_center = 0.5 * (latency_ylim[0] + latency_ylim[1])
    latency_span = max(latency_ylim[1] - latency_ylim[0], 0.20)
    latency_ylim = (
        max(0.0, latency_center - 0.5 * latency_span),
        latency_center + 0.5 * latency_span,
    )

    legend_handles = None
    legend_labels = None
    for idx, ((collective, title), ax) in enumerate(zip(op_order, latency_axes)):
        series = COLLECTIVE_SERIES.get(collective)
        if not series:
            ax.set_axis_off()
            continue

        small_points = [_collective_point_at_size(series, size) for size in small_sizes]
        if any(point is None for point in small_points):
            ax.set_axis_off()
            continue

        xtile_ms = [point["xtile_ms"] for point in small_points]
        nccl_ms = [point["nccl_ms"] for point in small_points]

        ax.plot(
            small_x,
            nccl_ms,
            "s--",
            color=COLORS[0],
            linewidth=1.35,
            markersize=4.5,
            label="NCCL",
        )
        ax.plot(
            small_x,
            xtile_ms,
            "o-",
            color=COLORS[1],
            linewidth=1.55,
            markersize=4.5,
            label="XTile",
        )

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        mean_delta_pct = (float(np.mean(xtile_ms)) / max(float(np.mean(nccl_ms)), 1e-9) - 1.0) * 100.0
        delta_direction = "higher" if mean_delta_pct >= 0 else "lower"
        ax.text(
            0.04,
            0.93,
            f"{abs(mean_delta_pct):.2f}% {delta_direction} vs NCCL",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=COLORS[1],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
        )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(small_x)
        ax.set_xticklabels(small_labels, fontsize=8)
        ax.set_ylim(*latency_ylim)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax.set_xlabel("Message Size")
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    anchor_points = []
    for collective, title in op_order:
        series = COLLECTIVE_SERIES.get(collective)
        point = _collective_point_at_size(series, anchor_size) if series else None
        if point is None:
            continue
        anchor_points.append((title, point["xtile_bw"], point["nccl_bw"]))

    if anchor_points:
        x = np.arange(len(anchor_points))
        width = 0.34
        anchor_labels = []
        xtile_bw = [item[1] for item in anchor_points]
        nccl_bw = [item[2] for item in anchor_points]
        for title, _, _ in anchor_points:
            anchor_labels.append("Reduce\nScatter" if title == "ReduceScatter" else title)
        anchor_ax.bar(
            x - width / 2,
            nccl_bw,
            width,
            color=COLORS[0],
            edgecolor="white",
            linewidth=0.5,
            label="NCCL",
        )
        anchor_ax.bar(
            x + width / 2,
            xtile_bw,
            width,
            color=COLORS[1],
            edgecolor="white",
            linewidth=0.5,
            label="XTile",
        )
        anchor_ax.set_title("Bandwidth at 256 KiB")
        anchor_ax.set_ylabel("Bandwidth (GB/s)")
        anchor_ax.set_xticks(x)
        anchor_ax.set_xticklabels(anchor_labels, fontsize=9)
        anchor_ax.set_xlabel("Collective")
        anchor_ax.set_ylim(0.0, max(max(xtile_bw), max(nccl_bw)) * 1.18)
        anchor_ax.grid(True, axis="y", alpha=0.25)
        anchor_ax.grid(False, axis="x")
        anchor_ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

        reduce_scatter_idx = next(
            (idx for idx, label in enumerate(anchor_labels) if label == "Reduce\nScatter"),
            None,
        )
        if reduce_scatter_idx is not None:
            y_offset = max(anchor_ax.get_ylim()[1] * 0.02, 0.0015)
            anchor_ax.text(
                x[reduce_scatter_idx] + width / 2,
                xtile_bw[reduce_scatter_idx] + y_offset,
                f"{xtile_bw[reduce_scatter_idx]:.5f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLORS[1],
            )

    allreduce_series = COLLECTIVE_SERIES.get("allreduce")
    if allreduce_series:
        sweep_points = [_collective_point_at_size(allreduce_series, size) for size in allreduce_large_sizes]
        if all(point is not None for point in sweep_points):
            sweep_x = np.arange(len(allreduce_large_sizes))
            sweep_labels = [
                _format_message_size_mib(size / (1024.0 * 1024.0))
                for size in allreduce_large_sizes
            ]
            sweep_ax.plot(
                sweep_x,
                [point["nccl_bw"] for point in sweep_points],
                "s--",
                color=COLORS[0],
                linewidth=1.35,
                markersize=4.8,
                label="NCCL",
            )
            sweep_ax.plot(
                sweep_x,
                [point["xtile_bw"] for point in sweep_points],
                "o-",
                color=COLORS[1],
                linewidth=1.55,
                markersize=4.8,
                label="XTile",
            )
            sweep_ax.set_xticks(sweep_x)
            sweep_ax.set_xticklabels(sweep_labels, fontsize=9)
            sweep_ax.set_xlabel("Message Size")
            sweep_ax.set_title("AllReduce Bandwidth Sweep")
            sweep_ax.set_ylim(0, max(point["nccl_bw"] for point in sweep_points) * 1.18)
            sweep_ax.grid(True, axis="y", alpha=0.25)
            sweep_ax.grid(False, axis="x")
            sweep_ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.952),
            ncol=2,
            columnspacing=1.4,
            handletextpad=0.6,
        )

    sns.despine(fig=fig)
    fig.suptitle("Communication-only Collectives: Latency and Bandwidth", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.18, wspace=0.55, hspace=0.5)

    footer = benchmark_footer_text(
        COLLECTIVE_PAYLOAD,
        source_name="collective_comm_only_latest.json",
        include_command=False,
    )
    if footer:
        fig.text(
            0.5,
            0.028,
            textwrap.fill(footer, width=120),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#555555",
        )
    _save(fig, "fig6_collective_comm_only")


# ===================================================================
# Figure 7: Communication-only speedup vs bulk_sync
# ===================================================================
def fig7_collective_bulk_sync():
    if not COLLECTIVE_BULK_SERIES:
        print("  Skipping fig7_collective_bulk_sync (no structured benchmark JSON)")
        return

    collective_order = ["allreduce", "allgather", "scatter", "reduce_scatter", "broadcast"]
    labels = [
        "allreduce",
        "allgather",
        "scatter",
        "reduce\nscatter",
        "broadcast",
    ]
    size_keys = sorted(
        {
            size
            for bucket in COLLECTIVE_BULK_SERIES.values()
            for size in bucket["size_kib"]
        }
    )

    x = np.arange(len(collective_order))
    width = 0.22 if len(size_keys) >= 3 else 0.28
    offsets = np.linspace(-width, width, len(size_keys))

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    best_collective = None
    best_size = None
    best_speedup = 0.0

    for idx, size_kib in enumerate(size_keys):
        values = []
        for collective in collective_order:
            bucket = COLLECTIVE_BULK_SERIES.get(collective, {})
            size_list = bucket.get("size_kib", [])
            speedup_list = bucket.get("speedup", [])
            speedup = next(
                (speedup_list[pos] for pos, size in enumerate(size_list) if size == size_kib),
                0.0,
            )
            values.append(speedup)
            if speedup > best_speedup:
                best_collective = collective
                best_size = size_kib
                best_speedup = speedup

        ax.bar(
            x + offsets[idx],
            values,
            width,
            label=_format_message_size_mib(size_kib / 1024.0),
            color=COLORS[idx],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(1.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.55)

    ymax = max(
        1.1,
        max(
            max(bucket["speedup"])
            for bucket in COLLECTIVE_BULK_SERIES.values()
            if bucket["speedup"]
        ) * 1.28,
    )
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Speedup vs bulk_sync")
    ax.set_xlabel("Communication-only Collective")
    ax.set_title("Communication-only XTile Speedup vs bulk_sync", fontsize=13)

    if best_collective is not None and best_size is not None:
        best_idx = collective_order.index(best_collective)
        best_on_right = best_idx >= (len(collective_order) // 2)
        text_dx = -0.42 if best_on_right else 0.42
        text_x = x[best_idx] + text_dx
        text_ha = "right" if best_on_right else "left"
        ax.annotate(
            f"best stable\n{best_speedup:.3f}×",
            xy=(x[best_idx], best_speedup),
            xytext=(text_x, min(best_speedup + 0.14, ymax - 0.16)),
            fontsize=8,
            fontweight="bold",
            color=COLORS[1],
            arrowprops=dict(arrowstyle="-", color=COLORS[1], lw=0.8),
            ha=text_ha,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
        )

    ax.legend(loc="upper left", ncol=min(3, len(size_keys)))

    _save_with_footer(
        fig,
        "fig7_collective_bulk_sync",
        benchmark_footer_text(
            COLLECTIVE_BULK_PAYLOAD,
            source_name="collective_bulk_sync_latest.json",
            include_command=False,
        ),
    )


# ===================================================================
# Main
# ===================================================================
def main():
    print("Generating XTile figures (current state)...")
    print(f"Output directory: {OUTDIR}\n")

    fig1_gemm_performance()
    fig2_p2p_bandwidth()
    fig3_pattern_overlap()
    fig4_architecture()
    fig5_roofline()
    fig6_collective_comm_only()
    fig7_collective_bulk_sync()

    print(f"\nAll 7 figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
