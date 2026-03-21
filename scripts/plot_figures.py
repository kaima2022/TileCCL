#!/usr/bin/env python3
"""XTile — publication-quality figures showing the latest validated status.

Generates 5 figures in Nature/Science style:
  1. GEMM: XTile vs cuBLAS (official helper, median of 3 repeats)
  2. P2P bandwidth vs transfer size (saturation curve)
  3. Pattern speedup vs bulk_sync (full 6-size rerun)
  4. 6-layer architecture diagram
  5. Roofline model (GEMM position analysis)

Output: figures/ directory — PDF (vector) + PNG (300 DPI)
"""

import os
import json
from pathlib import Path

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
PATTERN_BENCHMARK_JSON = REPO_ROOT / "figures" / "data" / "pattern_overlap_latest.json"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

# GEMM data: `tests/benchmarks/bench_gemm.py::_run_gemm_comparison`
# rerun 3 times on 2026-03-21, then median aggregated per size/dtype.
GEMM_BARS = {
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
    "cublas_tflops": [60.93, 81.33, 284.14, 363.06, 409.47, 470.00, 484.07, 487.90],
    "xtile_tflops": [27.92, 40.48, 157.26, 288.63, 408.75, 435.42, 381.98, 408.97],
    "ratio_pct": [45.8, 49.8, 55.0, 79.6, 99.8, 92.1, 78.3, 84.9],
}

# Pattern data fallback: `PYTHONPATH=. python tests/benchmarks/bench_patterns.py --warmup 3 --iters 10`
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
    "fused_sequential": [1.004, 0.876, 0.890, 0.779, 0.844, 0.959],
    "producer_consumer": [0.239, 0.856, 0.485, 0.763, 0.339, 0.360],
    "wg_specialized": [0.562, 0.817, 0.827, 0.849, 0.905, 0.854],
}


def _format_size_label(M, N, K):
    if M == N == K:
        return f"{M}³"
    return f"{M}×{N}\n×{K}"


def _load_pattern_speedups():
    if not PATTERN_BENCHMARK_JSON.exists():
        return _FALLBACK_PATTERN_SPEEDUPS, {"best_speedup_vs_bulk": 1.004}

    with PATTERN_BENCHMARK_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    environment = payload.get("environment", {})
    if environment.get("quick_mode") or len(payload.get("sizes", [])) < 6:
        return _FALLBACK_PATTERN_SPEEDUPS, {"best_speedup_vs_bulk": 1.004}

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
        return _FALLBACK_PATTERN_SPEEDUPS, {"best_speedup_vs_bulk": 1.004}

    return {
        "sizes": sizes,
        **series,
    }, payload.get("summary", {})


PATTERN_SPEEDUPS, PATTERN_SUMMARY = _load_pattern_speedups()


def _save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}.pdf + {name}.png")


# ===================================================================
# Figure 1: GEMM — XTile vs cuBLAS (current state)
# ===================================================================
def fig1_gemm_performance():
    sizes = GEMM_BARS["sizes"]
    cublas = GEMM_BARS["cublas_tflops"]
    xtile = GEMM_BARS["xtile_tflops"]
    ratio = GEMM_BARS["ratio_pct"]

    x = np.arange(len(sizes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(x - w / 2, cublas, w, label="cuBLAS (torch.matmul)", color=COLORS[0],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, xtile, w, label="XTile", color=COLORS[1],
           edgecolor="white", linewidth=0.5)

    # Annotate ratio above XTile bars
    for i, r in enumerate(ratio):
        color = COLORS[2] if r >= 90 else "#888888"
        weight = "bold" if r >= 90 else "normal"
        ax.text(x[i] + w / 2, xtile[i] + 12, f"{r:.0f}%",
                ha="center", va="bottom", fontsize=7, color=color, fontweight=weight)

    ax.set_ylabel("TFLOPS")
    ax.set_title("GEMM Performance: XTile vs cuBLAS (median of 3 runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_ylim(0, 580)
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, "fig1_gemm_performance")


# ===================================================================
# Figure 2: P2P Bandwidth vs Transfer Size
# ===================================================================
def fig2_p2p_bandwidth():
    sizes_mb = [1.0, 4.2, 16.8, 67.1, 134.2]
    read_bw  = [31.8, 111.7, 233.4, 245.9, 248.8]
    write_bw = [31.8, 109.6, 233.4, 245.1, 248.4]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(sizes_mb, read_bw, "o-", color=COLORS[0], label="Read", markersize=5, linewidth=1.5)
    ax.plot(sizes_mb, write_bw, "s--", color=COLORS[1], label="Write", markersize=5, linewidth=1.5)

    ax.fill_between(sizes_mb, read_bw, 300, alpha=0.06, color="gray")

    ax.axhline(300, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1.1, 305, "NV12 Peak (300 GB/s)", fontsize=7, color="red", va="bottom")

    ax.axhline(248.7, color=COLORS[0], linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(50, 240, "XTile ceiling\n(82.9%)", fontsize=7, color=COLORS[0], ha="center")

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

    fig.tight_layout()
    _save(fig, "fig2_p2p_bandwidth")


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

    ax.annotate(
        f"best stable\n{PATTERN_SUMMARY.get('best_speedup_vs_bulk', 1.004):.3f}×",
        xy=(0 - 0.5 * w, fused[0]),
        xytext=(0.15, 1.06),
        fontsize=8,
        fontweight="bold",
        color=COLORS[1],
        arrowprops=dict(arrowstyle="-", color=COLORS[1], lw=0.8),
        ha="center",
    )

    ax.set_xlabel("Problem Size (M × N × K)")
    ax.set_ylabel("Speedup vs bulk_sync")
    ax.set_title("Pattern Speedup vs bulk_sync (full 6-size rerun)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=7.5)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    _save(fig, "fig3_pattern_overlap")


# ===================================================================
# Figure 4: 6-Layer Architecture Diagram
# ===================================================================
def fig4_architecture():
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
    peak_flops = 756e12
    peak_bw = 2.04e12
    ridge_point = peak_flops / peak_bw

    sizes = [1024, 2048, 4096, 8192]
    intensity = []
    for n in sizes:
        flops = 2 * n * n * n
        bytes_moved = 2 * (n * n + n * n + n * n)
        intensity.append(flops / bytes_moved)

    cublas_tflops = [60.93, 284.14, 409.47, 484.07]
    xtile_tflops  = [27.92, 157.26, 408.75, 381.98]

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

    for i, n in enumerate(sizes):
        ax.annotate(f"{n}", (intensity[i], cublas_tflops[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=6, color=COLORS[0])
        suffix = "\n(99.8%)" if n == 4096 else ""
        ax.annotate(f"{n}{suffix}", (intensity[i], xtile_tflops[i]),
                    textcoords="offset points", xytext=(5, -10), fontsize=6, color=COLORS[1])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Roofline: GEMM (fp16)")
    ax.set_xlim(50, 5000)
    ax.set_ylim(10, 1000)
    ax.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig5_roofline")


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

    print(f"\nAll 5 figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
