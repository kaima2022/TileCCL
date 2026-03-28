#!/usr/bin/env bash
# TNCC Phase 3: Run all benchmarks and collect results.
#
# Usage:
#   ./run_benchmarks.sh          # full suite
#   ./run_benchmarks.sh --quick  # quick mode (fewer sizes)
#   ./run_benchmarks.sh p2p      # single benchmark
#
# Results are saved to results/ directory with timestamps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

RESULTS_DIR="results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
QUICK_FLAG=""

# Parse arguments
BENCH_TYPE="all"
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_FLAG="--quick" ;;
        p2p|collective|pattern|gemm|all) BENCH_TYPE="$arg" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  TNCC Benchmark Suite"
echo "  $(date)"
echo "  Mode: $BENCH_TYPE ${QUICK_FLAG:+(quick)}"
echo "============================================================"
echo

# Check GPU availability
python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}')"
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPU(s)"
echo

run_bench() {
    local name="$1"
    local script="$2"
    local extra_args="${3:-}"
    local outfile="$RESULTS_DIR/${name}_${TIMESTAMP}.txt"

    echo "============================================================"
    echo "  Running: $name"
    echo "  Output:  $outfile"
    echo "============================================================"

    if python3 -u "$script" $extra_args 2>&1 | tee "$outfile"; then
        echo "  -> $name: SUCCESS"
    else
        echo "  -> $name: FAILED (exit code $?)"
    fi
    echo
}

# Run benchmarks based on type
if [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "gemm" ]]; then
    run_bench "bench_gemm" "tests/benchmarks/bench_gemm.py"
fi

if [[ "$NUM_GPUS" -ge 2 ]]; then
    if [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "p2p" ]]; then
        run_bench "bench_p2p" "tests/benchmarks/bench_p2p_translate.py" "$QUICK_FLAG"
    fi

    if [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "collective" ]]; then
        run_bench "bench_collectives" "tests/benchmarks/bench_collectives.py"
    fi

    if [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "pattern" ]]; then
        run_bench "bench_patterns" "tests/benchmarks/bench_patterns.py" "$QUICK_FLAG"
    fi
else
    echo "(Skipping multi-GPU benchmarks: only $NUM_GPUS GPU detected)"
fi

echo "============================================================"
echo "  All benchmarks complete."
echo "  Results in: $RESULTS_DIR/"
echo "============================================================"
ls -la "$RESULTS_DIR"/*_${TIMESTAMP}.txt 2>/dev/null || echo "  (no output files)"
