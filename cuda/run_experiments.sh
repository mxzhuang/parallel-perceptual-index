#!/usr/bin/env bash
# Builds the CUDA extension and times it on the fixed inputs of the experiment campaign.
#
#   bash cuda/run_experiments.sh OUT_DIR [--reps N] [--threads N]
#
# For every image in data/inputs/ (or the bundled test images when that directory is empty)
# it writes OUT_DIR/<image>_t<threads>.report with the per-stage times of pi_eval_cuda and
# its differences from the CPU modules, for all CPU threads and for one CPU thread.
# Called by run_all.sh when nvcc and a GPU are present; can also be run by hand.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash cuda/run_experiments.sh OUT_DIR [--reps N] [--threads N]" >&2
    exit 1
fi
OUT=$1
shift
REPS=5
THREADS=$(nproc)
while [ $# -gt 0 ]; do
    case "$1" in
        --reps) REPS=$2; shift 2 ;;
        --threads) THREADS=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done

HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
mkdir -p "$OUT"

command -v nvcc >/dev/null || { echo "nvcc not found: skipping the CUDA extension" >&2; exit 3; }
command -v nvidia-smi >/dev/null && nvidia-smi >"$OUT/nvidia-smi.txt" 2>&1 || true
nvcc --version >"$OUT/nvcc-version.txt" 2>&1 || true

make -C cuda >"$OUT/build.log" 2>&1 || { echo "CUDA build failed, see $OUT/build.log" >&2; exit 4; }

shopt -s nullglob
INPUTS=(data/inputs/*.png)
PIRM=(--pirm)
if [ ${#INPUTS[@]} -eq 0 ]; then
    INPUTS=(tests/images/coffee_y.png tests/images/rocket_y.png)
    PIRM=()
fi

export OMP_PROC_BIND=close OMP_PLACES=cores
status=0
for t in "$THREADS" 1; do
    for img in "${INPUTS[@]}"; do
        name=$(basename "${img%.*}")_t$t
        if ./pi_eval_cuda "${PIRM[@]}" --threads "$t" --repeat "$REPS" --report "$OUT/$name.report" "$img" >"$OUT/$name.log" 2>&1; then
            echo "cuda: $name ok"
        else
            echo "cuda: $name FAILED, see $OUT/$name.log" >&2
            status=5
        fi
    done
    [ "$THREADS" = 1 ] && break
done
exit $status
