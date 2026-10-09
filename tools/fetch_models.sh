#!/usr/bin/env bash
# Download the official model files and convert them into models/ for the C++ port.
# The converted files (about 80 MB) are not kept in git.
#
#   pip install numpy scipy
#   bash tools/fetch_models.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 https://github.com/chaoma99/sr-metric.git "$TMP/sr-metric"
git clone --depth 1 https://github.com/roimehrez/PIRM2018.git "$TMP/PIRM2018"

mkdir -p "$HERE/models"
python3 "$HERE/tools/convert_models.py" \
    --ma "$TMP/sr-metric/model.mat" \
    --niqe "$TMP/PIRM2018/utils/niqe_release/modelparameters.mat" \
    --out "$HERE/models"
ls -l "$HERE/models"
