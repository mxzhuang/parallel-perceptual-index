#!/usr/bin/env bash
# Prepare GNU Octave to run the OFFICIAL Ma + NIQE code (used to produce the reference
# values in reference/octave_reference.txt). Tested with Octave 8.4 on Ubuntu 24.04.
#
#   sudo apt-get install octave octave-image octave-signal octave-dev
#   bash reference/setup_octave.sh <work_dir>
#
# What it does
#   1. clones the official code: chaoma99/sr-metric (Ma) and roimehrez/PIRM2018
#      (PI evaluation, includes the LIVE NIQE release in utils/niqe_release)
#   2. builds the two MEX files for Octave (random forest prediction, pyramid pointOp).
#      Two one-line source fixes are needed for Octave's MEX headers:
#        - mex_regressionRF_predict.cpp passes 0 instead of mxREAL to mxCreateNumericMatrix
#        - pointOp.c includes <matrix.h>, which Octave does not ship (mex.h is enough)
#   3. rewrites model.mat so that the random-forest node status is stored as int8.
#      The official file keeps it as raw bytes inside a MATLAB char array (2 bytes per
#      char); Octave stores chars as 1 byte, so the MEX would read garbage otherwise.
#
# Octave lacks or differs from MATLAB in four functions used by the official code; the
# MATLAB-compatible versions in reference/octave_shims are put first on the path by
# run_reference.m / time_official_modules.m:
#   imresize (MATLAB bicubic with antialiasing), blkproc (removed from Octave's image
#   package), nanmean / nancov (we avoid loading Octave's statistics package, which
#   shadows the core var function).
# MATLAB users do not need any of this: run the .m scripts with shim_dir = ''.
set -euo pipefail
WORK=${1:-octave_ref}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$WORK" && cd "$WORK"

[ -d sr-metric ] || git clone --depth 1 https://github.com/chaoma99/sr-metric.git
[ -d PIRM2018 ] || git clone --depth 1 https://github.com/roimehrez/PIRM2018.git
[ -d niqe_release ] || cp -r PIRM2018/utils/niqe_release .

mkdir -p shims
cp "$HERE"/reference/octave_shims/*.m shims/

RF=sr-metric/external/randomforest-matlab/RF_Reg_C
sed 's/mxDOUBLE_CLASS,0)/mxDOUBLE_CLASS,mxREAL)/' $RF/src/mex_regressionRF_predict.cpp > $RF/src/mex_predict_oct.cpp
( cd $RF && mkoctfile --mex -DMATLAB -include cstdio -fpermissive \
    src/cokus.cpp src/mex_predict_oct.cpp src/reg_RF.cpp -o mexRF_predict )

sed 's/#include <matrix.h>.*//' sr-metric/external/matlabPyrTools/MEX/pointOp.c > shims/pointOp_oct.c
( cd shims && mkoctfile --mex pointOp_oct.c -o pointOp )

[ -f sr-metric/model_original.mat ] || cp sr-metric/model.mat sr-metric/model_original.mat
python3 "$HERE"/tools/convert_models.py --ma sr-metric/model_original.mat \
    --niqe niqe_release/modelparameters.mat --out "$HERE"/models \
    --octave-model sr-metric/model.mat

echo "Octave reference environment ready in $(pwd)"
echo "Run, for example:"
echo "  octave-cli --eval \"run_reference('$HERE/tests/images', 'ref.txt', '$(pwd)/sr-metric', '$(pwd)/niqe_release', '$(pwd)/shims')\""
