#!/usr/bin/env bash
# Profile pi_eval with the tools introduced in HW0 (time, gprof, perf) plus repeated
# per-module timing. Results are written to profile_results/<variant>/.
#
#   bash tools/profile.sh [--faithful] [-n RUNS] [-c CORE] image1.png [image2.png ...]
#
#   --faithful  profile the faithful port instead of the optimized baseline (much slower;
#               use few images and -n 1)
#   -n RUNS     repetitions of the per-module timing; the median is reported (default 5)
#   -c CORE     CPU core to pin the program to with taskset (default 0)
#
# Steps
#   1. time        total wall-clock, user and system time, peak memory
#   2. --timing    per-module time (M1..M8), RUNS times, median per module
#   3. gprof       flat profile and call graph (pi_eval_gprof, built with -pg)
#   4. perf record hottest functions, including library code such as libm
#   5. perf stat   hardware events (cycles, instructions, cache and branch misses);
#                  needs hardware counters, which most virtual machines do not expose
set -uo pipefail

VARIANT=baseline
RUNS=5
CORE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --faithful) VARIANT=faithful; shift ;;
    -n) RUNS=$2; shift 2 ;;
    -c) CORE=$2; shift 2 ;;
    -*) echo "unknown option $1" >&2; exit 1 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [--faithful] [-n RUNS] [-c CORE] images..." >&2; exit 1; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGES=()
for f in "$@"; do IMAGES+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"); done
FLAGS=(--models "$ROOT/models")
[ "$VARIANT" = faithful ] && FLAGS+=(--faithful)

PIN=()
command -v taskset >/dev/null && PIN=(taskset -c "$CORE")
PERF=${PERF:-perf}
# fall back to a perf binary that matches an installed linux-tools package
if ! "$PERF" --version >/dev/null 2>&1; then
  PERF=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | tail -1)
fi

make -C "$ROOT" -s pi_eval pi_eval_gprof || exit 1
OUT="$ROOT/profile_results/$VARIANT"
mkdir -p "$OUT"
cd "$OUT" || exit 1
{
  echo "date:     $(date '+%Y-%m-%d %H:%M')"
  echo "host:     $(hostname)"
  echo "cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
  echo "compiler: $(${CXX:-g++} --version | head -1)"
  echo "variant:  $VARIANT, pinned to core $CORE, ${#IMAGES[@]} image(s), $RUNS run(s)"
} > summary.txt

# warm-up run (not measured), so that the model file is in the page cache
"${PIN[@]}" "$ROOT/pi_eval" "${FLAGS[@]}" "${IMAGES[0]}" > /dev/null

echo "[1/5] time"
if [ -x /usr/bin/time ]; then
  /usr/bin/time -f "elapsed %e s, user %U s, system %S s, max RSS %M KB" -o time.txt \
    "${PIN[@]}" "$ROOT/pi_eval" "${FLAGS[@]}" "${IMAGES[@]}" > /dev/null
else
  { TIMEFORMAT='elapsed %R s, user %U s, system %S s'; time "${PIN[@]}" "$ROOT/pi_eval" "${FLAGS[@]}" "${IMAGES[@]}" > /dev/null; } 2> time.txt
fi
{ echo; echo "== time"; cat time.txt; } >> summary.txt

echo "[2/5] per-module timing, $RUNS run(s)"
: > timing_runs.txt
for i in $(seq 1 "$RUNS"); do
  "${PIN[@]}" "$ROOT/pi_eval" "${FLAGS[@]}" --timing "${IMAGES[@]}" | grep '^Total time per module' >> timing_runs.txt
done
python3 - timing_runs.txt >> summary.txt <<'EOF'
import re, statistics, sys
runs = [[float(x) for x in re.findall(r"M\d ([0-9.]+) \(", line)] for line in open(sys.argv[1])]
io = statistics.median(float(re.search(r"I/O ([0-9.]+)", l).group(1)) for l in open(sys.argv[1]))
med = [statistics.median(r[m] for r in runs) for m in range(8)]
total = sum(med)
names = ["spatial pyramid", "block-DCT statistics", "patch SVD", "steerable pyramid",
         "divisive normalization", "subband statistics", "NIQE", "random forests"]
print(f"\n== per-module time, median of {len(runs)} run(s)")
for m in range(8):
    lo, hi = min(r[m] for r in runs), max(r[m] for r in runs)
    print(f"M{m+1} {names[m]:24s} {med[m]:10.3f} s  {100*med[m]/total:5.1f}%   (min {lo:.3f}, max {hi:.3f})")
print(f"   {'sum of M1-M8':24s} {total:10.3f} s")
print(f"   {'I/O (models, images)':24s} {io:10.3f} s   (M1-M8 + I/O should be close to the elapsed time above)")
EOF

echo "[3/5] gprof"
rm -f gmon.out
"${PIN[@]}" "$ROOT/pi_eval_gprof" "${FLAGS[@]}" "${IMAGES[@]}" > /dev/null
gprof -b "$ROOT/pi_eval_gprof" gmon.out > gprof.txt 2>/dev/null
{ echo; echo "== gprof flat profile (top 15; time in shared libraries such as libm is not counted)"
  sed -n '/^Flat profile/,/^$/p;/^ *%/,/^$/p' gprof.txt | sed -n '1,20p'; } >> summary.txt

echo "[4/5] perf record"
if [ -n "$PERF" ] && "$PERF" record -q -g -o perf.data -- "${PIN[@]}" "$ROOT/pi_eval" "${FLAGS[@]}" "${IMAGES[@]}" > /dev/null 2> perf_record.log; then
  "$PERF" report -i perf.data --stdio --no-children --sort symbol -g none 2>/dev/null \
    | grep -v '^$' | sed -E 's/[[:space:]-]+$//' > perf_report.txt
  { echo; echo "== perf record: hottest functions (top 15)"
    grep -E '^ +[0-9.]+%' perf_report.txt | head -15; } >> summary.txt
else
  { echo; echo "== perf record: not available (see perf_record.log)"; } >> summary.txt
fi

echo "[5/5] perf stat"
if [ -n "$PERF" ]; then
  "$PERF" stat -e cycles,instructions,cache-references,cache-misses,branch-misses -o perf_stat.txt \
    -- "${PIN[@]}" "$ROOT/pi_eval" "${FLAGS[@]}" "${IMAGES[@]}" > /dev/null 2>&1
  { echo; echo "== perf stat (\"not supported\" means this machine exposes no hardware counters)"
    grep -E 'cycles|instructions|cache|branch|elapsed' perf_stat.txt; } >> summary.txt
fi

echo
cat summary.txt
echo
echo "full results in $OUT"
