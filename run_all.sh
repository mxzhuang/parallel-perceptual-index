#!/usr/bin/env bash
# One command for the whole experiment campaign: set up, build, check, fetch data, measure,
# summarize, plot, and pack everything into one archive.
#
#   bash run_all.sh [options]
#
# Needs only g++ (C++17 with OpenMP), make, python3, git and curl or wget. No root access.
# Adapts to the number of CPUs of the machine. Safe to run again: finished measurements are
# kept and only missing ones are run.
#
# Result: results/pi-results-<hostname>-<YYYYmmdd-HHMMSS>.tar.gz
#         (CSV files, raw reports, logs, machine description, figures, tables; no datasets,
#         no models). The working copy of the same content stays in results/run-<hostname>/.
#
# Options
#   --quick               smoke test: bundled test images, 1 repetition, few thread counts, no
#                         downloads, reduced accuracy check; finishes in a few minutes and writes
#                         to results/run-<hostname>-quick/
#   --only LIST           comma-separated experiments to run, e.g. --only E2,E3 (default E1-E8)
#   --threads-max N       never use more than N threads
#   --reps N              repetitions per configuration (default 5)
#   --skip-faithful       skip the faithful sequential variant in E1 (it takes minutes)
#   --div2k-subset N      keep and use only the first N DIV2K validation images
#   --skip-optional-data  download DIV2K only (no benchmark sets, PIRM sets, published results)
#   --offline             no downloads; use data already present and the bundled test images
#   --with-pyiqa          E8: install pyiqa into the Python environment and time its PI
#   --with-perf           E8: perf stat / perf record through tools/profile.sh
#   --skip-check          do not run the accuracy checks (`make check`)
#   --no-pin              do not pin threads (default: OMP_PROC_BIND=close OMP_PLACES=cores)
#   --machine-id NAME     label of this machine in tables and figures (default: host name)
#   --results-dir DIR     working results directory (default results/run-<hostname>)
#   --force               repeat measurements that already have a report
#   -h, --help            this text
#
# Stages: 1 environment, 2 models, 3 build and check, 4 data, 5 experiments E1-E7,
# optional hooks E8 (CUDA, pyiqa, perf), 6 summaries, 7 figures and tables, 8 archive.
# A failed optional stage never stops the run; the final summary lists what ran, what was
# skipped and why. See docs/EXPERIMENTS.md.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

QUICK=0; ONLY=""; THREADS_MAX=0; REPS=0; SKIP_FAITHFUL=0; SUBSET=0; SKIP_OPTIONAL=0; OFFLINE=0
WITH_PYIQA=0; WITH_PERF=0; SKIP_CHECK=0; NO_PIN=0; MACHINE_ID=""; RES=""; FORCE=0
need() { [ $# -ge 2 ] || { echo "run_all.sh: $1 needs a value (see --help)" >&2; exit 1; }; }
ARGS="$*"
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --only) need "$@"; ONLY=$2; shift 2 ;;
    --threads-max) need "$@"; THREADS_MAX=$2; shift 2 ;;
    --reps) need "$@"; REPS=$2; shift 2 ;;
    --skip-faithful) SKIP_FAITHFUL=1; shift ;;
    --div2k-subset) need "$@"; SUBSET=$2; shift 2 ;;
    --skip-optional-data) SKIP_OPTIONAL=1; shift ;;
    --offline) OFFLINE=1; shift ;;
    --with-pyiqa) WITH_PYIQA=1; shift ;;
    --with-perf) WITH_PERF=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --no-pin) NO_PIN=1; shift ;;
    --machine-id) need "$@"; MACHINE_ID=$2; shift 2 ;;
    --results-dir) need "$@"; RES=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_all.sh: unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
done
for v in "$THREADS_MAX" "$REPS" "$SUBSET"; do
  case "$v" in ''|*[!0-9]*) echo "run_all.sh: numeric option expected, got '$v'" >&2; exit 1 ;; esac
done
ONLY=$(printf '%s' "$ONLY" | tr '[:lower:]' '[:upper:]' | tr -d ' ')
selected() { [ -z "$ONLY" ] || case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

HOST=$(hostname -s 2>/dev/null || uname -n)
HOST=$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9._-' '_')
[ -n "$MACHINE_ID" ] || MACHINE_ID=$HOST
STAMP=$(date +%Y%m%d-%H%M%S)
if [ -z "$RES" ]; then RES="results/run-$HOST"; [ "$QUICK" = 1 ] && RES="$RES-quick"; fi
mkdir -p "$RES/logs" "$RES/machine"
RES=$(cd "$RES" && pwd)
LOGS="$RES/logs"
STAGES="$RES/stages.tsv"
: > "$STAGES"
CHECK_STATUS="not run"
START=$(date +%s)

say() { printf '%s\n' "$*"; }
mark() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$STAGES"; printf '   -> %s: %s%s\n' "$1" "$2" "${3:+ ($3)}"; }

# ---- final steps: summaries, figures, archive, report (also used when a required stage fails)
finish() {
  local code=$1
  if [ -n "${PY:-}" ]; then
    {
      echo "machine_id=$MACHINE_ID"; echo "hostname=$HOST"; echo "quick=$QUICK"; echo "check=$CHECK_STATUS"
      echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
      echo "git_dirty=$(git status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
      echo "options=$ARGS"; echo "started=$STAMP"; echo "finished=$(date +%Y%m%d-%H%M%S)"; echo "exit_code=$code"
    } > "$RES/run_info.txt"
    "$PY" - "$RES" <<'EOF' || true
import json, sys
res = sys.argv[1]
info = dict(line.rstrip("\n").split("=", 1) for line in open(f"{res}/run_info.txt") if "=" in line)
info["quick"] = info.get("quick") == "1"
info["stages"] = [dict(zip(("stage", "status", "detail"), line.rstrip("\n").split("\t"))) for line in open(f"{res}/stages.tsv")]
json.dump(info, open(f"{res}/run_info.json", "w"), indent=1)
EOF
    rm -f "$RES/run_info.txt"
    if [ -d "$RES/raw" ]; then
      say "[6/8] Summaries"
      if "$PY" tools/summarize_results.py "$RES" > "$LOGS/06_summarize.log" 2>&1; then mark summaries ok "csv/, summary.txt"
      else mark summaries failed "see logs/06_summarize.log"; fi
      say "[7/8] Figures and tables"
      if "$PY" tools/plot_results.py "$RES" > "$LOGS/07_plot.log" 2>&1; then mark figures ok "figures/, tables/"
      else mark figures failed "see logs/07_plot.log; regenerate later with tools/plot_results.py"; fi
    fi
  fi
  say "[8/8] Archive"
  local name="pi-results-$HOST-$STAMP" archive
  archive="$(dirname "$RES")/$name.tar.gz"
  {
    echo "Run summary for $MACHINE_ID ($(date))"
    echo "options: ${ARGS:-(none)}"
    [ "$QUICK" = 1 ] && echo "QUICK RUN: smoke-test settings; not measurements to report."
    case "$CHECK_STATUS" in failed*) echo "*** ACCURACY CHECK FAILED: see logs/03_check.log ***" ;; esac
    echo
    awk -F'\t' '{ printf "  %-14s %-8s %s\n", $1, $2, $3 }' "$STAGES"
    echo "  elapsed        $(( ($(date +%s) - START) / 60 )) min"
  } > "$RES/RUN_SUMMARY.txt"
  rm -rf "$RES/raw/E7/work"
  if tar -C "$(dirname "$RES")" --transform "s,^$(basename "$RES"),$name," -czf "$archive" "$(basename "$RES")" 2> "$LOGS/08_archive.log"; then
    :
  else
    say "   could not write the archive (see $LOGS/08_archive.log)"; archive=""
  fi
  say
  say "================================ summary ================================"
  cat "$RES/RUN_SUMMARY.txt"
  [ -f "$RES/summary.txt" ] && say "  results overview: $RES/summary.txt"
  if [ -n "$archive" ]; then
    say
    say "Archive to send back: $archive ($(du -h "$archive" | cut -f1))"
  fi
  exit "$code"
}
die() { say "ERROR: $*"; mark "$STAGE" failed "$*"; finish 1; }

say "Parallel Perceptual Index: experiment run on $MACHINE_ID, $(nproc 2>/dev/null || echo '?') CPU(s)"
say "Results directory: $RES"
if [ "$QUICK" = 1 ]; then
  say "Quick mode: a smoke test of a few minutes; its numbers are not measurements."
else
  say "Expected time: roughly 45-75 minutes on a 20-core machine, plus downloads (about 0.8 GB)"
  say "and the faithful variant of E1 (skip it with --skip-faithful). A forecast for this machine"
  say "is printed when the experiments start."
fi
say

# ---- 1. environment ----------------------------------------------------------------------
STAGE=environment
say "[1/8] Environment"
missing=""
command -v g++ >/dev/null 2>&1 || missing="$missing g++"
command -v make >/dev/null 2>&1 || missing="$missing make"
command -v python3 >/dev/null 2>&1 || missing="$missing python3"
command -v git >/dev/null 2>&1 || missing="$missing git"
if [ -n "$missing" ]; then
  die "missing tool(s):$missing. Install them (Debian/Ubuntu: apt install build-essential python3 python3-venv git; without root: ask the administrator or use a conda/module environment) and run again"
fi
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 ||
  { [ "$OFFLINE" = 1 ] || [ "$QUICK" = 1 ] || die "neither curl nor wget found; install one of them or use --offline"; }
if ! printf '#include <omp.h>\n#include <optional>\nint main(){std::optional<int> x; return omp_get_max_threads() > 0 ? 0 : 1;}\n' |
     g++ -std=c++17 -fopenmp -x c++ - -o "$RES/.omp_test" > "$LOGS/01_env.log" 2>&1; then
  die "g++ cannot build a C++17 OpenMP program (see $LOGS/01_env.log); a g++ of version 7 or newer with libgomp is required"
fi
rm -f "$RES/.omp_test"

PKGS="numpy scipy pillow matplotlib"
have_pkgs() { "$1" -c 'import numpy, scipy, PIL, matplotlib' >/dev/null 2>&1; }
PY=""
{
  if [ ! -x .venv/bin/python ]; then
    echo "creating .venv"; python3 -m venv .venv || { echo "python3 -m venv failed"; rm -rf .venv; }
  fi
  if [ -x .venv/bin/python ]; then
    have_pkgs .venv/bin/python || .venv/bin/python -m pip install --disable-pip-version-check -q $PKGS || echo "pip install into .venv failed"
  fi
} >> "$LOGS/01_env.log" 2>&1
if [ -x .venv/bin/python ] && have_pkgs .venv/bin/python; then
  PY="$ROOT/.venv/bin/python"; export PATH="$ROOT/.venv/bin:$PATH"; PYENV=".venv"
else
  have_pkgs python3 || python3 -m pip install --user --disable-pip-version-check -q $PKGS >> "$LOGS/01_env.log" 2>&1 || true
  if have_pkgs python3; then PY=$(command -v python3); PYENV="system python (venv not usable)"; fi
fi
[ -n "$PY" ] || die "could not provide numpy, scipy, pillow and matplotlib (see $LOGS/01_env.log). Install python3-venv, or run: python3 -m pip install --user $PKGS"

M="$RES/machine"
{
  echo "$HOST" > "$M/hostname.txt"
  (lscpu || cat /proc/cpuinfo) > "$M/lscpu.txt" 2>&1
  (lscpu -e 2>/dev/null || true) > "$M/lscpu_topology.txt"
  (nproc; echo "nproc --all: $(nproc --all 2>/dev/null)") > "$M/nproc.txt" 2>&1
  head -n 12 /proc/meminfo > "$M/meminfo.txt" 2>&1
  uname -a > "$M/uname.txt" 2>&1
  g++ --version > "$M/gxx.txt" 2>&1
  make --version 2>&1 | head -n 1 > "$M/make.txt"
  { git rev-parse HEAD; git status --short --untracked-files=no; } > "$M/git.txt" 2>&1
  { "$PY" --version; "$PY" -c 'import numpy, scipy, PIL, matplotlib; print("numpy", numpy.__version__, "scipy", scipy.__version__, "pillow", PIL.__version__, "matplotlib", matplotlib.__version__)'; } > "$M/python.txt" 2>&1
  # virtual machine?
  if command -v systemd-detect-virt >/dev/null 2>&1; then systemd-detect-virt 2>/dev/null > "$M/virt.txt" || echo none > "$M/virt.txt"
  elif grep -qw hypervisor /proc/cpuinfo 2>/dev/null; then echo "virtual machine (hypervisor flag set)" > "$M/virt.txt"
  else echo "none detected" > "$M/virt.txt"; fi
  grep -i 'hypervisor' "$M/lscpu.txt" >> "$M/virt.txt" 2>/dev/null || true
  # GPU
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > "$M/nvidia-smi.txt" 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$M/gpu.txt" 2>/dev/null || true
  else echo "no NVIDIA GPU detected" > "$M/nvidia-smi.txt"; fi
  if command -v nvcc >/dev/null 2>&1; then nvcc --version > "$M/nvcc.txt" 2>&1; else echo "nvcc not found" > "$M/nvcc.txt"; fi
  {
    echo "Threads are set with --threads on the command line of every run."
    if [ "$NO_PIN" = 1 ]; then echo "Pinning: none (--no-pin)"
    else
      echo "Single-image runs (E1-E5): OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false"
      echo "Image-set runs (E6, E7):   OMP_PROC_BIND=spread,close OMP_PLACES=cores OMP_DYNAMIC=false"
    fi
    echo "The exact environment of each run is stored in its .meta.json file."
    echo "OpenMP variables set by the caller (removed for the runs):"
    env | grep -E '^(OMP_|GOMP_|KMP_)' || echo "  (none)"
  } > "$M/omp_env.txt"
} >> "$LOGS/01_env.log" 2>&1 || true
mark environment ok "python: $PYENV; $(head -n 1 "$M/gxx.txt")"

# ---- 2. models ---------------------------------------------------------------------------
STAGE=models
say "[2/8] Models"
if [ -s models/ma_model.bin ] && [ -s models/niqe_params.bin ]; then
  mark models present "models/*.bin"
elif bash tools/fetch_models.sh > "$LOGS/02_models.log" 2>&1; then
  mark models ok "downloaded and converted"
else
  die "tools/fetch_models.sh failed (see $LOGS/02_models.log); it needs access to github.com"
fi

# ---- 3. build and check ------------------------------------------------------------------
STAGE=build
say "[3/8] Build and accuracy check"
if make -j"$(nproc 2>/dev/null || echo 2)" > "$LOGS/03_build.log" 2>&1; then
  mark build ok "pi_eval, pi_eval_seq"
else
  tail -n 20 "$LOGS/03_build.log"
  die "make failed (see $LOGS/03_build.log)"
fi
if [ "$SKIP_CHECK" = 1 ]; then
  CHECK_STATUS="skipped"; mark check skipped "--skip-check"
else
  if [ "$QUICK" = 1 ]; then  # the steps of `make check` with the short list of parallel configurations
    check_cmd() {
      make tests/test_ggd && ./tests/test_ggd &&
        ./pi_eval_seq --dump tests/cpp_dump.txt tests/images/*_y.png > /dev/null &&
        python3 tests/compare.py reference/octave_reference.txt tests/cpp_dump.txt &&
        python3 tests/check_parallel.py --quick --threads 1,2
    }
    kind="reduced check (--quick)"
  else
    check_cmd() { make check; }
    kind="make check"
  fi
  if check_cmd > "$LOGS/03_check.log" 2>&1; then
    CHECK_STATUS="passed"; mark check ok "$kind passed"; rm -f "$RES/CHECK_FAILED.txt"
  else
    CHECK_STATUS="failed"
    { echo "The accuracy check ($kind) FAILED on this machine. Timings may still be valid,"
      echo "but scores must not be trusted until logs/03_check.log has been examined."; } > "$RES/CHECK_FAILED.txt"
    say "   *** ACCURACY CHECK FAILED; continuing. See $LOGS/03_check.log ***"
    mark check FAILED "$kind failed; see logs/03_check.log and CHECK_FAILED.txt"
  fi
fi

# ---- 4. data -----------------------------------------------------------------------------
STAGE=data
say "[4/8] Data"
data_args=(--python "$PY")
[ "$SUBSET" -gt 0 ] && data_args+=(--div2k-subset "$SUBSET")
[ "$SKIP_OPTIONAL" = 1 ] && data_args+=(--skip-optional)
if [ "$OFFLINE" = 1 ] || [ "$QUICK" = 1 ]; then data_args+=(--offline); fi
if bash tools/fetch_data.sh "${data_args[@]}" > "$LOGS/04_data.log" 2>&1; then
  cp data/fetch_status.tsv "$RES/data_status.tsv"
  cp data/inputs/manifest*.json "$RES/" 2>/dev/null || true
  skipped=$(awk -F'\t' '$2 == "skipped" || $2 == "failed" { printf "%s%s", sep, $1; sep = ", " }' data/fetch_status.tsv)
  mark data ok "${skipped:+not available: $skipped; }details in data_status.tsv"
else
  tail -n 5 "$LOGS/04_data.log"
  die "tools/fetch_data.sh could not prepare the inputs (see $LOGS/04_data.log)"
fi

# ---- 5. experiments E1-E7 ----------------------------------------------------------------
STAGE=experiments
say "[5/8] Experiments"
core=""
for e in E1 E2 E3 E4 E5 E6 E7; do if selected "$e"; then core="$core${core:+,}$e"; fi; done
if [ -n "$core" ]; then
  exp_args=(--results "$RES" --only "$core")
  [ "$QUICK" = 1 ] && exp_args+=(--quick)
  [ "$REPS" -gt 0 ] && exp_args+=(--reps "$REPS")
  [ "$THREADS_MAX" -gt 0 ] && exp_args+=(--threads-max "$THREADS_MAX")
  [ "$SKIP_FAITHFUL" = 1 ] && exp_args+=(--skip-faithful)
  [ "$NO_PIN" = 1 ] && exp_args+=(--no-pin)
  [ "$FORCE" = 1 ] && exp_args+=(--force)
  set +e
  "$PY" tools/run_experiments.py "${exp_args[@]}" 2>&1 | tee "$LOGS/05_experiments.log"
  rc=${PIPESTATUS[0]}
  set -e
  case "$rc" in
    0) mark experiments ok "$core" ;;
    3) mark experiments partial "$core; some runs failed, see logs/05_experiments.log and experiment_status.json" ;;
    *) mark experiments failed "exit $rc, see logs/05_experiments.log" ;;
  esac
else
  mark experiments skipped "none of E1-E7 selected"
fi

# ---- E8: optional hooks ------------------------------------------------------------------
if selected E8; then
  # (i) CUDA extension, when it is part of the checkout and a GPU toolchain exists
  if [ ! -f cuda/Makefile ] || [ ! -f cuda/run_experiments.sh ]; then mark E8-cuda skipped "no cuda/ directory in this checkout"
  elif ! command -v nvcc >/dev/null 2>&1; then mark E8-cuda skipped "nvcc not found"
  elif ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then mark E8-cuda skipped "no usable NVIDIA GPU"
  elif bash cuda/run_experiments.sh "$RES/cuda" > "$LOGS/08_cuda.log" 2>&1; then mark E8-cuda ok "cuda/"
  else mark E8-cuda failed "see logs/08_cuda.log"; fi

  # (ii) pyiqa comparison
  if [ "$WITH_PYIQA" = 0 ]; then mark E8-pyiqa skipped "off by default (--with-pyiqa)"
  elif [ "$PYENV" != ".venv" ]; then mark E8-pyiqa skipped "needs the .venv environment"
  elif { "$PY" -c 'import pyiqa' 2>/dev/null || "$PY" -m pip install --disable-pip-version-check -q pyiqa; } > "$LOGS/08_pyiqa.log" 2>&1 &&
       "$PY" tools/pyiqa_compare.py --manifest "data/inputs/$([ "$QUICK" = 1 ] && echo manifest_quick.json || echo manifest.json)" \
            --out "$RES/pyiqa" --reps "$([ "$QUICK" = 1 ] && echo 1 || echo 3)" >> "$LOGS/08_pyiqa.log" 2>&1; then
    mark E8-pyiqa ok "pyiqa/"
  else mark E8-pyiqa failed "see logs/08_pyiqa.log"; fi

  # (iii) hardware counters
  PERF=$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | tail -n 1 || true)
  if [ "$WITH_PERF" = 0 ]; then mark E8-perf skipped "off by default (--with-perf)"
  elif [ -z "$PERF" ] || ! "$PERF" stat -e task-clock true >/dev/null 2>&1; then mark E8-perf skipped "perf is not installed or not permitted (kernel.perf_event_paranoid)"
  else
    input=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["inputs"][0]["path"])' \
            "data/inputs/$([ "$QUICK" = 1 ] && echo manifest_quick.json || echo manifest.json)")
    if PERF="$PERF" bash tools/profile.sh --pirm -n "$([ "$QUICK" = 1 ] && echo 1 || echo 3)" -o "$RES/perf" "$input" > "$LOGS/08_perf.log" 2>&1; then
      rm -f "$RES"/perf/*/perf.data "$RES"/perf/*/gmon.out
      mark E8-perf ok "perf/"
    else mark E8-perf failed "see logs/08_perf.log"; fi
  fi
else
  mark E8 skipped "not selected"
fi

finish 0
