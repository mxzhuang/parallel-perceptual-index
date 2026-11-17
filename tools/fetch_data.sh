#!/usr/bin/env bash
# Download and prepare the image data used by the experiments. Everything goes into the
# untracked data/ directory; running the script again only does what is still missing.
#
#   bash tools/fetch_data.sh [options]
#
#   --div2k-subset N   keep only the first N DIV2K validation images (default: all 100)
#   --skip-optional    only DIV2K; no benchmark sets, PIRM sets or published results
#   --offline          no downloads at all; use what is already in data/ and the bundled
#                      test images (the pipeline still completes)
#   --data DIR         data directory (default: data)
#   --python EXE       Python interpreter with Pillow (default: python3)
#   -h, --help
#
# Layout
#   data/downloads/                 archives as downloaded
#   data/hr/<set>/                  original (HR) images: div2k_val, Set5, Set14, BSD100,
#                                   Urban100, pirm_*, bundled
#   data/sr/<set>/<method>/         images to score; bicubic_x4 is generated here by
#                                   tools/make_inputs.py, other methods are published results
#   data/inputs/                    the two fixed inputs of the scaling experiments
#   data/fetch_status.tsv           one line per item: name, status, detail
#
# Sources (required: DIV2K; every other item is optional and skipped with a logged reason)
#   DIV2K validation HR   http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
#   Set5, Set14, BSD100,  https://huggingface.co/datasets/eugenesiow/<set>  (data/<set>_HR.tar.gz)
#   Urban100                fallback: https://github.com/jbhuang0604/SelfExSR  (data/<set>/image_SRF_4)
#   PIRM 2018 sets        Dropbox archive linked from https://github.com/roimehrez/PIRM2018
#   PIRM 2018 baselines   https://pirm2018.org/files/PIRM_baseline_results.zip (linked from https://pirm.github.io)
#   ESRGAN                sample output stored in https://github.com/xinntao/ESRGAN; the complete
#                         result sets are only on Google Drive and are fetched when `gdown` is installed
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA="$ROOT/data"
PY=python3
SUBSET=0
OPTIONAL=1
OFFLINE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --div2k-subset) SUBSET=${2:?--div2k-subset needs a number}; shift 2 ;;
    --skip-optional) OPTIONAL=0; shift ;;
    --offline) OFFLINE=1; shift ;;
    --data) DATA=${2:?}; shift 2 ;;
    --python) PY=${2:?}; shift 2 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "fetch_data.sh: unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
done
case "$SUBSET" in ''|*[!0-9]*) echo "fetch_data.sh: --div2k-subset needs a non-negative integer" >&2; exit 1 ;; esac

DL="$DATA/downloads"
mkdir -p "$DL" "$DATA/hr" "$DATA/sr" "$DATA/inputs"
STATUS="$DATA/fetch_status.tsv"
: > "$STATUS"
MAKE_INPUTS=("$PY" "$ROOT/tools/make_inputs.py")

note() {  # note ITEM STATUS DETAIL
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$STATUS"
  printf '  [%-7s] %-22s %s\n' "$2" "$1" "$3"
}
count_images() { find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.bmp' -o -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l; }

# download URL FILE [MAX_SECONDS]: resumable, with retries; returns non-zero on failure
download() {
  local url=$1 out=$2 limit=${3:-1800}
  [ -s "$out" ] && return 0
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 5 --connect-timeout 20 --max-time "$limit" -C - -sS -o "$out.part" "$url" || return 1
  elif command -v wget >/dev/null 2>&1; then
    timeout "$limit" wget -q -c -t 3 -T 30 -O "$out.part" "$url" || return 1
  else
    echo "neither curl nor wget is installed" >&2; return 1
  fi
  [ -s "$out.part" ] || return 1
  mv "$out.part" "$out"
}
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-16; else echo unknown; fi; }

echo "Preparing data in $DATA"

# ---- DIV2K validation HR (required) ------------------------------------------------------
want=100; [ "$SUBSET" -gt 0 ] && [ "$SUBSET" -lt 100 ] && want=$SUBSET
have=$(count_images "$DATA/hr/div2k_val")
if [ "$have" -ge "$want" ]; then
  note div2k_val present "$have image(s)"
elif [ "$OFFLINE" = 1 ]; then
  note div2k_val skipped "offline mode; the bundled test images are used instead"
else
  zip="$DL/DIV2K_valid_HR.zip"
  ok=0
  for url in http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip; do
    echo "  downloading $url (about 450 MB)"
    if download "$url" "$zip" 7200; then ok=1; break; fi
  done
  if [ "$ok" = 1 ] && "${MAKE_INPUTS[@]}" extract "$zip" "$DATA/hr/div2k_val" --flat --images-only --first "$want" --expect "$want"; then
    note div2k_val ok "$(count_images "$DATA/hr/div2k_val") image(s); archive $(du -h "$zip" | cut -f1), sha256 $(sha256 "$zip")..."
  else
    [ "$ok" = 1 ] && rm -f "$zip"   # damaged archive: fetch it again next time
    note div2k_val failed "download or extraction failed; the bundled test images are used instead"
  fi
fi

# ---- classic benchmark sets (optional) ---------------------------------------------------
selfexsr_clone() {  # sparse, blob-less clone: only the x4 folders are downloaded
  local dir="$DL/SelfExSR"
  [ -d "$dir/.git" ] && return 0
  rm -rf "$dir"
  timeout 900 git clone -q --depth 1 --filter=blob:none --sparse https://github.com/jbhuang0604/SelfExSR.git "$dir" || { rm -rf "$dir"; return 1; }
}
fetch_benchmark() {
  local set=$1 minimum=$2 have
  have=$(count_images "$DATA/hr/$set")
  if [ "$have" -ge "$minimum" ]; then note "$set" present "$have image(s)"; return; fi
  if [ "$OFFLINE" = 1 ]; then note "$set" skipped "offline mode"; return; fi
  local tar="$DL/${set}_HR.tar.gz"
  if download "https://huggingface.co/datasets/eugenesiow/$set/resolve/main/data/${set}_HR.tar.gz" "$tar" 1800 &&
     "${MAKE_INPUTS[@]}" extract "$tar" "$DATA/hr/$set" --flat --images-only --expect "$minimum"; then
    note "$set" ok "$(count_images "$DATA/hr/$set") image(s) from huggingface.co/datasets/eugenesiow/$set"
    return
  fi
  rm -f "$tar" "$tar.part"
  if selfexsr_clone && (cd "$DL/SelfExSR" && timeout 1800 git sparse-checkout add "data/$set/image_SRF_4" >/dev/null 2>&1) &&
     "${MAKE_INPUTS[@]}" collect --src "$DL/SelfExSR/data/$set/image_SRF_4" --data "$DATA" --kind hr --prefix "$set" --match '_HR\.png$' >/dev/null &&
     [ "$(count_images "$DATA/hr/$set")" -ge "$minimum" ]; then
    note "$set" ok "$(count_images "$DATA/hr/$set") image(s) from github.com/jbhuang0604/SelfExSR (x4 HR crops)"
    return
  fi
  note "$set" skipped "not reachable (huggingface.co and github.com/jbhuang0604/SelfExSR both failed)"
}

# an optional archive whose inner layout is not fixed: unpack it, then turn every directory
# that holds images into one image set
fetch_tree() {  # fetch_tree ITEM URL FILE KIND PREFIX [SKIP_DIRS_REGEX]
  local item=$1 url=$2 file=$3 kind=$4 prefix=$5 skip=${6:-}
  local marker="$DATA/.done_$item"
  if [ -e "$marker" ]; then note "$item" present "already unpacked"; return; fi
  if [ "$OFFLINE" = 1 ]; then note "$item" skipped "offline mode"; return; fi
  if ! download "$url" "$DL/$file" 3600; then rm -f "$DL/$file.part"; note "$item" skipped "download failed: $url"; return; fi
  local args=(collect --src "$DATA/extracted/$item" --data "$DATA" --kind "$kind" --prefix "$prefix")
  [ -n "$skip" ] && args+=(--skip-dirs "$skip")
  if "${MAKE_INPUTS[@]}" extract "$DL/$file" "$DATA/extracted/$item" --images-only >/dev/null &&
     "${MAKE_INPUTS[@]}" "${args[@]}"; then
    touch "$marker"
    note "$item" ok "unpacked from $url"
  else
    rm -f "$DL/$file"
    note "$item" skipped "the downloaded file is not a usable archive: $url"
  fi
}

if [ "$OPTIONAL" = 1 ]; then
  fetch_benchmark Set5 5
  fetch_benchmark Set14 14
  fetch_benchmark BSD100 100
  fetch_benchmark Urban100 100

  # PIRM 2018: self-validation, validation and test images (low-resolution folders are ignored)
  fetch_tree pirm_dataset "https://www.dropbox.com/s/9xofxxsn6nwum2p/PIRM_dataset.zip?dl=1" PIRM_dataset.zip hr pirm '(4x|down|lr|low)'
  # PIRM 2018 baseline outputs (EDSR, EnhanceNet, CX) on the test set
  fetch_tree pirm_baselines "https://pirm2018.org/files/PIRM_baseline_results.zip" PIRM_baseline_results.zip sr pirm_test

  # ESRGAN: the one output image kept in the official repository
  dest="$DATA/sr/esrgan_sample/ESRGAN"
  if [ -s "$dest/baboon_ESRGAN.png" ]; then
    note esrgan_sample present "1 image"
  elif [ "$OFFLINE" = 1 ]; then
    note esrgan_sample skipped "offline mode"
  elif mkdir -p "$dest" && download "https://raw.githubusercontent.com/xinntao/ESRGAN/master/results/baboon_ESRGAN.png" "$dest/baboon_ESRGAN.png" 300; then
    note esrgan_sample ok "1 image from github.com/xinntao/ESRGAN"
  else
    rmdir "$dest" "$DATA/sr/esrgan_sample" 2>/dev/null
    note esrgan_sample skipped "download failed"
  fi
  # ESRGAN: complete published results (Google Drive folder linked from the ESRGAN README)
  gdrive="https://drive.google.com/drive/folders/1iaM-c6EgT1FNoJAOKmDrK7YhEhtlKcLx"
  if [ -e "$DATA/.done_esrgan_results" ]; then
    note esrgan_results present "already unpacked"
  elif [ "$OFFLINE" = 1 ]; then
    note esrgan_results skipped "offline mode"
  elif ! "$PY" -c 'import gdown' >/dev/null 2>&1; then
    note esrgan_results skipped "Google Drive only; install gdown (pip install gdown) or unpack the folder $gdrive into data/sr/<set>/ESRGAN/ by hand"
  elif timeout 3600 "$PY" -m gdown --folder "$gdrive" -O "$DATA/extracted/esrgan_results" --quiet &&
       "${MAKE_INPUTS[@]}" collect --src "$DATA/extracted/esrgan_results" --data "$DATA" --kind sr --prefix esrgan_published; then
    touch "$DATA/.done_esrgan_results"
    note esrgan_results ok "from $gdrive"
  else
    note esrgan_results skipped "gdown could not fetch $gdrive (large folders need archives inside; unpack by hand into data/sr/<set>/ESRGAN/)"
  fi
else
  note optional_sets skipped "--skip-optional"
fi

# ---- derived images ----------------------------------------------------------------------
# bundled fallback set: x4 bicubic versions of the bundled RGB originals
"${MAKE_INPUTS[@]}" bicubic --src "$ROOT/tests/images" --dst "$DATA/sr/bundled/bicubic_x4" --exclude '(_y|x4bicubic_rgb)\.png$' ||
  note bundled failed "could not generate the bundled fallback set (is Pillow installed?)"

for dir in "$DATA"/hr/*/; do
  [ -d "$dir" ] || continue
  set=$(basename "$dir")
  [ "$(count_images "$dir")" -gt 0 ] || continue
  "${MAKE_INPUTS[@]}" bicubic --src "$dir" --dst "$DATA/sr/$set/bicubic_x4" || note "bicubic_$set" failed "bicubic generation failed"
done

if [ -s "$DATA/hr/div2k_val/0801.png" ]; then
  "${MAKE_INPUTS[@]}" inputs --hr "$DATA/hr/div2k_val/0801.png" --fallback-dir "$ROOT/tests/images" --out "$DATA/inputs"
else
  "${MAKE_INPUTS[@]}" inputs --fallback-dir "$ROOT/tests/images" --out "$DATA/inputs"
fi || { note inputs failed "could not create the fixed inputs"; exit 1; }
"${MAKE_INPUTS[@]}" inputs --quick --fallback-dir "$ROOT/tests/images" --out "$DATA/inputs" || exit 1
note inputs ok "data/inputs/manifest.json"

echo "Data status written to $STATUS"
