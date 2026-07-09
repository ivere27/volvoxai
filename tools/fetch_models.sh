#!/usr/bin/env bash
#
# fetch_models.sh — download source models and export them to the VolvoxAI blueprint
# format (config.json + model.safetensors [+ labels.txt / vocab.bin / merges.txt]).
#
# The models/ directory is .gitignored (weights are large), so this script reproduces
# it from public sources. The VolvoxAI *runtime* has no Python dependency — Python is
# only needed here, to convert source models. See tools/requirements-export.txt.
#
# Usage:
#   tools/fetch_models.sh                 # everything (EfficientDet variants + TinyStories)
#   tools/fetch_models.sh efficientdet    # only the EfficientDet-Lite0 detector (fp32/fp16/int8)
#   tools/fetch_models.sh tinystories     # only the TinyStories-1M language model
#
# Env:
#   PY=python3        # override the Python interpreter
#   ONLY=int8         # (efficientdet) restrict to one precision: int8 | float16 | float32
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
EXPORT="$ROOT/tools/export_safetensors.py"
EXPORT_TOK="$ROOT/tools/export_tokenizer.py"
LABELS_SRC="$ROOT/tools/assets/coco_labels.txt"
MODELS="$ROOT/models"

MP_BASE="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0"
TINYSTORIES_ID="roneneldan/TinyStories-1M"

log()  { printf '\033[1;36m[fetch-models]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[fetch-models]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fetch-models] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

require_python() {
  command -v "$PY" >/dev/null 2>&1 || die "Python interpreter '$PY' not found."
  "$PY" - "$@" <<'PYEOF' || die "Missing Python deps. Install: $PY -m pip install -r tools/requirements-export.txt"
import importlib, sys
need = sys.argv[1:]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("missing modules:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PYEOF
}

download() {  # url dest
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  log "download $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    die "Need curl or wget to download models."
  fi
}

# ------------------------------------------------------------------ EfficientDet
# precision -> (model-dir-suffix, local tflite filename matching config.json "source.tflite")
export_efficientdet_one() {
  local prec="$1" dir_suffix="$2" tflite_name="$3" dtype="$4"
  local dir="$MODELS/efficientdet_lite0_${dir_suffix}"
  local tflite="$dir/$tflite_name"
  download "$MP_BASE/$prec/latest/efficientdet_lite0.tflite" "$tflite"
  log "export ($prec) -> $dir/config.json + model.safetensors"
  "$PY" "$EXPORT" --model "$tflite" --out "$dir/model.safetensors" --weight-dtype "$dtype"
  cp "$LABELS_SRC" "$dir/labels.txt"
  log "labels -> $dir/labels.txt"
}

fetch_efficientdet() {
  [ -f "$LABELS_SRC" ] || die "Missing labels asset: $LABELS_SRC"
  require_python numpy flatbuffers safetensors torch
  local only="${ONLY:-}"
  if [ -z "$only" ] || [ "$only" = "int8" ];    then export_efficientdet_one int8    int8 efficientdet_lite0.tflite          auto;    fi
  if [ -z "$only" ] || [ "$only" = "float16" ]; then export_efficientdet_one float16 fp16 efficientdet_lite0_float16.tflite  float16; fi
  if [ -z "$only" ] || [ "$only" = "float32" ]; then export_efficientdet_one float32 fp32 efficientdet_lite0_float32.tflite  float32; fi
  log "EfficientDet-Lite0 ready under $MODELS/efficientdet_lite0_*"
}

# ------------------------------------------------------------------ TinyStories
fetch_tinystories() {
  require_python torch transformers safetensors numpy
  local dir="$MODELS/tinystories_1m"
  log "export $TINYSTORIES_ID -> $dir (config.json + model.safetensors + tokens/positions)"
  "$PY" "$EXPORT" --model "$TINYSTORIES_ID" --out "$dir/model.safetensors"
  log "export tokenizer -> $dir (vocab.bin + merges.txt)"
  "$PY" "$EXPORT_TOK" "$TINYSTORIES_ID" "$dir"
  log "TinyStories-1M ready under $dir"
}

# ------------------------------------------------------------------ main
main() {
  local what="${1:-all}"
  case "$what" in
    all)          fetch_efficientdet; fetch_tinystories ;;
    efficientdet) fetch_efficientdet ;;
    tinystories)  fetch_tinystories ;;
    *) die "Unknown target '$what' (use: all | efficientdet | tinystories)" ;;
  esac
  log "Done."
}

main "$@"
